#!/usr/bin/env python3
import os, json, threading, time
import findspark
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from azure.storage.blob import BlobServiceClient

app = FastAPI(title="Sales Prediction w/ Background Load")

# Globals (to be populated)
spark = model = gbt_model = assembler = explainer = None
feature_names = model_metrics = None
_loaded = False

# Paths (exactly as mounted in Colab)
MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/model")
METRICS_PATH = os.environ.get("METRICS_PATH", "/tmp/metrics.json")
# Add these imports
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
import os, tempfile, zipfile, shutil, glob, traceback

# ─── Configuration ─────────────────────────────────────────────────────────────
# (Set these in your Function App settings or local env)
STORAGE_ACCOUNT_NAME = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME", "mymlstorage1")
# If you prefer connection strings, set AZURE_STORAGE_CONNECTION_STRING (or use AzureWebJobsStorage)
STORAGE_CONNECTION_STRING = (
    os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    or os.environ.get("AzureWebJobsStorage")
    or None
)
CONTAINER_NAME = os.environ.get("MODEL_CONTAINER", "models")

# Local paths
MODEL_PATH   = os.environ.get("MODEL_PATH",   "/tmp/model")
METRICS_PATH = os.environ.get("METRICS_PATH", "/tmp/metrics.json")


def download_and_extract_model():
    try:
        # ─── Create BlobServiceClient ─────────────────────────────────────────────
        if STORAGE_CONNECTION_STRING:
            blob_service = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
        else:
            account_url = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
            credential  = DefaultAzureCredential()
            blob_service = BlobServiceClient(account_url=account_url, credential=credential)

        container_client = blob_service.get_container_client(CONTAINER_NAME)

        # ─── Download & extract both ZIPs ────────────────────────────────────────
        tmp_dir = tempfile.gettempdir()
        downloads = {
            "model":   "complete_sales_model-20250419T055616Z-001.zip",
            "results": "results-20250419T120204Z-001.zip"
        }

        for key, blob_name in downloads.items():
            zip_path = os.path.join(tmp_dir, blob_name)
            blob = container_client.get_blob_client(blob_name)
            with open(zip_path, "wb") as f:
                f.write(blob.download_blob().readall())

            extract_to = (
                os.path.dirname(MODEL_PATH)
                if key == "model"
                else os.path.join(tmp_dir, "results")
            )
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(extract_to)

        # ─── Locate and copy metrics.json ───────────────────────────────────────
        candidates = glob.glob(os.path.join(tmp_dir, "results", "**", "metrics.json"), recursive=True)
        if candidates:
            os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
            shutil.copy2(candidates[0], METRICS_PATH)
        else:
            print("⚠️  metrics.json not found in results archive")

        print("✅ Downloaded and extracted model files from Azure Storage")
        return True

    except Exception as e:
        print(f"❌ Failed to download and extract model files: {e}")
        traceback.print_exc()
        return False

class PredictionRequest(BaseModel):
    year: int; month: int; day: int
    holiday_type: str; COST: float; Transaction: float
    dcoilwtico: float; CusNo: float; QUANTITY: float
    BRAND: str; LOCATION: str; Cat: str
    BIZ_TYPE1: str; AREA_2: str; BRANCH: str

def _background_load():
    """Runs in a daemon thread to initialize Spark, the model, and SHAP."""
    global spark, model, gbt_model, assembler, explainer, feature_names, model_metrics, _loaded
    try:
        findspark.init()
        from pyspark.sql import SparkSession
        from pyspark.ml import PipelineModel
        from pyspark.ml.regression import GBTRegressionModel
        from pyspark.ml.feature import VectorAssembler
        import shap

        spark = SparkSession.builder \
            .appName("SalesPredictionAPI") \
            .config("spark.driver.memory", "4g") \
            .config("spark.sql.shuffle.partitions", "50") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("WARN")

        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model path not found: {MODEL_PATH}")
        model = PipelineModel.load(MODEL_PATH)

        gbt_model   = next(s for s in model.stages if isinstance(s, GBTRegressionModel))
        assembler   = next(s for s in model.stages if isinstance(s, VectorAssembler))
        feature_names = assembler.getInputCols()

        explainer = shap.TreeExplainer(gbt_model)

        try:
            with open(METRICS_PATH) as mf:
                model_metrics = json.load(mf)
        except:
            model_metrics = None

        _loaded = True
        print("✅ Background load complete.")
    except Exception as e:
        print(f"❌ Background load failed: {e}")

@app.on_event("startup")
def startup_event():
    # Launch the heavy initialization in the background
    thread = threading.Thread(target=_background_load, daemon=True)
    thread.start()

@app.get("/")
def root():
    return {
        "message": "API running.  “/metrics”, “/predict”, “/explain”.",
        "ready": _loaded
    }

@app.get("/metrics")
def get_metrics() -> Dict[str, float]:
    if not _loaded:
        raise HTTPException(503, "Model is still loading, try again in a bit")
    if model_metrics is None:
        raise HTTPException(500, "Metrics file not found or invalid")
    return model_metrics

@app.post("/predict")
def predict(req: PredictionRequest) -> Dict[str, float]:
    if not _loaded:
        raise HTTPException(503, "Model is still loading, try again in a bit")
    df = spark.createDataFrame([req.dict()])
    row = model.transform(df).select("prediction").collect()
    if not row:
        raise HTTPException(500, "Prediction failed")
    return {"prediction": float(row[0]["prediction"])}

@app.post("/explain")
def explain(req: PredictionRequest) -> Dict:
    if not _loaded:
        raise HTTPException(503, "Model is still loading, try again in a bit")
    df = spark.createDataFrame([req.dict()])
    out = model.transform(df).select("prediction","features").collect()[0]
    vals = explainer.shap_values(out["features"].toArray())
    return {
        "prediction": float(out["prediction"]),
        "base_value": float(explainer.expected_value),
        "shap_values": dict(zip(feature_names, vals.tolist()))
    }