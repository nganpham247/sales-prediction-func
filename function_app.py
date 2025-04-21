import azure.functions as func
import logging
import json
from serve import app, startup_event
from fastapi.responses import JSONResponse

app_func = func.FunctionApp()

# Initialize the FastAPI app at startup
startup_event()

@app_func.route(route="metrics", auth_level=func.AuthLevel.FUNCTION)
def metrics_function(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Metrics endpoint called')
    fastapi_req = app.router.get("/metrics")
    response = fastapi_req.endpoint({})
    return func.HttpResponse(json.dumps(response), mimetype="application/json")

@app_func.route(route="predict", auth_level=func.AuthLevel.FUNCTION)
async def predict_function(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Predict endpoint called')
    try:
        request_body = req.get_json()
        # Pass the JSON data to your FastAPI endpoint
        response = await app.router.get("/predict").endpoint(request_body)
        return func.HttpResponse(json.dumps(response), mimetype="application/json")
    except ValueError:
        return func.HttpResponse("Invalid request body", status_code=400)

@app_func.route(route="explain", auth_level=func.AuthLevel.FUNCTION)
async def explain_function(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Explain endpoint called')
    try:
        request_body = req.get_json()
        # Pass the JSON data to your FastAPI endpoint
        response = await app.router.get("/explain").endpoint(request_body)
        return func.HttpResponse(json.dumps(response), mimetype="application/json")
    except ValueError:
        return func.HttpResponse("Invalid request body", status_code=400)