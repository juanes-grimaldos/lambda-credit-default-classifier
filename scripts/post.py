from scripts.simulate_values import values_simulation
import requests
import json
import os
from src.predict import lambda_handler


# docker build -t zoomcamp-test .
# docker run -it --rm -p 9696:9696 zoomcamp-test

def local_running_lambda():
    # 1. Initialize simulation and get payload
    sim = values_simulation(N=30)
    payload = sim.generate_json_payload(as_string=False)

    # 2. Format the payload to mimic AWS API Gateway proxy integration
    lambda_payload = {
        "body": json.dumps(payload)
    }

    # 3. DIRECT FUNCTION CALL (Replacing requests.post)
    print("Calling local lambda_handler directly...")
    
    # We pass 'lambda_payload' as the event, and 'None' as the context
    lambda_response_dict = lambda_handler(lambda_payload, None) # type: ignore
    # to use remove comments above!

    # 4. Print the API output
    print("--- Lambda Output ---")
    print("Status Code:", lambda_response_dict.get("statusCode"))
    print("Lambda Wrapper Response:", lambda_response_dict)
    
    # Extract and parse the inner body to see your actual predictions
    if "body" in lambda_response_dict:
        actual_predictions = json.loads(lambda_response_dict["body"])
        print("Actual ML Predictions:", actual_predictions)


def running_lambda(local_docker:bool=False):
    # 1. Initialize simulation and get payload
    sim = values_simulation(N=30)
    payload = sim.generate_json_payload(as_string=False)

    # 2. Format the payload to mimic AWS API Gateway proxy integration
    # The actual payload must be serialized into a JSON string inside the 'body' key
    lambda_payload = {
        "body": json.dumps(payload)
    }

    # 3. Send it to the local Lambda Runtime Interface Emulator (Port 9000)
    url = os.getenv("POST_URL", "http://localhost:9000/2015-03-31/functions/function/invocations" )
    if local_docker:
        url = "http://localhost:9000/2015-03-31/functions/function/invocations"

    response = requests.post(url, json=lambda_payload)

    # 4. Print the API output
    print("--- Lambda Output ---")
    print("Status Code:", response.status_code)
    
    # Lambda returns a dict with statusCode, headers, and body
    lambda_response_dict = response.json()
    print("Lambda Wrapper Response:", lambda_response_dict)
    
    # Extract and parse the inner body to see your actual predictions
    if "body" in lambda_response_dict:
        actual_predictions = json.loads(lambda_response_dict["body"])
        print("Actual ML Predictions:", actual_predictions)
