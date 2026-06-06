from simulate_values import values_simulation
import requests
import json
import os


# docker build -t zoomcamp-test .
# docker run -it --rm -p 9696:9696 zoomcamp-test

def running_to_flaks():
    # 1. Initialize simulation and get payload
    sim = values_simulation(N=10)
    payload = sim.generate_json_payload(as_string=False)

    # 2. Send it to your running Flask application
    url = os.getenv("POST_URL", "http://localhost:9696/predict" )
    response = requests.post(url, json=payload) # type: ignore

    # 3. Print the API output
    print("Status Code:", response.status_code)
    print("Predictions:", response.json())

def running_lambda():
    # 1. Initialize simulation and get payload
    sim = values_simulation(N=3000)
    payload = sim.generate_json_payload(as_string=False)

    # 2. Format the payload to mimic AWS API Gateway proxy integration
    # The actual payload must be serialized into a JSON string inside the 'body' key
    lambda_payload = {
        "body": json.dumps(payload)
    }

    # 3. Send it to the local Lambda Runtime Interface Emulator (Port 9000)
    url = os.getenv("POST_URL", "http://localhost:9000/2015-03-31/functions/function/invocations" )

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

if __name__ == '__main__':
    running_lambda()