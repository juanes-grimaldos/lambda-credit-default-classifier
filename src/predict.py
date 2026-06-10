import json
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent

pipeline = joblib.load(BASE_DIR / "product_pipeline.pkl")
threshold = joblib.load( BASE_DIR / "opt_threshold.pkl")

def lambda_handler(event, context):
    try:
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', event) # Fallback if event IS the payload

        # 2. Process data and predict
        df = pd.DataFrame(body)
        probs = pipeline.predict_proba(df)[:, 1]
        preds = (probs >= threshold).astype(int)

        results = {
            'probs': probs.tolist(),
            'predict': preds.tolist(),
            'opt_prob': float(threshold) # Ensure it's JSON serializable
        }


        # 3. Return API Gateway compatible response
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps(results)
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }