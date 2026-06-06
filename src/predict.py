import json
import joblib
import pandas as pd
from pathlib import Path

# Global scope: Loaded once during Lambda initialization (Warm Starts)
BASE_DIR = Path(__file__).parent

pipeline = joblib.load(BASE_DIR / "pipeline_produccion.pkl")
umbral_optimo = joblib.load( BASE_DIR / "umbral_optimo.pkl")

def lambda_handler(event, context):
    try:
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', event) # Fallback if event IS the payload

        # 2. Process data and predict
        df = pd.DataFrame(body)
        probabilidades = pipeline.predict_proba(df)[:, 1]
        predicciones = (probabilidades >= umbral_optimo).astype(int)

        results = {
            'probs': probabilidades.tolist(),
            'predict': predicciones.tolist(),
            'opt_prob': float(umbral_optimo) # Ensure it's JSON serializable
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