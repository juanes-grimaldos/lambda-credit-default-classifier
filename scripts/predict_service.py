from flask import Flask
import joblib
from scripts.simulate_values import values_simulation
from flask import request
from flask import jsonify
import pandas as pd


app = Flask('default_score')


@app.route('/predict', methods=['POST'])
def predict():
    pipeline      = joblib.load("pipeline_produccion.pkl")
    umbral_optimo = joblib.load("umbral_optimo.pkl")

    df = pd.DataFrame(request.get_json())
    probabilidades = pipeline.predict_proba(df)[:, 1]
    predicciones   = (probabilidades >= umbral_optimo).astype(int)

    results = {
        'probs': probabilidades.tolist(),
        'predict': predicciones.tolist(),
        'opt_prob': umbral_optimo
    }

    print("the probabilities are: ", probabilidades, " umbral optimo F1 score:", umbral_optimo)
    print('\n')
    print("the values are: ", predicciones)
    return jsonify(results)




