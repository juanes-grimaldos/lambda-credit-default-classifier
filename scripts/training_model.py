from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import make_scorer, f1_score, precision_recall_curve, accuracy_score, precision_score, recall_score

class TraininModel():

    def __init__(self):
        pass


    def geting_data(self):
        data_uci = fetch_ucirepo(id=350) 
  
        # data (as pandas dataframes) 
        self.X = data_uci.data.features  # type: ignore
        self.y = data_uci.data.targets  # type: ignore
        pass

    def modelling(self):
        X=self.X
        y = self.y
        # ==============================================================================
        # 1. DEFINICIÓN DE COLUMNAS Y PREPROCESADOR
        # ==============================================================================
        X_cat = ['X3', 'X2', 'X4', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11']

        X_num = [col for col in X.columns if col not in X_cat]
        X_cat = ['X3', 'X2', 'X4', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11']
        X_num = [col for col in X.columns if col not in X_cat]

        preprocesamiento = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), X_num),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), X_cat)
            ]
        )

        # ==============================================================================
        # 2. DIVISIÓN ESTRATIFICADA  (ahora sobre X sin escalar)
        # ==============================================================================
        X_train, X_test, y_train, y_test = train_test_split(
            X,           # ← raw, sin escalar
            y,
            test_size=0.2,
            stratify=y
        )

        # ==============================================================================
        # 3. ENSAMBLAJE DENTRO DE UN PIPELINE
        # ==============================================================================
        clf1 = LogisticRegression(class_weight='balanced')
        clf2 = RandomForestClassifier(class_weight='balanced', n_jobs=-1)
        clf3 = HistGradientBoostingClassifier(class_weight='balanced')

        ensemble_voting = VotingClassifier(
            estimators=[('lr', clf1), ('rf', clf2), ('hgb', clf3)],
            voting='soft'
        )

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocesamiento),
            ('model', ensemble_voting)
        ])

        # ==============================================================================
        # 4. GRIDSEARCH  (los parámetros ahora llevan el prefijo 'model__')
        # ==============================================================================
        param_grid = {
            'model__rf__max_depth':        [6, 8, 10],
            'model__rf__min_samples_leaf': [3, 5, 10],
            'model__hgb__max_depth':       [3, 5, 7],
            'model__hgb__l2_regularization': [0.5, 1.5, 5.0]
        }

        cv_estrategia = StratifiedKFold(n_splits=5, shuffle=True)
        f1_scorer     = make_scorer(f1_score)

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=f1_scorer,
            cv=cv_estrategia,
            n_jobs=-1,
            verbose=1
        )

        print("Iniciando la optimización con K-Folds...")
        grid_search.fit(X_train, y_train)

        mejor_pipeline = grid_search.best_estimator_   # ← preprocessor + model juntos

        print("\n--- ¡Optimización Completa! ---")
        print("Mejores parámetros:", grid_search.best_params_)
        print(f"Mejor F1 en K-Folds: {grid_search.best_score_:.4f}")

        # ==============================================================================
        # 5. UMBRAL ÓPTIMO  (igual que antes)
        # ==============================================================================
        probabilidades_train = mejor_pipeline.predict_proba(X_train)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_train, probabilidades_train)

        f1_scores_curva = 2 * (precision * recall) / (precision + recall + 1e-10)
        indice_optimo   = np.argmax(f1_scores_curva)
        umbral_optimo   = thresholds[indice_optimo] if indice_optimo < len(thresholds) else thresholds[-1]

        print(f"Umbral óptimo: {umbral_optimo:.4f}")

        # ==============================================================================
        # 6. EVALUACIÓN FINAL
        # ==============================================================================
        probabilidades_test = mejor_pipeline.predict_proba(X_test)[:, 1]
        y_pred_final        = (probabilidades_test >= umbral_optimo).astype(int)

        print("\n=============================================")
        print("  MÉTRICAS FINALES EN TEST                  ")
        print("=============================================")
        print(f"Accuracy : {accuracy_score(y_test, y_pred_final):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred_final):.4f}")
        print(f"Recall   : {recall_score(y_test, y_pred_final):.4f}")
        print(f"F1 Score : {f1_score(y_test, y_pred_final):.4f}")

        # ==============================================================================
        # 7. GUARDAR  (pipeline completo + umbral)
        # ==============================================================================
        joblib.dump(mejor_pipeline, "src/pipeline_produccion.pkl")
        joblib.dump(umbral_optimo,  "src/umbral_optimo.pkl")
        print("\nArtifacts guardados: pipeline_produccion.pkl, umbral_optimo.pkl")
        pass
