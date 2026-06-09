import os
import joblib
import logging
import warnings
import numpy as np
import pandas as pd
import optuna
import mlflow
import mlflow.sklearn
from ucimlrepo import fetch_ucirepo

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ==============================================================================
# CONFIGURATION & LOGGING SETUP
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress specific sklearn/lightgbm warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Found unknown categories")
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

# Suppress Optuna internal logs to avoid duplicate terminal clutter
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelTrainer:

    def __init__(self):
        self.X = pd.DataFrame({})
        self.y = np.array([])
        self.preprocessor = None
        self.X_train, self.X_test = pd.DataFrame({}), pd.DataFrame({})
        self.y_train, self.y_test = np.array([]), np.array([])

    def get_data(self):
        """Fetches data from the UCI repository and performs feature engineering."""
        logger.info("Loading data from UCI Repository...")
        data_uci = fetch_ucirepo(id=350) 
        
        X_raw = data_uci.data.features.copy()  # type: ignore
        self.y = np.ravel(data_uci.data.targets).astype(int)  # type: ignore

        logger.info("Performing feature engineering...")
        # 1. Over the 6 months has the client ever been late
        X_raw['ever_late'] = (X_raw[['X6', 'X7', 'X8', 'X9', 'X10', 'X11']] >= 1).any(axis=1).astype(int)

        # 2. The max delay observed
        X_raw['max_delay'] = X_raw[['X6', 'X7', 'X8', 'X9', 'X10', 'X11']].clip(lower=0).max(axis=1)

        # 3. Is this improving or not?
        X_raw['delay_trend'] = X_raw['X11'] - X_raw['X6']  

        # 4. Good performance ratio over the past 6 months
        good_mask = (X_raw[['X6', 'X7', 'X8', 'X9', 'X10', 'X11']] <= -1)
        X_raw['good_payment_ratio'] = good_mask.sum(axis=1) / 6

        self.X = X_raw

    def _build_preprocessor(self):
        """Identifies column groups and constructs the tailored preprocessor pipeline."""
        # Baseline Categorical and Repayment feature definitions
        X_cat = ['X3', 'X2', 'X4', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11']
        X_repayment = ['X6', 'X7', 'X8', 'X9', 'X10', 'X11']
        X_derived = ['ever_late', 'max_delay', 'delay_trend', 'good_payment_ratio']
        
        # Filter original categorical features to remove those moving to ordinal or missing
        X_cat_clean = [col for col in X_cat if col in self.X.columns and col not in X_repayment]
        
        # Numeric features are anything left over that isn't categorical, repayment, or derived
        excluded_cols = X_cat_clean + X_repayment + X_derived
        X_num = [col for col in self.X.columns if col not in excluded_cols]

        # Define ordinal repayment categories mapping
        single_repayment_order = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        repayment_categories = [single_repayment_order for _ in range(len(X_repayment))]

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), X_num),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), X_cat_clean),
                ('rep', OrdinalEncoder(categories=repayment_categories, handle_unknown='use_encoded_value', unknown_value=-1), X_repayment),
                ('der', StandardScaler(), X_derived),
            ]
        )
    
    def objective(self, trial):
        lgbm_lr = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        lgbm_n_est = trial.suggest_int("n_estimators", 50, 200)
        lgbm_num_leaves = trial.suggest_int("num_leaves", 15, 63)

        clf = LGBMClassifier(
            learning_rate=lgbm_lr,
            n_estimators=lgbm_n_est,
            num_leaves=lgbm_num_leaves,
            class_weight="balanced",
            n_jobs=-1,
            verbosity=-1
        )

        trial_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', clf)
        ])

        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Using roc_auc to find the best general performance configuration
        score = cross_val_score(
            trial_pipeline, self.X_train, self.y_train, 
            cv=cv_strategy, scoring="roc_auc", n_jobs=-1
        ).mean()

        # Keeps tracking perfectly to a clean single score log row
        logger.info(
            f"Trial {trial.number:02d} | "
            f"ROC AUC Score: {score:.4f} |"
        )
        return score

    def run_tuning_and_evaluation(self, n_trials=30):
        """Executes splitting, Optuna tuning, threshold optimization, and evaluation."""
        if self.X.empty or self.y.size == 0:
            raise ValueError("Data not loaded. Please run `get_data()` first.")

        # 1. Setup preprocessing and data splits
        self._build_preprocessor()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
        )

        # 2. Define Optuna Objective Function nested to access local splits safely


        # 3. Optimize Hyperparameters
        logger.info(f"Starting hyperparameter optimization with Optuna using {n_trials} trials...")
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)

        logger.info("Optimization complete!")
        logger.info(f"Best Cross-Validation ROC AUC: {study.best_value:.4f}")
        logger.info(f"Best Hyperparameters found: {study.best_params}")

        # 4. Train final model with the optimal parameters
        logger.info("Training final pipeline using optimal hyperparameters...")
        best_clf = LGBMClassifier(
            **study.best_params,
            class_weight="balanced",
            n_jobs=-1,
            verbosity=-1
        )
        
        best = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', best_clf)
        ])
        
        best.fit(self.X_train, self.y_train)

        # 5. Calculate Optimal Threshold (Using Test Split to prevent data leakage)
        probabilidades_test = best.predict_proba(self.X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(self.y_test, probabilidades_test)

        f1_scores_curva = 2 * (precision * recall) / np.maximum((precision + recall), 1e-10)
        indice_optimo = np.argmax(f1_scores_curva[:-1])  # Align lengths correctly
        opt_threshold = thresholds[indice_optimo]

        logger.info(f"Optimal classification threshold calculated (Max F1): {opt_threshold:.4f}")

        # 6. Final Performance Evaluation
        y_pred_final = (probabilidades_test >= opt_threshold).astype(int)

        logger.info("=============================================")
        logger.info("           FINAL OUT-OF-SAMPLE METRICS       ")
        logger.info("=============================================")
        logger.info(f"Accuracy  : {accuracy_score(self.y_test, y_pred_final):.4f}")
        logger.info(f"Precision : {precision_score(self.y_test, y_pred_final):.4f}")
        logger.info(f"Recall    : {recall_score(self.y_test, y_pred_final):.4f}")
        logger.info(f"F1 Score  : {f1_score(self.y_test, y_pred_final):.4f}")
        logger.info(f"ROC AUC   : {roc_auc_score(self.y_test, probabilidades_test):.4f}")



        pipeline_path = "src/product_pipeline.pkl"
        threshold_path = "src/opt_threshold.pkl"
        
        # Calculate the ROC AUC score of the newly trained model on the test probabilities
        new_model_auc = roc_auc_score(self.y_test, probabilidades_test)
        save_new_model = True

        # Check if a production model already exists to compare against
        if os.path.exists(pipeline_path):
            logger.info("An existing production model was found. Loading for comparison...")
            try:
                old_pipeline = joblib.load(pipeline_path)
                
                # Predict probabilities using the old model on the *current* test data
                probabilidades_old = old_pipeline.predict_proba(self.X_test)[:, 1]
                old_model_auc = roc_auc_score(self.y_test, probabilidades_old)
                
                logger.info(f"Existing Production Model ROC AUC: {old_model_auc:.4f}")
                logger.info(f"Newly Trained Model ROC AUC:       {new_model_auc:.4f}")
                
                # Check performance
                if old_model_auc >= new_model_auc:
                    logger.warning(
                        f"The existing production model performed better or equal "
                        f"({old_model_auc:.4f} >= {new_model_auc:.4f}). Aborting save operation."
                    )
                    save_new_model = False
                else:
                    logger.info("The newly trained model outperforms the current production model. Proceeding to overwrite...")
                    
            except Exception as e:
                logger.error(f"Failed to evaluate the existing production model due to error: {e}. Defaulting to save the new model.")

        # Save the new artifacts if conditions are met
        if save_new_model:
            os.makedirs("src", exist_ok=True)
            joblib.dump(best, pipeline_path)
            joblib.dump(opt_threshold, threshold_path)

            logger.info("=============================================")
            logger.info("           Loding artifacts to mlflow    ")
            logger.info("=============================================")
            try:
                import mlflow
                import mlflow.sklearn

                # Optional: Point to a local or remote server tracking URI
                # mlflow.set_tracking_uri("http://localhost:5000") 
                mlflow.set_experiment("Credit_Card_Default_Production")

                # Define explicit column names used for tracking
                X_cat = ['X3', 'X2', 'X4']
                X_repayment = ['X6', 'X7', 'X8', 'X9', 'X10', 'X11']
                X_derived = ['ever_late', 'max_delay', 'delay_trend', 'good_payment_ratio']
                all_features_used = list(self.X.columns)

                with mlflow.start_run(run_name="Production_Deployment_Update"):
                    # 1. Log metadata about the model framework
                    mlflow.log_param("model_type", "LGBMClassifier")
                    mlflow.log_param("optimal_threshold", opt_threshold)
                    
                    # 2. Log structural feature tracking 
                    mlflow.log_param("total_features_count", len(all_features_used))
                    mlflow.log_param("categorical_features", X_cat)
                    mlflow.log_param("repayment_features", X_repayment)
                    mlflow.log_param("derived_features", X_derived)
                    
                    # 3. Log the final optimized hyperparameters chosen by Optuna
                    mlflow.log_params(study.best_params)

                    # 4. Log the real out-of-sample metrics achieved on your test set
                    mlflow.log_metric("accuracy", accuracy_score(self.y_test, y_pred_final))
                    mlflow.log_metric("precision", precision_score(self.y_test, y_pred_final))
                    mlflow.log_metric("recall", recall_score(self.y_test, y_pred_final))
                    mlflow.log_metric("f1_score", f1_score(self.y_test, y_pred_final))
                    mlflow.log_metric("test_roc_auc", new_model_auc)

                    # 5. Save the pipeline binary directly to the MLflow local run registry
                    mlflow.sklearn.log_model(best, artifact_path="production_pipeline")
                    
                logger.info("Successfully registered the new champion model to MLflow server.")
                
            except ImportError:
                logger.error("MLflow library is not installed in this environment. Skipping tracking log.")
            except Exception as e:
                logger.error(f"Failed to push logs to MLflow due to error: {e}")

            logger.info(f"Production artifacts successfully updated at: {pipeline_path} and {threshold_path}")
        else:
            logger.info("Deployment skipped. Retained previous production artifacts.")


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.get_data()
    trainer.run_tuning_and_evaluation(n_trials=30)