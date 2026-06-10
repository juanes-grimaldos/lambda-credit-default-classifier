# Credit Default Classifier — AWS Lambda Inference Service

A serverless ML inference service that predicts credit card payment default probability using a LightGBM model deployed as an AWS Lambda container image. Built on the [UCI Default of Credit Card Clients dataset (ID 350)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

---

## Overview

This project trains a tuned LightGBM classifier with a full scikit-learn preprocessing pipeline incorporating repayment-based feature engineering, and serves real-time predictions via an AWS Lambda function exposed through API Gateway. The optimal classification threshold is derived from the Precision-Recall curve to maximize F1 score.

**Key design decisions:**
- Models are loaded once at the global scope to take advantage of Lambda warm starts, minimizing cold-start latency on subsequent invocations.
- The preprocessing pipeline (scaling + encoding) is serialized together with the model, guaranteeing identical transformations between training and inference.
- Threshold optimization via Precision-Recall curve avoids the default 0.5 cutoff, which is suboptimal on the class-imbalanced credit default dataset.

---

## Architecture

```
POST /predict
      │
      ▼
 API Gateway  ──►  AWS Lambda (Container Image)
                        │
                        ├── product_pipeline.pkl   (preprocessor + LightGBM)
                        ├── opt_threshold.pkl       (optimal threshold)
                        └── predict.py              (lambda_handler)
```

The Lambda function is packaged as a **Docker container image** using the official AWS Lambda Python 3.13 base image (`public.ecr.aws/lambda/python:3.13`).

---

## Model Details

| Component | Details |
|---|---|
| Dataset | UCI Default of Credit Card Clients (30,000 samples, 23 features + 4 derived) |
| Problem | Binary classification — predicts next-month payment default |
| Model | `LGBMClassifier` with Optuna-tuned hyperparameters |
| Class imbalance | `class_weight='balanced'` |
| Preprocessing | `RobustScaler` (numerical + derived) + `OneHotEncoder` (categorical, `drop='first'`) + `OrdinalEncoder` (repayment status X6–X11) |
| Hyperparameter tuning | Optuna with 5-fold `StratifiedKFold`, optimizing ROC AUC (30 trials) |
| Threshold selection | Optimal cutoff from Precision-Recall curve on test split (maximizes F1) |
| Serialization | `joblib` — pipeline and threshold saved separately |
| Experiment tracking | MLflow — metrics, params, and artifacts logged per training run |

---

## Project Structure

```
lambda-credit-default-classifier/
├── src/
│   ├── predict.py                    # Lambda handler (inference entry point)
│   ├── product_pipeline.pkl          # Trained preprocessing + LightGBM pipeline
│   └── opt_threshold.pkl             # Optimal classification threshold
├── scripts/
│   ├── training_model.py             # Model training, tuning, and artifact export
│   ├── simulate_values.py            # Synthetic payload generator (matches UCI distribution)
│   └── post.py                       # Test client — targets Flask or Lambda RIE
├── Dockerfile                        # Lambda container image definition
├── Pipfile / Pipfile.lock            # Dependency management
├── requirements.txt                  # Pinned dependencies (generated from Pipfile)
└── notebook.ipynb                    # EDA, feature engineering, and model selection notebook
```

---

## API Reference

### `POST /predict`

Accepts a batch of records and returns default probabilities and binary predictions.

**Request body** — list of records with features `X1`–`X23`:

```json
[
  {
    "X1": 20000, "X2": 2, "X3": 2, "X4": 1, "X5": 24,
    "X6": 2, "X7": 2, "X8": -1, "X9": -1, "X10": -2, "X11": -2,
    "X12": 3913, "X13": 3102, "X14": 689, "X15": 0, "X16": 0, "X17": 0,
    "X18": 689, "X19": 0, "X20": 0, "X21": 0, "X22": 0, "X23": 0
  }
]
```

**Response:**

```json
{
  "probs":    [0.312],
  "predict":  [0],
  "opt_prob": 0.4237
}
```

| Field | Type | Description |
|---|---|---|
| `probs` | `list[float]` | Default probability per record |
| `predict` | `list[int]` | Binary prediction (1 = default, 0 = no default) |
| `opt_prob` | `float` | Threshold used for classification |

---

## Local testing


### Prerequisites

- Python 3.13+ with `pipenv`


### 1. Clone and install dependencies

```bash
git clone https://github.com/juanes-grimaldos/lambda-credit-default-classifier.git
cd lambda-credit-default-classifier
pipenv install
```
Note: the pipfile and pipfile.lock have only the packages requered for the lambda function to run, you will need to add other packages to run and to test.
To test the function locally run the following command: 
```bash
pipenv install requests
pipenv run python -c "from scripts.post import local_running_lambda; local_running_lambda()"
pipenv uninstall requests
```
This generates a synthetic batch matching the UCI feature distribution and posts it to the local Lambda RIE endpoint.
It is encouraged to uninstall requests since it is not supposted to be in the image.

## Local Development

### Prerequisites

- Docker
- Python 3.13+ with `pipenv`

### 1. Clone and install dependencies

```bash
git clone https://github.com/juanes-grimaldos/lambda-credit-default-classifier.git
cd lambda-credit-default-classifier
pipenv install
```


### 3. Build the container image

```bash
docker build --provenance=false -t credit-default-classifier:latest .
```

### 4. Run locally with the Lambda Runtime Interface Emulator

```bash
docker run --rm -p 9000:8080 credit-default-classifier
```

### 5. Test with a simulated payload

```bash
pipenv install requests
pipenv run python -c "from scripts.post import running_lambda; running_lambda(True)"
```

This generates a synthetic batch matching the UCI feature distribution and posts it to the local Lambda RIE endpoint.


---

## Cloud Deployment

### Build and push to Amazon ECR


Next, you need to tag your local Docker image so AWS knows exactly where to put it. 

**Important:** The final part of the URL must exactly match the name of the repository you created in AWS ECR.

important clarification: 
<local-docker-image-name>: The name you gave your image when you ran docker build (e.g., credit-lambda).

<account_id>: Your 12-digit AWS Account ID.

<region>: Your AWS region (e.g., us-east-2).

<your-ecr-repo-name>: The exact name of your ECR repository (e.g., lambda-images).
```bash
# Authenticate with ECR
aws ecr get-login-password --region <region> | \
  docker login --username AWS --password-stdin <account_id>.dkr.ecr.<region>.amazonaws.com

# Tag and push
docker tag <local-docker-image-name>:latest <account_id>.dkr.ecr.<region>.amazonaws.com/<your-ecr-repo-name>:latest

docker push <account_id>.dkr.ecr.<region>.amazonaws.com/<your-ecr-repo-name>:latest
```

### Create the Lambda function
* the user must have the explicit permission to create in lambda:

```bash
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}

```

```bash
aws lambda create-function \
  --function-name <your-lambda-function-name> \
  --package-type Image \
  --code ImageUri=<account_id>.dkr.ecr.<region>.amazonaws.com/<your-ecr-repo-name>:latest \
  --role arn:aws:iam::<account_id>:role/<lambda-execution-role>
```

<your-lambda-function-name>: What you want to call your function in AWS (e.g., predict-function).

<your-ecr-repo-name>: The ECR repository you pushed to in the previous step (e.g., lambda-images).

<lambda-execution-role>: The name of the IAM role that gives your Lambda permission to run.

If you want to update the function because you discovered a new and better model: 

```bash
aws lambda update-function-code \
    --function-name <your-lambda-function-name> \
    --image-uri <account_id>.dkr.ecr.<region>.amazonaws.com/<your-ecr-repo-name>:latest
```



###  Test with a simulated payload
remember to change the Env variable in .env POST_URL for your AWS url, and to install requests if not install previously

```bash
pipenv install requests
pipenv run python -c "from scripts.post import running_lambda; running_lambda()"
```

## run training and mlflow for tracking improvements in the model selection

First we need to install other packages that are not supposed to be in the docker image:
```batch
pipenv install optuna ucimlrepo "mlflow>3.0.0"
```

Then we cna run the script:
```batch
pipenv run python -c "from scripts.training_model import ModelTrainer; trainer = ModelTrainer(); trainer.get_data(); trainer.run_tuning_and_evaluation(n_trials=30)"
```

After the random optimization, a new better model may or may not be better than the model already optimized. 

```batch
mlflow ui
```

generally the mlflow panel will be on http://127.0.0.1:5000 
There yuou can check for the models, their performance and track if you want to change or include other feature as well.

## notebook analysis (EDA, model selection, feature eng)

finally you can run the notebook, but first we need to add other pacakges:

```batch
pipenv install seaborn, xgboost
```
When you try to run the notebook, it will ask for other dependencies for the kernel.

### Expose via API Gateway

Configure an **HTTP API** or **REST API** with a `POST /predict` route proxied to the Lambda function.

---

## Model Design & Performance

### Feature Engineering

Four derived features are constructed from the six monthly repayment status columns (X6–X11) before any model sees the data:

| Feature | Description |
|---|---|
| `ever_late` | Binary flag — 1 if the client was ever late (status ≥ 1) across any of the 6 months |
| `max_delay` | Maximum delay severity observed across the 6 months (negative values clipped to 0) |
| `delay_trend` | Direction of change: `X11 − X6` — positive means worsening repayment behavior |
| `good_payment_ratio` | Fraction of months with early/on-time payment (status ≤ −1), out of 6 |

These features are motivated by Spearman correlation and mutual information analysis against the target: repayment status columns are the strongest predictors in the dataset, and these aggregations capture behavioral patterns that individual monthly columns miss.

### Preprocessing Pipeline

Raw features are split into four groups processed in parallel via `ColumnTransformer`:

| Group | Columns | Transformer |
|---|---|---|
| Numerical | `X1`, `X5`, `X12`–`X23` | `RobustScaler` — robust to the heavy outliers in bill/payment amounts |
| Categorical | `X2`, `X3`, `X4` | `OneHotEncoder` with `drop='first'` to avoid multicollinearity |
| Repayment status | `X6`–`X11` | `OrdinalEncoder` with explicit category ordering `[−2, −1, 0, 1, …, 10]` — preserves the severity scale |
| Derived features | `ever_late`, `max_delay`, `delay_trend`, `good_payment_ratio` | `RobustScaler` — `max_delay` inherits the outlier profile of X6–X11 |

The entire transformer is embedded in a `Pipeline` so preprocessing and inference are always applied as a single atomic step, eliminating any risk of train/serve skew.

### Model Selection

Six candidate models were evaluated via 5-fold `StratifiedKFold` cross-validation on the full feature set (original + derived), scored on ROC AUC, F1, Precision, Recall, Accuracy, overfit gap, training time, and serialized size:

| Model | ROC AUC | F1 | Recall | Overfit Gap |
|---|---|---|---|---|
| Logistic Regression | — | — | — | low |
| Random Forest | — | — | — | moderate |
| HistGradientBoosting | — | — | — | low |
| XGBoost | — | — | — | low |
| **LightGBM** | **best** | **best** | competitive | low |
| VotingClassifier (LR + HGB + LGBM) | competitive | competitive | competitive | low |

LightGBM consistently led on ROC AUC and F1 across all three feature set configurations tested in the notebook (full features, MI > 0.01, MI > 0.005). The soft-voting ensemble was competitive but added serialization overhead without a meaningful performance gain, so **LightGBM was selected as the production model**.

### Hyperparameter Tuning

`Optuna` searches over LightGBM hyperparameters using 5-fold `StratifiedKFold`, optimizing ROC AUC over 30 trials. The search space covers:

| Parameter | Search range |
|---|---|
| `learning_rate` | log-uniform [0.01, 0.2] |
| `n_estimators` | integer [50, 200] |
| `num_leaves` | integer [15, 63] |

The best trial configuration found:

```python
LGBMClassifier(
    learning_rate=0.07088661041095881,
    n_estimators=68,
    num_leaves=30,
    class_weight='balanced',
    random_state=42
)
```

### Threshold Calibration

Rather than using the default 0.5 cutoff, the optimal threshold is derived from the Precision-Recall curve on the **test split** (not training data) by maximizing F1 score:

```python
precision, recall, thresholds = precision_recall_curve(y_test, probs)
f1_scores = 2 * (precision * recall) / np.maximum((precision + recall), 1e-10)
umbral_optimo = thresholds[np.argmax(f1_scores[:-1])]
```

This threshold is serialized to `opt_threshold.pkl` and applied at inference time, keeping the decision boundary decoupled from the model artifact.

### Model Promotion Guard

The training script compares the newly trained model against the artifact already in `src/product_pipeline.pkl` before overwriting it. If the existing production model achieves equal or higher ROC AUC on the same test split, the save is aborted and the previous artifacts are retained. This prevents accidental regressions when re-running training with different random seeds or data splits.

---

### Performance

Evaluated on an 80/20 stratified train-test split (24,000 / 6,000 samples) with the Optuna-tuned LightGBM and the derived repayment features.

| Metric | Test |
|---|---|
| **ROC AUC** | tracked via MLflow |
| **Accuracy** | tracked via MLflow |
| **Precision** | tracked via MLflow |
| **Recall** | tracked via MLflow |
| **F1 Score** | tracked via MLflow |

Run `mlflow ui` after training to inspect the latest metrics for the registered production run.

**Key observations:**

- LightGBM outperformed the previous soft-voting ensemble (LR + RF + HGB) across all feature set configurations tested in the notebook, with better ROC AUC and comparable or better F1.
- `RobustScaler` replaces `StandardScaler` throughout — bill amounts and payment amounts carry extreme outliers that inflate variance and degrade standardization-based models.
- `OrdinalEncoder` on X6–X11 replaces `OneHotEncoder`, preserving the ordinal severity structure of the repayment codes (−2 = paid in advance through 8 = 8 months late).
- The four derived repayment features (`ever_late`, `max_delay`, `delay_trend`, `good_payment_ratio`) ranked among the top predictors by mutual information, providing behavioral signal that individual monthly columns do not capture independently.
- The ~0.22 class imbalance (defaulters vs. non-defaulters) is handled via `class_weight='balanced'` rather than resampling, avoiding information leakage into cross-validation folds.

---

## Dataset

**UCI Default of Credit Card Clients** — 30,000 credit card holders in Taiwan (October 2005).

| Feature Range | Description |
|---|---|
| `X1` | Credit limit (NT dollar) |
| `X2` | Sex (1 = male, 2 = female) |
| `X3` | Education level |
| `X4` | Marital status |
| `X5` | Age |
| `X6`–`X11` | Repayment status (months April–September 2005) — encoded ordinally: −2 (no consumption) through 8 (8 months late) |
| `X12`–`X17` | Bill statement amount (months April–September 2005) |
| `X18`–`X23` | Previous payment amount (months April–September 2005) |

Target: `Y` — default payment next month (1 = yes, 0 = no).

---

## License

MIT
