# Credit Default Classifier — AWS Lambda Inference Service

A serverless ML inference service that predicts credit card payment default probability using an ensemble model deployed as an AWS Lambda container image. Built on the [UCI Default of Credit Card Clients dataset (ID 350)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

---

## Overview

This project trains a soft-voting ensemble classifier (Logistic Regression + Random Forest + HistGradientBoosting), packages it alongside a full scikit-learn preprocessing pipeline, and serves real-time predictions via an AWS Lambda function exposed through API Gateway. The optimal classification threshold is tuned on the Precision-Recall curve to maximize F1 score.

**Key design decisions:**
- Models are loaded once at the global scope to take advantage of Lambda warm starts, minimizing cold-start latency on subsequent invocations.
- The preprocessing pipeline (scaling + one-hot encoding) is serialized together with the ensemble, guaranteeing identical transformations between training and inference.
- Threshold optimization via Precision-Recall curve avoids the default 0.5 cutoff, which is suboptimal on the class-imbalanced credit default dataset.

---

## Architecture

```
POST /predict
      │
      ▼
 API Gateway  ──►  AWS Lambda (Container Image)
                        │
                        ├── pipeline_produccion.pkl   (preprocessor + ensemble)
                        ├── umbral_optimo.pkl          (optimal threshold)
                        └── predict.py                 (lambda_handler)
```

The Lambda function is packaged as a **Docker container image** using the official AWS Lambda Python 3.13 base image (`public.ecr.aws/lambda/python:3.13`).

---

## Model Details

| Component | Details |
|---|---|
| Dataset | UCI Default of Credit Card Clients (30,000 samples, 23 features) |
| Problem | Binary classification — predicts next-month payment default |
| Ensemble | Soft-voting: Logistic Regression + Random Forest + HistGradientBoosting |
| Class imbalance | `class_weight='balanced'` on all base estimators |
| Preprocessing | `StandardScaler` (numerical) + `OneHotEncoder` (categorical, `drop='first'`) |
| Hyperparameter tuning | `GridSearchCV` with 5-fold `StratifiedKFold`, optimizing F1 |
| Threshold selection | Optimal cutoff from Precision-Recall curve (maximizes F1) |
| Serialization | `joblib` — pipeline and threshold saved separately |

---

## Project Structure

```
lambda-credit-default-classifier/
├── src/
│   ├── predict.py                    # Lambda handler (inference entry point)
│   ├── pipeline_produccion.pkl       # Trained preprocessing + ensemble pipeline
│   └── umbral_optimo.pkl             # Optimal classification threshold
├── scripts/
│   ├── training_model.py             # Model training, tuning, and artifact export
│   ├── simulate_values.py            # Synthetic payload generator (matches UCI distribution)
│   ├── predict_service.py            # Local Flask server for development testing
│   └── post.py                       # Test client — targets Flask or Lambda RIE
├── Dockerfile                        # Lambda container image definition
├── Pipfile / Pipfile.lock            # Dependency management
├── requirements.txt                  # Pinned dependencies (generated from Pipfile)
└── eda.ipynb                         # Exploratory data analysis notebook
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

### 2. (Optional) Retrain the model

```bash
pipenv run python scripts/training_model.py
```

This fetches the UCI dataset, trains the ensemble with GridSearchCV, computes the optimal threshold, and saves artifacts to `src/`.

### 3. Build the container image

```bash
docker build -t credit-default-classifier .
```

### 4. Run locally with the Lambda Runtime Interface Emulator

```bash
docker run --rm -p 9000:8080 credit-default-classifier
```

### 5. Test with a simulated payload

```bash
pipenv run python scripts/post.py
```

This generates a synthetic batch (N=3000) matching the UCI feature distribution and posts it to the local Lambda RIE endpoint.

### 6. Run the Flask development server (alternative)

```bash
pipenv run python scripts/predict_service.py
# Then, in another terminal:
POST_URL=http://localhost:9696/predict pipenv run python scripts/post.py
```

---

## Deployment

### Build and push to Amazon ECR

```bash
# Authenticate with ECR
aws ecr get-login-password --region <region> | \
  docker login --username AWS --password-stdin <account_id>.dkr.ecr.<region>.amazonaws.com

# Tag and push
docker tag credit-default-classifier:latest \
  <account_id>.dkr.ecr.<region>.amazonaws.com/credit-default-classifier:latest

docker push <account_id>.dkr.ecr.<region>.amazonaws.com/credit-default-classifier:latest
```

### Create the Lambda function

```bash
aws lambda create-function \
  --function-name credit-default-classifier \
  --package-type Image \
  --code ImageUri=<account_id>.dkr.ecr.<region>.amazonaws.com/credit-default-classifier:latest \
  --role arn:aws:iam::<account_id>:role/<lambda-execution-role>
```

### Expose via API Gateway

Configure an **HTTP API** or **REST API** with a `POST /predict` route proxied to the Lambda function.

---

## Model Design & Performance

### Design

The model follows a three-stage design: preprocessing, ensemble construction, and threshold calibration.

**Stage 1 — Preprocessing (`ColumnTransformer`)**

Raw features are split into two groups processed in parallel:

- Numerical (`X1`, `X5`, `X12`–`X23`): `StandardScaler` — zero mean, unit variance.
- Categorical (`X2`, `X3`, `X4`, `X6`–`X11`): `OneHotEncoder` with `drop='first'` to avoid multicollinearity.

The entire transformer is embedded inside a `Pipeline` so preprocessing and inference are always applied as a single atomic step — eliminating any risk of train/serve skew.

**Stage 2 — Soft Voting Ensemble**

Three base estimators with complementary inductive biases are combined via soft (probability-averaging) voting:

| Estimator | Rationale |
|---|---|
| `LogisticRegression` | Linear baseline; fast, interpretable, regularized |
| `RandomForestClassifier` | High-variance, non-linear; captures feature interactions |
| `HistGradientBoostingClassifier` | Gradient boosting; strong on tabular data with mixed feature types |

All estimators use `class_weight='balanced'` to compensate for the ~22/78 default/non-default class imbalance in the UCI dataset.

**Stage 3 — Threshold Calibration**

Rather than using the default 0.5 cutoff, the optimal threshold is derived from the Precision-Recall curve on the training set by maximizing F1 score:

```python
precision, recall, thresholds = precision_recall_curve(y_train, probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
umbral_optimo = thresholds[np.argmax(f1_scores)]
```

This threshold is serialized to `umbral_optimo.pkl` and applied at inference time, keeping the decision boundary decoupled from the model artifact.

**Hyperparameter Tuning**

`GridSearchCV` with 5-fold `StratifiedKFold` searches over Random Forest (`max_depth`, `min_samples_leaf`) and HistGradientBoosting (`max_depth`, `l2_regularization`) parameters, scoring on F1.

---

### Performance

Evaluated on an 80/20 stratified train-test split (24,000 / 6,000 samples).

| Metric | Train | Test |
|---|---|---|
| **Accuracy** | 0.7917 | 0.7911 |
| **Error Rate** | 0.2083 | 0.2089 |
| **Precision** | 0.5254 | 0.5245 |
| **Recall** | 0.6006 | 0.5960 |
| **F1 Score** | 0.5605 | 0.5580 |

**Key observations:**

- The near-identical train/test scores (e.g., F1 Δ = 0.0025) confirm the model generalizes well with no meaningful overfitting.
- Recall of ~0.60 means the model correctly identifies 60% of actual defaulters — prioritized over precision given the asymmetric cost of missed defaults in credit risk.
- The ~0.21 error rate reflects the difficulty of the task: the UCI dataset has significant noise in repayment status features, and the class prior is imbalanced (~22% positives).

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `scikit-learn` | 1.9.0 | ML pipeline, ensemble, preprocessing |
| `pandas` | 3.0.3 | Data manipulation |
| `numpy` | 2.4.6 | Numerical operations |
| `joblib` | 1.5.3 | Model serialization |
| `scipy` | 1.17.1 | Truncated normal sampling (simulation) |
| `flask` | 3.1.3 | Local development server |

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
| `X6`–`X11` | Repayment status (months April–September 2005) |
| `X12`–`X17` | Bill statement amount (months April–September 2005) |
| `X18`–`X23` | Previous payment amount (months April–September 2005) |

Target: `Y` — default payment next month (1 = yes, 0 = no).

---

## License

MIT
