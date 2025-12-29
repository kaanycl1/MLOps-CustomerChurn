# Customer Churn Prediction – MLOps Pipeline

This project implements an end-to-end MLOps pipeline for customer churn prediction, focusing on reproducibility, data versioning, and reliable model development rather than standalone model accuracy.

## Project Overview

The goal is to move beyond notebook-based experimentation and build a reproducible machine learning workflow that tracks data preparation, model inputs, and system behavior over time.

## Dataset

The project uses the Customer Churn Dataset from Kaggle.  
Raw data files are not included in this repository.

Dataset Source: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset

## Data Version Control

Data Version Control (DVC) is used to track both raw and processed datasets.
- Raw CSV files are tracked as DVC dependencies.
- Processed train/validation/test splits are generated via DVC pipeline stages.
- Given the same raw data, all processed datasets can be reproduced deterministically.

## Reproducibility

```bash
pip install -r requirements.txt
```

To reproduce the pipeline:

1. Download the raw dataset from Kaggle
2. Place the CSV files under `data/raw/`
3. Run:

```bash
dvc repro
```

## Periodic Retraining

The system implements a **threshold-based retraining policy**.

* Labeled feedback is collected incrementally after deployment
* Retraining is triggered only when a predefined number of new labeled samples is reached
* If the threshold is not met, retraining is safely skipped
* Model versions are updated only on successful retraining events

The full pipeline can be reproduced using the following commands:
```bash
dvc pull
dvc repro preprocess
dvc repro build_retrain_splits
dvc repro retrain_model
```

## Inference API (Dockerized Service)

```bash
docker build -t churn-api .
docker run -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts churn-api
```

The volume mount ensures that inference logs and model artifacts are persisted locally.

## Inference Logging and Monitoring

Inference-time inputs and prediction probabilities are logged and stored under `artifacts/`.
These logs are later compared against the training baseline using Evidently AI, enabling drift detection based on real inference traffic.

## UI

### Streamlit App (Recommended)

```bash
streamlit run streamlit_app.py
```

Open: [http://localhost:8501](http://localhost:8501)



**Pages:**

* **Prediction** – Churn predictions with SHAP explanations
* **Monitoring** – Data drift report generation and visualization

### HTML/JS UI (Alternative)

```bash
cd ui
python -m http.server 5500
```

Open: [http://localhost:5500](http://localhost:5500)

## Monitoring

Drift reports can be generated via the Streamlit UI or directly:

```bash
python monitoring.py
```

This produces `artifacts/drift_report.html`, comparing inference data with the training reference dataset.

```

