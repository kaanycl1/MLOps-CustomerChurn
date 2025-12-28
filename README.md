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
- Processed train/validation/test splits are generated via a DVC pipeline stage.
- Given the same raw data, all processed datasets can be reproduced deterministically.

### Reproducibility
```bash
pip install -r requirements.txt
```
To reproduce the data preparation pipeline:
1) Download the raw dataset from Kaggle.
2) Place the CSV files under data/raw/.
3) Run 
```bash
dvc repro
```
## Inference API (Dockerized Service)
```bash
docker build -t churn-api .
docker run -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts churn-api
```

The volume mount (`-v $(pwd)/artifacts:/app/artifacts`) ensures that inference logs are saved to your host machine's `artifacts/` directory, making them accessible for monitoring.

## UI (Optional Frontend)

### Streamlit App (Recommended)
```bash
streamlit run streamlit_app.py
```
Then open: http://localhost:8501

**Pages:**
- **📊 Prediction**: Make churn predictions with SHAP explanations
- **📈 Monitoring**: Generate and view data drift reports

### HTML/JS UI (Alternative)
Run a lightweight local web server:
```bash
cd ui
python -m http.server 5500
```
Then open: http://localhost:5500

## Monitoring

Data drift monitoring is available as a page in the Streamlit app, or run directly:
```bash
python monitoring.py
```

This generates `artifacts/drift_report.html` comparing current inference data with the reference dataset.

