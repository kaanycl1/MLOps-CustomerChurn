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
To reproduce the data preparation pipeline:
1) Download the raw dataset from Kaggle.
2) Place the CSV files under data/raw/.
3) Run 
```bash
dvc repro
```
