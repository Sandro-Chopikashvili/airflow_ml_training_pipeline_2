# End-to-End ML Pipeline (Airflow + MLflow + PostgreSQL + FastAPI)

An end-to-end machine learning pipeline that automates data ingestion, preprocessing, model training, hyperparameter tuning, experiment tracking, model selection, and model promotion.

---

## What it does

- Loads the **UCI Credit Card Default dataset**
- Stores data in PostgreSQL with **deduplication using row hashing**
- Cleans missing values (median for numeric, mode for categorical)
- Trains:
  - Random Forest Classifier
  - XGBoost Classifier
- Performs **hyperparameter tuning using RandomizedSearchCV**
- Tracks experiments with MLflow
- Evaluates models using:
  - ROC AUC
  - Log Loss
- Compares models and selects the best one
- Promotes the best model to `@champion` in MLflow Model Registry
- Creates a **unified production model**
- Stores model performance metrics in PostgreSQL
- (Optional serving layer present via FastAPI)

---

## Tech stack

- Apache Airflow  
- MLflow  
- PostgreSQL  
- FastAPI + Uvicorn  
- scikit-learn  
- XGBoost  
- Pandas / NumPy  
- Docker / Docker Compose  

---

## Pipeline Overview

### 1. Data Ingestion
- Reads dataset from:
  ```
  /opt/airflow/dags/data/UCI_Credit_Card.csv
  ```
- Adds `row_hash` for deduplication
- Creates table: `credit_card_clients`

---

### 2. Data Cleaning
- Removes duplicates
- Fills missing values:
  - Numeric → median
  - Categorical → mode

---

### 3. Model Training

#### Random Forest
- Hyperparameter tuning with RandomizedSearchCV
- Logs:
  - Best parameters
  - ROC AUC
  - Log Loss

#### XGBoost
- Same pipeline as Random Forest
- Fully tracked in MLflow

---

### 4. Model Comparison
- Compares models using **ROC AUC**
- Selects best model

---

### 5. Model Promotion
- Promotes best model to:
  ```
  @champion
  ```
- Creates unified production model:
  ```
  credit_card_model_classification
  ```

---

### 6. Metrics Storage
- Saves results into PostgreSQL table:
  ```
  model_metrics
  ```

---

## Project Structure

```
dags/
 ├── data/
 │   └── UCI_Credit_Card.csv
 └── first_dag.py

serving/
 ├── server.py
 └── dockerfile

config/
logs/
mlflow_artifact/
plugins/

.env
docker-compose.yaml
dockerfile
requirements.txt
README.md
```

---

## Airflow DAG

**DAG Name:** `training_pipeline`  
**Schedule:** Daily  

**Tasks:**
- create_tables
- load_data
- train_rf
- train_xgb
- compare_models
- promote_the_best_model
- save_metrics

---

## How to run

### 1. Start services
```
docker-compose up --build
```

---

### 2. Access tools

- Airflow UI: http://localhost:8080  
- MLflow UI: http://localhost:5000  
- FastAPI (if running): http://localhost:8000  

---

### 3. Trigger pipeline

- Open Airflow UI  
- Enable DAG `training_pipeline`  
- Trigger manually or wait for schedule  

---

## Key Features

- Idempotent data loading (no duplicates)
- Full experiment tracking
- Automated model selection
- Production-ready model registry workflow
- Database-backed metrics tracking
- Modular and scalable pipeline design

---

## Notes

- Target column:
  ```
  default.payment.next.month
  ```
- Uses stratified train-test split
- Suitable for imbalanced classification problems