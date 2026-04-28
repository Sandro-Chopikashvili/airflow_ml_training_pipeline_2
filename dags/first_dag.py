import numpy as np
import pandas as pd
import pendulum
from datetime import datetime
from sqlalchemy import MetaData, Table
from sqlalchemy.dialects.postgresql import insert
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from airflow.sdk import DAG, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from sklearn.model_selection import RandomizedSearchCV
import mlflow.xgboost
from mlflow import MlflowClient
from datetime import timedelta


url = "/opt/airflow/dags/data/UCI_Credit_Card.csv"

@DAG(
    dag_id='training_pipeline',
    tags = ['ml','whazaap'],
    default_args={'retries':1, 'retry_delay': timedelta(minutes=5)},
    schedule='@daily',
    start_date=pendulum.datetime(2026,5,1, tz='UTC'),
    catchup=False
)
def training_pipeline():
    
    @task
    def create_tables():
        hook = PostgresHook(postgres_conn_id='data-postgres')
        engine = hook.get_sqlalchemy_engine()
        df = pd.read_csv(url)
        df["row_hash"] = pd.util.hash_pandas_object(df, index=False).astype(str)
        df.head(0).to_sql('credit_card_clients', engine, if_exists='replace', index=False)

        with engine.begin() as conn:
            conn.exec_driver_sql("""
            ALTER TABLE credit_card_clients
            ADD CONSTRAINT IF NOT EXISTS unique_row_hash UNIQUE (row_hash)
            """)

            conn.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id SERIAL PRIMARY KEY,
                model_name TEXT,
                ROC_AUC FLOAT,
                LOG_LOSS FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
        return 'tables_created'
    
    @task
    def load_data():
        hook = PostgresHook(postgres_conn_id='data-postgres')
        engine = hook.get_sqlalchemy_engine()

        df = pd.read_csv(url)
        df["row_hash"] = pd.util.hash_pandas_object(df, index=False).astype(str)
        df = df.drop_duplicates()

        num_cols = df.select_dtypes(include=['number']).columns
        for num in num_cols:
            df[num] = df[num].fillna(df[num].median())


        cat_cols = df.select_dtypes(include=['object']).columns
        for cat in cat_cols:
            df[cat] = df[cat].fillna(df[cat].mode()[0])
        
        metadata = MetaData()

        table = Table('credit_card_clients', metadata, autoload_with=engine)

        stmt = insert(table).values(df.to_dict(orient='records'))
        stmt = stmt.on_conflict_do_nothing(constraint="unique_row_hash")

        with engine.begin() as conn:
            conn.execute(stmt)
        
        return "Data_Loaded"
    
    @task
    def train_rf(loaded):
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("model_compare_kaggle_dataset")

        hook = PostgresHook(postgres_conn_id='data-postgres')
        engine = hook.get_sqlalchemy_engine()
        df = pd.read_sql("SELECT * FROM credit_card_clients", engine)
        X = df.drop(columns=["default.payment.next.month", "row_hash"])
        y = df["default.payment.next.month"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        with mlflow.start_run(run_name='RandomForestClassifier'):
            model_RandomForest = RandomForestClassifier()
            param_dist_RFC = {
                "n_estimators": [100, 200, 300, 500, 800, 1000],
                "max_depth": [None, 5, 10, 20, 30, 50],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4, 8],
                "max_features": ["sqrt", "log2", None],
                "bootstrap": [True, False],
                "class_weight": [None, "balanced"]
            }

            randomforestclassifier = RandomizedSearchCV(
                estimator=model_RandomForest,
                param_distributions=param_dist_RFC,
                n_iter=30,
                scoring="roc_auc",
                cv=5,
                verbose=1,
                n_jobs=-1
            )

            randomforestclassifier.fit(X_train, y_train)
            mlflow.log_params(randomforestclassifier.best_params_)

            rfc_preds = randomforestclassifier.predict_proba(X_test)
            rfc_roc_auc = roc_auc_score(y_test, rfc_preds[:, 1])
            rfc_log_loss = log_loss(y_test, rfc_preds)

            mlflow.log_metric("RFC_ROC_AUC", rfc_roc_auc)
            mlflow.log_metric("RFC_LOG_LOSS", rfc_log_loss)
            mlflow.sklearn.log_model(randomforestclassifier.best_estimator_, "Model_RFC", registered_model_name="Random Forest Classifier")

        return {"ROC_AUC": rfc_roc_auc, "Log_Loss": rfc_log_loss, "model_name": "Random Forest Classifier"}


    @task
    def train_xgb(loaded):
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("model_compare_kaggle_dataset")
        hook = PostgresHook(postgres_conn_id='data-postgres')
        engine = hook.get_sqlalchemy_engine()
        df = pd.read_sql("SELECT * FROM credit_card_clients", engine)
        X = df.drop(columns=["default.payment.next.month", "row_hash"])
        y = df["default.payment.next.month"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        with mlflow.start_run(run_name='XGBoostClassifier'):
            model_XGBC = XGBClassifier()
            param_dist_XGBC = {
                "n_estimators": [100, 200, 300, 500, 800],
                "max_depth": [3, 4, 5, 6, 8, 10],
                "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "gamma": [0, 0.1, 0.3, 0.5, 1],
                "min_child_weight": [1, 3, 5, 7],
                "reg_alpha": [0, 0.01, 0.1, 1],
                "reg_lambda": [0.1, 1, 5, 10]
            }

            xgboost_classifier = RandomizedSearchCV(
                estimator=model_XGBC,
                param_distributions=param_dist_XGBC,
                n_iter=30,
                scoring="roc_auc",
                cv=5,
                verbose=1,
                n_jobs=-1
            )

            xgboost_classifier.fit(X_train, y_train)
            mlflow.log_params(xgboost_classifier.best_params_)

            xgb_preds = xgboost_classifier.predict_proba(X_test)
            xgbc_roc_auc = roc_auc_score(y_test, xgb_preds[:, 1])
            xgbc_log_loss = log_loss(y_test, xgb_preds)

            mlflow.log_metric("XGBC_ROC_AUC", xgbc_roc_auc)
            mlflow.log_metric("XGBC_LOG_LOSS", xgbc_log_loss)
            mlflow.xgboost.log_model(xgboost_classifier.best_estimator_, "XGBC_model", registered_model_name="XGBoost Classifier")

        return {"ROC_AUC": xgbc_roc_auc, "Log_Loss": xgbc_log_loss, "model_name": "XGBoost Classifier"}


    @task
    def compare_models(rf_metrics, xgb_metrics):
        if rf_metrics["ROC_AUC"] > xgb_metrics["ROC_AUC"]:
            best = rf_metrics
        else:
            best = xgb_metrics

        return {
            "RandomForestsClassifier": {"ROC_AUC": rf_metrics["ROC_AUC"], "Log_Loss": rf_metrics["Log_Loss"]},
            "XGBoostClassifier": {"ROC_AUC": xgb_metrics["ROC_AUC"], "Log_Loss": xgb_metrics["Log_Loss"]},
            "Best Model": {
                "Best Model": best["model_name"],
                "Best Model ROC_AUC": best["ROC_AUC"],
                "Best Model Log_Loss": best["Log_Loss"]
            }
        }

    
    @task
    def promote_the_best_model(metrics):
        client = MlflowClient()
        best = metrics["Best Model"]
        best_name = best["Best Model"]
        best_ROC_AUC = best["Best Model ROC_AUC"]

        source_model_name = best_name

        versions = client.search_model_versions(f"name='{source_model_name}'")

        if not versions:
            raise ValueError(f"No versions found for model {source_model_name}")
        
        latest = max(versions, key=lambda v: int(v.version))
        client.set_registered_model_alias(source_model_name, "champion", latest.version)

        unified_name = "credit_card_model_classification"

        try:
            client.create_registered_model(unified_name)
        except Exception:
            pass

        new_version = client.create_model_version(
            name=unified_name,
            source=latest.source,
            run_id=latest.run_id,
        ).version

        client.set_registered_model_alias(unified_name, "champion", new_version)

        print(f"Promoted {source_model_name} v{latest.version} → @champion (ROC_AUC: {best_ROC_AUC:.4f})")
        print(f"Unified model {unified_name} v{new_version} → @champion")

        return {"model_name": unified_name, "version": new_version, "alias": "champion"}

    @task
    def save_metrics(metrics):
        hook = PostgresHook(postgres_conn_id='data-postgres')
        engine = hook.get_sqlalchemy_engine()

        metadata = MetaData()
        table = Table("model_metrics", metadata, autoload_with=engine)

        rows = [
            {
                'model_name': "Random Forest Classifier",
                'ROC_AUC': metrics['RandomForestsClassifier']['ROC_AUC'],
                'LOG_LOSS': metrics["RandomForestsClassifier"]['Log_Loss'],
                'created_at': datetime.utcnow()
            },
            {
                'model_name': "XGBoost Classifier",
                'ROC_AUC': metrics['XGBoostClassifier']['ROC_AUC'],
                'LOG_LOSS': metrics["XGBoostClassifier"]['Log_Loss'],
                'created_at': datetime.utcnow()
            },

            {
                'model_name': f"Best Model:{metrics['Best Model']['Best Model']}",
                'ROC_AUC': metrics['Best Model']['Best Model ROC_AUC'],
                'LOG_LOSS': metrics['Best Model']['Best Model Log_Loss'],
                'created_at': datetime.utcnow()
            }
        ]

        stmt = insert(table).values(rows)

        with engine.begin() as conn:
            conn.execute(stmt)
        
        return 'metrics_saved'
    
    created = create_tables()
    loaded = load_data()
    created >> loaded

    rf_metrics = train_rf(loaded)
    xgb_metrics = train_xgb(loaded)
    metrics = compare_models(rf_metrics, xgb_metrics)

    promoted = promote_the_best_model(metrics)
    saved = save_metrics(metrics)
    promoted >> saved

training_pipeline()