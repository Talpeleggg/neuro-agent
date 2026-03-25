import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score


def setup_mlflow():
    """Configure the MLflow tracking URI and experiment name."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("NeuroData_Pipeline")


def train_xgboost_and_log(df, feature_cols, target_col):
    """
    Smart MLOps engine: auto-detects whether the task is regression
    (continuous numeric target) or classification (categorical / few unique values),
    trains the appropriate XGBoost model, and logs everything to MLflow.

    Returns
    -------
    metric_text : str  — human-readable result string, or None on failure
    fig         : matplotlib Figure with feature importances, or None on failure
    error_msg   : str error message, or None on success
    """
    setup_mlflow()

    try:
        # Drop rows where either features or target are missing
        train_df = df[feature_cols + [target_col]].dropna(subset=[target_col])
        X = train_df[feature_cols].fillna(0)  # impute remaining NaNs in features
        y = train_df[target_col]

        # Auto-detect task type: numeric target with many unique values → regression
        is_numeric = pd.api.types.is_numeric_dtype(y)
        unique_vals = y.nunique()
        task_type = 'regression' if (is_numeric and unique_vals > 10) else 'classification'

        with mlflow.start_run(run_name=f"XGBoost_{task_type.capitalize()}"):

            if task_type == 'classification':
                # Encode string/categorical labels to integers
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                n_classes = len(le.classes_)

                # Use mlogloss for multi-class, logloss for binary
                eval_metric = 'mlogloss' if n_classes > 2 else 'logloss'

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                )

                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    eval_metric=eval_metric,
                    random_state=42,
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                metric_val = accuracy_score(y_test, preds)
                metric_text = f"Accuracy: {metric_val * 100:.2f}%"
                mlflow.log_metric("accuracy", metric_val)
                mlflow.log_param("n_classes", n_classes)

            else:  # Regression
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    eval_metric='rmse',
                    random_state=42,
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                metric_val = r2_score(y_test, preds)
                metric_text = f"R² Score: {metric_val:.4f}"
                mlflow.log_metric("r2_score", metric_val)

            # Log shared hyperparameters and metadata
            mlflow.log_params({
                "task_type": task_type,
                "test_size": 0.2,
                "num_features": len(feature_cols),
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
            })
            mlflow.xgboost.log_model(model, "xgb_model")

            # Feature importance plot — use feature_importances_ (sklearn API)
            # instead of xgb.plot_importance() which crashes on small datasets
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]  # top 15
            top_features = [feature_cols[i] for i in indices]
            top_scores = importances[indices]

            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, max(4, len(top_features) * 0.45)))
            bars = ax.barh(top_features[::-1], top_scores[::-1], color='#4A78F5')
            ax.set_xlabel("Importance (gain)", color='white')
            ax.set_title(f"Feature Importances — {task_type.capitalize()}", color='white', fontsize=13)
            ax.tick_params(colors='white')
            for bar, score in zip(bars, top_scores[::-1]):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                        f'{score:.4f}', va='center', color='white', fontsize=8)
            plt.tight_layout()

            return metric_text, fig, None

    except Exception as e:
        # Return the error as a string so the UI can display it gracefully
        return None, None, str(e)
