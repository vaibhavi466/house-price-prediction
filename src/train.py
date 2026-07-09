import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE
from src.pipeline import get_preprocessing_pipeline

def evaluate_predictions(y_true, y_pred):
    """
    Computes RMSE, MAE, and R^2 metrics.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) if "mean_squared_error" in globals() else np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Calculate R2 manually or import it
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

def train_and_evaluate(model, encoding_type, X_train, y_train, X_test, y_test, threshold=10):
    """
    Fits the preprocessing + estimator pipeline and returns prediction metrics on test set.
    """
    preprocessor = get_preprocessing_pipeline(encoding_type=encoding_type, threshold=threshold)
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    full_pipeline.fit(X_train, y_train)
    y_pred = full_pipeline.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)
    return full_pipeline, metrics

def run_grid_search_experiments():
    # Setup MLflow experiment
    mlflow.set_experiment("Bangalore_House_Price_Prediction")
    
    # Load processed data
    train_path = PROCESSED_DATA_DIR / "train_cleaned.csv"
    test_path = PROCESSED_DATA_DIR / "test_cleaned.csv"
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Cleaned train/test splits not found. Run outlier removal first.")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Split features and target
    X_train = train_df.drop(columns=["price"])
    y_train = train_df["price"]
    X_test = test_df.drop(columns=["price"])
    y_test = test_df["price"]
    
    print(f"Loaded Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Define models and hyperparameter grids
    models_config = {
        "Linear Regression": {
            "estimator": LinearRegression(),
            "grid": {}
        },
        "Ridge": {
            "estimator": Ridge(random_state=RANDOM_STATE),
            "grid": {"model__alpha": [0.1, 1.0, 10.0]}
        },
        "Lasso": {
            "estimator": Lasso(random_state=RANDOM_STATE, max_iter=10000),
            "grid": {"model__alpha": [0.01, 0.1, 1.0]}
        },
        "Random Forest": {
            "estimator": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            "grid": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [6, 10, None]
            }
        },
        "XGBoost": {
            "estimator": XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            "grid": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [4, 6]
            }
        },
        "LightGBM": {
            "estimator": LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
            "grid": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [4, 6]
            }
        }
    }
    
    results = []
    best_r2 = -float("inf")
    best_pipeline = None
    best_model_name = ""
    best_encoding = ""
    
    for encoding in ["onehot", "target"]:
        print(f"\n--- Running experiments under {encoding.upper()} Encoding ---")
        preprocessor = get_preprocessing_pipeline(encoding_type=encoding, threshold=10)
        
        for name, config in models_config.items():
            run_name = f"{name}_{encoding}"
            print(f"Training {run_name} via GridSearchCV...")
            
            # Create a full pipeline with a dummy model that will be overwritten in grid search
            full_pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", config["estimator"])
            ])
            
            with mlflow.start_run(run_name=run_name) as run:
                # Log encoding type tag
                mlflow.set_tag("encoding", encoding)
                mlflow.set_tag("model_name", name)
                
                # Perform Grid Search
                grid_search = GridSearchCV(
                    full_pipeline,
                    param_grid=config["grid"],
                    cv=5,
                    scoring="neg_mean_squared_error",
                    n_jobs=-1
                )
                
                grid_search.fit(X_train, y_train)
                
                # Log best parameters to MLflow
                best_params = grid_search.best_params_
                for param_name, param_val in best_params.items():
                    # Strip "model__" prefix for cleaner MLflow logging
                    clean_param_name = param_name.replace("model__", "")
                    mlflow.log_param(clean_param_name, param_val)
                
                # Evaluate on test set
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)
                metrics = evaluate_predictions(y_test, y_pred)
                
                # Log test metrics
                mlflow.log_metric("test_rmse", metrics["rmse"])
                mlflow.log_metric("test_mae", metrics["mae"])
                mlflow.log_metric("test_r2", metrics["r2"])
                
                # Log cross validation training MSE
                cv_best_rmse = np.sqrt(-grid_search.best_score_)
                mlflow.log_metric("cv_best_rmse", cv_best_rmse)
                
                # Log Model Artifact
                mlflow.sklearn.log_model(best_model, "model", serialization_format="pickle")
                
                # Add to results list
                results.append({
                    "Model": name,
                    "Encoding": encoding,
                    "Validation R²": metrics["r2"],
                    "Validation RMSE": metrics["rmse"],
                    "Validation MAE": metrics["mae"],
                    "MLflow Run ID": run.info.run_id
                })
                
                print(f"Results: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}, MAE = {metrics['mae']:.4f}")
                
                # Check if this is the overall best model (based on test R²)
                if metrics["r2"] > best_r2:
                    best_r2 = metrics["r2"]
                    best_pipeline = best_model
                    best_model_name = name
                    best_encoding = encoding
    
    # Save the best model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = MODELS_DIR / "model_pipeline.joblib"
    joblib.dump(best_pipeline, model_save_path)
    print(f"\nSaved best pipeline ({best_model_name} with {best_encoding} encoding, R² = {best_r2:.4f}) to {model_save_path}")
    
    # Format and save comparison table to models/comparison_table.md
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values(by="Validation R²", ascending=False)
    
    # Create models/comparison_table.md
    comparison_md_path = MODELS_DIR / "comparison_table.md"
    with open(comparison_md_path, "w") as f:
        f.write("# Model Performance Comparison\n\n")
        f.write("All models were trained using 5-fold cross-validation on the filtered training set and evaluated on the unfiltered test set.\n\n")
        f.write(comparison_df.to_markdown(index=False))
        f.write(f"\n\n### Winning Model Selection\n")
        f.write(f"The chosen final model is **{best_model_name}** with **{best_encoding} encoding** (R² = {best_r2:.4f}), saved to `models/model_pipeline.joblib`.\n")
        
    print(f"Saved performance comparison table to {comparison_md_path}")
    
if __name__ == "__main__":
    run_grid_search_experiments()
