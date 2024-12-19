import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

def save_object(file_path: str, obj: object) -> None:
    """
    Save a Python object to a file using pickle
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param: dict) -> dict:
    """
    Evaluate multiple models and return their performance scores
    """
    try:
        report = {}
        
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            
            # Get parameters for current model
            params = param[model_name]
            
            # Special handling for XGBoost and CatBoost
            if model_name in ["XGBRegressor", "CatBoosting Regressor"]:
                best_score = float('-inf')
                best_model = None
                
                # Manual grid search for XGBoost/CatBoost
                for param_combo in _generate_param_combinations(params):
                    # Create a fresh model instance with the current parameters
                    if model_name == "XGBRegressor":
                        current_model = XGBRegressor(**param_combo)
                    else:  # CatBoosting Regressor
                        current_model = CatBoostRegressor(**param_combo, verbose=False)
                    
                    # Train the model
                    current_model.fit(X_train, y_train)
                    
                    # Evaluate
                    y_test_pred = current_model.predict(X_test)
                    score = r2_score(y_test, y_test_pred)
                    
                    if score > best_score:
                        best_score = score
                        best_model = current_model
                
                # Store best model back in models dictionary
                models[model_name] = best_model
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)
                
            else:
                # Standard sklearn models
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=params,
                    cv=3,
                    n_jobs=-1,
                    verbose=1
                )
                gs.fit(X_train, y_train)
                
                # Update the model with best parameters
                models[model_name] = gs.best_estimator_
                
                y_train_pred = gs.best_estimator_.predict(X_train)
                y_test_pred = gs.best_estimator_.predict(X_test)
            
            # Calculate scores
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)
            
            print(f"{model_name} - Train R2 Score: {train_score:.4f}")
            print(f"{model_name} - Test R2 Score: {test_score:.4f}")
            
            report[model_name] = test_score
        
        return report
        
    except Exception as e:
        print(f"Error in model evaluation: {str(e)}")
        raise CustomException(e, sys)

def _generate_param_combinations(params):
    """
    Generate all possible combinations of parameters
    """
    from itertools import product
    
    keys = params.keys()
    values = params.values()
    combinations = list(product(*values))
    
    return [dict(zip(keys, combo)) for combo in combinations]

def load_object(file_path: str) -> object:
    """
    Load a Python object from a pickle file
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)