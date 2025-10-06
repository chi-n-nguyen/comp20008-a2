"""
models.py
=========
Module for training machine learning models with hyperparameter tuning.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
# Set random seed for reproducibility
np.random.seed(42)

def train_random_forest(X_train, y_train):
    """Train Random Forest with hyperparameter tuning."""
    print("\n" + "=" * 70)
    print("MODEL 1: RANDOM FOREST CLASSIFIER")
    print("=" * 70)
    
    param_grid_rf = {
        'n_estimators': [50, 100, 200],     # number of trees
        'max_depth': [10, 20, 30, None],    # maximum dep of each tree
        'min_samples_split': [2, 5, 10],    # minimum sample to split a node
    }
    
    rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    grid_search_rf = GridSearchCV(
        rf_base, param_grid_rf, cv=5, scoring='f1', 
        n_jobs=-1, verbose=1
    )
    
    grid_search_rf.fit(X_train, y_train)
    
    print(f"\nBest Parameters: {grid_search_rf.best_params_}")
    print(f"Best Cross-Validation F1 Score: {grid_search_rf.best_score_:.4f}")
    
    return grid_search_rf.best_estimator_, grid_search_rf


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression with hyperparameter tuning."""
    print("\n" + "=" * 70)
    print("MODEL 2: LOGISTIC REGRESSION")
    print("=" * 70)
    
    param_grid_lr = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'class_weight': ['balanced', None]
    }
    
    lr_base = LogisticRegression(random_state=42,
        max_iter=500,  
        class_weight='balanced'
    )
    
    grid_search_lr = GridSearchCV(
        lr_base, param_grid_lr, cv=5, scoring='f1',
        n_jobs=-1, verbose=1
    )
    
    grid_search_lr.fit(X_train, y_train)
    
    print(f"\nBest Parameters: {grid_search_lr.best_params_}")
    print(f"Best Cross-Validation F1 Score: {grid_search_lr.best_score_:.4f}")
    
    best_C = grid_search_lr.best_params_['C']
    best_penalty = grid_search_lr.best_params_['penalty']
    print(f"\nParameter Interpretation:")
    print(f"  - C={best_C}: {'Weak' if best_C > 10 else 'Strong' if best_C < 0.1 else 'Moderate'} regularization")
    print(f"  - Penalty={best_penalty}: {'L1 (Lasso) - feature selection' if best_penalty == 'l1' else 'L2 (Ridge) - coefficient shrinkage'}")
    
    return grid_search_lr.best_estimator_, grid_search_lr
