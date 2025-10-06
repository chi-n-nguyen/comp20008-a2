"""
evaluation.py
=============
Module for evaluating machine learning models.
"""

from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)

def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation."""
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS: {model_name}")
    print("=" * 70)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
   
    print("\n1. CONFUSION MATRIX:")
    print(cm)
    print(f"\n   True Negatives:  {cm[0,0]}")
    print(f"   False Positives: {cm[0,1]}")
    print(f"   False Negatives: {cm[1,0]}")
    print(f"   True Positives:  {cm[1,1]}")
    
    print("\n2. CLASSIFICATION METRICS:")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   ROC-AUC:   {roc_auc:.4f}")
    
    print("\n3. DETAILED CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=['Non-WFH', 'WFH'], zero_division=0))
    
    return {
        'confusion_matrix': cm,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
