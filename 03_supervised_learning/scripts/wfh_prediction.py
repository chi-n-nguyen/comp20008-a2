"""
Work-from-Home (WFH) Prediction using Machine Learning
======================================================
This version uses data from 01_preprocessing folder

Files used:
- processed_household_master.csv
- processed_person_master.csv
- processed_journey_master.csv
- processed_morning_travel.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# 1. LOAD THE PROCESSED DATA
# ============================================================================

def load_and_merge_data():
    """
    Load and merge the pre_processed data.
    """
    print("Loading data files...")
    
    import os
    
    # data locations for the CSV files
    path = ['01_preprocessing/outputs/']  # preprocessing outputs from project root
    
    data_path = None
    for path in path:
        test_file = f'{path}processed_household_master.csv'
        if os.path.exists(test_file):
            data_path = path
            print(f"✓ Found data files in: {os.path.abspath(path)}")
            break
    
    if data_path is None:
        print("\n Error: Cannot find CSV files!")
        raise FileNotFoundError("CSV files not found. Please check the file paths.")
    
    try:
        # Load all CSV files with the correct path
        household_df = pd.read_csv(f'{data_path}processed_household_master.csv')
        person_df = pd.read_csv(f'{data_path}processed_person_master.csv')
        journey_df = pd.read_csv(f'{data_path}processed_journey_master.csv')
        morning_df = pd.read_csv(f'{data_path}processed_morning_travel.csv')
        
        # Display column names to help with merging
        print("\n" + "="*70)
        print("COLUMN NAMES IN EACH FILE:")
        print("="*70)
        print("\nHousehold columns:", list(household_df.columns))
        print("\nPerson columns:", list(person_df.columns))
        print("\nJourney columns:", list(journey_df.columns))
        print("\nMorning travel columns:", list(morning_df.columns))
        
        return household_df, person_df, journey_df, morning_df
        
    except FileNotFoundError as e:
        print(f"\n Error: Could not find file - {e}")
        print("\nMake sure you're running the script from the correct directory!")
        print("Current files should be in the same folder as this script.")
        raise

# ============================================================================
# 2. PREPROCESSING (ENCODING / FEATURE AND CANDIDATE SELECTION)
# ============================================================================

def create_wfh_target(person_df):
    """
    Create the work-from-home target variable using actual WFH columns.
    
    Available WFH columns in data:
    - anywfh: Any work from home (likely binary indicator)
    - wfhmon, wfhtue, wfhwed, wfhthu, wfhfri, wfhsat, wfhsun: Day-specific WFH
    - wfhtravday: WFH on travel day
    """
    print("\n" + "="*70)
    print("CREATING WFH TARGET VARIABLE")
    print("="*70)
    
    # Check which WFH columns are available
    wfh_columns = ['anywfh', 'wfhmon', 'wfhtue', 'wfhwed', 'wfhthu', 
                   'wfhfri', 'wfhsat', 'wfhsun', 'wfhtravday']
    
    available_wfh_cols = [col for col in wfh_columns if col in person_df.columns]
    print(f"Found WFH columns: {available_wfh_cols}")
    
    if 'anywfh' in person_df.columns:
        # Use 'anywfh' as the primary target (indicates if person works from home at all)
        person_df['wfh'] = person_df['anywfh'].copy()
        
        # Convert to binary if needed (handle different encodings)
        # Common encodings: Yes/No, 1/0, True/False, or actual days
        if person_df['wfh'].dtype == 'object':
            # String values like 'Yes'/'No'
            person_df['wfh'] = person_df['wfh'].map({
                'Yes': 1, 'No': 0, '1': 1, '0': 0
            })
        
        # Ensure binary (0 or 1)
        person_df['wfh'] = (person_df['wfh'] > 0).astype(int)
        
        print(f"\n✓ Using 'anywfh' as target variable")
        
    elif 'wfhtravday' in person_df.columns:
        # Alternative: Use WFH on travel day
        person_df['wfh'] = person_df['wfhtravday'].copy()
        if person_df['wfh'].dtype == 'object':
            person_df['wfh'] = person_df['wfh'].map({
                'Yes': 1, 'No': 0, '1': 1, '0': 0
            })
        person_df['wfh'] = (person_df['wfh'] > 0).astype(int)
        print(f"\n✓ Using 'wfhtravday' as target variable")
        
    else:
        # Fallback: Create target from any weekday WFH
        weekday_cols = [col for col in ['wfhmon', 'wfhtue', 'wfhwed', 'wfhthu', 'wfhfri'] 
                       if col in person_df.columns]
        
        if weekday_cols:
            # If person WFH on any weekday, mark as WFH
            person_df['wfh'] = person_df[weekday_cols].apply(
                lambda row: int(any(x > 0 for x in row if pd.notna(x))), axis=1
            )
            print(f"\n✓ Created target from weekday WFH columns: {weekday_cols}")
        else:
            raise ValueError("No WFH columns found in dataset!")
    
    # Handle missing values in target
    missing_count = person_df['wfh'].isnull().sum()
    if missing_count > 0:
        print(f"\n Warning: {missing_count} missing values in WFH target")
        print("Dropping rows with missing target values...")
        person_df = person_df.dropna(subset=['wfh'])
    
    # Display distribution
    wfh_counts = person_df['wfh'].value_counts()
    print(f"\nWFH Target Distribution:")
    print(f"  Non-WFH (0): {wfh_counts.get(0, 0)} ({wfh_counts.get(0, 0)/len(person_df)*100:.1f}%)")
    print(f"  WFH (1):     {wfh_counts.get(1, 0)} ({wfh_counts.get(1, 0)/len(person_df)*100:.1f}%)")
    
    return person_df

def prepare_features(household_df, person_df, journey_df, morning_df):
    """
    Merge datasets and decided features for modeling.
    """
    print("\n" + "="*70)
    print("PREPARING FEATURES")
    print("="*70)
    
    # Start with person-level data
    df = person_df.copy()
    
    # Identify common key columns for merging
    # Common possibilities: 'household_id', 'person_id', 'hhid', 'persid', etc.
    print("\nAttempting to merge datasets...")
    
    # Try to identify ID columns
    household_id_cols = [col for col in household_df.columns if 'household' in col.lower() or 'hh' in col.lower()]
    person_id_cols = [col for col in person_df.columns if 'person' in col.lower() or 'pers' in col.lower()]
    
    print(f"Potential household ID columns: {household_id_cols}")
    print(f"Potential person ID columns: {person_id_cols}")
    
    # Merge with household data (adjust key names as needed)
    if household_id_cols:
        try:
            # Try first household ID column found
            key_col = household_id_cols[0]
            if key_col in df.columns and key_col in household_df.columns:
                df = df.merge(household_df, on=key_col, how='left', suffixes=('', '_hh'))
                print(f"✓ Merged household data on '{key_col}'")
        except Exception as e:
            print(f"Could not merge household data: {e}")
    
    # Aggregate journey-level features (e.g., number of trips per person)
    if person_id_cols and len(person_id_cols) > 0:
        try:
            person_key = person_id_cols[0]
            if person_key in journey_df.columns:
                # Count trips per person
                trip_counts = journey_df.groupby(person_key).size().reset_index(name='num_trips')
                df = df.merge(trip_counts, on=person_key, how='left')
                df['num_trips'] = df['num_trips'].fillna(0)
                print(f"✓ Added trip count feature")
                
                # Add average trip distance if available
                distance_cols = [col for col in journey_df.columns if 'distance' in col.lower()]
                if distance_cols:
                    avg_distance = journey_df.groupby(person_key)[distance_cols[0]].mean().reset_index()
                    avg_distance.columns = [person_key, 'avg_trip_distance']
                    df = df.merge(avg_distance, on=person_key, how='left')
                    df['avg_trip_distance'] = df['avg_trip_distance'].fillna(0)
                    print(f"✓ Added average trip distance feature")
        except Exception as e:
            print(f"Could not process journey data: {e}")
    
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Total columns: {len(df.columns)}")
    
    return df

def select_features(df):
    """
    Select relevant features for modeling.
    Automatically detects numeric and categorical features.
    """
    print("\n" + "="*70)
    print("FEATURE SELECTION")
    print("="*70)
    
    # Ensure we have the target variable
    if 'wfh' not in df.columns:
        raise ValueError("Target variable 'wfh' not found. Check create_wfh_target() function.")
    
    # Separate features and target
    target = 'wfh'
    
    # Exclude non-feature columns (IDs, dates, target)
    exclude_keywords = ['id', 'date', 'time', 'name', target]
    
    feature_cols = [col for col in df.columns 
                   if col != target 
                   and not any(keyword in col.lower() for keyword in exclude_keywords)]
    
    print(f"\nInitial features selected: {len(feature_cols)}")
    
    # Separate numeric and categorical
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"  - Numeric features: {len(numeric_features)}")
    print(f"  - Categorical features: {len(categorical_features)}")
    
    # Encode categorical variables
    print("\nEncoding categorical variables...")
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        label_encoders[col] = le

def prepare_train_test_split(df, target_col='wfh', test_size=0.2):
    """
    Split data into training and testing sets.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Check for class imbalance
    class_counts = y.value_counts()
    print(f"\nClass distribution:")
    print(class_counts)
    
    if len(class_counts) < 2:
        raise ValueError("Target variable must have at least 2 classes!")
    
    # Stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# ============================================================================
# 3. MODEL 1: RANDOM FOREST CLASSIFIER
# ============================================================================

def train_random_forest(X_train, y_train):
    """Train Random Forest with hyperparameter tuning."""
    print("\n" + "=" * 70)
    print("MODEL 1: RANDOM FOREST CLASSIFIER")
    print("=" * 70)
    print("\nHyperparameter Tuning in Progress...")
    
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
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

# ============================================================================
# 4. MODEL 2: LOGISTIC REGRESSION
# ============================================================================

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression with hyperparameter tuning."""
    print("\n" + "=" * 70)
    print("MODEL 2: LOGISTIC REGRESSION")
    print("=" * 70)
    print("\nHyperparameter Tuning in Progress...")
    
    param_grid_lr = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [100, 200, 500, 1000],
        'class_weight': ['balanced', None]
    }
    
    lr_base = LogisticRegression(random_state=42)
    
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

# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation."""
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS: {model_name}")
    print("=" * 70)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n1. CONFUSION MATRIX:")
    print(cm)
    print(f"\n   True Negatives:  {cm[0,0]}")
    print(f"   False Positives: {cm[0,1]}")
    print(f"   False Negatives: {cm[1,0]}")
    print(f"   True Positives:  {cm[1,1]}")
    
    print("\n2. CLASSIFICATION METRICS:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   ROC-AUC:   {roc_auc:.4f}")
    
    print("\n3. DETAILED CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=['Non-WFH', 'WFH'], zero_division=0))
    
    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

# ============================================================================
# 6. VISUALIZATION
# ============================================================================

def get_output_directory():
    """
    Determine the correct output directory based on where script is run from.
    Always saves to 03_supervised_learning/outputs/
    """
    import os
    # Possible output locations
    possible_outputs = [
        '03_supervised_learning/outputs/',  # From project root
        'outputs/',  # From 03_supervised_learning folder
        '../outputs/',  # From scripts folder
    ]
    
    # Try to create/find the output directory
    for output_dir in possible_outputs:
        try:
            os.makedirs(output_dir, exist_ok=True)
            # Test if we can write to this directory
            test_file = os.path.join(output_dir, '.test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return output_dir
        except:
            continue
    
    # Fallback: current directory
    return ''

def plot_results(results_rf, results_lr, y_test):
    """Create comprehensive visualizations."""
    fig = plt.figure(figsize=(18, 12))
    
    # Confusion Matrices
    ax1 = plt.subplot(2, 3, 1)
    sns.heatmap(results_rf['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-WFH', 'WFH'], yticklabels=['Non-WFH', 'WFH'])
    plt.title('Random Forest - Confusion Matrix', fontsize=12, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    ax2 = plt.subplot(2, 3, 2)
    sns.heatmap(results_lr['confusion_matrix'], annot=True, fmt='d', cmap='Greens',
                xticklabels=['Non-WFH', 'WFH'], yticklabels=['Non-WFH', 'WFH'])
    plt.title('Logistic Regression - Confusion Matrix', fontsize=12, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Metrics Comparison
    ax3 = plt.subplot(2, 3, 3)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    rf_scores = [results_rf['accuracy'], results_rf['precision'], 
                 results_rf['recall'], results_rf['f1_score'], results_rf['roc_auc']]
    lr_scores = [results_lr['accuracy'], results_lr['precision'], 
                 results_lr['recall'], results_lr['f1_score'], results_lr['roc_auc']]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax3.bar(x - width/2, rf_scores, width, label='Random Forest', color='steelblue')
    ax3.bar(x + width/2, lr_scores, width, label='Logistic Regression', color='seagreen')
    ax3.set_ylabel('Score')
    ax3.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, rotation=45, ha='right')
    ax3.legend()
    ax3.set_ylim([0, 1])
    ax3.grid(axis='y', alpha=0.3)
    
    # ROC Curves
    ax4 = plt.subplot(2, 3, 4)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, results_rf['y_pred_proba'])
    fpr_lr, tpr_lr, _ = roc_curve(y_test, results_lr['y_pred_proba'])
    
    ax4.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={results_rf['roc_auc']:.3f})", 
             color='steelblue', linewidth=2)
    ax4.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={results_lr['roc_auc']:.3f})", 
             color='seagreen', linewidth=2)
    ax4.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('ROC Curves Comparison', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # Prediction Distribution
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(results_rf['y_pred_proba'], bins=30, alpha=0.6, label='Random Forest', color='steelblue')
    ax5.hist(results_lr['y_pred_proba'], bins=30, alpha=0.6, label='Logistic Regression', color='seagreen')
    ax5.set_xlabel('Predicted Probability')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Prediction Probability Distribution', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    import os
    output_dir = get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'wfh_prediction_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    plt.show()

def plot_feature_importance(model_rf, model_lr, feature_names, top_n=15):
    """Plot feature importance for both models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Random Forest
    importance_rf = pd.DataFrame({
        'feature': feature_names,
        'importance': model_rf.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    ax1.barh(importance_rf['feature'], importance_rf['importance'], color='steelblue')
    ax1.set_xlabel('Importance')
    ax1.set_title(f'Random Forest - Top {top_n} Features', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    
    # Logistic Regression
    importance_lr = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model_lr.coef_[0],
        'abs_coefficient': np.abs(model_lr.coef_[0])
    }).sort_values('abs_coefficient', ascending=False).head(top_n)
    
    colors = ['seagreen' if x > 0 else 'coral' for x in importance_lr['coefficient']]
    ax2.barh(importance_lr['feature'], importance_lr['abs_coefficient'], color=colors)
    ax2.set_xlabel('Absolute Coefficient')
    ax2.set_title(f'Logistic Regression - Top {top_n} Features\n(Green=Positive, Red=Negative)', 
                  fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    import os
    output_dir = get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.show()

# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""
    print("\n" + "=" * 70)
    print("WORK-FROM-HOME PREDICTION - REAL DATA ANALYSIS")
    print("=" * 70)
    
    # Load data
    household_df, person_df, journey_df, morning_df = load_and_merge_data()
    
    # Create WFH target
    person_df = create_wfh_target(person_df)
    
    # Prepare features
    df = prepare_features(household_df, person_df, journey_df, morning_df)
    
    # Select and clean features
    df_clean, label_encoders = select_features(df)
    
    # Split data
    print("\n" + "="*70)
    print("TRAIN-TEST SPLIT")
    print("="*70)
    X_train, X_test, y_train, y_test, scaler = prepare_train_test_split(df_clean)
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    # Train models
    model_rf, grid_rf = train_random_forest(X_train, y_train)
    model_lr, grid_lr = train_logistic_regression(X_train, y_train)
    
    # Evaluate models
    results_rf = evaluate_model(model_rf, X_test, y_test, "Random Forest")
    results_lr = evaluate_model(model_lr, X_test, y_test, "Logistic Regression")
    
    # Compare models
    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)
    
    comparison_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'Random Forest': [
            results_rf['accuracy'], results_rf['precision'],
            results_rf['recall'], results_rf['f1_score'], results_rf['roc_auc']
        ],
        'Logistic Regression': [
            results_lr['accuracy'], results_lr['precision'],
            results_lr['recall'], results_lr['f1_score'], results_lr['roc_auc']
        ]
    })
    
    comparison_df['Difference'] = comparison_df['Logistic Regression'] - comparison_df['Random Forest']
    print("\n", comparison_df.to_string(index=False))
    
    # Visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    plot_results(results_rf, results_lr, y_test)
    plot_feature_importance(model_rf, model_lr, X_train.columns)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()