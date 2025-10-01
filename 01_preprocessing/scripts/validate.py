from typing import List, Dict
import pandas as pd
import config

def assert_required_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

def handle_missing_values(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    print(f"\n{'='*70}")
    print(f"MISSING VALUE HANDLING: {dataset_name}")
    print("="*70)

    missing_summary = df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)

    if len(missing_summary) == 0:
        print("No missing values found!")
        return df

    df_clean = df.copy()

    for col in missing_summary.index:
        missing_pct = (missing_summary[col] / len(df)) * 100
        print(f"\n{col}: {missing_pct:.2f}% missing")

        if missing_pct > config.MISSING_DROP_THRESHOLD:
            print(f"  -> DROP column (too many missing)")
            df_clean = df_clean.drop(columns=[col])
        elif str(df_clean[col].dtype) in ['object', 'category']:
            print(f"  -> Fill with 'Unknown'")
            df_clean[col] = df_clean[col].fillna('Unknown')
        elif str(df_clean[col].dtype) in ['int64', 'float64']:
            if missing_pct < config.NUMERIC_MEDIAN_THRESHOLD:
                print(f"  -> Fill with median")
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                print(f"  -> Fill with 0 and create missing indicator")
                df_clean[f'{col}_missing'] = df_clean[col].isnull().astype(int)
                df_clean[col] = df_clean[col].fillna(0)

    print(f"\nCleaning complete. Shape: {df_clean.shape}")
    return df_clean

def detect_outliers(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Dict[str, float]]:
    print(f"\n{'='*70}")
    print("OUTLIER DETECTION")
    print("="*70)
    outlier_summary: Dict[str, Dict[str, float]] = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        numeric_data = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(numeric_data) == 0:
            continue
        Q1 = float(numeric_data.quantile(0.25))
        Q3 = float(numeric_data.quantile(0.75))
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = int(((numeric_data < lower) | (numeric_data > upper)).sum())
        pct = (outliers / len(numeric_data)) * 100
        outlier_summary[col] = {'count': outliers, 'percent': pct, 'lower_bound': lower, 'upper_bound': upper}
        print(f"{col}: {outliers} outliers ({pct:.2f}%)")
    return outlier_summary

def cap_outliers(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    df_capped = df.copy()
    for col in numeric_cols:
        if col in df_capped.columns:
            numeric_data = pd.to_numeric(df_capped[col], errors='coerce')
            p99 = float(numeric_data.quantile(0.99))
            df_capped[col] = numeric_data.clip(upper=p99)
            print(f"Capped {col} at 99th percentile: {p99:.2f}")
    return df_capped


