"""Data integration orchestrator.

Coordinates dataset loading, feature engineering, aggregation,
validation, and exporting of processed datasets.

Usage:
  cd 01_preprocessing/scripts && python data_integration.py
"""
import os
import json
import warnings
import pandas as pd

import config
from features import create_wfh_features, create_household_wfh_metrics
from aggregates import (
    analyze_travel_start_times,
    create_person_level_dataset,
    create_household_level_dataset,
    create_journey_level_dataset,
)
from validate import handle_missing_values, detect_outliers, cap_outliers
from weights import apply_weights_to_person_data
from variable_mapping import save_readable_dataset, get_ml_ready_features


def load_vista_datasets():
    print("=" * 70)
    print("VISTA 2023-2024 DATA INTEGRATION PIPELINE")
    print("Research Question: WFH adoption factors")
    print("=" * 70)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    datasets = {}
    print("Loading datasets...")
    print("-" * 50)
    for name, path in config.FILENAMES.items():
        try:
            print(f"Loading {name}...", end=" ")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = pd.read_csv(path, low_memory=False)
            datasets[name] = df
            memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
            print(f"LOADED: {df.shape[0]:,} rows x {df.shape[1]} cols ({memory_mb:.1f} MB)")
        except FileNotFoundError:
            print(f"FILE NOT FOUND: {path}")
            datasets[name] = None
        except Exception as e:
            print(f"ERROR: {str(e)}")
            datasets[name] = None
    return datasets


def main():
    # Load
    datasets = load_vista_datasets()

    persons_df = datasets['persons']
    households_df = datasets['households']
    trips_df = datasets['trips']
    journey_work_df = datasets['journey_work']

    if persons_df is None or households_df is None:
        raise ValueError("Required datasets (persons/households) are missing")

    # Weights
    persons_weighted = apply_weights_to_person_data(persons_df)

    # Features
    persons_with_wfh = create_wfh_features(persons_weighted)
    households_enhanced = create_household_wfh_metrics(persons_with_wfh, households_df)

    # Morning travel
    if trips_df is not None:
        morning_travel = analyze_travel_start_times(persons_with_wfh, trips_df)
    else:
        morning_travel = pd.DataFrame()

    # Masters
    person_master = create_person_level_dataset(persons_with_wfh, households_enhanced)
    hh_master = create_household_level_dataset(households_enhanced, trips_df if trips_df is not None else pd.DataFrame())
    if journey_work_df is not None:
        journey_master = create_journey_level_dataset(journey_work_df, persons_with_wfh)
    else:
        journey_master = pd.DataFrame()

    # Missing values and data quality report
    print(f"\n{'='*70}")
    print("DATA QUALITY REPORT")
    print("="*70)
    print(f"Raw persons dataset: {len(persons_df):,} records")
    print(f"After worker filtering: {len(person_master):,} records")
    print(f"Records dropped: {len(persons_df) - len(person_master):,} ({((len(persons_df) - len(person_master))/len(persons_df)*100):.1f}%)")
    
    person_master_clean = handle_missing_values(person_master, "Person Master")
    household_master_clean = handle_missing_values(hh_master, "Household Master")
    journey_master_clean = handle_missing_values(journey_master, "Journey Master")

    # Outliers on journey
    journey_numeric_cols = ['journey_travel_time', 'journey_distance', 'journey_elapsed_time']
    if not journey_master_clean.empty:
        _ = detect_outliers(journey_master_clean, journey_numeric_cols)
        journey_master_clean = cap_outliers(journey_master_clean, journey_numeric_cols)

    # Export
    print(f"\n{'='*70}")
    print("EXPORTING PROCESSED DATASETS")
    print("="*70)

    person_master_clean.to_csv(f"{config.OUTPUT_DIR}/processed_person_master.csv", index=False)
    print(f"Saved: processed_person_master.csv ({person_master_clean.shape})")
    
    # Save ML-friendly version with readable column names
    save_readable_dataset(
        person_master_clean, 
        f"{config.OUTPUT_DIR}/processed_person_master_readable.csv",
        include_dictionary=True
    )

    household_master_clean.to_csv(f"{config.OUTPUT_DIR}/processed_household_master.csv", index=False)
    print(f"Saved: processed_household_master.csv ({household_master_clean.shape})")
    
    # Save ML-friendly household version
    save_readable_dataset(
        household_master_clean,
        f"{config.OUTPUT_DIR}/processed_household_master_readable.csv",
        include_dictionary=True
    )

    journey_master_clean.to_csv(f"{config.OUTPUT_DIR}/processed_journey_master.csv", index=False)
    print(f"Saved: processed_journey_master.csv ({journey_master_clean.shape})")

    if not morning_travel.empty:
        morning_travel.to_csv(f"{config.OUTPUT_DIR}/processed_morning_travel.csv", index=False)
        print(f"Saved: processed_morning_travel.csv ({morning_travel.shape})")

    # Parquet optional
    try:
        person_master_clean.to_parquet(f"{config.OUTPUT_DIR}/processed_person_master.parquet", index=False)
        household_master_clean.to_parquet(f"{config.OUTPUT_DIR}/processed_household_master.parquet", index=False)
        journey_master_clean.to_parquet(f"{config.OUTPUT_DIR}/processed_journey_master.parquet", index=False)
        if not morning_travel.empty:
            morning_travel.to_parquet(f"{config.OUTPUT_DIR}/processed_morning_travel.parquet", index=False)
        print("Saved Parquet versions of processed datasets")
    except Exception as e:
        print(f"Parquet save skipped or failed: {e}")

    # Data dictionary
    data_dict = {
        'person_master': {
            'rows': len(person_master_clean),
            'columns': person_master_clean.columns.tolist(),
            'dtypes': {c: str(person_master_clean[c].dtype) for c in person_master_clean.columns},
            'purpose': 'Individual-level WFH prediction (supervised learning)',
            'weight_column': 'analysis_weight'
        },
        'household_master': {
            'rows': len(household_master_clean),
            'columns': household_master_clean.columns.tolist(),
            'dtypes': {c: str(household_master_clean[c].dtype) for c in household_master_clean.columns},
            'purpose': 'Household WFH profiling (clustering)',
            'weight_column': 'analysis_weight'
        },
        'journey_master': {
            'rows': len(journey_master_clean),
            'columns': journey_master_clean.columns.tolist(),
            'dtypes': {c: str(journey_master_clean[c].dtype) for c in journey_master_clean.columns},
            'purpose': 'Journey correlation analysis',
            'weight_column': 'analysis_weight'
        }
    }
    with open(f"{config.OUTPUT_DIR}/data_dictionary.json", 'w') as f:
        json.dump(data_dict, f, indent=2)
    print("Saved: data_dictionary.json")

    print(f"\n{'='*70}")
    print("DATA INTEGRATION COMPLETE!")
    print("="*70)
    print(f"Person master: {len(person_master_clean)} rows")
    print(f"Household master: {len(household_master_clean)} rows")
    print(f"Journey master: {len(journey_master_clean)} rows")
    print(f"Data retention rate: {(len(person_master_clean)/len(persons_df)*100):.1f}%")
    print(f"All files saved to: {config.OUTPUT_DIR}/")
    
    # WFH validation report
    if len(person_master_clean) > 0:
        wfh_summary = person_master_clean.groupby(['anywfh', 'wfh_adopter']).size().reset_index(name='count')
        print(f"\nWFH Validation Summary:")
        for _, row in wfh_summary.iterrows():
            print(f"  anywfh={row['anywfh']}, wfh_adopter={row['wfh_adopter']}: {row['count']} records")


if __name__ == "__main__":
    main()


