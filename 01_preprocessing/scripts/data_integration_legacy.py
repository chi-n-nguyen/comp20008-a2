import pandas as pd
import os
import warnings
from typing import Dict, List, Union, Optional, Any
import json

# =====================================
# VISTA WFH Data Integration Pipeline
# Responsible: Nhat Chi Nguyen, 1492182
# =====================================

def load_vista_datasets() -> Dict[str, Optional[pd.DataFrame]]:
    """Load all VISTA 2023-2024 datasets with proper error handling"""
    
    print("="*70)
    print("VISTA 2023-2024 DATA INTEGRATION PIPELINE")
    print("Research Question: WFH adoption factors and household travel behavior")
    print("="*70)
    
    # Create outputs directory
    os.makedirs('../../01_preprocessing/outputs', exist_ok=True)
    
    # Dataset file paths
    datasets: Dict[str, Optional[pd.DataFrame]] = {}
    file_paths = {
        'households': '../../00_raw_data/household_vista_2023_2024.csv',
        'persons': '../../00_raw_data/person_vista_2023_2024.csv', 
        'trips': '../../00_raw_data/trips_vista_2023_2024.csv',
        'stops': '../../00_raw_data/stops_vista_2023_2024.csv',
        'journey_work': '../../00_raw_data/journey_to_work_vista_2023_2024.csv',
        'journey_edu': '../../00_raw_data/journey_to_education_vista_2023_2024.csv'
    }
    
    print("Loading datasets...")
    print("-" * 50)
    
    for name, path in file_paths.items():
        try:
            print(f"Loading {name}...", end=" ")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = pd.read_csv(path, low_memory=False)
            datasets[name] = df
            memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
            print(f"LOADED: {df.shape[0]:,} rows x {df.shape[1]} cols ({memory_mb:.1f} MB)")
            
        except FileNotFoundError:
            print(f"FILE NOT FOUND: {path}")
            datasets[name] = None
        except Exception as e:
            print(f"ERROR: {str(e)}")
            datasets[name] = None
    
    return datasets

def explore_dataset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Create detailed summary for each dataset"""
    print(f"\n{'='*50}")
    print(f"Dataset: {name}")
    print(f"{'='*50}")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    missing_vals = df.isnull().sum()
    missing_vals = missing_vals[missing_vals > 0]
    if len(missing_vals) > 0:
        print(f"\nMissing values:\n{missing_vals}")
    else:
        print("\nNo missing values found")
    
    print(f"\nData types:\n{df.dtypes.value_counts()}")
    return df.describe()

def analyze_wfh_variables(persons_df: pd.DataFrame) -> None:
    """Analyze WFH-related variables in person dataset"""
    print("\n" + "="*70)
    print("WFH VARIABLES ANALYSIS")
    print("="*70)
    
    # WFH columns to examine
    wfh_cols = ['anywfh', 'wfhmon', 'wfhtue', 'wfhwed', 'wfhthu', 
                'wfhfri', 'wfhsat', 'wfhsun', 'wfhtravday']
    
    # Check which columns exist
    existing_wfh_cols = [col for col in wfh_cols if col in persons_df.columns]
    print(f"WFH columns found: {existing_wfh_cols}")
    
    if existing_wfh_cols:
        print(f"\nFirst 20 rows of WFH data:")
        print(persons_df[existing_wfh_cols].head(20))
        
        print(f"\nValue counts for key WFH variables:")
        for col in existing_wfh_cols[:3]:  # Show first 3 for brevity
            print(f"\n{col}:")
            print(persons_df[col].value_counts())
    
    # Check employment status
    employment_cols = ['anywork', 'fulltimework', 'parttimework', 'casualwork']
    existing_emp_cols = [col for col in employment_cols if col in persons_df.columns]
    print(f"\nEmployment columns found: {existing_emp_cols}")
    
    if 'anywork' in persons_df.columns:
        print(f"\nEmployment status distribution:")
        print(persons_df['anywork'].value_counts())

def apply_weights_to_person_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply perspoststratweight to person-level variables"""
    print("\n" + "="*70)
    print("APPLYING EXPANSION WEIGHTS")
    print("="*70)
    
    df_weighted = df.copy()
    
    # Store weight for analysis
    if 'perspoststratweight' in df.columns:
        df_weighted['weight'] = df_weighted['perspoststratweight']
        print(f"Applied person weights - Total weighted population: {df_weighted['weight'].sum():,.0f}")
    else:
        print("WARNING: perspoststratweight not found in person dataset")
        df_weighted['weight'] = 1.0
    
    return df_weighted

def create_wfh_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer WFH-related features for analysis"""
    print("\n" + "="*70)
    print("WFH FEATURE ENGINEERING")
    print("="*70)
    
    df_features = df.copy()
    
    # 1. WFH Intensity Score (0-5 scale based on weekdays)
    wfh_weekdays = ['wfhmon', 'wfhtue', 'wfhwed', 'wfhthu', 'wfhfri']
    existing_weekdays = [col for col in wfh_weekdays if col in df_features.columns]
    
    print(f"Processing WFH weekday columns: {existing_weekdays}")
    
    # Convert to numeric (handle 'Yes'/'No' and other labels explicitly)
    for col in existing_weekdays:
        # Check current values
        unique_vals = df_features[col].unique()
        print(f"{col} unique values: {unique_vals}")
        
        # Map to binary with explicit handling of survey labels
        mapping = {
            'Yes': 1,
            'Y': 1,
            'No': 0,
            'N': 0,
            'Not in Work Force': 0,
            'Missing/Refused': 0,
            'Unknown': 0,
            'NA': 0,
            'N/A': 0
        }
        df_features[col] = df_features[col].map(mapping)
        df_features[col] = pd.to_numeric(df_features[col], errors='coerce').fillna(0)
    
    # Calculate WFH intensity
    if existing_weekdays:
        df_features['wfh_intensity'] = df_features[existing_weekdays].sum(axis=1)
        print(f"WFH intensity distribution:\n{df_features['wfh_intensity'].value_counts().sort_index()}")
    else:
        df_features['wfh_intensity'] = 0
    
    # 2. WFH Adoption Binary (any WFH vs none)
    df_features['wfh_adopter'] = (df_features['wfh_intensity'] > 0).astype(int)
    print(f"WFH adopters: {df_features['wfh_adopter'].sum():,} out of {len(df_features):,} ({df_features['wfh_adopter'].mean()*100:.1f}%)")
    
    # 3. WFH Frequency Category
    def categorize_wfh(score: Union[int, float]) -> str:
        if score == 0:
            return 'No_WFH'
        elif score <= 2:
            return 'Occasional_WFH'
        elif score <= 4:
            return 'Frequent_WFH'
        else:
            return 'Full_WFH'
    
    df_features['wfh_category'] = df_features['wfh_intensity'].apply(categorize_wfh)
    print(f"WFH categories:\n{df_features['wfh_category'].value_counts()}")
    
    # 4. Worker identification
    if 'anywork' in df_features.columns:
        df_features['is_worker'] = (df_features['anywork'] == 'Yes').astype(int)
        print(f"Workers identified: {df_features['is_worker'].sum():,} out of {len(df_features):,}")
    else:
        print("WARNING: anywork column not found, setting all as workers")
        df_features['is_worker'] = 1
    
    return df_features

def create_household_wfh_metrics(persons_df: pd.DataFrame, households_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate WFH metrics to household level"""
    print("\n" + "="*70)
    print("HOUSEHOLD WFH METRICS CREATION")
    print("="*70)
    
    # Group by household
    hh_wfh = persons_df.groupby('hhid').agg({
        'wfh_adopter': ['sum', 'mean'],  # Total adopters & proportion
        'wfh_intensity': ['mean', 'max'],  # Average & max intensity
        'is_worker': 'sum',  # Total workers in household
        'perspoststratweight': 'first'  # Carry weight forward
    }).reset_index()
    
    # Flatten column names
    hh_wfh.columns = ['hhid', 'total_wfh_adopters', 'prop_wfh_adopters',
                      'avg_wfh_intensity', 'max_wfh_intensity', 
                      'total_workers', 'hhweight']
    
    # Flags and robust saturation metric
    hh_wfh['has_worker'] = (hh_wfh['total_workers'] > 0).astype(int)
    hh_wfh['hh_wfh_saturation'] = hh_wfh.apply(
        lambda r: (r['total_wfh_adopters'] / r['total_workers']) if r['total_workers'] > 0 else 0.0,
        axis=1
    )
    
    print(f"Household WFH metrics created for {len(hh_wfh):,} households")
    print(f"Average WFH saturation: {hh_wfh['hh_wfh_saturation'].mean():.3f}")
    
    # Merge with household data
    households_enhanced = households_df.merge(hh_wfh, on='hhid', how='left')
    print(f"Enhanced household dataset: {households_enhanced.shape}")
    
    return households_enhanced

def analyze_travel_start_times(persons_df: pd.DataFrame, trips_df: pd.DataFrame, _: pd.DataFrame) -> pd.DataFrame:
    """Analyze how WFH affects travel start times"""
    print("\n" + "="*70)
    print("TRAVEL START TIME ANALYSIS")
    print("="*70)
    
    # Merge person data with trips
    person_cols = ['persid', 'hhid', 'wfh_intensity', 'wfh_adopter', 
                   'wfh_category', 'perspoststratweight']
    
    # Add wfhtravday if it exists
    if 'wfhtravday' in persons_df.columns:
        person_cols.append('wfhtravday')
    
    trips_with_person = trips_df.merge(
        persons_df[person_cols],
        on='persid',
        how='left'
    )
    
    print(f"Merged trips with person data: {trips_with_person.shape}")
    
    # Convert start hour to numeric
    if 'starthour' in trips_with_person.columns:
        trips_with_person['starthour'] = pd.to_numeric(trips_with_person['starthour'], errors='coerce')
        
        # Filter to morning trips (before 10 AM)
        morning_trips = trips_with_person[trips_with_person['starthour'] < 10].copy()
        print(f"Morning trips (before 10 AM): {len(morning_trips):,}")
        
        # Create time period categories
        def categorize_start_time(hour: Union[float, int]) -> str:
            if pd.isna(hour):
                return 'Unknown'
            elif hour < 6:
                return 'Very_Early'
            elif hour < 7:
                return 'Early'
            elif hour < 8:
                return 'Peak_Early'
            elif hour < 9:
                return 'Peak_Late'
            else:
                return 'Late'
        
        morning_trips['start_time_category'] = morning_trips['starthour'].apply(categorize_start_time)
        print(f"Start time categories:\n{morning_trips['start_time_category'].value_counts()}")
        
        # Apply weights
        morning_trips['weight'] = morning_trips['perspoststratweight']
        
        return morning_trips
    else:
        print("WARNING: starthour column not found in trips data")
        return pd.DataFrame()

def create_person_level_dataset(persons_df: pd.DataFrame, households_df: pd.DataFrame) -> pd.DataFrame:
    """Merge person and household data for classification/regression"""
    print("\n" + "="*70)
    print("CREATING PERSON-LEVEL ANALYSIS DATASET")
    print("="*70)
    
    # Select relevant household features
    hh_cols = ['hhid', 'hhsize', 'dwelltype', 'totalvehs', 'hhinc_group', 'homelga', 'homeregion_ASGS']
    
    # Add household weight if it exists
    if 'hhpoststratweight' in households_df.columns:
        hh_cols.append('hhpoststratweight')
    
    # Filter to existing columns
    existing_hh_cols = [col for col in hh_cols if col in households_df.columns]
    print(f"Using household features: {existing_hh_cols}")
    
    hh_features = households_df[existing_hh_cols]
    
    # Merge
    person_analysis = persons_df.merge(hh_features, on='hhid', how='left')
    print(f"Initial merged dataset: {person_analysis.shape}")
    
    # Filter to workers only
    person_analysis = person_analysis[person_analysis['is_worker'] == 1].copy()
    print(f"Workers only: {person_analysis.shape}")
    
    # Set analysis weight
    person_analysis['analysis_weight'] = person_analysis['perspoststratweight']
    
    return person_analysis

def create_household_level_dataset(households_enhanced: pd.DataFrame, _: pd.DataFrame, trips_df: pd.DataFrame) -> pd.DataFrame:
    """Rich household dataset with travel patterns"""
    print("\n" + "="*70)
    print("CREATING HOUSEHOLD-LEVEL ANALYSIS DATASET")
    print("="*70)
    
    # Aggregate trip statistics by household
    trip_agg_cols = {}
    if 'tripno' in trips_df.columns:
        trip_agg_cols['tripno'] = 'count'
    if 'duration' in trips_df.columns:
        trip_agg_cols['duration'] = 'mean'
    if 'starthour' in trips_df.columns:
        trip_agg_cols['starthour'] = 'mean'
    
    if trip_agg_cols:
        trip_stats = trips_df.groupby('hhid').agg(trip_agg_cols).reset_index()
        
        # Rename columns
        new_names = ['hhid']
        if 'tripno' in trip_agg_cols:
            new_names.append('total_trips')
        if 'duration' in trip_agg_cols:
            new_names.append('avg_trip_duration')
        if 'starthour' in trip_agg_cols:
            new_names.append('avg_start_hour')
        
        trip_stats.columns = new_names
        print(f"Trip statistics aggregated: {trip_stats.shape}")
        
        # Merge with household data
        hh_master = households_enhanced.merge(trip_stats, on='hhid', how='left')
    else:
        print("WARNING: No trip columns found for aggregation")
        hh_master = households_enhanced.copy()
    
    # Set analysis weight
    if 'hhpoststratweight' in hh_master.columns:
        hh_master['analysis_weight'] = hh_master['hhpoststratweight']
    else:
        hh_master['analysis_weight'] = 1.0
    
    print(f"Final household dataset: {hh_master.shape}")
    return hh_master

def create_journey_level_dataset(journey_work_df: pd.DataFrame, persons_df: pd.DataFrame) -> pd.DataFrame:
    """Work journey dataset with person characteristics"""
    print("\n" + "="*70)
    print("CREATING JOURNEY-LEVEL ANALYSIS DATASET")
    print("="*70)
    
    # Select person characteristics
    person_cols = ['persid', 'wfh_intensity', 'wfh_adopter', 'wfh_category', 'perspoststratweight']
    
    # Add additional columns if they exist
    optional_cols = ['agegroup', 'sex', 'carlicence', 'emptype', 'persinc']
    for col in optional_cols:
        if col in persons_df.columns:
            person_cols.append(col)
    
    print(f"Using person characteristics: {person_cols}")
    
    journey_analysis = journey_work_df.merge(
        persons_df[person_cols],
        on='persid',
        how='left'
    )
    
    print(f"Journey analysis dataset: {journey_analysis.shape}")
    
    # Set analysis weight with provenance
    if 'journey_weight' in journey_analysis.columns:
        journey_analysis['analysis_weight'] = journey_analysis['journey_weight']
        journey_analysis['weight_source'] = 'journey_weight'
    else:
        print("WARNING: journey_weight not found, using person weight")
        journey_analysis['analysis_weight'] = journey_analysis['perspoststratweight']
        journey_analysis['weight_source'] = 'perspoststratweight'
    
    return journey_analysis

def handle_missing_values(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Systematic missing value handling"""
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
        
        if missing_pct > 50:
            print(f"  -> DROP column (too many missing)")
            df_clean = df_clean.drop(columns=[col])
        
        elif str(df_clean[col].dtype) in ['object', 'category']:
            print(f"  -> Fill with 'Unknown'")
            df_clean[col] = df_clean[col].fillna('Unknown')
        
        elif str(df_clean[col].dtype) in ['int64', 'float64']:
            if missing_pct < 5:
                print(f"  -> Fill with median")
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                print(f"  -> Fill with 0 and create missing indicator")
                df_clean[f'{col}_missing'] = df_clean[col].isnull().astype(int)
                df_clean[col] = df_clean[col].fillna(0)
    
    print(f"\nCleaning complete. Shape: {df_clean.shape}")
    return df_clean

def detect_outliers(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Dict[str, Union[int, float]]]:
    """IQR-based outlier detection"""
    print(f"\n{'='*70}")
    print("OUTLIER DETECTION")
    print("="*70)
    
    outlier_summary: Dict[str, Dict[str, Union[int, float]]] = {}
    
    for col in numeric_cols:
        if col not in df.columns:
            continue
            
        # Remove non-numeric values
        numeric_data = pd.to_numeric(df[col], errors='coerce').dropna()
        
        if len(numeric_data) == 0:
            continue
            
        Q1 = float(numeric_data.quantile(0.25))
        Q3 = float(numeric_data.quantile(0.75))
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = int(((numeric_data < lower_bound) | (numeric_data > upper_bound)).sum())
        outlier_pct = (outliers / len(numeric_data)) * 100
        
        outlier_summary[col] = {
            'count': outliers,
            'percent': outlier_pct,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        
        print(f"{col}: {outliers} outliers ({outlier_pct:.2f}%)")
    
    return outlier_summary

def cap_outliers(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Cap outliers at 99th percentile"""
    df_capped = df.copy()
    
    for col in numeric_cols:
        if col in df_capped.columns:
            numeric_data = pd.to_numeric(df_capped[col], errors='coerce')
            p99 = float(numeric_data.quantile(0.99))
            df_capped[col] = numeric_data.clip(upper=p99)
            print(f"Capped {col} at 99th percentile: {p99:.2f}")
    
    return df_capped

def create_integration_validation_report(person_master_clean: pd.DataFrame, 
                                       household_master_clean: pd.DataFrame, 
                                       journey_master_clean: pd.DataFrame) -> str:
    """Final validation checks"""
    print(f"\n{'='*70}")
    print("INTEGRATION VALIDATION")
    print("="*70)
    
    report: List[str] = []
    
    # Check 1: Weight preservation
    report.append("Weight Validation:")
    if 'analysis_weight' in person_master_clean.columns:
        report.append(f"  Person weights sum: {person_master_clean['analysis_weight'].sum():,.0f}")
    if 'analysis_weight' in household_master_clean.columns:
        report.append(f"  HH weights sum: {household_master_clean['analysis_weight'].sum():,.0f}")
    
    # Check 2: WFH distribution
    report.append("\nWFH Distribution:")
    report.append(f"  Total workers: {len(person_master_clean)}")
    if 'wfh_adopter' in person_master_clean.columns:
        report.append(f"  WFH adopters: {person_master_clean['wfh_adopter'].sum()}")
        report.append(f"  Adoption rate: {person_master_clean['wfh_adopter'].mean()*100:.1f}%")
    
    # Check 3: Missing values
    report.append("\nMissing Values After Cleaning:")
    for df, name in [(person_master_clean, 'Person'), 
                     (household_master_clean, 'Household'),
                     (journey_master_clean, 'Journey')]:
        missing = df.isnull().sum().sum()
        report.append(f"  {name}: {missing} total missing values")
    
    # Check 4: Key merges successful
    report.append("\nMerge Validation:")
    if 'hhid' in person_master_clean.columns:
        report.append(f"  Unique households in person data: {person_master_clean['hhid'].nunique()}")
    if 'persid' in journey_master_clean.columns:
        report.append(f"  Unique persons in journey data: {journey_master_clean['persid'].nunique()}")
    
    validation_text = "\n".join(report)
    print(validation_text)
    return validation_text

def main() -> None:
    """Main execution function for data integration pipeline"""
    
    # Phase 1: Load datasets
    datasets = load_vista_datasets()
    
    # Explore each dataset
    for name, df in datasets.items():
        if df is not None:
            explore_dataset(df, name)
    
    # Phase 2: Analyze WFH variables
    persons_df = datasets['persons']
    if persons_df is not None:
        analyze_wfh_variables(persons_df)
    
    # Phase 3: Apply weights
    if persons_df is None:
        raise ValueError("Persons dataset is required but not loaded")
    persons_weighted = apply_weights_to_person_data(persons_df)
    
    # Apply weights to journey datasets
    journey_work_df = datasets['journey_work']
    journey_edu_df = datasets['journey_edu']
    
    if journey_work_df is not None and 'journey_weight' in journey_work_df.columns:
        journey_work_df['weight'] = journey_work_df['journey_weight']
    
    if journey_edu_df is not None and 'journey_weight' in journey_edu_df.columns:
        journey_edu_df['weight'] = journey_edu_df['journey_weight']
    
    # Phase 4: Feature engineering
    persons_with_wfh = create_wfh_features(persons_weighted)
    
    # Phase 5: Create household metrics
    households_df = datasets['households']
    if households_df is None:
        raise ValueError("Households dataset is required but not loaded")
    households_enhanced = create_household_wfh_metrics(persons_with_wfh, households_df)
    
    # Phase 6: Analyze travel patterns
    trips_df = datasets['trips']
    stops_df = datasets['stops']
    if trips_df is None or stops_df is None:
        print("Warning: trips or stops dataset not available, skipping travel analysis")
        morning_travel = pd.DataFrame()
    else:
        morning_travel = analyze_travel_start_times(persons_with_wfh, trips_df, stops_df)
    
    # Phase 7: Create master datasets
    person_master = create_person_level_dataset(persons_with_wfh, households_enhanced)
    household_master = create_household_level_dataset(households_enhanced, persons_with_wfh, trips_df if trips_df is not None else pd.DataFrame())
    
    if journey_work_df is None:
        print("Warning: journey_work dataset not available, creating empty journey master")
        journey_master = pd.DataFrame()
    else:
        journey_master = create_journey_level_dataset(journey_work_df, persons_with_wfh)
    
    # Phase 8: Handle missing values
    person_master_clean = handle_missing_values(person_master, "Person Master")
    household_master_clean = handle_missing_values(household_master, "Household Master")
    journey_master_clean = handle_missing_values(journey_master, "Journey Master")
    
    # Phase 9: Outlier detection and handling
    # Identify numeric columns for outlier detection
    journey_numeric_cols = ['journey_travel_time', 'journey_distance', 'journey_elapsed_time']
    if not journey_master_clean.empty:
        _ = detect_outliers(journey_master_clean, journey_numeric_cols)
        
        # Cap outliers
        journey_master_clean = cap_outliers(journey_master_clean, journey_numeric_cols)
    
    # Phase 10: Export datasets
    print(f"\n{'='*70}")
    print("EXPORTING PROCESSED DATASETS")
    print("="*70)
    
    # Save processed datasets
    output_dir = '../../01_preprocessing/outputs'
    
    person_master_clean.to_csv(f'{output_dir}/processed_person_master.csv', index=False)
    print(f"Saved: processed_person_master.csv ({person_master_clean.shape})")
    
    household_master_clean.to_csv(f'{output_dir}/processed_household_master.csv', index=False)
    print(f"Saved: processed_household_master.csv ({household_master_clean.shape})")
    
    journey_master_clean.to_csv(f'{output_dir}/processed_journey_master.csv', index=False)
    print(f"Saved: processed_journey_master.csv ({journey_master_clean.shape})")
    
    if not morning_travel.empty:
        morning_travel.to_csv(f'{output_dir}/processed_morning_travel.csv', index=False)
        print(f"Saved: processed_morning_travel.csv ({morning_travel.shape})")
    
    # Also save Parquet for efficiency
    try:
        person_master_clean.to_parquet(f'{output_dir}/processed_person_master.parquet', index=False)
        household_master_clean.to_parquet(f'{output_dir}/processed_household_master.parquet', index=False)
        journey_master_clean.to_parquet(f'{output_dir}/processed_journey_master.parquet', index=False)
        if not morning_travel.empty:
            morning_travel.to_parquet(f'{output_dir}/processed_morning_travel.parquet', index=False)
        print("Saved Parquet versions of processed datasets")
    except Exception as e:
        print(f"Parquet save skipped or failed: {e}")

    # Create data dictionary
    data_dict: Dict[str, Dict[str, Any]] = {
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
    
    with open(f'{output_dir}/data_dictionary.json', 'w') as f:
        json.dump(data_dict, f, indent=2)
    print("Saved: data_dictionary.json")
    
    # Phase 11: Validation report
    validation_report = create_integration_validation_report(
        person_master_clean, household_master_clean, journey_master_clean
    )
    
    with open(f'{output_dir}/integration_validation_report.txt', 'w') as f:
        f.write(validation_report)
    print("Saved: integration_validation_report.txt")
    
    print(f"\n{'='*70}")
    print("DATA INTEGRATION COMPLETE!")
    print("="*70)
    print(f"Person master: {len(person_master_clean)} rows")
    print(f"Household master: {len(household_master_clean)} rows")
    print(f"Journey master: {len(journey_master_clean)} rows")
    print(f"All files saved to: {output_dir}/")

if __name__ == "__main__":
    main()