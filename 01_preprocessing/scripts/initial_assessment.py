"""Initial data assessment script.

Loads VISTA datasets and produces:
- dataset shapes, missing data, and duplicates
- key linkage analysis (household-person, person-journey)
- WFH-related column discovery and weight summaries
- overview visualization saved to outputs

Usage:
  cd 01_preprocessing/scripts && python initial_assessment.py
"""
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt

# =====================================
# Research Question: "What factors predict working-from-home (WFH) adoption?"
# Responsible: Nhat Chi Nguyen, 1492182
# =====================================

def load_vista_datasets():
    """Load all VISTA 2023-2024 datasets with proper error handling"""
    
    print("="*70)
    print("VISTA 2023-2024 DATASET LOADING")
    print("Research Question: WFH adoption factors")
    print("="*70)
    
    # Create outputs directory
    os.makedirs('../../01_preprocessing/outputs', exist_ok=True)
    
    # Dataset file paths - corrected to match actual structure
    datasets = {}
    file_paths = {
        'household': '../../00_raw_data/household_vista_2023_2024.csv',
        'person': '../../00_raw_data/person_vista_2023_2024.csv', 
        'trips': '../../00_raw_data/trips_vista_2023_2024.csv',
        'stops': '../../00_raw_data/stops_vista_2023_2024.csv',
        'work_journeys': '../../00_raw_data/journey_to_work_vista_2023_2024.csv',
        'edu_journeys': '../../00_raw_data/journey_to_education_vista_2023_2024.csv'
    }
    
    print("Loading datasets...")
    print("-" * 50)
    
    for name, path in file_paths.items():
        try:
            print(f"Loading {name}...", end=" ")
            # Suppress dtype warnings for cleaner output
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

def immediate_data_quality_assessment(datasets):
    """Immediate data quality assessment including expansion weights analysis"""
    
    print("\n" + "="*70)
    print("IMMEDIATE DATA QUALITY ASSESSMENT")
    print("="*70)
    
    # Focus on household and person datasets as specified
    priority_datasets = ['household', 'person']
    
    for df_name in priority_datasets:
        df = datasets.get(df_name)
        if df is None:
            print(f"\n{df_name.upper()}: DATASET NOT AVAILABLE")
            continue
            
        print(f"\n{df_name.upper()} Dataset:")
        print("-" * 30)
        
        # Basic shape info
        print(f"{df_name}: {df.shape}")
        
        # Missing values analysis
        total_missing = df.isnull().sum().sum()
        print(f"Missing values: {total_missing:,}")
        
        # Check for ID columns and duplicates
        if df_name == 'household':
            id_col = 'hhid'
        elif df_name == 'person':
            id_col = 'hhid'  # Using hhid as specified in requirements
        else:
            id_col = None
            
        if id_col and id_col in df.columns:
            duplicate_ids = df.duplicated([id_col]).sum()
            print(f"Duplicate IDs ({id_col}): {duplicate_ids}")
            
            # Unique households/persons
            unique_ids = df[id_col].nunique()
            print(f"Unique {id_col}s: {unique_ids:,}")
        
        # Data types summary
        print(f"Data types: {df.dtypes.value_counts().to_dict()}")
        
        # Expansion weights analysis
        weight_cols = [col for col in df.columns if 'weight' in col.lower()]
        if weight_cols:
            print(f"Expansion weights found: {weight_cols}")
            for weight_col in weight_cols:
                weight_stats = df[weight_col].describe()
                print(f"  {weight_col}: mean={weight_stats['mean']:.2f}, range=({weight_stats['min']:.2f}-{weight_stats['max']:.2f})")
        
        # Key columns preview for WFH research
        if df_name == 'person':
            wfh_related_cols = [col for col in df.columns if any(keyword in col.lower() 
                               for keyword in ['wfh', 'work', 'employ', 'main'])]
            if wfh_related_cols:
                print(f"WFH-related columns found: {len(wfh_related_cols)}")
                print(f"Sample: {wfh_related_cols[:5]}")
        
        if df_name == 'household':
            socio_cols = [col for col in df.columns if any(keyword in col.lower()
                         for keyword in ['inc', 'vehicle', 'dwell', 'size'])]
            if socio_cols:
                print(f"Socioeconomic columns found: {len(socio_cols)}")
                print(f"Sample: {socio_cols[:5]}")

def analyze_dataset_relationships(datasets):
    """Analyze relationships between datasets for integration planning"""
    
    print("\n" + "="*70)
    print("DATASET RELATIONSHIP ANALYSIS")
    print("="*70)
    
    # Check key linking columns
    household_df = datasets.get('household')
    person_df = datasets.get('person')
    work_journeys_df = datasets.get('work_journeys')
    # trips_df = datasets.get('trips')  # Not used in this function
    
    if household_df is not None and person_df is not None:
        print("\nHOUSEHOLD <-> PERSON Linkage:")
        print("-" * 30)
        
        # Check hhid overlap
        hh_ids = set(household_df['hhid'].unique()) if 'hhid' in household_df.columns else set()
        person_hh_ids = set(person_df['hhid'].unique()) if 'hhid' in person_df.columns else set()
        
        print(f"Households in household_df: {len(hh_ids):,}")
        print(f"Households in person_df: {len(person_hh_ids):,}")
        
        if hh_ids and person_hh_ids:
            overlap = len(hh_ids.intersection(person_hh_ids))
            print(f"Overlapping households: {overlap:,}")
            print(f"Match rate: {overlap/len(hh_ids)*100:.1f}%")
    
    if person_df is not None and work_journeys_df is not None:
        print("\nPERSON <-> WORK JOURNEYS Linkage:")
        print("-" * 30)
        
        # Check person ID overlap for work analysis
        person_ids: set = set()
        work_person_ids: set = set()
        
        if 'hhid' in person_df.columns and 'persid' in person_df.columns:
            person_ids = set(zip(person_df['hhid'], person_df['persid']))
        
        if 'hhid' in work_journeys_df.columns and 'persid' in work_journeys_df.columns:
            work_person_ids = set(zip(work_journeys_df['hhid'], work_journeys_df['persid']))
        
        if person_ids and work_person_ids:
            work_overlap = len(person_ids.intersection(work_person_ids))
            print(f"People in person_df: {len(person_ids):,}")
            print(f"People in work_journeys_df: {len(work_person_ids):,}")
            print(f"Working people overlap: {work_overlap:,}")

def preview_key_variables(datasets):
    """Preview key variables for WFH research"""
    
    print("\n" + "="*70)
    print("KEY VARIABLES PREVIEW FOR WFH RESEARCH")
    print("="*70)
    
    person_df = datasets.get('person')
    
    if person_df is not None:
        print("\nWORK-FROM-HOME VARIABLES:")
        print("-" * 40)
        
        # Look for WFH-related columns
        wfh_columns = [col for col in person_df.columns if 'wfh' in col.lower()]
        work_columns = [col for col in person_df.columns if any(keyword in col.lower() 
                       for keyword in ['anywork', 'anywfh', 'mainact', 'emptype'])]
        
        all_relevant = list(set(wfh_columns + work_columns))
        
        print(f"Found {len(all_relevant)} relevant columns:")
        for col in all_relevant[:10]:  # Show first 10
            if col in person_df.columns:
                unique_vals = person_df[col].value_counts().head(3)
                print(f"  {col}: {unique_vals.to_dict()}")
        
        if len(all_relevant) > 10:
            print(f"  ... and {len(all_relevant)-10} more columns")
    
    # Preview work journey characteristics
    work_df = datasets.get('work_journeys')
    if work_df is not None:
        print(f"\nWORK JOURNEY CHARACTERISTICS:")
        print("-" * 40)
        print(f"Work journeys: {len(work_df):,} records")
        
        # Look for travel-related columns
        travel_cols = [col for col in work_df.columns if any(keyword in col.lower()
                      for keyword in ['time', 'distance', 'mode', 'transport'])]
        
        if travel_cols:
            print(f"Travel variables: {travel_cols[:5]}")

def analyze_expansion_weights(datasets):
    """Analyze expansion weights across datasets"""
    
    print("\n" + "="*70)
    print("EXPANSION WEIGHTS ANALYSIS")
    print("="*70)
    
    # Check person weights
    person_df = datasets.get('person')
    if person_df is not None and 'perspoststratweight' in person_df.columns:
        print("\nPERSON EXPANSION WEIGHTS:")
        print("-" * 30)
        weight_col = 'perspoststratweight'
        total_weighted_pop = person_df[weight_col].sum()
        unweighted_count = len(person_df)
        print(f"Unweighted sample size: {unweighted_count:,}")
        print(f"Weighted population estimate: {total_weighted_pop:,.0f}")
        print(f"Average expansion factor: {total_weighted_pop/unweighted_count:.2f}")
        
        # WFH analysis with weights
        if 'anywfh' in person_df.columns:
            wfh_yes = person_df[person_df['anywfh'] == 'Yes']
            wfh_weighted = wfh_yes[weight_col].sum()
            total_workers = person_df[person_df['anywork'] == 'Yes'][weight_col].sum()
            print(f"Weighted WFH adoption rate: {(wfh_weighted/total_workers)*100:.1f}%")
    
    # Check work journey weights
    work_df = datasets.get('work_journeys')
    if work_df is not None and 'journey_weight' in work_df.columns:
        print("\nWORK JOURNEY EXPANSION WEIGHTS:")
        print("-" * 30)
        weight_col = 'journey_weight'
        total_weighted_journeys = work_df[weight_col].sum()
        unweighted_count = len(work_df)
        print(f"Unweighted journey sample: {unweighted_count:,}")
        print(f"Weighted journey population: {total_weighted_journeys:,.0f}")
        print(f"Average expansion factor: {total_weighted_journeys/unweighted_count:.2f}")

def create_data_overview_plot(datasets):
    """Create initial data overview visualization"""
    
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('VISTA 2023-2024 Dataset Overview', fontsize=18, fontweight='bold', y=0.98)
    
    # Define vibrant color palette
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
    
    # Dataset sizes
    dataset_names = []
    dataset_sizes = []
    for name, df in datasets.items():
        if df is not None:
            dataset_names.append(name.replace('_', ' ').title())
            dataset_sizes.append(len(df))
    
    bars1 = ax1.bar(dataset_names, dataset_sizes, color=colors[:len(dataset_names)], alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('Dataset Record Counts', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('Number of Records', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    ax1.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars1, dataset_sizes):
        ax1.text(bar.get_x() + bar.get_width()/2, v + max(dataset_sizes)*0.01, 
                f'{v:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Missing data analysis for key datasets
    key_datasets = ['household', 'person']
    missing_data = []
    for name in key_datasets:
        df = datasets.get(name)
        if df is not None:
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            missing_data.append(missing_pct)
        else:
            missing_data.append(0)
    
    bars2 = ax2.bar(key_datasets, missing_data, color=['#E67E22', '#8E44AD'], alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_title('Missing Data Percentage (Key Datasets)', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('Missing Data (%)', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='y', labelsize=10)
    ax2.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars2, missing_data):
        max_missing = max(missing_data) if missing_data and max(missing_data) > 0 else 1
        ax2.text(bar.get_x() + bar.get_width()/2, v + max_missing*0.05, 
                f'{v:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Data types distribution for person dataset
    person_df = datasets.get('person')
    if person_df is not None:
        dtype_counts = person_df.dtypes.value_counts()
        pie_colors = ['#FF5733', '#33A1FF', '#FFD700', '#FF69B4', '#32CD32', '#FF4500']
        _, texts, autotexts = ax3.pie(dtype_counts.values, 
                                      labels=[str(x) for x in dtype_counts.index], 
                                      autopct='%1.1f%%', 
                                      startangle=90,
                                      colors=pie_colors[:len(dtype_counts)],
                                      explode=[0.05]*len(dtype_counts),
                                      shadow=True)
        ax3.set_title('Person Dataset: Data Types Distribution', fontsize=14, fontweight='bold', pad=20)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        for text in texts:
            text.set_fontsize(10)
            text.set_fontweight('bold')
    
    # Memory usage by dataset
    memory_usage = []
    for name, df in datasets.items():
        if df is not None:
            memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
            memory_usage.append(memory_mb)
        else:
            memory_usage.append(0)
    
    bars4 = ax4.bar(dataset_names, memory_usage, color=colors[:len(dataset_names)], alpha=0.8, edgecolor='black', linewidth=1)
    ax4.set_title('Memory Usage by Dataset', fontsize=14, fontweight='bold', pad=20)
    ax4.set_ylabel('Memory Usage (MB)', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45, labelsize=10)
    ax4.tick_params(axis='y', labelsize=10)
    ax4.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars4, memory_usage):
        max_memory = max(memory_usage) if memory_usage else 1
        ax4.text(bar.get_x() + bar.get_width()/2, v + max_memory*0.01, 
                f'{v:.1f}MB', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout(pad=3.0)
    plt.savefig('../../01_preprocessing/outputs/initial_data_overview.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    


def main():
    """Main execution function for initial dataset assessment"""
    
    
    # Execute initial assessment pipeline
    datasets = load_vista_datasets()
    immediate_data_quality_assessment(datasets)
    analyze_dataset_relationships(datasets)
    preview_key_variables(datasets)
    analyze_expansion_weights(datasets)
    create_data_overview_plot(datasets)
    
    
    return datasets

if __name__ == "__main__":
    datasets = main()
