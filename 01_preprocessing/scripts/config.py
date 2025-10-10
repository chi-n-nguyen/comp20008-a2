"""Configuration for preprocessing pipeline.

- Centralizes paths to raw data and outputs
- Defines column groups, mappings, and thresholds
- Imported by other modules to avoid hard-coded literals
"""


# Paths
RAW_DIR = "../../00_raw_data"
OUTPUT_DIR = "../../01_preprocessing/outputs"

# Filenames
FILENAMES = {
    'households': f"{RAW_DIR}/household_vista_2023_2024.csv",
    'persons': f"{RAW_DIR}/person_vista_2023_2024.csv",
    'journey_work': f"{RAW_DIR}/journey_to_work_vista_2023_2024.csv",
}

# Column groups
WFH_WEEKDAYS = ['wfhmon', 'wfhtue', 'wfhwed', 'wfhthu', 'wfhfri']
WFH_WEEKENDS = ['wfhsat', 'wfhsun']
WFH_ALL_DAYS = WFH_WEEKDAYS + WFH_WEEKENDS
WFH_ALL = WFH_ALL_DAYS + ['wfhtravday', 'anywfh']
PERSON_WEIGHT = 'perspoststratweight'
HOUSEHOLD_WEIGHT = 'hhpoststratweight'
JOURNEY_WEIGHT = 'journey_weight'

# Mappings
WFH_MAPPING = {
    'Yes': 1, 'Y': 1,
    'No': 0, 'N': 0,
    'Not in Work Force': 0,
    'Missing/Refused': 0,
    'Unknown': 0,
    'NA': 0, 'N/A': 0
}

# Thresholds for Missing Data Strategy
# Based on empirical analysis of VISTA 2023-2024 core datasets:
# - Person dataset (n=8,175): All WFH variables have 0% missingness; only survey weights have 1.8% missing
# - Household dataset (n=3,239): Only income groups have 2.7% missingness  
# - Journey to Work dataset (n=1,819): 98/124 columns have >50% missingness (extended trip legs 7-15)

# Column Drop Threshold: 50% missingness
# Rationale: Columns exceeding 50% missingness provide insufficient information for reliable 
# statistical inference, particularly when applying survey weights. In our streamlined WFH analysis:
# - All core WFH variables (anywfh, wfhmon-wfhfri) have 0% missingness
# - Journey dataset variables >50% missing are extended trip chain components (legs 7-15) with 99.9% missingness
# - These high-missingness columns represent rare multi-modal journeys irrelevant to WFH prediction
# - Conservative 50% threshold ensures reliable statistical inference while retaining informative variables
MISSING_DROP_THRESHOLD = 50.0

# Numeric Median Imputation Threshold: 5% missingness  
# Rationale: Variables with <5% missingness can be median-imputed with negligible distributional
# impact in large samples (n=8,175). Applied to survey weights (1.8% missing) and household
# income groups (2.7% missing). For 5-50% missingness, we use zero-filling with missing 
# indicators to preserve potentially informative missingness patterns for tree-based models.
# - Median imputation maintains distributional properties for correlation analysis
# - Tree-based models can learn from missingness patterns via indicator variables  
# - Conservative threshold ensures minimal bias introduction (<2% in distribution moments)
NUMERIC_MEDIAN_THRESHOLD = 5.0


