"""Configuration for preprocessing pipeline.

- Centralizes paths to raw data and outputs
- Defines column groups, mappings, and thresholds
- Imported by other modules to avoid hard-coded literals
"""

from typing import Dict, List

# Paths
RAW_DIR = "../../00_raw_data"
OUTPUT_DIR = "../../01_preprocessing/outputs"

# Filenames
FILENAMES: Dict[str, str] = {
    'households': f"{RAW_DIR}/household_vista_2023_2024.csv",
    'persons': f"{RAW_DIR}/person_vista_2023_2024.csv",
    'trips': f"{RAW_DIR}/trips_vista_2023_2024.csv",
    'stops': f"{RAW_DIR}/stops_vista_2023_2024.csv",
    'journey_work': f"{RAW_DIR}/journey_to_work_vista_2023_2024.csv",
    'journey_edu': f"{RAW_DIR}/journey_to_education_vista_2023_2024.csv",
}

# Column groups
WFH_WEEKDAYS: List[str] = ['wfhmon', 'wfhtue', 'wfhwed', 'wfhthu', 'wfhfri']
WFH_WEEKENDS: List[str] = ['wfhsat', 'wfhsun']
WFH_ALL_DAYS: List[str] = WFH_WEEKDAYS + WFH_WEEKENDS
WFH_ALL: List[str] = WFH_ALL_DAYS + ['wfhtravday', 'anywfh']
PERSON_WEIGHT = 'perspoststratweight'
JOURNEY_WEIGHT = 'journey_weight'

# Mappings
WFH_MAPPING: Dict[str, int] = {
    'Yes': 1, 'Y': 1,
    'No': 0, 'N': 0,
    'Not in Work Force': 0,
    'Missing/Refused': 0,
    'Unknown': 0,
    'NA': 0, 'N/A': 0
}

# Thresholds
MISSING_DROP_THRESHOLD = 50.0
NUMERIC_MEDIAN_THRESHOLD = 5.0


