# Data Preprocessing Pipeline

This directory contains the preprocessing pipeline for the VISTA 2023-2024 travel survey analysis.

## Overview

The preprocessing stage handles data cleaning, integration, and feature engineering across multiple VISTA datasets:
- Households
- Persons
- Trips
- Journey to work
- Journey to education
- Stops

## Directory Structure

```
01_preprocessing/
├── outputs/              # Processed datasets and reports
│   ├── data_dictionary.json
│   ├── initial_data_overview.png
│   ├── integration_validation_report.txt
│   ├── processed_household_master.csv
│   ├── processed_journey_master.csv
│   ├── processed_morning_travel.csv
│   └── processed_person_master.csv
└── scripts/             # Processing scripts
    ├── aggregates.py    # Dataset construction utilities
    ├── config.py        # Configuration parameters
    ├── data_integration.py  # Main pipeline orchestrator
    ├── features.py      # Feature engineering functions
    ├── initial_assessment.py  # Data exploration
    ├── validate.py      # Data validation and cleaning
    └── weights.py       # Weight handling utilities
```

## Key Features

### Survey Weights (`weights.py`)
VISTA uses complex weighting schemes for population representativeness:

#### Weight Types
- **`perspoststratweight`**: Individual-level population expansion (use for person analysis)
- **`hhpoststratweight`**: Household-level population expansion (use for household analysis)  
- **`analysis_weight`**: Unified weight field auto-selected based on dataset type

#### Usage Example
```python
# CORRECT: Weighted analysis
weighted_avg = (df['variable'] * df['analysis_weight']).sum() / df['analysis_weight'].sum()

# INCORRECT: Unweighted analysis (not population-representative)
unweighted_avg = df['variable'].mean()
```

### Data Validation (`validate.py`)
- Column presence validation
- Missing value handling with configurable thresholds
- Outlier detection using IQR method
- Value capping at 99th percentile

### Data Integration (`aggregates.py`)
- Morning travel time categorization
- Person-level dataset creation with household attributes
- Household-level aggregation with trip statistics
- Work journey dataset with person characteristics

## Output Datasets

The pipeline produces four main processed datasets:
1. `processed_person_master.csv` - Individual-level data for WFH analysis (weight: `analysis_weight`)
2. `processed_household_master.csv` - Household profiles for clustering (weight: `analysis_weight`)
3. `processed_journey_master.csv` - Work journey analysis (weight: `journey_weight`)
4. `processed_morning_travel.csv` - Morning travel patterns

All datasets include appropriate survey weights for population-representative analysis.

## Usage

1. Ensure raw data files are present in `00_raw_data/`
2. Run the preprocessing pipeline:

```python
python scripts/data_integration.py
```

3. Review validation reports in `outputs/`

## Configuration

Key parameters can be adjusted in `config.py`:
- `MISSING_DROP_THRESHOLD`: Column drop threshold for missing values
- `NUMERIC_MEDIAN_THRESHOLD`: Threshold for median imputation

## Important Notes

**Survey Weights**: Always use appropriate weights in analysis to ensure population representativeness. The `analysis_weight` column is pre-configured for each dataset type. Unweighted analysis will not reflect true Melbourne population patterns.