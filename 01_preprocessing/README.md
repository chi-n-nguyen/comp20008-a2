# Data Preprocessing Pipeline

Rigorous data integration and quality control for VISTA 2023-2024 WFH adoption analysis.

## Use of Gen AI Declaration
I acknowledge my use of Claude [claude.ai] to assist with code development for this project's data preprocessing. 

## Overview

Multi-dataset integration with methodological quality controls:
- **Data sources**: Households, persons, trips, journeys, stops
- **Integration**: Hierarchical identifiers (person -> household -> journey)
- **Quality assurance**: Target variable validation, survey weighting, zero missing values
- **Output**: Analysis-ready datasets with human-readable variable names

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

## Pipeline Components

### Survey Weighting (`weights.py`)
**Population representativeness** via post-stratification weights:
- `perspoststratweight`: Individual-level analysis
- `hhpoststratweight`: Household-level analysis  
- `analysis_weight`: Auto-selected unified field

**Critical**: Always weight analyses. Unweighted results misrepresent Melbourne population.

### Quality Control (`validate.py`)
**Methodological rigor**:
- Missing value handling: Drop >80% missing columns, median imputation
- Outlier management: IQR detection, 99th percentile capping
- Data type standardization across all WFH variables

### Feature Engineering (`features.py`)
**WFH metrics creation**:
- Target variable: `wfh_adopter` (includes travel day workers)
- Intensity scales: 0-7 days/week (consistent across datasets)
- Household aggregations: Worker count, saturation, intensity averages

### Integration (`aggregates.py`)
**Multi-level dataset construction**:
- Person-household merging with conflict resolution
- Trip pattern aggregation to household level  
- Journey-to-work specialized dataset creation

## Output Datasets

**Production-ready data** with quality guarantees:

### Core Datasets
- **`processed_person_master.csv`** (4,361 workers): Individual WFH analysis
- **`processed_household_master.csv`** (3,239 households): Clustering analysis  
- **`processed_journey_master.csv`** (1,819 commuters): Journey correlation analysis
- **`processed_morning_travel.csv`** (5,742 trips): Travel pattern analysis

### ML-Ready Versions  
- **`*_readable.csv`**: Human-interpretable variable names
- **`*_feature_dictionary.json`**: Complete feature documentation

**Variable transformation examples**:
- `hhpoststratweight_GROUP_1` -> `weight_demo_group_1`
- `anzsco1` -> `occupation_major_group`
- `agegroup` -> `age_group`

### Quality Metrics
- **Zero missing values** across all datasets
- **Consistent data types** (all WFH variables as int64)
- **Validated target variables** (WFH adoption consistency: 99.86%)
- **Survey weights applied** throughout

## Usage

**Execute pipeline**:
```bash
cd scripts && python data_integration.py
```

**Validation outputs**:
- Console logs with data quality metrics
- `data_dictionary.json` with complete dataset metadata
- Processed datasets ready for analysis

## Quality Assurance

**Critical fixes implemented**:
1. **WFH target consistency**: Resolved 10 cases where `anywfh='Yes'` but `wfh_adopter=0`
2. **Survey weight application**: Proper weights for population representativeness  
3. **Data type standardization**: All WFH variables converted to consistent numeric format
4. **Duplicate column resolution**: Eliminated merge conflicts in geographic variables

**Result**: Publication-ready datasets with methodological rigor supporting robust analysis.
