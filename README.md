# Working From Home Adoption Analysis

**Research Question: "What factors predict working-from-home (WFH) adoption?"**

## Team Members

| Name | Role | Student ID | GitHub |
|------|------|------------|--------|
| Nhat Chi Nguyen | Team Lead + Data Integration | 1492182 | chi-n-nguyen |
| Wanhui Li | Supervised Learning Models | 1450755 | Waiola77 |
| Chang Lu | Correlation Analysis | 1551448 | 9uchang |
| Jingxuan Bai | Clustering & Visualisation | 1566889 | BBrianUU |

## Overview

We investigate WFH adoption predictors using VISTA 2023-2024 data representing 8,175 Melbourne residents across 3,239 households. Analysis integrates multi-level survey data through correlation analysis, supervised learning (Random Forest, Logistic Regression), and K-means clustering to identify systematic patterns in post-pandemic remote work adoption.

## **Project Structure**

```
comp20008-a2/
├── 00_raw_data/                    # Original VISTA 2023-2024 datasets
│   ├── household_vista_2023_2024.csv
│   ├── person_vista_2023_2024.csv
│   ├── trips_vista_2023_2024.csv
│   ├── stops_vista_2023_2024.csv
│   ├── journey_to_work_vista_2023_2024.csv
│   └── journey_to_education_vista_2023_2024.csv
├── 01_preprocessing/               # Data cleaning and feature engineering
│   ├── README.md                   # Preprocessing pipeline documentation
│   ├── scripts/
│   │   ├── initial_assessment.py
│   │   ├── data_integration.py     # Main pipeline orchestrator
│   │   ├── config.py
│   │   ├── features.py
│   │   ├── aggregates.py
│   │   ├── validate.py
│   │   ├── variable_mapping.py
│   │   └── weights.py
│   └── outputs/
│       ├── initial_data_overview.png
│       ├── processed_person_master.csv
│       ├── processed_person_master_readable.csv
│       ├── processed_person_master_readable_feature_dictionary.json
│       ├── processed_household_master.csv
│       ├── processed_household_master_readable.csv
│       ├── processed_household_master_readable_feature_dictionary.json
│       ├── processed_journey_master.csv
│       ├── processed_morning_travel.csv
│       ├── processed_stops_analysis.csv
│       ├── data_dictionary.json
│       └── integration_validation_report.txt
├── 02_correlation_analysis/        # Statistical correlation analysis
│   ├── scripts/
│   │   ├── Person_Factor.py        # Person-level factor analysis
│   │   └── Commute_Factor.py       # Commute pattern analysis
│   └── outputs/
│       ├── Age_vs_Occupation.png
│       └── Journey_Mode_Purpose_Time.png
├── 03_supervised_learning/         # Machine learning models
│   ├── scripts/
│   │   ├── main.py                 # Main WFH prediction pipeline
│   │   ├── data_loader.py          # Data loading utilities
│   │   ├── preprocessing.py        # Feature preprocessing
│   │   ├── models.py               # ML model implementations
│   │   ├── evaluation.py           # Model evaluation metrics
│   │   └── visualization.py        # Results visualization
│   └── outputs/
│       ├── feature_importance.png
│       └── wfh_prediction_results.png
├── 04_clustering/                  # Unsupervised learning and segmentation
│   ├── 1.py                        # Clustering implementation script 1
│   ├── 2.py                        # Clustering implementation script 2
│   ├── 3.py                        # Clustering implementation script 3
│   └── outputs/
│       ├── household_clusters_pca.png
│       ├── household_geographic_cluster_distribution.png
│       ├── househould_cluster_optimization.png
│       ├── person_cluster_optimization.png
│       └── person_household_clusters_pca.png
├── cluster_optimization.png        # Additional clustering outputs
├── geographic_cluster_distribution.png
├── household_clusters_pca.png
└── README.md
```

## Data

**VISTA 2023-2024** comprehensive travel diary representing Melbourne through stratified sampling:

- **Household** (3,239): Dwelling, vehicles, composition
- **Person** (8,175): Demographics, employment, WFH frequency  
- **Trips** (24,457): Modes, purposes, timing
- **Stops** (27,862): Locations, durations
- **Journey-to-work** (1,819): Complete commute chains
- **Journey-to-education** (684): Student travel

**Survey Weights**: Post-stratification weights (`perspoststratweight`, `hhpoststratweight`) enable population-level inference. All analyses weight appropriately to ensure Melbourne representativeness.

## Analysis Methods

### 1. Data Preprocessing
Rigorous pipeline with quality controls:
- Multi-dataset integration using hierarchical identifiers
- WFH target variable validation (resolved inconsistencies)
- Survey weight application throughout
- Zero missing values, consistent data types

### 2. Correlation Analysis
Statistical association testing:
- **Person Factor Analysis**: Demographics, occupation, and WFH adoption patterns
- **Commute Factor Analysis**: Journey patterns, transport modes, and travel characteristics
- Survey-weighted analyses for population validity

### 3. Supervised Learning
Predictive modeling with hyperparameter optimization:
- **Random Forest**: Feature importance, handles mixed data types
- **Logistic Regression**: Interpretable coefficients, baseline performance
- 5-fold cross-validation, F1-score optimization, sample weighting

### 4. Clustering Analysis
Household behavioral segmentation:
- **K-means clustering**: Household WFH profiles using composition, travel, adoption metrics
- **Geographic validation**: Chi-square tests for spatial clustering patterns
- Optimal k via elbow method and silhouette analysis

## Key Features

**Demographics**: Age groups, gender, income, employment type, occupation (ANZSCO)  
**Work Patterns**: WFH frequency (0-7 days/week), employment status, main activity  
**Household**: Size, vehicles, dwelling type, income group, composition  
**Travel**: Commute distance/time, transport modes, trip patterns  
**Geographic**: Melbourne regions, local government areas

## Requirements

**Python 3.11+** with dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## Quick Start

1. **Process data**:
   ```bash
   cd 01_preprocessing/scripts && python data_integration.py
   ```

2. **Run correlation analysis**:
   ```bash
   cd 02_correlation_analysis/scripts
   python Person_Factor.py
   python Commute_Factor.py
   ```

3. **Execute ML models**:
   ```bash
   cd 03_supervised_learning/scripts && python main.py
   ```

4. **Perform clustering**:
   ```bash
   cd 04_clustering
   python 1.py
   python 2.py
   python 3.py
   ```

## Output Files

**Preprocessing**:
- `processed_person_master.csv` (4,361 workers)
- `processed_household_master.csv` (3,239 households)  
- `processed_journey_master.csv` (1,819 commuters)
- Readable versions with human-interpretable variable names

**Analysis Results**:
- Correlation plots with statistical measures
- ML model performance and feature importance
- Household cluster profiles and geographic distributions

## Expected Findings

- **High predictability**: 99.9-100% ML accuracy suggests structural WFH determination
- **Occupational dominance**: Job requirements outweigh personal preferences
- **Household patterns**: Distinct WFH adoption profiles across household types
- **Policy insights**: Targeted interventions for specific demographic/occupational groups

---

**Course**: COMP20008 Elements of Data Processing | **Semester**: 2, 2025 | **Group**: W04G5
