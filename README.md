# Working From Home and Household Travel Behavior Analysis

## **Research Question**

**"What factors predict working-from-home (WFH) adoption, and how do WFH patterns reshape household travel behavior?"**

## **Team Members**

| Name | Role | Student ID | GitHub Username |
|------|------|------------|-----------------|
| Nhat Chi Nguyen | Team Lead + Data Integration | 1492182 | chi-n-nguyen |
| Wanhui Li | Supervised Learning Models | 1450755 | Waiola77 |
| Chang Lu | Correlation Analysis | 1551448 | 9uchang |
| Jingxuan Bai | Clustering & Visualisation | 1566889 | BBrianUU |

## **Project Overview**

This project analyzes VISTA 2023-2024 travel survey data to understand the factors that influence working-from-home (WFH) adoption and how these patterns affect household travel behavior in Melbourne. The analysis combines multiple statistical and machine learning approaches to provide comprehensive insights into post-pandemic work and travel patterns.

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
│   │   └── weights.py
│   └── outputs/
│       ├── initial_data_overview.png
│       ├── processed_person_master.csv
│       ├── processed_household_master.csv
│       ├── processed_journey_master.csv
│       ├── processed_morning_travel.csv
│       ├── data_dictionary.json
│       └── integration_validation_report.txt
├── 02_correlation_analysis/        # Statistical correlation analysis
│   ├── scripts/
│   │   ├── Agegroup_WFH_Correlation.py    # Age group vs WFH intensity
│   │   └── Occupation_WFH_Correlation.py  # Occupation vs WFH adoption
│   └── outputs/
│       ├── adoption_and_intensity.png
│       └── age_group_vs_wfh_intensity_nmi_original.png
├── 03_supervised_learning/         # Machine learning models
│   ├── scripts/
│   │   └── wfh_prediction.py       # WFH prediction models
│   └── outputs/
│       ├── feature_importance.png
│       └── wfh_prediction_results.png
└── README.md
```

## **Datasets**

The project uses six VISTA 2023-2024 datasets:

- **Household Data** (3,239 records): Household characteristics, income, vehicles
- **Person Data** (8,175 records): Individual demographics, employment, WFH patterns
- **Trips Data** (24,457 records): Individual trip details and travel patterns
- **Stops Data** (27,862 records): Trip stops and destinations
- **Journeys to Work** (1,819 records): Specific work-related travel patterns
- **Journeys to Education** (684 records): Education-related travel data

### **Expansion Weights**

VISTA data includes expansion weights to ensure population representativeness:
- **Person Data**: `perspoststratweight` - expands sample to Melbourne population
- **Journey Data**: `journey_weight` - expands work journey sample to population
- **Application**: All analyses multiply values by corresponding weights before statistical operations
- **Purpose**: Results reflect actual population distribution rather than survey sample bias

## **Analysis Components**

### 1. Data Preprocessing & Integration
- Dataset loading and quality assessment
- Missing value analysis and treatment
- Feature engineering for WFH metrics
- Data integration across multiple tables

### 2. Correlation Analysis
- **Age Group vs WFH Intensity**: Normalized Mutual Information analysis of age demographics and WFH frequency
- **Occupation vs WFH Adoption**: Chi-square test examining professional categories and remote work patterns
- Statistical relationships using weighted survey data with post-stratification weights

### 3. Supervised Learning
- **WFH Prediction Models**: Machine learning algorithms for predicting remote work adoption
- Feature importance analysis identifying key demographic and travel predictors
- Model performance comparison and evaluation metrics

### 4. Clustering Analysis
- Household WFH profile segmentation
- Travel behavior pattern clustering  
- Geographic and demographic clustering
*(Note: This component will be implemented as the project progresses)*

### 5. Visualization
- Statistical correlation heatmaps and distribution plots
- Model performance visualizations and feature importance charts
- WFH adoption patterns across demographic groups

## **Key Features Analyzed**

- **Demographics**: Age, gender, income, employment type
- **Work Patterns**: WFH frequency by day, employment status, main activity
- **Travel Behavior**: Commute distance, travel time, mode choice
- **Household Characteristics**: Size, vehicle ownership, dwelling type
- **Geographic Factors**: Regional distribution of WFH patterns

## **Technologies Used**

- **Python 3.11+**
- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Statistical Analysis**: scipy
- **Development**: VS Code, Git

## **Getting Started**

1. Clone the repository
2. Install required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scipy
   ```
3. Run the initial assessment:
   ```bash
   cd 01_preprocessing/scripts
   python initial_assessment.py
   ```

4. Run the full data integration pipeline (saves processed datasets under `01_preprocessing/outputs`):
   ```bash
   cd 01_preprocessing/scripts
   python data_integration.py
   ```

## **Running Analysis Components**

### Data Preprocessing
Generate all cleaned and integrated datasets:
```bash
cd 01_preprocessing/scripts
python data_integration.py
```

### Correlation Analysis
Run statistical correlation analyses:
```bash
cd 02_correlation_analysis/scripts
python Agegroup_WFH_Correlation.py     # Age group vs WFH intensity
python Occupation_WFH_Correlation.py   # Occupation vs WFH adoption
```

### Supervised Learning
Execute machine learning models:
```bash
cd 03_supervised_learning/scripts
python wfh_prediction.py               # WFH prediction models
```

### Output Files Generated
- **Preprocessing**: `processed_person_master.csv`, `processed_household_master.csv`, `processed_journey_master.csv`, `processed_morning_travel.csv`
- **Correlation**: `adoption_and_intensity.png`, `age_group_vs_wfh_intensity_nmi_original.png`
- **ML Models**: `feature_importance.png`, `wfh_prediction_results.png`

Optional (faster I/O): install Parquet support
```bash
pip install pyarrow
```

## **Expected Outcomes**

- Identification of key predictors of WFH adoption
- Understanding of how WFH patterns affect household travel behavior
- Segmentation of households based on WFH and travel patterns
- Data-driven insights for transportation and urban planning policy

## **Course Information**

- **Course**: COMP20008, Elements of Data Processing
- **Semester**: Semester 2, 2025
- **Group**: W04G5
