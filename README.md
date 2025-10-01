# Working From Home and Household Travel Behavior Analysis

## **Research Question**

**"What factors predict working-from-home adoption, and how do WFH patterns reshape household travel behavior?"**

## **Team Members**

| Name | Role | Student ID | GitHub Username |
|------|------|------------|-----------------|
| Nhat Chi Nguyen | Data Integration | 1492182 | chi-n-nguyen |
| Wanhui Li | Supervised Learning Models | 1450755 | Waiola77 |
| Chang Lu | Correlation Analysis | 1551448 | 9uchang |
| Jingxuan Bai | Clustering & Visualization | 1566889 | - |

## **Project Overview**

This project analyzes VISTA 2023-2024 travel survey data to understand the factors that influence working-from-home (WFH) adoption and how these patterns affect household travel behavior in Melbourne. The analysis combines multiple statistical and machine learning approaches to provide comprehensive insights into post-pandemic work and travel patterns.

## **Project Structure**

```
eodp-assignment-2/
├── 00_raw_data/                    # Original VISTA 2023-2024 datasets
│   ├── household_vista_2023_2024.csv
│   ├── person_vista_2023_2024.csv
│   ├── trips_vista_2023_2024.csv
│   ├── stops_vista_2023_2024.csv
│   ├── journey_to_work_vista_2023_2024.csv
│   └── journey_to_education_vista_2023_2024.csv
├── 01_preprocessing/               # Data cleaning and feature engineering
│   ├── scripts/
│   │   ├── 11_initial_assessment.py
│   │   └── 12_feature_engineering.py
│   └── outputs/
│       └── initial_data_overview.png
├── 02_correlation_analysis/        # Statistical correlation analysis
├── 03_supervised_learning/         # Machine learning models
├── 04_clustering_analysis/         # Unsupervised learning and segmentation
├── 05_visualisation/              # Final visualizations and plots
├── 06_documentation/              # Reports and documentation
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

## **Analysis Components**

### 1. Data Preprocessing & Integration
- Dataset loading and quality assessment
- Missing value analysis and treatment
- Feature engineering for WFH metrics
- Data integration across multiple tables

### 2. Correlation Analysis
- Statistical relationships between demographics and WFH adoption
- Commute characteristics vs WFH frequency analysis
- Household factors vs travel behavior correlations

### 3. Supervised Learning
- Binary classification for WFH adoption prediction
- Regression models for WFH intensity prediction
- Feature importance analysis

### 4. Clustering Analysis
- Household WFH profile segmentation
- Travel behavior pattern clustering
- Geographic and demographic clustering

### 5. Visualization
- Comprehensive data overview plots
- Statistical analysis visualizations
- Model performance and results visualization

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
   python 11_initial_assessment.py
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
