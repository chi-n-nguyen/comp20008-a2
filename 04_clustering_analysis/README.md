# Clustering Analysis

This directory contains unsupervised learning analysis for identifying household WFH adoption patterns.

## Overview

The clustering analysis addresses the research question by segmenting households based on WFH adoption patterns to identify distinct household profiles and key factors that predict WFH adoption.

## Directory Structure

```
04_clustering_analysis/
├── README.md                           # This documentation
├── scripts/
│   ├── household_wfh_clustering.py     # Main clustering implementation
│   └── geographic_cluster_analysis.py  # Geographic distribution analysis
└── outputs/
    ├── cluster_optimization.png        # Elbow and silhouette analysis
    ├── household_clusters_pca.png      # PCA visualization of clusters
    ├── cluster_profiles_heatmap.png    # Feature comparison heatmap
    ├── household_cluster_assignments.csv  # Cluster assignments
    ├── clustering_summary_report.txt   # Detailed cluster analysis
    ├── geographic_cluster_*.png        # Geographic distribution plots
    └── geographic_analysis_report.txt  # Geographic clustering analysis
```

## Analysis Components

### 1. Household WFH Profile Clustering (`household_wfh_clustering.py`)

**Objective**: Identify distinct household profiles based on WFH adoption and travel patterns

**Features Used**:
- `hhsize`: Household size
- `totalvehs`: Number of vehicles owned
- `num_workers`: Number of working household members
- `num_wfh_adopters`: Number of WFH adopters
- `total_trips`: Total household trip frequency
- `avg_trip_duration`: Average trip duration
- `wfh_saturation`: Proportion of workers who WFH

**Method**:
- K-means clustering with standardized features
- Optimal cluster selection using elbow method and silhouette analysis
- Principal Component Analysis (PCA) for visualization

**Outputs**:
- Cluster optimization plots showing optimal K selection
- PCA scatter plot of households colored by cluster
- Cluster profile comparison heatmap
- Individual household cluster assignments

### 2. Geographic Distribution Analysis (`geographic_cluster_analysis.py`)

**Objective**: Examine how WFH clusters are distributed across geographic areas

**Analysis Methods**:
- Chi-square test for independence between geography and clusters
- Herfindahl-Hirschman Index (HHI) for geographic concentration measurement
- Cross-tabulation analysis of cluster-geography relationships

**Visualizations**:
- Stacked bar charts showing cluster distribution by area
- Heatmaps of geographic-cluster proportions
- Grouped bar charts for direct comparison
- Concentration index visualization

## Key Research Insights

### Cluster Interpretation Framework

The clustering analysis identifies household segments that represent different responses to WFH opportunities:

1. **High WFH Adoption Households**: Large households with multiple workers, high WFH saturation
2. **Traditional Commuter Households**: Car-dependent, low WFH adoption, high travel frequency
3. **Small Flexible Households**: Small size, moderate WFH, efficient travel patterns
4. **Mixed Pattern Households**: Variable WFH adoption with diverse travel behaviors

### Geographic Patterns

- **Urban Core**: Higher concentration of flexible WFH households
- **Suburban Areas**: Traditional commuter patterns dominate
- **Transport Corridors**: Mixed adoption patterns along public transport routes

## Usage Instructions

### Running the Analysis

1. **Ensure prerequisites are met**:
   ```bash
   # Verify processed data exists
   ls ../01_preprocessing/outputs/processed_household_master.csv
   ```

2. **Execute main clustering analysis**:
   ```bash
   cd scripts/
   python household_wfh_clustering.py
   ```

3. **Run geographic distribution analysis**:
   ```bash
   python geographic_cluster_analysis.py
   ```

### Dependencies

- pandas, numpy: Data manipulation
- scikit-learn: Clustering algorithms and preprocessing
- matplotlib, seaborn: Visualization
- scipy: Statistical testing

### Configuration Options

**Clustering Parameters** (in `household_wfh_clustering.py`):
- `K_range`: Range of cluster numbers to test (default: 2-8)
- `random_state`: Reproducibility seed (default: 42)
- `clustering_features`: List of features used for clustering

**Analysis Features** (customizable):
- Add/remove features based on research focus
- Adjust missing value handling strategies
- Modify visualization color schemes

## Interpretation Guidelines

### Cluster Validation

- **Silhouette Score**: Measures cluster separation quality (higher = better)
- **Elbow Method**: Identifies optimal number of clusters
- **Geographic Distribution**: Tests if clusters have meaningful spatial patterns

### Statistical Significance

- **Chi-square Test**: Tests independence between geography and clusters
- **p < 0.05**: Significant geographic clustering pattern
- **HHI Index**: Measures geographic concentration (higher = more concentrated)

### Business/Policy Implications

1. **Transportation Planning**: Identify areas needing infrastructure adaptation
2. **Urban Development**: Guide mixed-use development in high WFH areas
3. **Public Transport**: Adjust services based on changing travel patterns
4. **Policy Targeting**: Focus WFH support programs on specific household types

## Limitations and Considerations

### Data Limitations
- Geographic granularity depends on available location variables
- Sample size may limit cluster stability in some areas
- Cross-sectional data limits temporal pattern analysis

### Methodological Considerations
- K-means assumes spherical clusters (may not suit all data patterns)
- Feature standardization affects cluster formation
- Optimal K selection involves subjective interpretation

### Future Enhancements
- Temporal clustering to identify changing patterns
- Integration with external datasets (demographics, transport infrastructure)
- Advanced clustering methods (hierarchical, density-based)
- Cluster stability analysis across different time periods

## Results Integration

The clustering results support the research question by:

1. **Identifying WFH Predictors**: Cluster profiles reveal household characteristics that predict WFH adoption
2. **Household Pattern Recognition**: Different clusters show distinct household composition and work arrangements
3. **Geographic Insights**: Spatial distribution reveals how location influences WFH adoption patterns
4. **Policy Relevance**: Actionable insights for supporting remote work adoption