# Clustering Analysis

Household WFH behavioral segmentation using K-means clustering to identify distinct adoption patterns and demographic profiles.

## Overview

**Objective**: Segment 3,239 Melbourne households into behavioral clusters based on WFH adoption, composition, and travel patterns to inform targeted policy interventions.

**Method**: K-means clustering with standardized features, optimal k via elbow method and silhouette analysis, geographic validation via chi-square tests.

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

### 1. Household WFH Clustering (`household_wfh_clustering.py`)

**Features** (7 dimensions, StandardScaler normalized):
- **Composition**: `household_size`, `vehicles_owned`, `num_working_members`
- **WFH patterns**: `num_wfh_workers`, `household_wfh_saturation`  
- **Travel behavior**: `total_trips`, `avg_trip_duration`

**Algorithm**: K-means clustering
- **Rationale**: Scalable, interpretable centroids, established for behavioral segmentation
- **Alternatives considered**: Hierarchical (computational cost), DBSCAN (arbitrary cluster shapes)

**Optimization**:
- **K selection**: Elbow method (WCSS) + silhouette analysis (k∈{2-8})
- **Validation**: Minimum cluster size >5%, silhouette score >0.5
- **Reproducibility**: Fixed random_state=42

**Outputs**:
- Cluster optimization metrics and visualizations
- PCA scatter plot with cluster assignments
- Cluster profile comparison heatmap
- Individual household assignments with labels
- Individual household cluster assignments

### 2. Geographic Validation (`geographic_cluster_analysis.py`)

**Statistical testing** for spatial clustering patterns:

**Methods**:
- **Chi-square test**: Independence between geography and behavioral clusters
- **Herfindahl-Hirschman Index**: Geographic concentration measurement  
- **Cross-tabulation**: Cluster-geography relationship matrices

**Outputs**:
- Statistical significance tests (χ², p-values)
- Geographic concentration indices
- Spatial distribution visualizations
- Evidence for/against geographic clustering

## Expected Cluster Profiles

**Hypothesis-driven interpretation framework**:

1. **High WFH Households**: Large, multi-worker, high saturation, professional-heavy
2. **Traditional Commuters**: Car-dependent, low WFH, high travel frequency, suburban
3. **Flexible Small Households**: Compact, moderate WFH, efficient travel patterns
4. **Mixed Adoption Households**: Variable patterns reflecting diverse occupational mix

**Geographic patterns** (if significant):
- Urban core: Higher flexible WFH concentration
- Suburban: Traditional commuter dominance  
- Transport corridors: Mixed patterns along public transport

## Usage

**Execute clustering pipeline**:
```bash
cd scripts/
python household_wfh_clustering.py    # Main clustering analysis
python geographic_cluster_analysis.py  # Geographic validation
```

**Prerequisites**: Processed household data from preprocessing pipeline.

**Dependencies**: pandas, numpy, scikit-learn, matplotlib, seaborn, scipy

**Configuration**: 
- K range: 2-8 clusters (adjustable in script)
- Features: 7-dimensional household profile (customizable)
- Reproducibility: random_state=42

## Policy Applications

**Transport planning**:
- Identify high-WFH areas for reduced infrastructure demand
- Prioritize routes serving traditional commuter clusters
- Adjust service frequency based on cluster geographic distribution

**Urban development**:
- Zone mixed-use development in flexible WFH areas
- Preserve office density where traditional clusters concentrate
- Plan home office infrastructure in high-adoption neighborhoods

**Workplace policy**:
- Target WFH expansion programs to specific household profiles
- Design commute subsidies for unavoidably traditional households
- Develop cluster-specific flexible work arrangements

## Limitations

**Methodological**:
- K-means assumes spherical clusters (may miss complex shapes)
- Feature selection impacts cluster formation
- Cross-sectional data limits temporal pattern analysis

**Analytical scope**:
- Household-level aggregation loses individual variation
- Limited geographic granularity (LGA-level only)
- No causal inference regarding cluster formation drivers
- Sample size may limit cluster stability in some areas

## Expected Contributions

**Research question support**:
1. **WFH predictors**: Cluster profiles reveal household characteristics driving adoption
2. **Behavioral patterns**: Distinct household WFH adoption and travel profiles  
3. **Geographic insights**: Spatial distribution of household cluster types
4. **Policy targeting**: Evidence-based segmentation for intervention design

**Methodological contribution**: Demonstrates household-level clustering for WFH analysis, complementing individual-level predictive modeling.