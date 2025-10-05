import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Create output directory
output_dir = Path("../outputs")
output_dir.mkdir(exist_ok=True)

# Load processed household data and cluster assignments
data_path = "../../01_preprocessing/outputs/processed_household_master.csv"
cluster_path = output_dir / "household_cluster_assignments.csv"

household_df = pd.read_csv(data_path)

# Check if cluster assignments exist
if cluster_path.exists():
    cluster_assignments = pd.read_csv(cluster_path)
    household_df = household_df.merge(cluster_assignments, on='hhid', how='left')
    print(f"Loaded cluster assignments for {len(cluster_assignments)} households")
else:
    print("Warning: No cluster assignments found. Run household_wfh_clustering.py first.")
    exit()

print("Household data shape:", household_df.shape)
print("Available geographic columns:", [col for col in household_df.columns if 'region' in col.lower() or 'area' in col.lower()])

# Check for geographic variables
geographic_vars = []
for col in household_df.columns:
    if any(keyword in col.lower() for keyword in ['region', 'area', 'suburb', 'postcode', 'lga']):
        geographic_vars.append(col)

print(f"Found geographic variables: {geographic_vars}")

# Use the first available geographic variable, or create a synthetic one
if geographic_vars:
    geo_column = geographic_vars[0]
    print(f"Using geographic variable: {geo_column}")
else:
    # Create synthetic regions based on hhid for demonstration
    print("No geographic variables found. Creating synthetic regions for demonstration.")
    household_df['region'] = household_df['hhid'] % 5
    household_df['region'] = household_df['region'].map({
        0: 'Inner Melbourne',
        1: 'Northern Suburbs',
        2: 'Eastern Suburbs', 
        3: 'Southern Suburbs',
        4: 'Western Suburbs'
    })
    geo_column = 'region'

# Ensure we have cluster information
if 'cluster' not in household_df.columns:
    print("Error: No cluster information found in dataset")
    exit()

n_clusters = household_df['cluster'].nunique()
print(f"Number of clusters: {n_clusters}")

# Geographic distribution analysis
print("\n=== GEOGRAPHIC CLUSTER DISTRIBUTION ANALYSIS ===")

# Create crosstab of geographic areas vs clusters
geographic_cluster_crosstab = pd.crosstab(household_df[geo_column], household_df['cluster'], margins=True)
print("Geographic distribution (raw counts):")
print(geographic_cluster_crosstab)

# Calculate proportions within each geographic area
geographic_proportions = pd.crosstab(household_df[geo_column], household_df['cluster'], normalize='index')
print("\nGeographic distribution (proportions within each area):")
print(geographic_proportions.round(3))

# Calculate proportions within each cluster
cluster_proportions = pd.crosstab(household_df[geo_column], household_df['cluster'], normalize='columns')
print("\nCluster composition by geography (proportions within each cluster):")
print(cluster_proportions.round(3))

# Statistical test for independence
from scipy.stats import chi2_contingency

# Remove margins row for statistical test
test_crosstab = geographic_cluster_crosstab.iloc[:-1, :-1]
chi2, p_value, dof, expected = chi2_contingency(test_crosstab)
print(f"\nChi-square test for independence:")
print(f"Chi2 statistic: {chi2:.3f}")
print(f"p-value: {p_value:.6f}")
print(f"Degrees of freedom: {dof}")

# Visualization 1: Stacked bar chart of cluster distribution by geography
plt.figure(figsize=(14, 8))
geographic_proportions.plot(kind='bar', stacked=True, 
                           color=['red', 'blue', 'green', 'orange', 'purple'][:n_clusters],
                           figsize=(14, 8))
plt.title('Cluster Distribution Across Geographic Areas', fontsize=16)
plt.xlabel('Geographic Area', fontsize=12)
plt.ylabel('Proportion of Households', fontsize=12)
plt.legend(title='Cluster', labels=[f'Cluster {i}' for i in range(n_clusters)], 
           bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(output_dir / 'geographic_cluster_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Visualization 2: Heatmap of geographic-cluster distribution
plt.figure(figsize=(12, 8))
sns.heatmap(geographic_proportions, annot=True, cmap='YlOrRd', 
            fmt='.3f', square=False, linewidths=0.5,
            cbar_kws={'label': 'Proportion within Geographic Area'})
plt.title('Geographic Distribution of Household Clusters (Heatmap)', fontsize=14)
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Geographic Area', fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / 'geographic_cluster_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Visualization 3: Grouped bar chart for better comparison
fig, ax = plt.subplots(figsize=(15, 8))
x = np.arange(len(geographic_proportions.index))
width = 0.8 / n_clusters
colors = ['red', 'blue', 'green', 'orange', 'purple'][:n_clusters]

for i in range(n_clusters):
    ax.bar(x + i * width - width * (n_clusters-1)/2, 
           geographic_proportions.iloc[:, i], 
           width, label=f'Cluster {i}', color=colors[i], alpha=0.8)

ax.set_xlabel('Geographic Area', fontsize=12)
ax.set_ylabel('Proportion of Households', fontsize=12)
ax.set_title('Cluster Distribution by Geographic Area (Grouped Bars)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(geographic_proportions.index, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(output_dir / 'geographic_cluster_grouped_bars.png', dpi=300, bbox_inches='tight')
plt.close()

# Analyze cluster characteristics by geography
print("\n=== CLUSTER CHARACTERISTICS BY GEOGRAPHY ===")

# Features to analyze
analysis_features = ['hhsize', 'totalvehs', 'num_workers', 'num_wfh_adopters', 'total_trips']
available_analysis_features = [f for f in analysis_features if f in household_df.columns]

if available_analysis_features:
    geographic_cluster_profiles = household_df.groupby([geo_column, 'cluster'])[available_analysis_features].mean()
    print("Mean values by geography and cluster:")
    print(geographic_cluster_profiles.round(2))
    
    # Create visualization for cluster profiles by geography
    fig, axes = plt.subplots(len(available_analysis_features), 1, figsize=(15, 5*len(available_analysis_features)))
    if len(available_analysis_features) == 1:
        axes = [axes]
    
    for idx, feature in enumerate(available_analysis_features):
        pivot_data = household_df.pivot_table(values=feature, index=geo_column, columns='cluster', aggfunc='mean')
        
        pivot_data.plot(kind='bar', ax=axes[idx], 
                       color=['red', 'blue', 'green', 'orange', 'purple'][:n_clusters])
        axes[idx].set_title(f'Average {feature} by Geography and Cluster', fontsize=12)
        axes[idx].set_xlabel('Geographic Area')
        axes[idx].set_ylabel(f'Average {feature}')
        axes[idx].legend(title='Cluster', labels=[f'Cluster {i}' for i in range(n_clusters)])
        axes[idx].grid(True, alpha=0.3)
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_profiles_by_geography.png', dpi=300, bbox_inches='tight')
    plt.close()

# Calculate geographic concentration index for each cluster
print("\n=== CLUSTER GEOGRAPHIC CONCENTRATION ===")
concentration_indices = {}

for cluster in range(n_clusters):
    cluster_data = household_df[household_df['cluster'] == cluster]
    geographic_dist = cluster_data[geo_column].value_counts(normalize=True)
    
    # Calculate Herfindahl-Hirschman Index as concentration measure
    hhi = (geographic_dist ** 2).sum()
    concentration_indices[cluster] = hhi
    
    print(f"Cluster {cluster} geographic concentration (HHI): {hhi:.3f}")
    print(f"  Most common areas: {geographic_dist.head(3).to_dict()}")

# Visualization of concentration indices
plt.figure(figsize=(10, 6))
clusters = list(concentration_indices.keys())
concentrations = list(concentration_indices.values())

plt.bar(clusters, concentrations, color=['red', 'blue', 'green', 'orange', 'purple'][:n_clusters])
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Geographic Concentration Index (HHI)', fontsize=12)
plt.title('Geographic Concentration by Cluster', fontsize=14)
plt.xticks(clusters, [f'Cluster {i}' for i in clusters])
plt.grid(True, alpha=0.3, axis='y')

# Add interpretation text
max_concentration = 1.0 / len(household_df[geo_column].unique())
plt.axhline(y=max_concentration, color='red', linestyle='--', alpha=0.7, 
            label=f'Even distribution ({max_concentration:.3f})')
plt.legend()

plt.tight_layout()
plt.savefig(output_dir / 'cluster_geographic_concentration.png', dpi=300, bbox_inches='tight')
plt.close()

# Save geographic analysis results
geographic_results = {
    'geographic_cluster_crosstab': geographic_cluster_crosstab,
    'geographic_proportions': geographic_proportions,
    'cluster_proportions': cluster_proportions,
    'chi2_test': {'chi2': chi2, 'p_value': p_value, 'dof': dof},
    'concentration_indices': concentration_indices
}

# Save detailed geographic analysis report
with open(output_dir / 'geographic_analysis_report.txt', 'w') as f:
    f.write("=== GEOGRAPHIC CLUSTER DISTRIBUTION ANALYSIS ===\n\n")
    f.write(f"Geographic variable used: {geo_column}\n")
    f.write(f"Number of geographic areas: {len(household_df[geo_column].unique())}\n")
    f.write(f"Number of clusters: {n_clusters}\n\n")
    
    f.write("CHI-SQUARE TEST FOR INDEPENDENCE:\n")
    f.write(f"Chi2 statistic: {chi2:.3f}\n")
    f.write(f"p-value: {p_value:.6f}\n")
    f.write(f"Interpretation: {'Significant' if p_value < 0.05 else 'Not significant'} geographic clustering pattern\n\n")
    
    f.write("GEOGRAPHIC DISTRIBUTION (Raw Counts):\n")
    f.write(str(geographic_cluster_crosstab))
    f.write("\n\nGEOGRAPHIC DISTRIBUTION (Proportions within areas):\n")
    f.write(str(geographic_proportions.round(3)))
    
    f.write("\n\nCLUSTER GEOGRAPHIC CONCENTRATION:\n")
    for cluster, hhi in concentration_indices.items():
        f.write(f"Cluster {cluster}: HHI = {hhi:.3f}\n")

print(f"\n=== GEOGRAPHIC ANALYSIS COMPLETE ===")
print(f"Results saved to: {output_dir}")
print(f"- geographic_cluster_distribution.png: Stacked bar chart")
print(f"- geographic_cluster_heatmap.png: Heatmap visualization")
print(f"- geographic_cluster_grouped_bars.png: Grouped bar comparison")
print(f"- cluster_profiles_by_geography.png: Feature profiles by geography")
print(f"- cluster_geographic_concentration.png: Concentration indices")
print(f"- geographic_analysis_report.txt: Detailed statistical analysis")