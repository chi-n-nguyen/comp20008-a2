import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Create output directory
output_dir = Path("../outputs")
output_dir.mkdir(exist_ok=True)

# Load processed household data
data_path = "../../01_preprocessing/outputs/processed_household_master.csv"
household_df = pd.read_csv(data_path)

print("Household data shape:", household_df.shape)
print("Available columns:", household_df.columns.tolist())

# Define clustering features based on research question
# Focus on WFH patterns and household travel behavior
clustering_features = [
    'hhsize',           # Household size
    'totalvehs',        # Number of vehicles
    'total_workers',    # Number of working members (from data_dictionary.json)
    'total_wfh_adopters', # Number who work from home (from data_dictionary.json)
    'total_trips',      # Total household trips
    'avg_trip_duration' # Average trip duration
]

# Check feature availability and create fallback features if needed
available_features = []
for feature in clustering_features:
    if feature in household_df.columns:
        available_features.append(feature)
    else:
        print(f"Warning: {feature} not found in dataset")

# Create WFH saturation feature if we have the necessary components
if 'total_wfh_adopters' in household_df.columns and 'total_workers' in household_df.columns:
    household_df['wfh_saturation'] = household_df['total_wfh_adopters'] / household_df['total_workers'].replace(0, np.nan)
    available_features.append('wfh_saturation')
elif 'hh_wfh_saturation' in household_df.columns:
    available_features.append('hh_wfh_saturation')

print(f"Using features for clustering: {available_features}")

# Prepare clustering dataset
clustering_data = household_df[available_features].copy()

# Handle missing values
clustering_data = clustering_data.fillna(clustering_data.median())

# Remove any infinite values
clustering_data = clustering_data.replace([np.inf, -np.inf], np.nan).fillna(clustering_data.median())

print(f"Clustering data shape after preprocessing: {clustering_data.shape}")
print("Data summary:")
print(clustering_data.describe())

# Standardize features for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(clustering_data)

print(f"Scaled data shape: {X_scaled.shape}")

# Determine optimal number of clusters using elbow method and silhouette analysis
K_range = range(2, 8)
inertias = []
silhouette_scores = []

print("Evaluating optimal number of clusters...")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(sil_score)
    print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}")

# Plot elbow curve and silhouette scores
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Elbow plot
ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_title('Elbow Method for Optimal K', fontsize=14)
ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
ax1.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
ax1.grid(True, alpha=0.3)

# Silhouette plot
ax2.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
ax2.set_title('Silhouette Score vs Number of Clusters', fontsize=14)
ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'cluster_optimization.png', dpi=300, bbox_inches='tight')
plt.close()

# Select optimal k (choose based on highest silhouette score for simplicity)
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"Selected optimal K: {optimal_k} (highest silhouette score: {max(silhouette_scores):.3f})")

# Apply final clustering
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
household_df['cluster'] = kmeans_final.fit_predict(X_scaled)

# Cluster profiling
print("\n=== CLUSTER ANALYSIS ===")
cluster_profiles = household_df.groupby('cluster')[available_features].agg(['mean', 'std']).round(3)
cluster_sizes = household_df['cluster'].value_counts().sort_index()

print("Cluster sizes:")
for i in range(optimal_k):
    print(f"Cluster {i}: {cluster_sizes[i]} households ({cluster_sizes[i]/len(household_df)*100:.1f}%)")

print("\nCluster profiles (mean ± std):")
print(cluster_profiles)

# Create cluster interpretation
cluster_interpretations = {}
for i in range(optimal_k):
    cluster_data = household_df[household_df['cluster'] == i]
    interpretation = f"Cluster {i}: "
    
    # Analyze key characteristics
    if 'hhsize' in available_features:
        avg_size = cluster_data['hhsize'].mean()
        if avg_size > household_df['hhsize'].mean():
            interpretation += "Large households, "
        else:
            interpretation += "Small households, "
    
    if 'wfh_saturation' in available_features:
        avg_wfh = cluster_data['wfh_saturation'].mean()
        if avg_wfh > 0.5:
            interpretation += "High WFH adoption, "
        elif avg_wfh > 0.2:
            interpretation += "Moderate WFH adoption, "
        else:
            interpretation += "Low WFH adoption, "
    
    if 'totalvehs' in available_features:
        avg_cars = cluster_data['totalvehs'].mean()
        if avg_cars > household_df['totalvehs'].mean():
            interpretation += "Car-dependent"
        else:
            interpretation += "Limited vehicle access"
    
    cluster_interpretations[i] = interpretation
    print(interpretation)

# Visualize clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 9))
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown'][:optimal_k]

for i in range(optimal_k):
    mask = household_df['cluster'] == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                c=colors[i], label=f'Cluster {i} (n={cluster_sizes[i]})', 
                alpha=0.6, s=50)

plt.xlabel(f'First Principal Component (explains {pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
plt.ylabel(f'Second Principal Component (explains {pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
plt.title('Household WFH Clusters (PCA Visualization)', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# Add cluster centers
cluster_centers_pca = pca.transform(kmeans_final.cluster_centers_)
plt.scatter(cluster_centers_pca[:, 0], cluster_centers_pca[:, 1], 
            c='black', marker='x', s=200, linewidths=3, label='Centroids')

plt.tight_layout()
plt.savefig(output_dir / 'household_clusters_pca.png', dpi=300, bbox_inches='tight')
plt.close()

# Create cluster profile comparison visualization
cluster_profiles_mean = household_df.groupby('cluster')[available_features].mean()

plt.figure(figsize=(14, 8))
cluster_profiles_normalized = cluster_profiles_mean.div(cluster_profiles_mean.max())

# Create heatmap
sns.heatmap(cluster_profiles_normalized.T, annot=True, cmap='RdYlBu_r', 
            center=0.5, square=False, linewidths=0.5,
            xticklabels=[f'Cluster {i}' for i in range(optimal_k)],
            yticklabels=available_features)

plt.title('Cluster Profile Comparison (Normalized)', fontsize=14)
plt.xlabel('Clusters', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / 'cluster_profiles_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Save cluster results
cluster_results = household_df[['hhid', 'cluster']].copy()
cluster_results.to_csv(output_dir / 'household_cluster_assignments.csv', index=False)

# Save cluster summary
summary_data = {
    'optimal_k': optimal_k,
    'silhouette_score': max(silhouette_scores),
    'cluster_sizes': cluster_sizes.to_dict(),
    'cluster_interpretations': cluster_interpretations,
    'features_used': available_features
}

# Create summary report
with open(output_dir / 'clustering_summary_report.txt', 'w') as f:
    f.write("=== HOUSEHOLD WFH CLUSTERING ANALYSIS SUMMARY ===\n\n")
    f.write(f"Dataset: {len(household_df)} households\n")
    f.write(f"Features used: {', '.join(available_features)}\n")
    f.write(f"Optimal number of clusters: {optimal_k}\n")
    f.write(f"Silhouette score: {max(silhouette_scores):.3f}\n\n")
    
    f.write("CLUSTER SIZES:\n")
    for i in range(optimal_k):
        f.write(f"Cluster {i}: {cluster_sizes[i]} households ({cluster_sizes[i]/len(household_df)*100:.1f}%)\n")
    
    f.write("\nCLUSTER INTERPRETATIONS:\n")
    for i, interpretation in cluster_interpretations.items():
        f.write(f"{interpretation}\n")
    
    f.write("\nCLUSTER PROFILES (MEANS):\n")
    f.write(str(cluster_profiles_mean))

print(f"\n=== CLUSTERING ANALYSIS COMPLETE ===")
print(f"Results saved to: {output_dir}")
print(f"- cluster_optimization.png: Elbow and silhouette analysis")
print(f"- household_clusters_pca.png: PCA visualization of clusters")
print(f"- cluster_profiles_heatmap.png: Feature comparison across clusters")
print(f"- household_cluster_assignments.csv: Cluster assignments for each household")
print(f"- clustering_summary_report.txt: Detailed analysis summary")