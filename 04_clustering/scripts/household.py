from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pandas as pd
import sys


try:
    household_df = pd.read_csv("../../01_preprocessing/outputs/processed_household_master_readable.csv")
    print(f"Loaded household data successfully: {household_df.shape[0]} rows, {household_df.shape[1]} columns")
except FileNotFoundError:
    print("Error: processed_household_master_readable.csv not found. Run preprocessing pipeline first.")
    exit(1)
except Exception as e:
    print(f"Error loading household data: {e}")
    exit(1)


household_features = [
    'household_wfh_saturation',      
    'num_wfh_workers',
    'proportion_wfh_workers',
    'avg_household_wfh_intensity',
    'max_household_wfh_intensity',
    'num_working_members',
    'has_worker',
    'household_income_group',   
]

def average_yearly(s):
    if pd.isna(s):
        return None
    else:
        num_list = [int(x.replace(",", "")) for x in re.findall(r"[\d,]+", s)]
        if len(num_list) == 2:
            return num_list[1]
        elif len(num_list) == 4:
            average = (num_list[2] + num_list[3]) / 2
            return average
        
household_df['household_income_group'] = household_df['household_income_group'].apply(average_yearly)
        

# Normalize features
median = household_df[household_features].median()
household_df[household_features] = household_df[household_features].fillna(median)
normalized_data = MinMaxScaler().fit_transform(household_df[household_features])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(household_df[household_features])


# Determine optimal number of clusters
inertias = []
silhouette_scores = []
k_range = range(2, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))

# Plot elbow curve
ax1.plot(k_range, inertias, 'bo-')
ax1.set_title('Elbow Method for Optimal K')
ax1.set_xlabel('Number of Clusters')
ax1.set_ylabel('Inertia')

# Plot silhouette score curv
ax2.plot(k_range, silhouette_scores, 'ro-')
ax2.set_title('Silhouette Score vs Number of Clusters')
ax2.set_xlabel('Number of Clusters')
ax2.set_ylabel('Silhouette Score')
plt.tight_layout()

plt.savefig('../outputs/househould_cluster_optimization.png', dpi=300)

# Determine optimal k automatically
# Find the k with highest silhouette score
optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]
print(f"Optimal k selected: {optimal_k} (Silhouette Score: {max(silhouette_scores):.3f})")
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
household_df['cluster'] = kmeans_final.fit_predict(normalized_data)

# Cluster profiling
cluster_profiles = household_df.groupby('cluster')[household_features].mean()
cluster_sizes = household_df['cluster'].value_counts().sort_index()

print("Cluster Profiles:")
print(cluster_profiles)
print("\nCluster Sizes:")
print(cluster_sizes)


# Visualize clusters (PCA for dimensionality reduction)
from sklearn.decomposition import PCA

sklearn_pca = PCA(n_components=2)
X_pca = sklearn_pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
sns.scatterplot(x = X_pca[:,0],
                y = X_pca[:,1],
                hue = kmeans_final.labels_)
plt.xlabel('Overall WFH adoption & intensity')
plt.ylabel('Work structure & income vs WFH')
plt.title('Household WFH Clusters (PCA Visualization)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../outputs/household_clusters_pca.png', dpi=300)

