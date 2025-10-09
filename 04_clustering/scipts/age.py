from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder


person_df = pd.read_csv("01_preprocessing/outputs/processed_person_master_readable.csv")

person_df['wfh_frequency'] = person_df[['wfh_monday', 'wfh_tuesday', 'wfh_wednesday', 'wfh_thursday', 'wfh_friday', 'wfh_saturday', 'wfh_sunday']].sum(axis=1)
person_df['wfh_consistency'] = person_df[['wfh_monday', 'wfh_tuesday', 'wfh_wednesday', 'wfh_thursday', 'wfh_friday']].std(axis=1) 

person_features = [
    'wfh_intensity',
    'wfh_adopter',
    'wfh_frequency',
    'wfh_consistency'
]

median = person_df[person_features].median()
person_df[person_features] = person_df[person_features].fillna(median)

le = LabelEncoder()
person_df['agegroup_encoded'] = le.fit_transform(person_df['age_group'].fillna('Unknown'))

person_features.append('agegroup_encoded')

normalized_data = MinMaxScaler().fit_transform(person_df[person_features])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(person_df[person_features])


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

plt.savefig('age_cluster_optimization.png', dpi=300)

# Apply optimal clustering (from the two graphs above shows optimal k is 3)
optimal_k = 5
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
person_df['cluster'] = kmeans_final.fit_predict(normalized_data)

# Cluster profiling
cluster_profiles = person_df.groupby('cluster')[person_features].mean()
cluster_sizes = person_df['cluster'].value_counts().sort_index()

print("Cluster Profiles:")
print(cluster_profiles)
print("\nCluster Sizes:")
print(cluster_sizes)


# Visualize clusters (PCA for dimensionality reduction)
from sklearn.decomposition import PCA

sklearn_pca = PCA(n_components=2)
X_pca = sklearn_pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0],
                y=X_pca[:, 1],
                hue=kmeans_final.labels_,
                palette='Set2',
                alpha=0.6,
                s=50)
plt.xlabel('Overall WFH engagement')
plt.ylabel('WFH stability vs age group')
plt.title('Age WFH Clusters (PCA Visualization)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('age_household_clusters_pca.png', dpi=300)