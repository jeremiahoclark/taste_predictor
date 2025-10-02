import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

# Optional visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# Cluster labels from the app
CLUSTER_LABELS = {
    0: "Romance-Infused, Suspense-Driven Stories",
    1: "Celebrity-and Culture-Driven Comedy",
    2: "Reality-Driven, Socially Chaotic Comedy",
    3: "Hidden Identity, Crime, and Supernatural Thrills",
    4: "Adult Relationship and Life-Stage Stories",
    5: "Community, Music, and Relationship Stories",
    6: "Ambition, Fame, and Erotic Thrillers",
    7: "Faith-Tinged, Family-Focused Comedy",
    8: "Star-and Legacy-Centered Documentary",
    9: "Crime and Dynasty Family Sagas",
    10: "Legal Battles and Female Bonds",
    11: "Holiday Romance and Wish Fulfillment"
}

# Load centroids
print("Loading centroids...")
centroids = joblib.load('centroids.pkl')

print(f"\nNumber of clusters: {len(centroids)}")
print(f"Centroid dimension: {centroids[0].shape[0]}")

# Calculate pairwise cosine similarity
n_clusters = len(centroids)
similarity_matrix = np.zeros((n_clusters, n_clusters))

for i in range(n_clusters):
    for j in range(n_clusters):
        if i == j:
            similarity_matrix[i, j] = 1.0
        else:
            # Cosine similarity = 1 - cosine distance
            similarity_matrix[i, j] = 1 - cosine(centroids[i], centroids[j])

print("\n" + "="*80)
print("CENTROID SIMILARITY ANALYSIS")
print("="*80)

# Create DataFrame for better visualization
cluster_names = [CLUSTER_LABELS[i] for i in range(n_clusters)]
df_similarity = pd.DataFrame(similarity_matrix,
                             index=cluster_names,
                             columns=cluster_names)

print("\nSimilarity Matrix (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite):")
print(df_similarity.round(3))

# Find most similar pairs (excluding diagonal)
print("\n" + "="*80)
print("MOST SIMILAR CLUSTER PAIRS (excluding self-similarity)")
print("="*80)

similar_pairs = []
for i in range(n_clusters):
    for j in range(i+1, n_clusters):
        similar_pairs.append({
            'Cluster 1': CLUSTER_LABELS[i],
            'Cluster 2': CLUSTER_LABELS[j],
            'Similarity': similarity_matrix[i, j]
        })

df_pairs = pd.DataFrame(similar_pairs).sort_values('Similarity', ascending=False)
print("\nTop 20 Most Similar Pairs:")
print(df_pairs.head(20).to_string(index=False))

# Statistics
print("\n" + "="*80)
print("SIMILARITY STATISTICS")
print("="*80)

# Get upper triangle (excluding diagonal) for statistics
upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]

print(f"\nMean similarity (excluding diagonal): {upper_triangle.mean():.4f}")
print(f"Median similarity: {np.median(upper_triangle):.4f}")
print(f"Max similarity: {upper_triangle.max():.4f}")
print(f"Min similarity: {upper_triangle.min():.4f}")
print(f"Std deviation: {upper_triangle.std():.4f}")

# Count problematic pairs
very_high = (upper_triangle > 0.9).sum()
high = ((upper_triangle > 0.8) & (upper_triangle <= 0.9)).sum()
medium = ((upper_triangle > 0.7) & (upper_triangle <= 0.8)).sum()

print(f"\nPairs with similarity > 0.9 (very problematic): {very_high}")
print(f"Pairs with similarity 0.8-0.9 (problematic): {high}")
print(f"Pairs with similarity 0.7-0.8 (concerning): {medium}")

# Show which specific pairs are very problematic
print("\n" + "="*80)
print("PROBLEMATIC PAIRS (similarity > 0.8)")
print("="*80)
problematic = df_pairs[df_pairs['Similarity'] > 0.8]
print(problematic.to_string(index=False))

# Calculate average similarity for each cluster
print("\n" + "="*80)
print("CLUSTER DISTINCTIVENESS (lower = more distinct)")
print("="*80)

cluster_avg_similarity = []
for i in range(n_clusters):
    # Average similarity with all other clusters (excluding self)
    mask = np.ones(n_clusters, dtype=bool)
    mask[i] = False
    avg_sim = similarity_matrix[i, mask].mean()
    cluster_avg_similarity.append({
        'Cluster': CLUSTER_LABELS[i],
        'Avg Similarity to Others': avg_sim
    })

df_distinctiveness = pd.DataFrame(cluster_avg_similarity).sort_values('Avg Similarity to Others', ascending=False)
print(df_distinctiveness.to_string(index=False))

# Save visualization
if HAS_PLOTTING:
    print("\n" + "="*80)
    print("Creating heatmap visualization...")
    print("="*80)

    plt.figure(figsize=(14, 12))
    sns.heatmap(df_similarity, annot=True, fmt='.2f', cmap='RdYlGn_r',
                vmin=0, vmax=1, center=0.5, square=True,
                cbar_kws={'label': 'Cosine Similarity'})
    plt.title('Centroid Similarity Heatmap\n(Red = High Similarity, Green = Low Similarity)',
              fontsize=14, pad=20)
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Cluster', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('centroid_similarity_heatmap.png', dpi=150, bbox_inches='tight')
    print("Saved heatmap to: centroid_similarity_heatmap.png")
else:
    print("\nNote: matplotlib/seaborn not available - skipping visualization")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
