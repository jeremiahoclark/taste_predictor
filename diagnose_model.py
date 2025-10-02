"""
CENTROID SIMILARITY DIAGNOSIS & RECOMMENDATIONS
================================================

Based on the analysis, your clusters have SERIOUS overlap issues.
Here are the key findings and recommendations:

KEY FINDINGS
============

1. EXTREMELY HIGH SIMILARITY (> 0.9):
   - animated_comedy_satire ↔ holiday_family: 0.98
   - animated_comedy_satire ↔ legacy_crime_empire: 0.97
   - animated_comedy_satire ↔ celebrity_doc_music_drama: 0.95
   - celebrity_doc_music_drama ↔ holiday_family: 0.95
   - legacy_crime_empire ↔ holiday_family: 0.94

   These clusters are essentially identical in embedding space!

2. OVERALL STATISTICS:
   - Mean similarity: 0.747 (concerning - should be < 0.5)
   - 10 pairs with similarity > 0.9 (very problematic)
   - 22 pairs with similarity 0.8-0.9 (problematic)
   - 54 out of 66 total pairs have similarity > 0.7

3. LEAST DISTINCT CLUSTERS (highest avg similarity to others):
   - legacy_crime_empire: 0.853
   - animated_comedy_satire: 0.848
   - holiday_family: 0.819

4. MOST DISTINCT CLUSTER:
   - romance_thriller_darkcomedy: 0.356 (this one is actually good!)

DIAGNOSIS
=========

Your clustering model likely has one or more of these issues:

1. TOO MANY CLUSTERS for the data complexity
   - 12 clusters may be too many if your content doesn't naturally separate
   - The model is forced to split similar content into arbitrary groups

2. INSUFFICIENT FEATURE DIVERSITY in embeddings
   - The all-MiniLM-L6-v2 model may not capture enough nuance
   - Your metadata text may be too similar across content types

3. POOR INITIAL CLUSTERING
   - The original K-means clustering may have converged poorly
   - Similar content was split rather than grouped

4. DATA IMBALANCE or LIMITED TRAINING DATA
   - Some cluster types may have very few examples
   - The model learned to predict high adoption for similar content

RECOMMENDATIONS (Ranked by Impact)
==================================

IMMEDIATE FIXES (High Impact, Medium Effort)
---------------------------------------------

1. REDUCE NUMBER OF CLUSTERS
   - Start with 5-6 clusters instead of 12
   - Merge highly similar clusters:
     * Merge: animated_comedy_satire + holiday_family + legacy_crime_empire
     * Merge: celebrity_doc_music_drama into above
     * Merge: animated_superhero + madea_comedy
     * Merge: reality_comedy + reality_glam_conflict
     * Keep: romance_thriller_darkcomedy (it's distinct!)
     * Keep: black_romance, biopic_drama, legal_justice_truth as one "serious drama" cluster

2. RE-CLUSTER WITH BETTER PARAMETERS
   - Use hierarchical clustering instead of K-means
   - Set minimum inter-cluster distance threshold
   - Use silhouette score to find optimal K (likely 4-7)

3. IMPROVE METADATA GENERATION
   - Current metadata may be too generic
   - Add more distinctive features:
     * Specific character archetypes
     * Plot structure details
     * Tone descriptors (gritty, whimsical, dark, light)
     * Target demographic
     * Pacing (fast, slow, episodic)
   - Use longer, more detailed metadata text

MEDIUM-TERM IMPROVEMENTS (High Impact, High Effort)
----------------------------------------------------

4. USE A BETTER EMBEDDING MODEL
   - Try domain-specific models:
     * 'all-mpnet-base-v2' (higher quality)
     * Fine-tune on your own content data
   - Or try multi-modal embeddings:
     * Include poster images, trailers if available
     * Combine text + visual features

5. ADD FEATURE ENGINEERING
   - Don't rely solely on raw embeddings
   - Add explicit categorical features:
     * Genre (one-hot encoded)
     * Content type (film/TV)
     * Target age rating
     * Production budget tier
   - Concatenate these with embeddings before clustering

6. USE CONTRASTIVE LEARNING
   - Train embeddings to maximize cluster separation
   - Triplet loss: anchor (content) + positive (same cluster) + negative (different cluster)
   - This forces the model to learn more discriminative features

LONG-TERM IMPROVEMENTS (Medium Impact, High Effort)
----------------------------------------------------

7. COLLECT MORE DIVERSE TRAINING DATA
   - Ensure each cluster has 100+ examples
   - Balance cluster sizes
   - Include edge cases and diverse content

8. USE SUPERVISED CLUSTERING
   - If you have ground truth labels (user preferences)
   - Train a classification model instead of unsupervised clustering
   - This directly optimizes for your prediction task

9. IMPLEMENT HIERARCHICAL ARCHITECTURE
   - Two-level clustering:
     * Level 1: Broad genres (3-4 clusters)
     * Level 2: Subgenres within each broad cluster
   - This naturally enforces separation

QUICK WIN (Low Effort)
----------------------

10. POST-PROCESS PREDICTIONS
    - Add a "distinctiveness penalty"
    - If multiple clusters predict high adoption (> 0.6)
    - Only keep top 3, set others to 0.3 * original
    - This compensates for cluster overlap without retraining

TESTING & VALIDATION
=====================

After implementing changes, validate with:

1. Silhouette Score (target: > 0.5)
2. Davies-Bouldin Index (target: < 1.0)
3. Calinski-Harabasz Index (higher is better)
4. Manual inspection: Do cluster members actually share taste?
5. Prediction correlation: Are predictions too similar across clusters?

RECOMMENDED ACTION PLAN
========================

Phase 1 (This Week):
- Implement recommendation #10 (quick win)
- Reduce to 6-7 clusters (recommendation #1)
- Re-evaluate metrics

Phase 2 (Next Week):
- Improve metadata generation (recommendation #3)
- Switch to hierarchical clustering (recommendation #2)
- Test with better embedding model (recommendation #4)

Phase 3 (Following Weeks):
- Add feature engineering (recommendation #5)
- Collect more training data (recommendation #7)
- Consider contrastive learning (recommendation #6)

SPECIFIC CODE CHANGES NEEDED
=============================

See the generated files:
- test_centroids.py: Analysis script (already created)
- improve_clustering.py: Implementation of fixes (to be created)
- revalidate_clusters.py: Validation suite (to be created)

"""

if __name__ == "__main__":
    print(__doc__)
