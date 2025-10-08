#!/usr/bin/env python3
"""
Diagnostic script to test if different content generates different embeddings and predictions.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import joblib

# Load the model and data
print("Loading models...")
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
trained_models = joblib.load('trained_models.pkl')
centroids = joblib.load('centroids.pkl')
path_b_model = trained_models.get('path_b_rfr')

# Test with different content
test_cases = [
    {
        "name": "Horror Movie",
        "embedding_text": "The Conjuring | Genre: Horror | Sub: Supernatural thriller | Tonal: The Exorcist, Insidious, Sinister | Tropes: Haunted house, Demonic possession, Family in danger | Diff: Based on true case files of paranormal investigators | Protagonist: Married couple, 40s, paranormal investigators"
    },
    {
        "name": "Romantic Comedy",
        "embedding_text": "When Harry Met Sally | Genre: Romance | Sub: Romantic comedy | Tonal: Sleepless in Seattle, You've Got Mail, Notting Hill | Tropes: Friends to lovers, Will they won't they, Meet cute | Diff: Explores whether men and women can be just friends | Protagonist: Woman, 30s, journalist and man, 30s, political consultant"
    },
    {
        "name": "Action Thriller",
        "embedding_text": "Die Hard | Genre: Action | Sub: Action thriller | Tonal: Speed, The Rock, Lethal Weapon | Tropes: Wrong place wrong time, Hostage situation, One man army | Diff: Action movie set during Christmas in a single building | Protagonist: Male, 30s, New York cop"
    },
    {
        "name": "Black Drama",
        "embedding_text": "Moonlight | Genre: Drama | Sub: Coming-of-age drama | Tonal: Boyhood, Pariah, Beasts of the Southern Wild | Tropes: Black LGBTQ+, Three act structure, Miami setting | Diff: Tender exploration of Black masculinity and queer identity | Protagonist: Black male, shown at three life stages"
    }
]

print("\n" + "="*80)
print("TESTING EMBEDDING GENERATION")
print("="*80)

embeddings = {}
for test in test_cases:
    print(f"\n{test['name']}:")
    print(f"  Input text: {test['embedding_text'][:100]}...")

    # Generate embedding
    embedding = embedder.encode([test['embedding_text']])[0]
    embedding_normalized = embedding / (np.linalg.norm(embedding) + 1e-12)
    embeddings[test['name']] = embedding_normalized

    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
    print(f"  Normalized embedding norm: {np.linalg.norm(embedding_normalized):.4f}")
    print(f"  First 10 values: {embedding_normalized[:10]}")

# Compare embeddings
print("\n" + "="*80)
print("EMBEDDING SIMILARITY COMPARISON")
print("="*80)

test_names = list(embeddings.keys())
print("\nCosine similarity matrix:")
print(f"{'':20s}", end="")
for name in test_names:
    print(f"{name:20s}", end="")
print()

for name1 in test_names:
    print(f"{name1:20s}", end="")
    for name2 in test_names:
        similarity = np.dot(embeddings[name1], embeddings[name2])
        print(f"{similarity:20.4f}", end="")
    print()

# Test predictions
print("\n" + "="*80)
print("TESTING PREDICTIONS")
print("="*80)

CLUSTER_LABELS = {
    0: "romance_thriller_darkcomedy",
    1: "animated_comedy_satire",
    2: "reality_comedy",
    3: "animated_superhero",
    4: "reality_glam_conflict",
    5: "black_romance",
    6: "biopic_drama",
    7: "madea_comedy",
    8: "celebrity_doc_music_drama",
    9: "legacy_crime_empire",
    10: "legal_justice_truth",
    11: "holiday_family"
}

for test in test_cases:
    print(f"\n{test['name']}:")
    embedding = embeddings[test['name']]

    cluster_scores = []
    for cluster_id, centroid in sorted(centroids.items()):
        X_B = np.concatenate([centroid, embedding]).reshape(1, -1)
        p_adopt = np.clip(path_b_model.predict(X_B)[0], 0.0, 1.0)
        cluster_scores.append({
            'cluster_id': int(cluster_id),
            'cluster_name': CLUSTER_LABELS[cluster_id],
            'p_adopt': p_adopt
        })

    # Sort by score
    cluster_scores.sort(key=lambda x: x['p_adopt'], reverse=True)

    # Show top 5
    print("  Top 5 clusters:")
    for i, score in enumerate(cluster_scores[:5], 1):
        print(f"    {i}. {score['cluster_name']:30s} {score['p_adopt']*100:5.1f}%")

    # Show engagement index
    top_3_avg = sum(s['p_adopt'] for s in cluster_scores[:3]) / 3
    print(f"  Engagement Index (avg of top 3): {top_3_avg*100:.1f}%")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
print("\nIf all predictions are similar despite different content:")
print("1. Check if the embedder is working correctly")
print("2. Check if the trained model is overfitting or poorly calibrated")
print("3. Check if the centroids are too similar to each other")
print("4. Check the model training data and features")
