import pandas as pd
import networkx as nx
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rapidfuzz.distance import JaroWinkler, Levenshtein
import time
from itertools import combinations
import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

print("Connecting to Qdrant...")
client = QdrantClient("http://127.0.0.1:6333")
collection_name = "profiles"

print("Loading dataset for evaluation ground truth...")
df = pd.read_parquet('data/identity_dataset.parquet')
df = df.fillna('')
records = df.to_dict('records')

# Build Ground Truth Pairs
print("Building Ground Truth Pairs...")
ground_truth_pairs = set()
# Group by entity_id
entity_groups = df.groupby('entity_id')
for _, group in entity_groups:
    ids = group['record_id'].tolist()
    # All pairs in the same entity are true matches
    for u, v in combinations(sorted(ids), 2):
        ground_truth_pairs.add((u, v))

def calculate_ensemble_score(source, cand, vector_score):
    # 1. Name Similarity (Jaro-Winkler is great for typos in names)
    s_name = source.get('full_name', '')
    c_name = cand.get('full_name', '')
    name_score = JaroWinkler.normalized_similarity(s_name, c_name) if s_name and c_name else 0.0
    
    # 2. Username Similarity (Levenshtein is good for character changes)
    s_user = source.get('username', '')
    c_user = cand.get('username', '')
    user_score = Levenshtein.normalized_similarity(s_user, c_user) if s_user and c_user else 0.0
    
    # 3. Exact Match Boosts (Phone / Email)
    s_phone = source.get('phone_number', '')
    c_phone = cand.get('phone_number', '')
    phone_match = 1.0 if (s_phone and c_phone and s_phone == c_phone) else 0.0
    
    s_email = source.get('email', '')
    c_email = cand.get('email', '')
    email_match = 1.0 if (s_email and c_email and s_email == c_email) else 0.0
    
    # 4. Penalty for logical mismatches (e.g., completely different birth year)
    penalty = 0.0
    s_year = source.get('birth_year')
    c_year = cand.get('birth_year')
    if s_year and c_year:
        diff = abs(int(s_year) - int(c_year))
        if diff > 5:
            penalty -= 0.3 # Heavy penalty for large age gaps
            
    # Weighted Ensemble
    if phone_match or email_match:
        final_score = 1.0 # Absolute match overrides
    else:
        # Base weights: 40% Bio Vector, 40% Name, 20% Username
        final_score = (vector_score * 0.4) + (name_score * 0.4) + (user_score * 0.2) + penalty
        
    return max(0.0, min(1.0, final_score))

print("Starting Candidate Retrieval and Scoring (Multi-Path)...")
G = nx.Graph()

start_time = time.time()
predicted_pairs = set()

# We will iterate through all records. 
# In a real system, you'd only process "new" records against the DB, but here we cluster the whole DB.
for i, record in enumerate(records):
    source_id = record['record_id']
    G.add_node(source_id) # Ensure all nodes are in graph
    
    # 1. Semantic Retrieval using Qdrant (fetch vector then search)
    try:
        retrieved = client.retrieve(
            collection_name=collection_name,
            ids=[source_id],
            with_vectors=True
        )
        if not retrieved or not retrieved[0].vector:
            continue
            
        semantic_hits_response = client.query_points(
            collection_name=collection_name,
            query=retrieved[0].vector,
            limit=10 # Get top 10 similar profiles
        )
        semantic_hits = semantic_hits_response.points
    except Exception as e:
        print(f"Error in search for {source_id}: {e}")
        continue
        
    candidates = {}
    for hit in semantic_hits:
        candidates[hit.id] = {"payload": hit.payload, "vector_score": hit.score}
        
    # 2. Deterministic Retrieval (Exact match on Phone or Email via Scroll)
    phone = record.get('phone_number')
    if phone:
        exact_hits, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(must=[FieldCondition(key="phone_number", match=MatchValue(value=phone))]),
            limit=5
        )
        for hit in exact_hits:
            if hit.id != source_id and hit.id not in candidates:
                candidates[hit.id] = {"payload": hit.payload, "vector_score": 0.5} # Default vector score if found via exact match
                
    # 3. Ensemble Scoring
    for cand_id, cand_data in candidates.items():
        if source_id == cand_id: continue
        
        score = calculate_ensemble_score(record, cand_data['payload'], cand_data['vector_score'])
        
        if score >= 0.75: # Matching Threshold
            # Add Edge to Graph
            G.add_edge(source_id, cand_id, weight=score)
            
            # Keep track for Pairwise Evaluation
            u, v = sorted([source_id, cand_id])
            predicted_pairs.add((u, v))
            
    if (i+1) % 1000 == 0:
        print(f"Processed {i+1}/{len(records)} records...")

print(f"Resolution completed in {time.time() - start_time:.2f} seconds.")

print("\n--- Running Evaluation ---")
# 1. Pairwise Evaluation
true_positives = len(ground_truth_pairs.intersection(predicted_pairs))
false_positives = len(predicted_pairs - ground_truth_pairs)
false_negatives = len(ground_truth_pairs - predicted_pairs)

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Pairwise Precision: {precision:.4f}")
print(f"Pairwise Recall:    {recall:.4f}")
print(f"Pairwise F1-Score:  {f1_score:.4f}")

# 2. Graph Clustering (Connected Components)
clusters = list(nx.connected_components(G))
print(f"\nTotal unique entities identified (Clusters): {len(clusters)}")
print(f"Ground Truth entities: {len(entity_groups)}")

# Save clustered results
print("Generating final resolution map...")
cluster_map = []
for cluster_id, cluster_nodes in enumerate(clusters):
    for node in cluster_nodes:
        cluster_map.append({"record_id": node, "predicted_entity_id": f"CLUSTER_{cluster_id}"})
        
results_df = pd.DataFrame(cluster_map)
results_df.to_csv('data/resolution_results.csv', index=False)
print("Results saved to data/resolution_results.csv")
