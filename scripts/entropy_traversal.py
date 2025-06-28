import argparse
import pickle
import numpy as np
import os
import json
from collections import deque, defaultdict
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(emb_path, nodes_path):
    """
    Loads node embeddings and returns:
    - node2idx: maps node name to embedding index
    - idx2node: maps embedding index to node name
    - embeddings: numpy array of shape (num_nodes, dim)
    """
    embeddings = np.load(emb_path)
    with open(nodes_path, "r", encoding="utf-8") as f:
        idx2node = [line.strip() for line in f]
    node2idx = {node: idx for idx, node in enumerate(idx2node)}
    assert embeddings.shape[0] == len(idx2node)
    return node2idx, idx2node, embeddings

def load_start_nodes(path):
    """Loads starting nodes, one per line."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def compute_entropy(neigh_embs, curr_emb):
    """
    Computes a simple entropy-like measure:
    - Similarity of current node to all neighbor nodes
    - Higher entropy (low similarity) = likely boundary
    You can replace with more advanced/statistical entropy if desired
    """
    if len(neigh_embs) == 0:
        return 1.0  # No neighbors, max uncertainty
    similarities = cosine_similarity([curr_emb], neigh_embs)[0]
    avg_sim = np.mean(similarities)
    entropy = 1.0 - avg_sim  # Higher value = more "different" (possible boundary)
    return entropy

def traverse_with_entropy(G, node2idx, embeddings, start_node, entropy_threshold=0.35, max_depth=10):
    """
    Traverse KG from start_node using entropy to decide when to halt (boundary).
    Returns:
        - visited_nodes: list of nodes predicted to be in the same sentence
        - entropies: dict of node -> entropy
    """
    visited = set()
    entropies = dict()
    q = deque()
    q.append((start_node, 0))
    visited.add(start_node)
    nodes_in_sentence = [start_node]
    entropies[start_node] = 0.0  # Start node entropy is 0

    while q:
        curr, depth = q.popleft()
        if depth >= max_depth:
            continue
        neighbors = set(G.successors(curr)) | set(G.predecessors(curr))
        for neigh in neighbors:
            if neigh in visited:
                continue
            curr_emb = embeddings[node2idx[curr]]
            neigh_emb = embeddings[node2idx[neigh]]
            # Entropy as "semantic divergence" from current node to neighbor
            entropy = compute_entropy([neigh_emb], curr_emb)
            entropies[neigh] = entropy
            if entropy < entropy_threshold:
                visited.add(neigh)
                q.append((neigh, depth + 1))
                nodes_in_sentence.append(neigh)
            # If entropy >= threshold, we halt traversal to that neighbor (boundary detected)
    return nodes_in_sentence, entropies

def main():
    parser = argparse.ArgumentParser(description="Traverse KG using entropy to detect sentence boundaries.")
    parser.add_argument("--kg", type=str, required=True, help="Knowledge graph pickle file (outputs/kg.pkl)")
    parser.add_argument("--embeddings", type=str, required=True, help="Embeddings .npy file (outputs/embeddings.npy)")
    parser.add_argument("--nodes", type=str, required=True, help="Node order .txt file (outputs/embeddings_nodes.txt)")
    parser.add_argument("--start_nodes", type=str, required=True, help="File with starting nodes (data/nodes_to_start.txt)")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file for predicted boundaries (outputs/boundaries.json)")
    parser.add_argument("--entropy_threshold", type=float, default=0.35, help="Entropy threshold for boundary detection")
    parser.add_argument("--max_depth", type=int, default=10, help="Max BFS depth")
    args = parser.parse_args()

    # Load KG
    with open(args.kg, "rb") as f:
        G = pickle.load(f)
    # Load embeddings
    node2idx, idx2node, embeddings = load_embeddings(args.embeddings, args.nodes)
    # Load starting nodes
    start_nodes = load_start_nodes(args.start_nodes)

    # Traverse from each starting node
    boundaries = dict()
    for start in start_nodes:
        if start not in node2idx:
            print(f"[WARN] Start node '{start}' not found in KG/embeddings. Skipping.")
            continue
        nodes, entropies = traverse_with_entropy(
            G, node2idx, embeddings, start,
            entropy_threshold=args.entropy_threshold,
            max_depth=args.max_depth
        )
        # For each node, store its entropy. Mark nodes with highest entropy as likely boundaries.
        boundaries[start] = {
            "nodes_in_sentence": nodes,
            "entropies": {n: entropies[n] for n in nodes if n in entropies}
        }
        print(f"Start node '{start}': found {len(nodes)} nodes in predicted sentence.")

    # Save output
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(boundaries, f, indent=2)
    print(f"Boundaries written to {args.output}")

if __name__ == "__main__":
    main()