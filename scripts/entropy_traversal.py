import argparse
import pickle
import numpy as np
from collections import deque
from sklearn.metrics.pairwise import cosine_similarity
import json
import sys
import os

# Import the updated entropy logic
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from entropy import node_entropy

def load_embeddings(emb_path, nodes_path):
    embeddings = np.load(emb_path)
    with open(nodes_path, "r", encoding="utf-8") as f:
        node_names = [line.strip() for line in f if line.strip()]
    node2idx = {n: i for i, n in enumerate(node_names)}
    idx2node = {i: n for n, i in node2idx.items()}
    return node2idx, idx2node, embeddings

def load_start_nodes(path):
    """Loads starting nodes, one per line."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def traverse_with_entropy(
    G, node2idx, embeddings, start_node,
    entropy_threshold=0.8, max_depth=10,
    method="blt", temperature=1.0
):
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
        # Collect both successors and predecessors
        neighbors = set(G.successors(curr)) | set(G.predecessors(curr))
        for neigh in neighbors:
            if neigh in visited:
                continue
            # Use BLT-style node entropy
            entropy = node_entropy(
                embeddings, node2idx, curr, [neigh],
                method=method, temperature=temperature
            )
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
    parser.add_argument("--entropy_threshold", type=float, default=0.8, help="Entropy threshold for boundary detection (BLT-style: 0.8 is a good start)")
    parser.add_argument("--max_depth", type=int, default=10, help="Max BFS depth")
    parser.add_argument("--entropy_method", type=str, default="blt", choices=["blt", "cosine"], help="Entropy method: blt (default) or cosine (legacy)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for BLT-style entropy")
    args = parser.parse_args()

    # Load KG
    with open(args.kg, "rb") as f:
        G = pickle.load(f)
    # Load embeddings and node mappings
    node2idx, idx2node, embeddings = load_embeddings(args.embeddings, args.nodes)
    # Load starting nodes
    start_nodes = load_start_nodes(args.start_nodes)

    results = {}
    for start in start_nodes:
        if start not in node2idx:
            print(f"Warning: start node '{start}' not in node2idx, skipping.", file=sys.stderr)
            continue
        nodes_in_sentence, entropies = traverse_with_entropy(
            G, node2idx, embeddings, start,
            entropy_threshold=args.entropy_threshold,
            max_depth=args.max_depth,
            method=args.entropy_method,
            temperature=args.temperature
        )
        results[start] = {
            "nodes_in_sentence": nodes_in_sentence,
            "entropies": {k: float(v) for k, v in entropies.items()},
        }

    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved sentence boundary predictions to {args.output}")

if __name__ == "__main__":
    main()