import argparse
import pickle
import numpy as np
import ast
import networkx as nx  # Graph library for handling KGs
import os
import sys
from collections import deque  # Efficient FIFO queue implementation

# Making sure we can import from the sibling src directory
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    # node_entropy is expected to compute entropy for a given node and its neighbors
    from entropy import node_entropy
except ImportError:
    # Fail fast if the helper module isn't found
    raise ImportError("Could not import node_entropy from src/entropy.py. Please ensure it exists.")


def load_graph(kg_path):
    """Load a pickled NetworkX graph from disk."""
    with open(kg_path, "rb") as f:
        G = pickle.load(f)
    return G


def load_embeddings(emb_path, nodes_path):
    """Load numpy embeddings and corresponding node names (one per line)."""
    embeddings = np.load(emb_path)
    # Each line in nodes_path is a Python literal representation of the node (str, tuple, etc.)
    with open(nodes_path, "r", encoding="utf-8") as f:
        node_names = [ast.literal_eval(line) for line in f if line.strip()]
    # Map node -> index into the embeddings array
    node2idx = {n: i for i, n in enumerate(node_names)}
    return embeddings, node2idx


def load_start_nodes(starts_path):
    """Read starting nodes (one per line) from a text file."""
    with open(starts_path, "r", encoding="utf-8") as f:
        nodes = [line.strip() for line in f if line.strip()]
    return nodes


def traverse_sentence_boundary(
    G,
    embeddings,
    node2idx,
    start_node,
    entropy_threshold=0.8,
    method="blt",
    temperature=1.0,
    max_nodes=30,
    debug=False,
):
    """Breadth‑first traversal that stops expanding from a node once its entropy
    exceeds the specified threshold. Returns a list of dictionaries with
    node names and their entropy values.
    """
    # Ensure start node exists in embedding map
    if start_node not in node2idx:
        raise ValueError(
            f"Start node '{start_node}' does not have an embedding. Check node list."
        )

    visited = set()
    queue = deque()
    queue.append(start_node)
    result = []

    # Continue until queue empty or max_nodes reached
    while queue and len(visited) < max_nodes:
        curr_node = queue.popleft()
        if curr_node in visited:
            continue  # Skip duplicate entries

        # Combine successors and predecessors to treat graph as undirected here
        neighbors = set(G.successors(curr_node)) | set(G.predecessors(curr_node))
        # Keep only neighbors that have embeddings
        neighbors = [n for n in neighbors if n in node2idx]

        if debug:
            print(f"\nCurrent node: {repr(curr_node)}")
            print(f"Neighbors: {[repr(n) for n in neighbors]}")

        # Compute entropy for the current node with respect to its neighbors
        entropy = node_entropy(
            embeddings, node2idx, curr_node, list(neighbors), method=method, temperature=temperature
        )

        if debug:
            print(f"Entropy: {entropy}")

        # Store the result as a plain Python float for JSON serialisation later
        result.append({"node": curr_node, "entropy": float(entropy)})
        visited.add(curr_node)

        # Decide whether to stop expanding from this node
        if entropy > entropy_threshold:
            if debug:
                print("Stopping traversal from this node due to high entropy.")
            continue  # Do not enqueue neighbors if entropy too high

        # Enqueue unvisited neighbors for further traversal
        for nb in neighbors:
            if nb not in visited and nb not in queue:
                queue.append(nb)

    return result


def main():
    parser = argparse.ArgumentParser(description="Detect sentence boundaries in a KG via entropy-based traversal.")
    parser.add_argument('--kg', type=str, required=True, help="Path to KG pickle file (e.g., outputs/kg.pkl)")
    parser.add_argument('--embeddings', type=str, required=True, help="Path to node embeddings (.npy)")
    parser.add_argument('--nodes', type=str, required=True, help="Path to node list (.txt)")
    parser.add_argument('--starts', type=str, required=True, help="Path to starting nodes file (one per line)")
    parser.add_argument('--entropy_threshold', type=float, default=0.8, help="Entropy threshold for sentence boundary detection (tune this!)")
    parser.add_argument('--entropy_method', type=str, default='blt', choices=['blt', 'cosine'], help='Entropy method')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for BLT-style entropy')
    parser.add_argument('--max_nodes', type=int, default=30, help='Maximum nodes to traverse per sentence')
    parser.add_argument('--output', type=str, default='outputs/predicted_boundaries.json', help='File to save predicted boundaries')
    parser.add_argument('--debug', action='store_true', help='Enable debug printing')
    args = parser.parse_args()

    G = load_graph(args.kg)
    embeddings, node2idx = load_embeddings(args.embeddings, args.nodes)
    start_nodes = load_start_nodes(args.starts)

    parsed_start_nodes = []
    for node in start_nodes:
        try:
            parsed_node = ast.literal_eval(node)
        except Exception:
            parsed_node = node
        parsed_start_nodes.append(parsed_node)

    all_results = {}
    for start_node in parsed_start_nodes:
        try:
            nodes_with_entropy = traverse_sentence_boundary(
                G, embeddings, node2idx, start_node,
                entropy_threshold=args.entropy_threshold,
                method=args.entropy_method,
                temperature=args.temperature,
                max_nodes=args.max_nodes,
                debug=args.debug
            )
            all_results[repr(start_node)] = nodes_with_entropy
            print(f"\nStart node: {start_node}")
            print("Nodes in predicted sentence (until boundary):")
            for d in nodes_with_entropy:
                print(f"  {repr(d['node'])} (entropy={d['entropy']:.4f})")
        except Exception as e:
            print(f"Error processing start node {repr(start_node)}: {e}")

    import json
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nPredicted boundaries for each start node saved to {args.output}")

if __name__ == "__main__":
    main()