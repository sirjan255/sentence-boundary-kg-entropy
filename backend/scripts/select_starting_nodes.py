import pickle
import networkx as nx
import random
import argparse
import numpy as np
import os
import sys
import ast

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    from entropy import node_entropy
except ImportError:
    node_entropy = None

def load_graph(path):
    """
    Load the knowledge graph from a pickle file.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

def select_top_degree_nodes(G, n=3):
    """
    Select the top n nodes with the highest degree in the graph.
    """
    degree_sorted = sorted(G.degree, key=lambda x: x[1], reverse=True)
    return [node for node, degree in degree_sorted[:n]]

def select_random_nodes(G, n=3, seed=42):
    """
    Select n random nodes from the graph for unbiased sampling.
    """
    random.seed(seed)
    return random.sample(list(G.nodes), min(n, G.number_of_nodes()))

def load_embeddings(emb_path, nodes_path):
    """
    Load node embeddings (.npy) and node-to-index mapping (.txt).
    """
    embeddings = np.load(emb_path)
    with open(nodes_path, "r", encoding="utf-8") as f:
        node_names = [ast.literal_eval(line) for line in f if line.strip()]
    node2idx = {n: i for i, n in enumerate(node_names)}
    return embeddings, node2idx

def select_high_entropy_nodes(G, n=3, emb_path=None, nodes_path=None, method="blt", temperature=1.0):
    """
    Select n nodes with the highest entropy, as computed using the provided method.
    This uses node embeddings and the BLT (or cosine) entropy defined in src/entropy.py.
    Only considers nodes that have corresponding embeddings.
    """
    if node_entropy is None or emb_path is None or nodes_path is None:
        raise ValueError("Entropy-based selection requires node embeddings and src/entropy.py.")
    embeddings, node2idx = load_embeddings(emb_path, nodes_path)
    entropy_dict = {}
    for node in G.nodes:
        if node not in node2idx:
            continue  # Skip nodes not present in the embedding mapping
        neighbors = set(G.successors(node)) | set(G.predecessors(node))
        entropy = node_entropy(
            embeddings, node2idx, node, list(neighbors), method=method, temperature=temperature
        )
        entropy_dict[node] = entropy
    if not entropy_dict:
        raise ValueError("No nodes with embeddings found in the graph!")
    entropy_sorted = sorted(entropy_dict.items(), key=lambda x: x[1], reverse=True)
    return [node for node, entropy in entropy_sorted[:n]]

def main():
    parser = argparse.ArgumentParser(description="Select starting nodes using various strategies.")
    parser.add_argument('--strategy', type=str, default='degree', choices=['degree', 'random', 'entropy', 'all'],
                        help='Node selection strategy: degree, random, entropy, or all')
    parser.add_argument('--num', type=int, default=3, help='Number of starting nodes to select')
    parser.add_argument('--kg_path', type=str, default='outputs/kg.pkl', help='Path to KG pickle file')
    parser.add_argument('--embeddings', type=str, default='outputs/embeddings.npy', help='Path to node embeddings (.npy)')
    parser.add_argument('--nodes', type=str, default='outputs/embeddings_nodes.txt', help='Path to node list (.txt)')
    parser.add_argument('--entropy_method', type=str, default='blt', choices=['blt', 'cosine'], help='Entropy method')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for BLT-style entropy')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for random selection')
    args = parser.parse_args()

    G = load_graph(args.kg_path)

    # Select by highest degree
    if args.strategy in ['degree', 'all']:
        degree_nodes = select_top_degree_nodes(G, args.num)
        print(f"\nTop {args.num} nodes by degree:")
        for node in degree_nodes:
            print(node)
    # Select randomly
    if args.strategy in ['random', 'all']:
        random_nodes = select_random_nodes(G, args.num, args.seed)
        print(f"\n{args.num} random nodes:")
        for node in random_nodes:
            print(node)
    # Select by entropy
    if args.strategy in ['entropy', 'all']:
        try:
            entropy_nodes = select_high_entropy_nodes(
                G, args.num, args.embeddings, args.nodes, method=args.entropy_method, temperature=args.temperature
            )
            print(f"\nTop {args.num} nodes by entropy ({args.entropy_method}):")
            for node in entropy_nodes:
                print(node)
        except Exception as e:
            print(f"Entropy-based selection failed: {e}")

if __name__ == "__main__":
    main()