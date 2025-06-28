import argparse
import pickle
import numpy as np
import os

import networkx as nx

try:
    from node2vec import Node2Vec
except ImportError:
    print("node2vec not found. Please install with `pip install node2vec`")
    raise

def generate_node2vec_embeddings(kg_path, dimensions=64, walk_length=30, num_walks=200, workers=2, seed=42, output_path=None):
    # Load the KG
    with open(kg_path, "rb") as f:
        G = pickle.load(f)
    print(f"Loaded KG with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Node2Vec expects an undirected graph, but for semantic boundary detection, directed might be useful
    # We'll use directed here, but you can also try G.to_undirected() if needed
    node2vec = Node2Vec(
        G, dimensions=dimensions, walk_length=walk_length,
        num_walks=num_walks, workers=workers, seed=seed, quiet=True
    )

    # Fit model
    model = node2vec.fit(window=10, min_count=1, batch_words=4, seed=seed)
    node_list = list(G.nodes())
    embeddings = np.zeros((len(node_list), dimensions), dtype=np.float32)
    node2idx = {node: idx for idx, node in enumerate(node_list)}

    for node in node_list:
        embeddings[node2idx[node]] = model.wv[node]

    # Save embeddings as .npy and node list as .txt for mapping
    if output_path is not None:
        base = os.path.splitext(output_path)[0]
        np.save(base + ".npy", embeddings)
        with open(base + "_nodes.txt", "w", encoding="utf-8") as f:
            for node in node_list:
                f.write(f"{node}\n")
        print(f"Saved embeddings to {base + '.npy'}")
        print(f"Saved node ordering to {base + '_nodes.txt'}")

    return embeddings, node_list

def main():
    parser = argparse.ArgumentParser(description="Generate node2vec embeddings from a knowledge graph.")
    parser.add_argument("--kg", type=str, required=True, help="Input KG pickle file (outputs/kg.pkl).")
    parser.add_argument("--output", type=str, required=True, help="Output path prefix for embeddings (e.g. outputs/embeddings.npy).")
    parser.add_argument("--dimensions", type=int, default=64, help="Embedding dimension size.")
    parser.add_argument("--walk_length", type=int, default=30, help="Length of walk per source.")
    parser.add_argument("--num_walks", type=int, default=200, help="Number of walks per node.")
    parser.add_argument("--workers", type=int, default=2, help="Number of worker threads.")
    args = parser.parse_args()

    generate_node2vec_embeddings(
        kg_path=args.kg,
        dimensions=args.dimensions,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        workers=args.workers,
        output_path=args.output,
    )

if __name__ == "__main__":
    main()