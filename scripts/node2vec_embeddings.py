import argparse
import pickle
import numpy as np
from node2vec import Node2Vec
import networkx as nx
import os

def save_embeddings(embeddings, idx2node, output_prefix):
    np.save(f"{output_prefix}.npy", embeddings)
    with open(f"{output_prefix}_nodes.txt", "w", encoding="utf-8") as f:
        for node in idx2node:
            f.write(f"{node}\n")

def main():
    parser = argparse.ArgumentParser(description="Generate node2vec embeddings from a knowledge graph.")
    parser.add_argument("--kg", type=str, required=True, help="Input KG pickle file (outputs/kg.pkl).")
    parser.add_argument("--output", type=str, required=True, help="Output path prefix for embeddings (e.g. outputs/embeddings).")
    parser.add_argument("--dimensions", type=int, default=64, help="Embedding dimension size.")
    parser.add_argument("--walk_length", type=int, default=30, help="Length of walk per source.")
    parser.add_argument("--num_walks", type=int, default=200, help="Number of walks per node.")
    parser.add_argument("--workers", type=int, default=2, help="Number of worker threads.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--directed", action="store_true", help="Use directed graph for embedding (default).")
    parser.add_argument("--undirected", action="store_true", help="Force undirected graph for embedding.")
    parser.add_argument("--p", type=float, default=1.0, help="Node2Vec return parameter p.")
    parser.add_argument("--q", type=float, default=1.0, help="Node2Vec in-out parameter q.")
    parser.add_argument("--weighted", action="store_true", help="Use edge weights if present.")
    parser.add_argument("--normalize", action="store_true", help="L2 normalize embeddings after training.")
    args = parser.parse_args()

    # Load the KG
    with open(args.kg, "rb") as f:
        G = pickle.load(f)
    print(f"Loaded KG with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # BLT hint: Try both directed and undirected graphs for embedding
    if args.undirected:
        G = G.to_undirected()
        print("Converted graph to undirected for embedding.")
    else:
        # Node2Vec defaults to directed. If both are false, keep original.
        print("Using original graph directionality for embedding.")

    # Optionally handle edge weights (BLT suggestion: weighted edges may better capture SVO importance)
    if args.weighted:
        for u, v, d in G.edges(data=True):
            if "weight" not in d:
                d["weight"] = 1.0  # Default to 1.0 if no weights present

    # BLT hint: Tune p (return) and q (in-out) parameters for walk bias
    node2vec = Node2Vec(
        G,
        dimensions=args.dimensions,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        workers=args.workers,
        seed=args.seed,
        quiet=True,
        p=args.p,
        q=args.q,
        weight_key="weight" if args.weighted else None
    )

    print(f"Training Node2Vec embeddings (dims={args.dimensions}, walks={args.num_walks}, walk_len={args.walk_length}, p={args.p}, q={args.q}) ...")
    model = node2vec.fit(window=10, min_count=1, batch_words=4, seed=args.seed)

    # Save embeddings in order of node index for downstream entropy calculation
    idx2node = list(model.wv.index_to_key)
    embeddings = np.stack([model.wv[node] for node in idx2node])

    if args.normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-8, None)
        print("L2-normalized embeddings.")

    save_embeddings(embeddings, idx2node, args.output)
    print(f"Saved embeddings to {args.output}.npy and node order to {args.output}_nodes.txt")

if __name__ == "__main__":
    main()