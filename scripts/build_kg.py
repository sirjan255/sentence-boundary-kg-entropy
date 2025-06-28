import argparse
import csv
import pickle
import networkx as nx
import os

def build_kg_from_triplets(triplet_path):
    """
    Build a directed knowledge graph from a CSV of SVO triplets.
    Each unique subject/object becomes a node.
    Each (subject, verb, object) triplet becomes a directed edge (subject -> object) with verb as edge attribute.
    Returns: networkx.DiGraph
    """
    G = nx.DiGraph()
    with open(triplet_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            subj = row["subject"].strip()
            obj = row["object"].strip()
            verb = row["verb"].strip()
            sentence = row["sentence"].strip() if "sentence" in row else ""
            # Add nodes (if not already present)
            if subj:
                G.add_node(subj)
            if obj:
                G.add_node(obj)
            # Add edge with attributes
            if subj and obj and verb:
                G.add_edge(subj, obj, verb=verb, sentence=sentence)
    return G

def main():
    parser = argparse.ArgumentParser(description="Build a Knowledge Graph from SVO triplets CSV.")
    parser.add_argument("--triplets", type=str, required=True, help="Input CSV file with SVO triplets (output from extract_svo.py).")
    parser.add_argument("--output", type=str, required=True, help="Output file path for serialized KG (e.g., outputs/kg.pkl).")
    args = parser.parse_args()

    if not os.path.exists(args.triplets):
        print(f"Triplets file {args.triplets} does not exist.")
        return

    # Build the KG
    kg = build_kg_from_triplets(args.triplets)
    print(f"Knowledge graph constructed with {kg.number_of_nodes()} nodes and {kg.number_of_edges()} edges.")

    # Save the KG
    with open(args.output, "wb") as f:
        pickle.dump(kg, f)
    print(f"Knowledge graph saved to {args.output}")

if __name__ == "__main__":
    main()