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
            if subj:
                G.add_node(subj)
            if obj:
                G.add_node(obj)
            if subj and obj and verb:
                G.add_edge(subj, obj, verb=verb, sentence=sentence)
    return G

def save_kg(G, output_path):
    """Save a NetworkX graph to a pickle file."""
    with open(output_path, "wb") as f:
        pickle.dump(G, f)

def load_kg(kg_path):
    """Load a NetworkX graph from a pickle file."""
    with open(kg_path, "rb") as f:
        return pickle.load(f)

def ensure_triplets_file_exists(triplet_path):
    """Check if triplet CSV exists."""
    return os.path.exists(triplet_path)