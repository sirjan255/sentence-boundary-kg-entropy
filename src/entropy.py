import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_entropy(neigh_embs, curr_emb):
    """
    Computes a simple entropy-like score based on cosine similarity
    between the current node embedding and its neighbor(s).
    Higher value = more different (higher uncertainty/boundary likelihood).
    """
    if len(neigh_embs) == 0:
        return 1.0  # No neighbors, max uncertainty
    similarities = cosine_similarity([curr_emb], neigh_embs)[0]
    avg_sim = np.mean(similarities)
    entropy = 1.0 - avg_sim
    return entropy

def local_entropy(embeddings, node2idx, curr, neighbors):
    """
    For a node and a set of neighbors, computes the average cosine-entropy
    between the current node and all its neighbors.
    """
    if not neighbors:
        return 1.0
    curr_emb = embeddings[node2idx[curr]]
    neigh_embs = [embeddings[node2idx[n]] for n in neighbors if n in node2idx]
    return cosine_entropy(neigh_embs, curr_emb)

def node_entropy(embeddings, node2idx, node, neighbor_nodes, method="cosine"):
    """
    General entropy interface. More methods can be added if needed.
    """
    if method == "cosine":
        return local_entropy(embeddings, node2idx, node, neighbor_nodes)
    else:
        raise NotImplementedError(f"Unknown entropy method: {method}")
