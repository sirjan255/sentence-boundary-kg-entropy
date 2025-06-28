import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax
from scipy.stats import entropy as entropy_func

def blt_style_entropy(neigh_embs, curr_emb, temperature=1.0):
    """
    BLT-inspired entropy: Compute softmax over similarity scores (as logits),
    then calculate the entropy of the resulting distribution.
    This is a more principled, information-theoretic approach than mean similarity.
    """
    if len(neigh_embs) == 0:
        return np.log(2.0)  # Max entropy for binary case; can also use 1.0
    # Compute cosine similarities as logits for softmax
    similarities = cosine_similarity([curr_emb], neigh_embs)[0]  # shape: (N,)
    logits = similarities / temperature
    probs = softmax(logits)
    ent = entropy_func(probs)
    return float(ent)

def local_entropy(embeddings, node2idx, curr, neighbors, method="blt", temperature=1.0):
    """
    For a node and a set of neighbors, computes the entropy using the selected method.
    method:
        - "blt": BLT-inspired entropy over softmax(similarities)
        - "cosine": Old mean-based entropy
    """
    if not neighbors:
        return np.log(2.0) if method == "blt" else 1.0
    curr_emb = embeddings[node2idx[curr]]
    neigh_embs = [embeddings[node2idx[n]] for n in neighbors if n in node2idx]
    if method == "blt":
        return blt_style_entropy(neigh_embs, curr_emb, temperature=temperature)
    elif method == "cosine":
        # Legacy fallback: 1 - average cosine similarity
        similarities = cosine_similarity([curr_emb], neigh_embs)[0]
        avg_sim = np.mean(similarities)
        return 1.0 - avg_sim
    else:
        raise NotImplementedError(f"Unknown entropy method: {method}")

def node_entropy(embeddings, node2idx, node, neighbor_nodes, method="blt", temperature=1.0):
    """
    General entropy interface.
    method: "blt" (default), or "cosine" for legacy.
    """
    return local_entropy(
        embeddings, node2idx, node, neighbor_nodes, method=method, temperature=temperature
    )