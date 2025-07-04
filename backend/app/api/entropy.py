import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax
from scipy.stats import entropy as entropy_func

def blt_style_entropy(neigh_embs, curr_emb, temperature=1.0, debug=False):
    if len(neigh_embs) == 0:
        if debug:
            print("No neighbors: returning max entropy (log2).")
        return np.log(2.0)
    similarities = cosine_similarity([curr_emb], neigh_embs)[0]
    logits = similarities / temperature
    probs = softmax(logits)
    ent = entropy_func(probs)
    if debug:
        print(f"Similarities: {similarities}")
        print(f"Softmax probs: {probs}")
        print(f"Entropy: {ent}")
    return float(ent)

def local_entropy(embeddings, node2idx, curr, neighbors, method="blt", temperature=1.0, debug=False):
    if not neighbors:
        if debug:
            print("No neighbors for local_entropy: returning max entropy (log2 or 1.0).")
        return np.log(2.0) if method == "blt" else 1.0
    curr_emb = embeddings[node2idx[curr]]
    neigh_embs = [embeddings[node2idx[n]] for n in neighbors if n in node2idx]
    if method == "blt":
        return blt_style_entropy(neigh_embs, curr_emb, temperature=temperature, debug=debug)
    elif method == "cosine":
        similarities = cosine_similarity([curr_emb], neigh_embs)[0]
        avg_sim = np.mean(similarities)
        if debug:
            print(f"Cosine similarities: {similarities}, avg: {avg_sim}")
        return 1.0 - avg_sim
    else:
        raise NotImplementedError(f"Unknown entropy method: {method}")

def node_entropy(embeddings, node2idx, node, neighbor_nodes, method="blt", temperature=1.0, debug=False):
    return local_entropy(
        embeddings, node2idx, node, neighbor_nodes, method=method, temperature=temperature, debug=debug
    )