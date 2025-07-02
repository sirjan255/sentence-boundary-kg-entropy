"""
FastAPI API: Select Starting Nodes from a Knowledge Graph

- Accepts a pickled NetworkX graph (KG) and optional embeddings/node list from the frontend.
- Returns nodes selected by user-chosen strategy:
    - 'degree': top N by degree
    - 'random': random N nodes
    - 'entropy': top N by entropy (requires embeddings and entropy method)
    - 'all': all three strategies (combined)
- All results are returned as JSON. No local file I/O.

Frontend usage:
    - POST multipart/form-data with:
        - kg: Pickle file (networkx graph)
        - embeddings: .npy node embeddings (required for entropy)
        - nodes: .txt node list (required for entropy)
        - strategy: str ('degree', 'random', 'entropy', 'all')
        - num: int (number of nodes per strategy)
        - entropy_method: str (default 'blt')
        - temperature: float (default 1.0)
        - seed: int (default 42)
    - Receives JSON with selected nodes per strategy.
"""

import pickle
import numpy as np
import io
import random
import ast
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import sys
import os

router = APIRouter()

# Trying to import node_entropy from src/entropy.py (relative to backend root)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
try:
    from entropy import node_entropy
except ImportError:
    node_entropy = None

def load_graph_from_filelike(kg_file: bytes):
    return pickle.loads(kg_file)

def load_embeddings_from_filelike(emb_file: bytes, nodes_file: bytes):
    embeddings = np.load(io.BytesIO(emb_file))
    node_names = [ast.literal_eval(line) for line in io.BytesIO(nodes_file).read().decode("utf-8").splitlines() if line.strip()]
    node2idx = {n: i for i, n in enumerate(node_names)}
    return embeddings, node2idx

def select_top_degree_nodes(G, n=3):
    degree_sorted = sorted(G.degree, key=lambda x: x[1], reverse=True)
    return [node for node, degree in degree_sorted[:n]]

def select_random_nodes(G, n=3, seed=42):
    random.seed(seed)
    nodes = list(G.nodes)
    if not nodes:
        return []
    return random.sample(nodes, min(n, len(nodes)))

def select_high_entropy_nodes(G, n=3, embeddings=None, node2idx=None, method="blt", temperature=1.0):
    if node_entropy is None or embeddings is None or node2idx is None:
        raise ValueError("Entropy-based selection requires node embeddings and src/entropy.py.")
    entropy_dict = {}
    for node in G.nodes:
        if node not in node2idx:
            continue
        neighbors = set(G.successors(node)) | set(G.predecessors(node))
        entropy = node_entropy(
            embeddings, node2idx, node, list(neighbors), method=method, temperature=temperature
        )
        entropy_dict[node] = float(entropy)
    if not entropy_dict:
        raise ValueError("No nodes with embeddings found in the graph!")
    entropy_sorted = sorted(entropy_dict.items(), key=lambda x: x[1], reverse=True)
    return [node for node, entropy in entropy_sorted[:n]]

@router.post("/select_starting_nodes/")
async def select_starting_nodes(
    kg: UploadFile = File(...),
    strategy: str = Form("degree"),
    num: int = Form(3),
    embeddings: UploadFile = File(None),
    nodes: UploadFile = File(None),
    entropy_method: str = Form("blt"),
    temperature: float = Form(1.0),
    seed: int = Form(42)
):
    """
    Select starting nodes from user-uploaded KG using the specified strategy.
    Returns: JSON with the selected nodes per strategy.
    """
    try:
        kg_bytes = await kg.read()
        G = load_graph_from_filelike(kg_bytes)
        results = {}

        if strategy in ['degree', 'all']:
            degree_nodes = select_top_degree_nodes(G, num)
            results['degree'] = degree_nodes

        if strategy in ['random', 'all']:
            random_nodes = select_random_nodes(G, num, seed)
            results['random'] = random_nodes

        if strategy in ['entropy', 'all']:
            if embeddings is None or nodes is None:
                results['entropy'] = {"error": "Embeddings and node list required for entropy selection."}
            else:
                try:
                    emb_bytes = await embeddings.read()
                    node_bytes = await nodes.read()
                    embeddings_arr, node2idx = load_embeddings_from_filelike(emb_bytes, node_bytes)
                    entropy_nodes = select_high_entropy_nodes(
                        G, num, embeddings=embeddings_arr, node2idx=node2idx, method=entropy_method, temperature=temperature
                    )
                    results['entropy'] = entropy_nodes
                except Exception as e:
                    results['entropy'] = {"error": f"Entropy-based selection failed: {str(e)}"}

        return JSONResponse(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Node selection failed: {str(e)}")