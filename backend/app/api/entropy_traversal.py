"""
FastAPI API: Entropy-Based Traversal for Sentence Boundary Detection in Knowledge Graph

- Accepts all input as in-memory file uploads (no paths on disk).
- No filesystem dependencies: works directly with user-uploaded files.
- Returns JSON: {start_node: {nodes_in_sentence: [...], entropies: {...}}, ...}
- Designed for easy integration with a modern frontend.

Inputs:
    - kg: Pickled NetworkX DiGraph (.pkl)
    - embeddings: Numpy embeddings file (.npy)
    - nodes: Node names file (.txt, one per line)
    - start_nodes: Starting nodes file (.txt, one per line)
    - entropy_threshold: float (default=0.8)
    - max_depth: int (default=10)
    - entropy_method: str (default="blt")
    - temperature: float (default=1.0)

How to use:
    - POST multipart/form-data to /api/entropy_traversal/
    - All files must be uploaded as form-data fields.

Dependencies:
    - NetworkX, numpy, src/entropy.py (node_entropy)
"""

import io
import ast
import pickle
import numpy as np
from collections import deque
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from entropy import node_entropy

router = APIRouter()

def strip_quotes(s):
    return s.strip().strip('"').strip("'")

async def load_embeddings_from_filelike(emb_file: UploadFile, nodes_file: UploadFile):
    # Read the uploaded files asynchronously
    emb_bytes = await emb_file.read()
    node_bytes = await nodes_file.read()
    # Load embeddings from bytes
    emb_buf = io.BytesIO(emb_bytes)
    embeddings = np.load(emb_buf)
    # Decode node names from bytes
    node_names = [
        line.strip() for line in node_bytes.decode("utf-8").splitlines() if line.strip()
    ]
    node2idx = {strip_quotes(n): i for i, n in enumerate(node_names)}
    idx2node = {i: n for n, i in node2idx.items()}
    return node2idx, idx2node, embeddings


async def load_start_nodes_from_filelike(start_nodes_file: UploadFile):
    file_bytes = await start_nodes_file.read()
    return [
        line.strip() for line in file_bytes.decode("utf-8").splitlines() if line.strip()
    ]


@router.post("/entropy_traversal/")
async def entropy_traversal(
    kg: UploadFile = File(...),
    embeddings: UploadFile = File(...),
    nodes: UploadFile = File(...),
    start_nodes: UploadFile = File(...),
    entropy_threshold: float = Form(0.8),
    max_depth: int = Form(10),
    entropy_method: str = Form("blt"),
    temperature: float = Form(1.0),
):
    """
    Run entropy-based traversal for each start node.
    Returns: {start_node: {nodes_in_sentence: [...], entropies: {...}}, ...}
    """
    try:
        # Load KG
        kg_bytes = await kg.read()
        G = pickle.loads(kg_bytes)

        # Load embeddings and nodes
        node2idx, idx2node, embeddings_arr = await load_embeddings_from_filelike(
            embeddings, nodes
        )
        
        # Load starting nodes
        start_nodes_list = [strip_quotes(n) for n in await load_start_nodes_from_filelike(start_nodes)]

        results = {}
        for start in start_nodes_list:
            if start not in node2idx:
                # Skip this start node, but tell the user
                results[start] = {
                    "error": f"Start node '{start}' not in node2idx, skipping."
                }
                continue
            visited = set()
            entropies = dict()
            q = deque()
            q.append((start, 0))
            visited.add(start)
            nodes_in_sentence = [start]
            entropies[start] = 0.0  # Start node entropy is 0

            while q:
                curr, depth = q.popleft()
                if depth >= max_depth:
                    continue
                neighbors = set(G.successors(curr)) | set(G.predecessors(curr))
                for neigh in neighbors:
                    if neigh in visited:
                        continue
                    entropy = node_entropy(
                        embeddings_arr,
                        node2idx,
                        curr,
                        [neigh],
                        method=entropy_method,
                        temperature=temperature,
                    )
                    entropies[neigh] = float(entropy)
                    if entropy < entropy_threshold:
                        visited.add(neigh)
                        q.append((neigh, depth + 1))
                        nodes_in_sentence.append(neigh)
            results[start] = {
                "nodes_in_sentence": nodes_in_sentence,
                "entropies": entropies,
            }
        return JSONResponse(results)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Entropy traversal failed: {str(e)}"
        )
