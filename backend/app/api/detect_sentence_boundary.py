"""
FastAPI API to detect sentence boundaries in a knowledge graph using entropy-based traversal.

- Accepts as input:
    - KG pickle file (.pkl)
    - Node embeddings (.npy)
    - Node names file (.txt)
    - Starting nodes file (.txt)
    - Optional: entropy threshold, method, temperature, max_nodes

- Returns:
    - JSON: {start_node: [{"node": ..., "entropy": ...}, ...], ...}
    - Ready for frontend display/analysis.

Dependencies:
    - NetworkX, numpy, ast, your src/entropy.py (node_entropy function)
    - Place this file as backend/app/api/detect_sentence_boundary.py
    - Register the router in your main.py

Example frontend POST:
    formData.append("kg", ...file...);
    formData.append("embeddings", ...file...);
    formData.append("nodes", ...file...);
    formData.append("starts", ...file...);
    formData.append("entropy_threshold", 0.8);
    formData.append("entropy_method", "blt");
    formData.append("temperature", 1.0);
    formData.append("max_nodes", 30);
"""

import os
import io
import ast
import pickle
import numpy as np
import networkx as nx
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

import sys
import os

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
    
from entropy import node_entropy

router = APIRouter()


def strip_quotes(s):
    return s.strip().strip('"').strip("'")


def load_graph_from_filelike(kg_file):
    kg_file = io.BytesIO(kg_file)
    kg_file.seek(0)
    return pickle.load(kg_file)


def load_embeddings_from_filelike(emb_file, nodes_file):
    emb_file = io.BytesIO(emb_file)
    emb_file.seek(0)
    embeddings = np.load(emb_file)
    nodes_file = io.BytesIO(nodes_file)
    nodes_file.seek(0)
    node_names = [
        ast.literal_eval(line)
        for line in nodes_file.read().decode("utf-8").splitlines()
        if line.strip()
    ]
    node2idx = {strip_quotes(n): i for i, n in enumerate(node_names)}
    return embeddings, node2idx


def load_start_nodes_from_filelike(starts_file):
    starts_file = io.BytesIO(starts_file)
    starts_file.seek(0)
    nodes = [
        line.strip()
        for line in starts_file.read().decode("utf-8").splitlines()
        if line.strip()
    ]
    return nodes


def traverse_sentence_boundary(
    G,
    embeddings,
    node2idx,
    start_node,
    entropy_threshold=0.8,
    method="blt",
    temperature=1.0,
    max_nodes=30,
):
    from collections import deque

    if start_node not in node2idx:
        raise ValueError(
            f"Start node '{start_node}' does not have an embedding. Check node list."
        )
    visited = set()
    queue = deque()
    queue.append(start_node)
    result = []
    while queue and len(visited) < max_nodes:
        curr_node = queue.popleft()
        if curr_node in visited:
            continue
        neighbors = set(G.successors(curr_node)) | set(G.predecessors(curr_node))
        neighbors = [n for n in neighbors if n in node2idx]
        entropy = node_entropy(
            embeddings,
            node2idx,
            curr_node,
            list(neighbors),
            method=method,
            temperature=temperature,
        )
        result.append({"node": curr_node, "entropy": float(entropy)})
        visited.add(curr_node)
        if entropy > entropy_threshold:
            continue
        for nb in neighbors:
            if nb not in visited and nb not in queue:
                queue.append(nb)
    return result


@router.post("/detect_sentence_boundary/")
async def detect_sentence_boundary(
    kg: UploadFile = File(...),
    embeddings: UploadFile = File(...),
    nodes: UploadFile = File(...),
    starts: UploadFile = File(...),
    entropy_threshold: float = Form(0.8),
    entropy_method: str = Form("blt"),
    temperature: float = Form(1.0),
    max_nodes: int = Form(30),
):
    """
    Detect sentence boundaries for each provided start node.
    Returns: JSON {start_node: [{"node":..., "entropy":...}, ...], ...}
    """
    try:
        # Load graph and embeddings
        G = load_graph_from_filelike(await kg.read())
        embeddings_bytes = await embeddings.read()
        nodes_bytes = await nodes.read()
        starts_bytes = await starts.read()

        embeddings_arr, node2idx = load_embeddings_from_filelike(
            embeddings_bytes, nodes_bytes
        )
        start_nodes = [
            strip_quotes(n) for n in load_start_nodes_from_filelike(starts_bytes)
        ]

        parsed_start_nodes = []
        for node in start_nodes:
            try:
                parsed_node = ast.literal_eval(node)
            except Exception:
                parsed_node = node
            parsed_start_nodes.append(parsed_node)

        all_results = {}
        for start_node in parsed_start_nodes:
            try:
                nodes_with_entropy = traverse_sentence_boundary(
                    G,
                    embeddings_arr,
                    node2idx,
                    start_node,
                    entropy_threshold=entropy_threshold,
                    method=entropy_method,
                    temperature=temperature,
                    max_nodes=max_nodes,
                )
                all_results[repr(start_node)] = nodes_with_entropy
            except Exception as e:
                all_results[repr(start_node)] = {"error": str(e)}

        return JSONResponse(all_results)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Sentence boundary detection failed: {str(e)}"
        )
