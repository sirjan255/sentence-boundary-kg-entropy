"""
FastAPI API: Generate node2vec embeddings from a knowledge graph

- Accepts a pickled NetworkX graph, and user parameters, from the frontend.
- Runs node2vec (via gensim or node2vec package) all in-memory.
- Returns the embeddings and node list as downloadable files (npy and txt) or as JSON.
- Does NOT read/write any local files.

Frontend usage:
    - POST multipart/form-data with:
        - kg: Pickle file (networkx graph)
        - dimensions: int (default 64)
        - walk_length: int (default 30)
        - num_walks: int (default 200)
        - workers: int (default 2)
        - seed: int (default 42)
        - directed: bool (default True)
        - undirected: bool (default False)
        - p: float (default 1.0)
        - q: float (default 1.0)
        - weighted: bool (default False)
        - normalize: bool (default False)
    - Receives npy as bytes and nodes as plain text, or JSON.
"""

import numpy as np
import pickle
import io
import networkx as nx
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
from node2vec import Node2Vec
import tempfile
import zipfile

router = APIRouter()

def serialize_embeddings(embeddings, idx2node):
    # Embeddings as bytes (npy)
    emb_bytes = io.BytesIO()
    np.save(emb_bytes, embeddings)
    emb_bytes.seek(0)
    # Nodes as text (repr per line)
    nodes_bytes = io.StringIO()
    for node in idx2node:
        nodes_bytes.write(repr(node) + "\n")
    nodes_bytes.seek(0)
    return emb_bytes, nodes_bytes

@router.post("/node2vec_embeddings/")
async def node2vec_embeddings(
    kg: UploadFile = File(...),
    dimensions: int = Form(64),
    walk_length: int = Form(30),
    num_walks: int = Form(200),
    workers: int = Form(2),
    seed: int = Form(42),
    directed: bool = Form(True),
    undirected: bool = Form(False),
    p: float = Form(1.0),
    q: float = Form(1.0),
    weighted: bool = Form(False),
    normalize: bool = Form(False),
    output_format: str = Form("zip")  # or "json"
):
    """
    Generate node2vec embeddings from a pickled NetworkX graph.
    Returns embeddings and node list as a .zip or JSON.
    """
    try:
        # Load graph from uploaded pickle
        kg_bytes = await kg.read()
        G = pickle.loads(kg_bytes)

        # Convert to undirected if requested
        if undirected:
            G = G.to_undirected()
        # Optionally handle edge weights
        if weighted:
            for u, v, d in G.edges(data=True):
                if "weight" not in d:
                    d["weight"] = 1.0

        node2vec = Node2Vec(
            G,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=workers,
            seed=seed,
            quiet=True,
            p=p,
            q=q,
            weight_key="weight" if weighted else None
        )

        model = node2vec.fit(window=10, min_count=1, batch_words=4, seed=seed)
        idx2node = list(model.wv.index_to_key)
        embeddings = np.stack([model.wv[node] for node in idx2node])

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-8, None)

        emb_bytes, nodes_bytes = serialize_embeddings(embeddings, idx2node)

        if output_format == "json":
            # Return as JSON-friendly structure
            return JSONResponse({
                "embeddings": embeddings.tolist(),
                "nodes": idx2node
            })
        else:
            # Return as .zip file (npy and txt)
            zip_bytes = io.BytesIO()
            with zipfile.ZipFile(zip_bytes, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("embeddings.npy", emb_bytes.getvalue())
                zf.writestr("embeddings_nodes.txt", nodes_bytes.getvalue())
            zip_bytes.seek(0)
            return StreamingResponse(
                zip_bytes,
                media_type="application/zip",
                headers={"Content-Disposition": "attachment; filename=embeddings.zip"}
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Node2vec embedding failed: {str(e)}")