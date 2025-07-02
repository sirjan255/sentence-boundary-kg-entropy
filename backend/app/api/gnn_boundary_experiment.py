"""
FastAPI API: GNN Boundary Classifier Experiments

- Accepts all KG data and experiment hyperparameters as form upload/fields from the frontend.
- Runs GNN boundary classification experiments using PyTorch Geometric.
- Returns results (metrics summary and per-run details) as JSON.
- Does NOT use or save any local files; all data is in-memory.

Frontend usage:
    - POST multipart/form-data with:
        - kg: Pickle file (networkx graph)
        - embeddings: .npy node embeddings
        - nodes: .txt node list (one per line, Python literal if needed)
        - gold: .json gold standard boundaries (see flatten_gold_boundaries)
        - epochs: int (default=30)
        - lr: float (default=0.01)
        - test_ratios: str (comma-separated, e.g. "0.1,0.2")
        - hidden_dims: str (comma-separated, e.g. "32,64")
        - seeds: str (comma-separated, e.g. "42,2024")
        - runs_per_setting: int (default=3)
    - Receives JSON summary and all per-run metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import json
import io
import random
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

# PyTorch Geometric imports
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

router = APIRouter()

def flatten_gold_boundaries(gold):
    flat_gold = set()
    if isinstance(gold, dict):
        for v in gold.values():
            if isinstance(v, list):
                flat_gold.update(v)
            elif isinstance(v, dict) and "nodes_in_sentence" in v:
                flat_gold.update(v["nodes_in_sentence"])
            else:
                flat_gold.add(v)
    elif isinstance(gold, list):
        flat_gold.update(gold)
    return flat_gold

def build_pyg_data(G, embeddings, node2idx, gold_boundaries, test_ratio=0.2, seed=42):
    x = torch.tensor(embeddings, dtype=torch.float)
    edges = []
    for u, v in G.edges():
        if u in node2idx and v in node2idx:
            edges.append([node2idx[u], node2idx[v]])
            edges.append([node2idx[v], node2idx[u]])
    if len(edges) == 0:
        raise ValueError("No edges found in the graph for the given node list.")
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    y = torch.zeros(x.size(0), dtype=torch.long)
    gold_boundary_nodes = flatten_gold_boundaries(gold_boundaries)
    for n in gold_boundary_nodes:
        if n in node2idx:
            y[node2idx[n]] = 1
    random.seed(seed)
    all_indices = list(range(x.size(0)))
    random.shuffle(all_indices)
    test_size = int(test_ratio * len(all_indices))
    test_indices = all_indices[:test_size]
    train_indices = all_indices[test_size:]
    test_mask = torch.zeros(x.size(0), dtype=torch.bool)
    train_mask = torch.zeros(x.size(0), dtype=torch.bool)
    test_mask[test_indices] = True
    train_mask[train_indices] = True
    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)

class GNNBoundaryClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.lin(x)
        return x

def train(model, data, epochs=30, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

def evaluate(model, data, mask=None):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        y = data.y
        if mask is not None:
            pred = pred[mask]
            y = y[mask]
        tp = ((pred == 1) & (y == 1)).sum().item()
        fp = ((pred == 1) & (y == 0)).sum().item()
        fn = ((pred == 0) & (y == 1)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return precision, recall, f1, pred

@router.post("/gnn_boundary_experiment/")
async def gnn_boundary_experiment(
    kg: UploadFile = File(...),
    embeddings: UploadFile = File(...),
    nodes: UploadFile = File(...),
    gold: UploadFile = File(...),
    epochs: int = Form(30),
    lr: float = Form(0.01),
    test_ratios: str = Form("0.1,0.2,0.3,0.5"),
    hidden_dims: str = Form("32,64,128"),
    seeds: str = Form("42,2024,7"),
    runs_per_setting: int = Form(3)
):
    """
    Run GNN boundary classifier experiments with all parameters provided by the user.
    Returns JSON with all results (no file I/O).
    """
    try:
        # Load KG
        kg_bytes = await kg.read()
        G = pickle.loads(kg_bytes)
        # Load embeddings
        embeddings_np = np.load(io.BytesIO(await embeddings.read()))
        # Load node list
        node_lines = (await nodes.read()).decode("utf-8").splitlines()
        node_names = [eval(line.strip()) for line in node_lines if line.strip()]
        node2idx = {n: i for i, n in enumerate(node_names)}
        idx2node = {i: n for n, i in node2idx.items()}
        # Load gold boundaries
        gold_boundaries = json.loads((await gold.read()).decode("utf-8"))

        test_ratios_arr = [float(x) for x in test_ratios.split(",")]
        hidden_dims_arr = [int(x) for x in hidden_dims.split(",")]
        seeds_arr = [int(x) for x in seeds.split(",")]

        results = []
        per_run = []

        for test_ratio in test_ratios_arr:
            for hidden_dim in hidden_dims_arr:
                for seed in seeds_arr:
                    for run in range(runs_per_setting):
                        run_seed = seed + run
                        # Prepare data and model
                        data = build_pyg_data(G, embeddings_np, node2idx, gold_boundaries, test_ratio=test_ratio, seed=run_seed)
                        model = GNNBoundaryClassifier(in_dim=embeddings_np.shape[1], hidden_dim=hidden_dim)
                        train(model, data, epochs=epochs, lr=lr)
                        precision, recall, f1, pred = evaluate(model, data, mask=data.test_mask)
                        test_indices = data.test_mask.nonzero(as_tuple=True)[0].tolist()
                        gold_test_count = int(((data.y[data.test_mask]) == 1).sum().item())
                        per_run.append({
                            "test_ratio": test_ratio,
                            "hidden_dim": hidden_dim,
                            "seed": run_seed,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                            "test_size": len(test_indices),
                            "gold_test_count": gold_test_count
                        })

        # Summarize results per (test_ratio, hidden_dim)
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in per_run:
            key = (r["test_ratio"], r["hidden_dim"])
            grouped[key].append(r)
        summary = []
        for key, vals in grouped.items():
            n = len(vals)
            avg_prec = sum(v["precision"] for v in vals)/n
            avg_rec = sum(v["recall"] for v in vals)/n
            avg_f1 = sum(v["f1"] for v in vals)/n
            avg_tsize = sum(v["test_size"] for v in vals)//n
            avg_goldc = sum(v["gold_test_count"] for v in vals)//n
            summary.append({
                "test_ratio": key[0],
                "hidden_dim": key[1],
                "avg_precision": avg_prec,
                "avg_recall": avg_rec,
                "avg_f1": avg_f1,
                "avg_test_size": avg_tsize,
                "avg_gold_test_count": avg_goldc,
                "runs": n
            })

        return JSONResponse({
            "summary": summary,
            "runs": per_run
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GNN experiment failed: {str(e)}")