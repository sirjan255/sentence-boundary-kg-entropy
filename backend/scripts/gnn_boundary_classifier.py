import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import numpy as np
import pickle
import json
import argparse
import os
import random

def load_graph_and_embeddings(kg_path, embeddings_path, nodes_path):
    with open(kg_path, "rb") as f:
        G = pickle.load(f)
    embeddings = np.load(embeddings_path)
    with open(nodes_path, "r", encoding="utf-8") as f:
        node_names = [eval(line.strip()) for line in f if line.strip()]
    node2idx = {n: i for i, n in enumerate(node_names)}
    idx2node = {i: n for n, i in node2idx.items()}
    return G, embeddings, node2idx, idx2node

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
        if (epoch+1) % 5 == 0 or epoch == 0:
            pred = out.argmax(dim=1)
            acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
            print(f"Epoch {epoch+1:02d} Loss: {loss.item():.4f} Train Acc: {acc:.4f}")

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
        print(f"Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}")
        return pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kg', type=str, required=True)
    parser.add_argument('--embeddings', type=str, required=True)
    parser.add_argument('--nodes', type=str, required=True)
    parser.add_argument('--gold', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--save_model', type=str, default="gnn_boundary.pt")
    parser.add_argument('--predict', action='store_true', help="Run prediction only")
    parser.add_argument('--output', type=str, default="predicted_gnn_boundaries.json")
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    G, embeddings, node2idx, idx2node = load_graph_and_embeddings(args.kg, args.embeddings, args.nodes)
    with open(args.gold, "r", encoding="utf-8") as f:
        gold_boundaries = json.load(f)

    data = build_pyg_data(G, embeddings, node2idx, gold_boundaries, test_ratio=args.test_ratio, seed=args.seed)
    model = GNNBoundaryClassifier(in_dim=embeddings.shape[1], hidden_dim=args.hidden_dim)

    if not args.predict:
        print("Training GNN boundary classifier...")
        train(model, data, epochs=args.epochs)
        torch.save(model.state_dict(), args.save_model)
    else:
        print("Loading trained model...")
        model.load_state_dict(torch.load(args.save_model))

    print("Evaluating on TEST set...")
    pred = evaluate(model, data, mask=data.test_mask)
    # Save predicted boundaries as a list of boundary node names (ONLY for test nodes)
    test_indices = data.test_mask.nonzero(as_tuple=True)[0].tolist()
    boundary_idxs = (pred == 1).nonzero(as_tuple=True)[0].tolist()
    test_idx_map = {i: test_indices[i] for i in range(len(test_indices))}
    pred_test_idxs = [test_idx_map[i] for i in boundary_idxs if i in test_idx_map]
    boundary_nodes = [idx2node[i] for i in pred_test_idxs]
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(boundary_nodes, f, indent=2, ensure_ascii=False)
    print(f"Predicted boundary nodes saved to {args.output}")

    # Optionally, show gold/test node counts
    test_gold_nodes = [i for i in test_indices if data.y[i]==1]
    print(f"Gold test boundary nodes: {len(test_gold_nodes)} / {len(test_indices)}")

if __name__ == "__main__":
    main()