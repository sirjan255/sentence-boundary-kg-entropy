import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv  # GraphSAGE layer
import numpy as np
import pickle
import json
import argparse
import os

def load_graph_and_embeddings(kg_path, embeddings_path, nodes_path):
    # Load graph
    with open(kg_path, "rb") as f:
        G = pickle.load(f)
    # Load embeddings
    embeddings = np.load(embeddings_path)
    # Load node list
    with open(nodes_path, "r", encoding="utf-8") as f:
        node_names = [eval(line.strip()) for line in f if line.strip()]
    node2idx = {n: i for i, n in enumerate(node_names)}
    idx2node = {i: n for n, i in node2idx.items()}
    return G, embeddings, node2idx, idx2node

def build_pyg_data(G, embeddings, node2idx, gold_boundaries):
    # Create node features (embeddings)
    x = torch.tensor(embeddings, dtype=torch.float)
    # Build edge index (undirected for simplicity)
    edges = []
    for u, v in G.edges():
        if u in node2idx and v in node2idx:
            edges.append([node2idx[u], node2idx[v]])
            edges.append([node2idx[v], node2idx[u]])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    # Label: 1 if node is a boundary in any gold group, else 0
    y = torch.zeros(x.size(0), dtype=torch.long)
    for group in gold_boundaries.values():
        for n in group:
            if n in node2idx:
                y[node2idx[n]] = 1
    return Data(x=x, edge_index=edge_index, y=y)

class GNNBoundaryClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, 2)  # 2 classes: not boundary, boundary

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
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 5 == 0:
            pred = out.argmax(dim=1)
            acc = (pred == data.y).float().mean().item()
            print(f"Epoch {epoch+1:02d} Loss: {loss.item():.4f} Acc: {acc:.4f}")

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        y = data.y
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
    args = parser.parse_args()

    G, embeddings, node2idx, idx2node = load_graph_and_embeddings(args.kg, args.embeddings, args.nodes)
    with open(args.gold, "r", encoding="utf-8") as f:
        gold_boundaries = json.load(f)

    data = build_pyg_data(G, embeddings, node2idx, gold_boundaries)
    model = GNNBoundaryClassifier(in_dim=embeddings.shape[1], hidden_dim=args.hidden_dim)

    if not args.predict:
        print("Training GNN boundary classifier...")
        train(model, data, epochs=args.epochs)
        torch.save(model.state_dict(), args.save_model)
    else:
        print("Loading trained model...")
        model.load_state_dict(torch.load(args.save_model))

    print("Evaluating...")
    pred = evaluate(model, data)
    # Save predicted boundaries as a list of boundary node names
    boundary_idxs = (pred == 1).nonzero(as_tuple=True)[0].tolist()
    boundary_nodes = [idx2node[i] for i in boundary_idxs]
    with open("predicted_gnn_boundaries.json", "w", encoding="utf-8") as f:
        json.dump(boundary_nodes, f, indent=2, ensure_ascii=False)
    print(f"Predicted boundary nodes saved to predicted_gnn_boundaries.json")

if __name__ == "__main__":
    main()