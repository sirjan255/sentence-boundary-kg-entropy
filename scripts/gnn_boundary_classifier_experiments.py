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
import random
from tqdm import tqdm

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

def flatten_gold_boundaries(gold):
    """Converts dict-of-list or dict-of-dict gold boundaries to a flat set of node names."""
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
    # Create node features (embeddings)
    x = torch.tensor(embeddings, dtype=torch.float)
    # Build edge index (undirected for simplicity)
    edges = []
    for u, v in G.edges():
        if u in node2idx and v in node2idx:
            edges.append([node2idx[u], node2idx[v]])
            edges.append([node2idx[v], node2idx[u]])
    if len(edges) == 0:
        raise ValueError("No edges found in the graph for the given node list.")
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    # Label: 1 if node is a boundary, else 0
    y = torch.zeros(x.size(0), dtype=torch.long)
    gold_boundary_nodes = flatten_gold_boundaries(gold_boundaries)
    for n in gold_boundary_nodes:
        if n in node2idx:
            y[node2idx[n]] = 1
    # Split nodes into train/test
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
        self.lin = nn.Linear(hidden_dim, 2)  # 2 classes: not boundary, boundary

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.lin(x)
        return x

def train(model, data, epochs=30, lr=0.01, verbose=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if verbose and ((epoch+1) % 5 == 0 or epoch == 0):
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
        return precision, recall, f1, pred

def experiment_run(G, embeddings, node2idx, idx2node, gold_boundaries, test_ratio, seed, hidden_dim, epochs, lr, verbose=False):
    data = build_pyg_data(G, embeddings, node2idx, gold_boundaries, test_ratio=test_ratio, seed=seed)
    model = GNNBoundaryClassifier(in_dim=embeddings.shape[1], hidden_dim=hidden_dim)
    train(model, data, epochs=epochs, lr=lr, verbose=verbose)
    precision, recall, f1, pred = evaluate(model, data, mask=data.test_mask)
    # For analysis: count test gold boundaries
    test_indices = data.test_mask.nonzero(as_tuple=True)[0].tolist()
    gold_test_count = int(((data.y[data.test_mask]) == 1).sum().item())
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "test_size": len(test_indices),
        "gold_test_count": gold_test_count,
        "test_indices": test_indices,
        "pred": pred,
        "data": data,
        "model": model
    }

def run_experiments(
    kg, embeddings, nodes, gold,
    test_ratios=[0.1, 0.2, 0.3, 0.5],
    seeds=[42, 2024, 7],
    hidden_dims=[32, 64, 128],
    epochs=30,
    lr=0.01,
    runs_per_setting=3,
    output_dir="gnn_experiment_results",
    verbose=False
):
    os.makedirs(output_dir, exist_ok=True)
    # Load everything once
    G, emb, node2idx, idx2node = load_graph_and_embeddings(kg, embeddings, nodes)
    with open(gold, "r", encoding="utf-8") as f:
        gold_boundaries = json.load(f)

    results = []
    print("Starting GNN boundary classifier experiments...")
    # Try every combination
    for test_ratio in test_ratios:
        for hidden_dim in hidden_dims:
            for seed in seeds:
                for run in range(runs_per_setting):
                    run_seed = seed + run  # vary seed per run
                    desc = f"test_ratio={test_ratio}, hidden_dim={hidden_dim}, seed={run_seed}"
                    if verbose:
                        print(f"\nRunning {desc}")
                    res = experiment_run(
                        G, emb, node2idx, idx2node, gold_boundaries,
                        test_ratio=test_ratio,
                        seed=run_seed,
                        hidden_dim=hidden_dim,
                        epochs=epochs,
                        lr=lr,
                        verbose=verbose
                    )
                    res_row = {
                        "test_ratio": test_ratio,
                        "hidden_dim": hidden_dim,
                        "seed": run_seed,
                        "precision": res["precision"],
                        "recall": res["recall"],
                        "f1": res["f1"],
                        "test_size": res["test_size"],
                        "gold_test_count": res["gold_test_count"]
                    }
                    results.append(res_row)
                    # Save predictions for this experiment
                    pred = res["pred"]
                    test_indices = res["test_indices"]
                    boundary_idxs = (pred == 1).nonzero(as_tuple=True)[0].tolist()
                    test_idx_map = {i: test_indices[i] for i in range(len(test_indices))}
                    pred_test_idxs = [test_idx_map[i] for i in boundary_idxs if i in test_idx_map]
                    boundary_nodes = [idx2node[i] for i in pred_test_idxs]
                    pred_filename = os.path.join(
                        output_dir,
                        f"predicted_{test_ratio}_{hidden_dim}_{run_seed}.json"
                    )
                    with open(pred_filename, "w", encoding="utf-8") as f:
                        json.dump(boundary_nodes, f, indent=2, ensure_ascii=False)
                    if verbose:
                        print(f"Saved test predictions to {pred_filename}")
    # Save results to CSV
    import csv
    csv_path = os.path.join(output_dir, "experiment_results.csv")
    with open(csv_path, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nAll experiment results saved to {csv_path}")
    # Print a summary table
    print("\nSummary of results (average per setting):")
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        key = (r["test_ratio"], r["hidden_dim"])
        grouped[key].append(r)
    print(f"{'Test Ratio':<10} {'HiddenDim':<10} {'Prec':<7} {'Rec':<7} {'F1':<7} {'TestSize':<9} {'GoldTestCt':<10} {'Runs':<4}")
    for key, vals in grouped.items():
        n = len(vals)
        avg_prec = sum(v["precision"] for v in vals)/n
        avg_rec = sum(v["recall"] for v in vals)/n
        avg_f1 = sum(v["f1"] for v in vals)/n
        avg_tsize = sum(v["test_size"] for v in vals)//n
        avg_goldc = sum(v["gold_test_count"] for v in vals)//n
        print(f"{key[0]:<10} {key[1]:<10} {avg_prec:.3f}  {avg_rec:.3f}  {avg_f1:.3f}  {avg_tsize:<9} {avg_goldc:<10} {n:<4}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kg', type=str, required=True)
    parser.add_argument('--embeddings', type=str, required=True)
    parser.add_argument('--nodes', type=str, required=True)
    parser.add_argument('--gold', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--test_ratios', type=str, default="0.1,0.2,0.3,0.5")
    parser.add_argument('--hidden_dims', type=str, default="32,64,128")
    parser.add_argument('--seeds', type=str, default="42,2024,7")
    parser.add_argument('--runs_per_setting', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default="gnn_experiment_results")
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    test_ratios = [float(x) for x in args.test_ratios.split(",")]
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    run_experiments(
        kg=args.kg,
        embeddings=args.embeddings,
        nodes=args.nodes,
        gold=args.gold,
        test_ratios=test_ratios,
        seeds=seeds,
        hidden_dims=hidden_dims,
        epochs=args.epochs,
        lr=args.lr,
        runs_per_setting=args.runs_per_setting,
        output_dir=args.output_dir,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()