import argparse
import os
import json
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt

def load_boundaries(path):
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # If data is a dict
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            # Dict of dicts with 'nodes_in_sentence'
            if isinstance(v, dict) and "nodes_in_sentence" in v:
                result[k] = set(v["nodes_in_sentence"])
            # Dict of lists (already just node names)
            elif isinstance(v, list):
                result[k] = set(v)
            else:
                # Unexpected value type, just coerce to set
                result[k] = set([v])
        return result, data

    # If data is a list (flat list of node names)
    elif isinstance(data, list):
        # Treat as a single group with all boundaries
        return {"all": set(data)}, data

    else:
        raise ValueError(f"Unknown boundaries format in {path}")

def evaluate_sentence_grouping(pred, gold):
    """
    For each start node, computes F1, precision, and recall between predicted and gold node sets.
    Returns macro-averaged metrics.
    """
    f1s, precisions, recalls = [], [], []
    missing = 0
    for start in gold.keys():
        gold_set = gold[start]
        pred_set = pred.get(start, set())
        if not pred_set:
            f1s.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
            missing += 1
            continue
        all_nodes = sorted(gold_set | pred_set)
        y_true = [1 if n in gold_set else 0 for n in all_nodes]
        y_pred = [1 if n in pred_set else 0 for n in all_nodes]
        f1s.append(f1_score(y_true, y_pred))
        precisions.append(precision_score(y_true, y_pred))
        recalls.append(recall_score(y_true, y_pred))
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    macro_prec = sum(precisions) / len(precisions) if precisions else 0.0
    macro_rec = sum(recalls) / len(recalls) if recalls else 0.0
    return macro_f1, macro_prec, macro_rec, missing

def boundary_precision(pred, gold):
    """
    Boundary precision: Of the predicted 'boundary' nodes (highest-entropy in group),
    how many are true sentence endpoints?
    Assumes the last node in predicted nodes_in_sentence is the predicted endpoint.
    """
    correct = 0
    total = 0
    for start in gold.keys():
        gold_nodes = list(gold[start])
        pred_nodes = pred.get(start, [])
        if not pred_nodes:
            continue
        pred_last = None
        if isinstance(pred_nodes, set):
            pred_nodes = sorted(pred_nodes)
        if isinstance(pred_nodes, list):
            pred_last = pred_nodes[-1] if pred_nodes else None
        if pred_last and pred_last in gold_nodes:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0

def traversal_efficiency(pred, gold):
    """
    Traversal efficiency: Average number of nodes visited vs. gold.
    Lower is better, but should cover all gold nodes.
    """
    ratios = []
    for start in gold.keys():
        pred_set = set(pred.get(start, []))
        gold_set = set(gold[start])
        if len(gold_set) == 0:
            continue
        ratios.append(len(pred_set) / len(gold_set))
    return sum(ratios) / len(ratios) if ratios else 0.0

def entropy_analysis(pred_entropies, pred, gold):
    """
    Analyze entropy at predicted boundary nodes vs. inner nodes.
    Print and plot statistics for insight into BLT-style entropy effectiveness.
    """
    boundary_entropies = []
    non_boundary_entropies = []
    misses = 0

    for start in gold.keys():
        pred_nodes = pred.get(start, [])
        entropies = pred_entropies.get(start, {})
        if not pred_nodes or not entropies:
            misses += 1
            continue
        if isinstance(pred_nodes, set):
            pred_nodes = sorted(pred_nodes)
        boundary_node = pred_nodes[-1]
        # Entropy at the predicted boundary
        boundary_entropy = entropies.get(boundary_node, None)
        if boundary_entropy is not None:
            boundary_entropies.append(boundary_entropy)
        # Entropy at inner nodes
        for n in pred_nodes[:-1]:
            e = entropies.get(n, None)
            if e is not None:
                non_boundary_entropies.append(e)
    print("\nEntropy Analysis (BLT-style):")
    print(f"  Number of samples: {len(boundary_entropies)} boundaries, {len(non_boundary_entropies)} inner nodes")
    print(f"  Misses (no prediction or entropies): {misses}")
    print(f"  Boundary entropy: mean={np.mean(boundary_entropies):.3f}, std={np.std(boundary_entropies):.3f}")
    print(f"  Non-boundary entropy: mean={np.mean(non_boundary_entropies):.3f}, std={np.std(non_boundary_entropies):.3f}")

    # Optional: plot
    try:
        plt.figure(figsize=(7,4))
        plt.hist(boundary_entropies, bins=20, alpha=0.7, label="Boundary nodes")
        plt.hist(non_boundary_entropies, bins=20, alpha=0.7, label="Non-boundary nodes")
        plt.xlabel("Entropy")
        plt.ylabel("Count")
        plt.title("Entropy distribution at boundaries vs. inner nodes")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Plotting failed (no display?): {e}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate sentence boundary detection on KG traversal results.")
    parser.add_argument("--gold", type=str, required=True, help="Gold standard boundaries JSON file")
    parser.add_argument("--pred", type=str, required=True, help="Predicted boundaries JSON file")
    parser.add_argument("--entropy_analysis", action="store_true", help="Analyze and plot entropy statistics (BLT-style)")
    args = parser.parse_args()

    if not os.path.exists(args.gold):
        print(f"Gold file {args.gold} does not exist.")
        return
    if not os.path.exists(args.pred):
        print(f"Prediction file {args.pred} does not exist.")
        return

    gold, _ = load_boundaries(args.gold)
    pred, pred_entropies = load_boundaries(args.pred)

    macro_f1, macro_prec, macro_rec, missing = evaluate_sentence_grouping(pred, gold)
    bound_prec = boundary_precision(pred, gold)
    eff = traversal_efficiency(pred, gold)

    print(f"\nEvaluation Results:")
    print(f"  Macro F1: {macro_f1:.3f} | Macro Prec: {macro_prec:.3f} | Macro Rec: {macro_rec:.3f}")
    print(f"  Boundary Precision: {bound_prec:.3f}")
    print(f"  Traversal Efficiency: {eff:.3f}")
    print(f"  Missing predictions: {missing} / {len(gold)}")

    if args.entropy_analysis:
        entropy_analysis(pred_entropies, pred, gold)

if __name__ == "__main__":
    main()