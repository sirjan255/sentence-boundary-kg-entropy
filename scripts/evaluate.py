import argparse
import json
import os
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score

def load_boundaries(path):
    """
    Loads predicted or gold boundaries in the format:
    {
      "start_node1": {
        "nodes_in_sentence": [node1, node2, ...],
        ...
      },
      ...
    }
    Returns a dict: start_node -> set(nodes_in_sentence)
    """
    with open(path, "r", encoding="utf-8") as f:
        boundaries = json.load(f)
    result = dict()
    for k, v in boundaries.items():
        result[k] = set(v["nodes_in_sentence"])
    return result

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
            # If no prediction, treat as all false (worst case)
            f1s.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
            missing += 1
            continue
        # Convert sets to binary labels over the union of both sets
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
        # If prediction is a set, convert to ordered list (for deterministic order)
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

def main():
    parser = argparse.ArgumentParser(description="Evaluate sentence boundary detection on KG traversal results.")
    parser.add_argument("--gold", type=str, required=True, help="Gold standard boundaries JSON file")
    parser.add_argument("--pred", type=str, required=True, help="Predicted boundaries JSON file")
    args = parser.parse_args()

    if not os.path.exists(args.gold):
        print(f"Gold file {args.gold} not found.")
        return
    if not os.path.exists(args.pred):
        print(f"Prediction file {args.pred} not found.")
        return

    gold = load_boundaries(args.gold)
    pred = load_boundaries(args.pred)

    macro_f1, macro_prec, macro_rec, missing = evaluate_sentence_grouping(pred, gold)
    b_prec = boundary_precision(pred, gold)
    eff = traversal_efficiency(pred, gold)

    print("==== Sentence Boundary Detection Evaluation ====")
    print(f"Macro F1-score (sentence grouping): {macro_f1:.4f}")
    print(f"Macro Precision:                   {macro_prec:.4f}")
    print(f"Macro Recall:                      {macro_rec:.4f}")
    print(f"Boundary Precision (endpoint):     {b_prec:.4f}")
    print(f"Traversal Efficiency (ratio):      {eff:.4f}")
    print(f"Num. missing predictions:          {missing} / {len(gold)}")

if __name__ == "__main__":
    main()