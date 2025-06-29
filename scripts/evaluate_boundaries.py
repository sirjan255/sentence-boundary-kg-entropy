import json
import argparse

def extract_node_list(entry):
    """Extract a list of node strings from either a list of dicts (with 'node' key) or a list of strings."""
    if not entry:
        return []
    if isinstance(entry[0], dict) and "node" in entry[0]:
        return [x["node"] for x in entry if "node" in x]
    return [str(x) for x in entry]

def group_metrics(pred, gold):
    """
    Computes macro-averaged precision, recall, and F1-score over all groups.
    For each key (start node), compares predicted and gold sets.
    """
    precisions, recalls, f1s = [], [], []
    all_keys = set(gold.keys()) | set(pred.keys())
    for key in all_keys:
        gold_set = set(extract_node_list(gold.get(key, [])))
        pred_set = set(extract_node_list(pred.get(key, [])))
        if not gold_set and not pred_set:
            continue  # skip if both are empty
        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    macro_p = sum(precisions) / len(precisions) if precisions else 0
    macro_r = sum(recalls) / len(recalls) if recalls else 0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0
    return macro_p, macro_r, macro_f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predicted', required=True, help='Path to predicted boundaries JSON')
    parser.add_argument('--actual', required=True, help='Path to actual/ground-truth boundaries JSON')
    args = parser.parse_args()

    pred = json.load(open(args.predicted, "r", encoding="utf-8"))
    gold = json.load(open(args.actual, "r", encoding="utf-8"))
    macro_p, macro_r, macro_f1 = group_metrics(pred, gold)
    print(f"Macro Precision: {macro_p:.4f}")
    print(f"Macro Recall:    {macro_r:.4f}")
    print(f"Macro F1-score:  {macro_f1:.4f}")

if __name__ == "__main__":
    main()