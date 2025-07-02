import json
import argparse

def normalize_key(key):
    # Strip single quotes and whitespace
    k = key.strip()
    if k.startswith("'") and k.endswith("'"):
        k = k[1:-1]
    return k.strip()

def normalize_node(node):
    # Remove leading/trailing whitespace, commas, normalize spaces and newlines
    return ' '.join(node.replace('\n', ' ').replace('\r', ' ').split()).strip(' ,')

def extract_node_list(entry):
    if not entry:
        return []
    if isinstance(entry[0], dict) and "node" in entry[0]:
        return [normalize_node(x["node"]) for x in entry if "node" in x]
    return [normalize_node(x) for x in entry]

def group_metrics(pred, gold):
    precisions, recalls, f1s = [], [], []
    all_keys = set(pred.keys()) | set(gold.keys())
    for key in all_keys:
        gold_set = set(extract_node_list(gold.get(key, [])))
        pred_set = set(extract_node_list(pred.get(key, [])))
        if not gold_set and not pred_set:
            continue
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
    parser.add_argument('--predicted', required=True)
    parser.add_argument('--actual', required=True)
    args = parser.parse_args()

    with open(args.predicted, "r", encoding="utf-8") as f:
        pred_raw = json.load(f)
    with open(args.actual, "r", encoding="utf-8") as f:
        gold_raw = json.load(f)

    # Normalize keys
    pred = {normalize_key(k): v for k, v in pred_raw.items()}
    gold = {normalize_key(k): v for k, v in gold_raw.items()}

    macro_p, macro_r, macro_f1 = group_metrics(pred, gold)
    print(f"Macro Precision: {macro_p:.4f}")
    print(f"Macro Recall:    {macro_r:.4f}")
    print(f"Macro F1-score:  {macro_f1:.4f}")

if __name__ == "__main__":
    main()