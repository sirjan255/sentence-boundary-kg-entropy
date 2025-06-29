import argparse
import json
from sklearn.metrics import precision_score, recall_score, f1_score

def load_boundaries(path):
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

def flatten_gold_boundaries(gold):
    """Converts dict-of-list gold boundaries to a flat set of node names."""
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

def flatten_pred_boundaries(pred):
    """Flattens predicted boundaries to a set of node names."""
    flat_pred = set()
    if isinstance(pred, dict):
        for v in pred.values():
            if isinstance(v, list):
                flat_pred.update(v)
            elif isinstance(v, dict) and "nodes_in_sentence" in v:
                flat_pred.update(v["nodes_in_sentence"])
            else:
                flat_pred.add(v)
    elif isinstance(pred, list):
        flat_pred.update(pred)
    return flat_pred

def evaluate(gold_set, pred_set):
    all_nodes = sorted(list(gold_set | pred_set))
    gold_labels = [1 if n in gold_set else 0 for n in all_nodes]
    pred_labels = [1 if n in pred_set else 0 for n in all_nodes]

    prec = precision_score(gold_labels, pred_labels, zero_division=0)
    rec = recall_score(gold_labels, pred_labels, zero_division=0)
    f1 = f1_score(gold_labels, pred_labels, zero_division=0)

    missing = gold_set - pred_set
    extra = pred_set - gold_set

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "missing": missing,
        "extra": extra,
        "missing_count": len(missing),
        "gold_count": len(gold_set)
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate predicted boundary nodes against gold.")
    parser.add_argument('--gold', type=str, required=True, help='Gold boundaries JSON file')
    parser.add_argument('--output', type=str, required=True, help='Predicted boundaries JSON file')
    args = parser.parse_args()

    gold_dict, gold_data = load_boundaries(args.gold)
    pred_dict, pred_data = load_boundaries(args.output)

    # Flatten for global evaluation
    gold_set = flatten_gold_boundaries(gold_data)
    pred_set = flatten_pred_boundaries(pred_data)

    results = evaluate(gold_set, pred_set)

    print("\nEvaluation Results:")
    print(f"  Precision: {results['precision']:.3f}")
    print(f"  Recall:    {results['recall']:.3f}")
    print(f"  F1:        {results['f1']:.3f}")
    print(f"  Missing predictions: {results['missing_count']} / {results['gold_count']}")
    if results['missing']:
        print(f"    Missing: {sorted(list(results['missing']))}")
    if results['extra']:
        print(f"    Extra predictions: {sorted(list(results['extra']))}")

if __name__ == "__main__":
    main()