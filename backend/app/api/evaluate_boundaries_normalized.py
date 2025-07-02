"""
FastAPI API for Evaluating Sentence Boundary Predictions with Normalization

- Accepts two JSON files via form upload:
    - predicted: predicted boundaries (JSON)
    - actual: ground truth boundaries (JSON)
- Computes macro precision, recall, F1 with normalization (for noisy/variant keys/nodes)
- Returns metrics as JSON

How to use in frontend:
    - POST multipart/form-data with two JSON files: predicted, actual
    - Receive response: { "macro_precision": ..., "macro_recall": ..., "macro_f1": ... }  
"""

import json
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

def normalize_key(key):
    k = key.strip()
    if k.startswith("'") and k.endswith("'"):
        k = k[1:-1]
    return k.strip()

def normalize_node(node):
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

@router.post("/evaluate_boundaries_normalized/")
async def evaluate_boundaries_normalized(
    predicted: UploadFile = File(...),
    actual: UploadFile = File(...)
):
    """
    Evaluate predicted vs actual sentence boundaries with normalization.
    Returns macro precision, recall, F1.
    """
    try:
        pred_raw = json.loads((await predicted.read()).decode("utf-8"))
        gold_raw = json.loads((await actual.read()).decode("utf-8"))
        pred = {normalize_key(k): v for k, v in pred_raw.items()}
        gold = {normalize_key(k): v for k, v in gold_raw.items()}
        macro_p, macro_r, macro_f1 = group_metrics(pred, gold)
        return JSONResponse({
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "macro_f1": round(macro_f1, 4)
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Evaluation failed: {str(e)}")