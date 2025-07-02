"""
FastAPI API for Evaluating Global Boundary Node Predictions

- Accepts two JSON files via form upload:
    - gold: ground truth boundaries (JSON)
    - output: predicted boundaries (JSON)
- Computes global precision, recall, F1 (all nodes, no per-group breakdown)
- Reports missing and extra predictions
- Returns metrics and error analysis as JSON

How to use in frontend:
    - POST multipart/form-data with two files: gold, output
    - Receives JSON response with metrics and details

"""

import json
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from sklearn.metrics import precision_score, recall_score, f1_score

router = APIRouter()

def load_boundaries_from_upload(upload: UploadFile):
    data = json.loads(upload)
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
        raise ValueError("Unknown boundaries format in uploaded file.")

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

def flatten_pred_boundaries(pred):
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
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "missing": sorted(list(missing)),
        "extra": sorted(list(extra)),
        "missing_count": len(missing),
        "gold_count": len(gold_set)
    }

@router.post("/evaluate/")
async def evaluate_boundaries(
    gold: UploadFile = File(...),
    output: UploadFile = File(...)
):
    """
    Evaluate predicted vs gold boundary nodes globally (no per-group breakdown).
    Returns precision, recall, F1, missing/extra analysis as JSON.
    """
    try:
        gold_raw = await gold.read()
        output_raw = await output.read()
        gold_dict, gold_data = load_boundaries_from_upload(gold_raw.decode("utf-8"))
        pred_dict, pred_data = load_boundaries_from_upload(output_raw.decode("utf-8"))
        gold_set = flatten_gold_boundaries(gold_data)
        pred_set = flatten_pred_boundaries(pred_data)
        results = evaluate(gold_set, pred_set)
        return JSONResponse(results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Evaluation failed: {str(e)}")