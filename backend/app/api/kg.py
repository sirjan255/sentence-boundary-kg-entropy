import csv
import pickle
import networkx as nx
import io
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

router = APIRouter()

def build_kg_from_triplets_filelike(triplet_file):
    """
    Build a directed knowledge graph from a file-like object containing SVO triplets (CSV).
    Each unique subject/object becomes a node.
    Each (subject, verb, object) triplet becomes a directed edge (subject -> object) with verb as edge attribute.
    Returns: networkx.DiGraph
    """
    G = nx.DiGraph()
    triplet_file.seek(0)
    reader = csv.DictReader(io.TextIOWrapper(triplet_file, encoding="utf-8"))
    for row in reader:
        subj = row["subject"].strip()
        obj = row["object"].strip()
        verb = row["verb"].strip()
        sentence = row["sentence"].strip() if "sentence" in row else ""
        if subj:
            G.add_node(subj)
        if obj:
            G.add_node(obj)
        if subj and obj and verb:
            G.add_edge(subj, obj, verb=verb, sentence=sentence)
    return G

@router.post("/build_kg/")
async def build_kg(triplets: UploadFile = File(...)):
    """
    Accepts a CSV file of SVO triplets and returns the pickled Knowledge Graph.
    """
    if triplets.filename and not triplets.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    try:
        file_bytes = await triplets.read()
        file_like = io.BytesIO(file_bytes)
        G = build_kg_from_triplets_filelike(file_like)
        # Serialize with pickle and return as a streaming response
        buf = io.BytesIO()
        pickle.dump(G, buf)
        buf.seek(0)
        return StreamingResponse(buf, media_type="application/octet-stream", headers={
            "Content-Disposition": "attachment; filename=kg.pkl"
        })
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"KG construction failed: {str(e)}")

@router.post("/build_kg/preview/")
async def build_kg_preview(triplets: UploadFile = File(...)):
    """
    Accepts a CSV file of SVO triplets and returns a lightweight JSON preview of the KG.
    """
    if triplets.filename and not triplets.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    try:
        file_bytes = await triplets.read()
        file_like = io.BytesIO(file_bytes)
        G = build_kg_from_triplets_filelike(file_like)
        nodes = list(G.nodes)
        edges = [{"source": u, "target": v, "verb": d.get("verb", "")} for u, v, d in G.edges(data=True)]
        return JSONResponse({"num_nodes": len(nodes), "num_edges": len(edges), "nodes": nodes, "edges": edges})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KG preview failed: {str(e)}")