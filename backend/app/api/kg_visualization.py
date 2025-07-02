"""
FastAPI API: Visualize a Knowledge Graph

Features:
- Accepts a pickled NetworkX graph (as file upload), from any user source—no local files.
- Generates a PNG visualization of the full graph with:
    - Spring layout
    - Node/edge coloring and labels
    - Edge labels if present ('verb' or 'label' attribute)
- Returns the PNG as base64 string (for inline display in frontend).
- All in-memory, supports any user-uploaded KG.

Frontend: POST KG file to this endpoint.
"""

import io
import pickle
import base64
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

def draw_kg(G):
    # Large figure for the whole graph
    plt.figure(figsize=(20, 15))
    pos = nx.spring_layout(G, seed=42, k=0.5)
    # Draw nodes and edges
    nx.draw(
        G, pos, with_labels=True, node_color='skyblue',
        edge_color='gray', node_size=2000, font_size=9, alpha=0.85
    )
    # Edge label logic
    # If all edges have 'verb', use that; else if all have 'label', use that; else none
    edge_attr = None
    if all('verb' in d for _,_,d in G.edges(data=True)):
        edge_attr = 'verb'
    elif all('label' in d for _,_,d in G.edges(data=True)):
        edge_attr = 'label'
    if edge_attr:
        edge_labels = nx.get_edge_attributes(G, edge_attr)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
    plt.title("Full Knowledge Graph")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_bytes = buf.read()
    return img_bytes

@router.post("/visualize_kg/")
async def visualize_kg(
    kg: UploadFile = File(...)
):
    """
    Visualize the uploaded pickled NetworkX graph and return the PNG as base64.
    """
    try:
        # Load graph from uploaded pickle
        kg_bytes = await kg.read()
        G = pickle.loads(kg_bytes)
        # Validate it's a NetworkX graph
        if not isinstance(G, nx.Graph) and not isinstance(G, nx.DiGraph):
            raise ValueError("Uploaded object is not a NetworkX Graph or DiGraph.")
        img_bytes = draw_kg(G)
        b64 = base64.b64encode(img_bytes).decode()
        return JSONResponse({"image_b64": b64})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"KG visualization failed: {str(e)}")