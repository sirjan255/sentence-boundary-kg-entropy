from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import all API modules
from api import embeddings
from api import convert
from api import nebula
from api import neo4j
from api import detect_sentence_boundary
from api import entropy_traversal
from api import evaluate_boundaries_normalized
from api import evaluate
from api import extract_svo
from api import generate_nodes_to_start
from api import gnn_boundary_experiment
from api import node2vec_embeddings
from api import select_starting_nodes
from api import patch_classifier
from api import patch_embedding
from api import kg_visualization
from api import analytics_dashboard
from api import kg

# Create FastAPI app
app = FastAPI(
    title="Sentence Boundary KG API",
    description="API for building and visualizing Knowledge Graphs from SVO triplets.",
    version="1.0.0",
)

# CORS middleware for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Will adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(convert.router, prefix="/api", tags=["convert"])
app.include_router(embeddings.router, prefix="/api", tags=["embeddings"])
app.include_router(nebula.router, prefix="/api", tags=["nebula"])
app.include_router(neo4j.router, prefix="/api", tags=["neo4j"])
app.include_router(
    detect_sentence_boundary.router, prefix="/api", tags=["sentence_boundary"]
)
app.include_router(entropy_traversal.router, prefix="/api", tags=["entropy_traversal"])
app.include_router(
    evaluate_boundaries_normalized.router, prefix="/api", tags=["evaluation"]
)
app.include_router(evaluate.router, prefix="/api", tags=["evaluation"])
app.include_router(extract_svo.router, prefix="/api", tags=["svo"])
app.include_router(generate_nodes_to_start.router, prefix="/api", tags=["svo"])
app.include_router(gnn_boundary_experiment.router, prefix="/api", tags=["gnn"])
app.include_router(node2vec_embeddings.router, prefix="/api", tags=["embeddings"])
app.include_router(select_starting_nodes.router, prefix="/api", tags=["nodes"])
app.include_router(patch_classifier.router, prefix="/api", tags=["patch-classifier"])
app.include_router(patch_embedding.router, prefix="/api", tags=["patch-embedding"])
app.include_router(kg_visualization.router, prefix="/api", tags=["kg-visualization"])
app.include_router(
    analytics_dashboard.router, prefix="/api", tags=["analytics-dashboard"]
)
app.include_router(kg.router, prefix="/api", tags=["kg"])


@app.get("/")
def read_root():
    return {"msg": "KG API is running!"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "Sentence Boundary KG API is running"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
