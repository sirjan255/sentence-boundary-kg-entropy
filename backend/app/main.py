from fastapi.middleware.cors import CORSMiddleware
from app.api import embeddings 
from app.api import convert  
from app.api import nebula
from app.api import neo4j
from app.api import detect_sentence_boundary
from app.api import entropy_traversal
from app.api import evaluate_boundaries_normalized
from app.api import evaluate
from app.api import extract_svo
from app.api import generate_nodes_to_start
from app.api import gnn_boundary_experiment

app.include_router(convert.router, prefix="/api", tags=["convert"])
app.include_router(embeddings.router, prefix="/api", tags=["embeddings"])
app.include_router(nebula.router, prefix="/api", tags=["nebula"])
app.include_router(neo4j.router, prefix="/api", tags=["neo4j"])
app.include_router(detect_sentence_boundary.router, prefix="/api", tags=["sentence_boundary"])
app.include_router(entropy_traversal.router, prefix="/api", tags=["entropy_traversal"])
app.include_router(evaluate_boundaries_normalized.router, prefix="/api", tags=["evaluation"])
app.include_router(evaluate.router, prefix="/api", tags=["evaluation"])
app.include_router(extract_svo.router, prefix="/api", tags=["svo"])
app.include_router(generate_nodes_to_start.router, prefix="/api", tags=["svo"])
app.include_router(gnn_boundary_experiment.router, prefix="/api", tags=["gnn"])
from app.api import kg  

app = FastAPI(
    title="Sentence Boundary KG API",
    description="API for building and visualizing Knowledge Graphs from SVO triplets.",
    version="1.0.0"
)

# CORS for frontend development - allow localhost:5173 etc
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Will Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(kg.router, prefix="/api", tags=["kg"])

@app.get("/")
def read_root():
    return {"msg": "KG API is running!"}