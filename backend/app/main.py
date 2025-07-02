from fastapi.middleware.cors import CORSMiddleware
from app.api import embeddings 
from app.api import convert  
from app.api import nebula
from app.api import neo4j
from app.api import detect_sentence_boundary

app.include_router(convert.router, prefix="/api", tags=["convert"])
app.include_router(embeddings.router, prefix="/api", tags=["embeddings"])
app.include_router(nebula.router, prefix="/api", tags=["nebula"])
app.include_router(neo4j.router, prefix="/api", tags=["neo4j"])
app.include_router(detect_sentence_boundary.router, prefix="/api", tags=["sentence_boundary"])
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