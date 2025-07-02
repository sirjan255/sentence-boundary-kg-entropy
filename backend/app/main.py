from fastapi.middleware.cors import CORSMiddleware

from app.api import kg  # Import your router

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