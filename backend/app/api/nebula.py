"""
Nebula Graph API Endpoint for SVO Triplet Ingestion and Visualization (for Frontend)

This FastAPI router exposes an endpoint to:
- Accept a CSV file of SVO triplets (subject, verb, object) uploaded by the user.
- Insert those triplets as nodes/edges into a running Nebula Graph database.
- Query the resulting subgraph and return nodes/edges as JSON for frontend visualization.


**How to Use This Endpoint in App:**
1. Ensure Nebula Graph and Nebula Studio are running and accessible (see below).
2. POST a CSV file (with columns: "subject", "verb", "object") to `/api/nebula/upload_triplets/`
3. The endpoint will:
   - Ingest the triplets into the specified Nebula Graph space.
   - Query all (up to 100) directed edges and nodes.
   - Return: {"nodes": [...], "edges": [...]} for use with a frontend graph library.

**Nebula Graph Setup Instructions**
You must have a Nebula Graph server running and accessible from your backend host!
This API DOES NOT start/stop Nebula for you -- it only connects to an existing server.

**For Local Development:**
- Recommended: Use Docker Compose (see Nebula docs: https://docs.nebula-graph.io/3.6.0/)

Quick Start with Docker Compose (Linux/macOS):
1. Download Nebula's docker-compose.yml: 
   wget https://raw.githubusercontent.com/vesoft-inc/nebula-docker-compose/master/docker-compose.yaml
2. Run Nebula (from the directory containing docker-compose.yaml):
   docker-compose up -d
3. Confirm Nebula services are running:
   docker ps
   # You should see: metad, storaged, graphd containers
4. Nebula Graph server will be accessible at 127.0.0.1:9669 (by default).
5. (Optional) Start Nebula Studio for browser visualization:
   - See https://docs.nebula-graph.io/3.6.0/4.deployment-and-installation/4.4.install-studio/
   - Studio runs on 127.0.0.1:7000 or 127.0.0.1:7001

**Environment Variables:**
- NEBULA_USER: Username for Nebula DB (default: "root")
- NEBULA_PWD:  Password for Nebula DB (default: "nebula")
- NEBULA_HOST: Host/IP (default: "127.0.0.1")
- NEBULA_PORT: Port (default: 9669)
- NEBULA_SPACE: Graph space name to use (default: "sentence_kg")

**For Production:**
- Deploy Nebula Graph server and Studio on production host(s).
- Make sure the backend API server can connect to Nebula's graphd endpoint.

--------------------------------------------------------------------------------------
**Frontend Integration:**
- After calling this endpoint, use the returned "nodes" and "edges" to render an interactive graph in the React app
- There is no need to show Nebula Studio to users -- it's just for admin/dev use.
- All ingestion/visualization can be handled through the custom UI only

**Troubleshooting:**
- If you get a "Nebula Graph is not running" error: Make sure docker-compose containers are up and NEBULA_HOST/PORT are correct.
- If you get connection/auth errors: Check your NEBULA_USER, NEBULA_PWD, and Nebula's logs.
- To reset the graph, you may manually drop the space or restart the containers.

"""

import os
import io
import csv
import time
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

# Nebula DB connection parameters
NEBULA_USER = os.getenv("NEBULA_USER", "root")
NEBULA_PWD = os.getenv("NEBULA_PWD", "nebula")
NEBULA_HOST = os.getenv("NEBULA_HOST", "127.0.0.1")
NEBULA_PORT = int(os.getenv("NEBULA_PORT", 9669))
NEBULA_SPACE = os.getenv("NEBULA_SPACE", "sentence_kg")

router = APIRouter()

def ensure_nebula_running():
    """
    Checks if the Nebula Graph server is listening on the configured host/port.
    Raises HTTPException if not available.
    """
    import socket
    s = socket.socket()
    try:
        s.connect((NEBULA_HOST, NEBULA_PORT))
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Nebula Graph is not running on host/port specified. Please check Nebula Graph server."
        )
    finally:
        s.close()

def connect_nebula():
    """
    Connects to Nebula Graph using nebula3-python client.
    Returns a session object for executing queries.
    """
    config = Config()
    config.max_connection_pool_size = 10
    pool = ConnectionPool()
    ok = pool.init([(NEBULA_HOST, NEBULA_PORT)], config)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to connect to Nebula Graph.")
    session = pool.get_session(NEBULA_USER, NEBULA_PWD)
    return session

def create_space_and_schema(session, space=NEBULA_SPACE):
    """
    Creates the Nebula space, node tag, and edge type if they don't already exist.
    """
    session.execute(
        f"CREATE SPACE IF NOT EXISTS {space} (partition_num=1, replica_factor=1);"
    )
    session.execute(f"USE {space};")
    session.execute("CREATE TAG IF NOT EXISTS Entity(name string);")
    session.execute("CREATE EDGE IF NOT EXISTS Rel(verb string);")
    time.sleep(1)

def load_triplets_to_nebula(session, triplets, space=NEBULA_SPACE):
    """
    Inserts SVO triplets as nodes and edges into Nebula Graph.
    triplets: list of (subject, verb, object)
    """
    session.execute(f"USE {space};")
    entities = set()
    for subj, verb, obj in triplets:
        if subj not in entities:
            session.execute(f'INSERT VERTEX Entity(name) VALUES "{subj}":("{subj}");')
            entities.add(subj)
        if obj not in entities:
            session.execute(f'INSERT VERTEX Entity(name) VALUES "{obj}":("{obj}");')
            entities.add(obj)
        session.execute(f'INSERT EDGE Rel(verb) VALUES "{subj}"->"{obj}":("{verb}");')

def fetch_graph(session, space=NEBULA_SPACE, limit=100):
    """
    Queries up to `limit` triples from Nebula Graph and returns as nodes/edges.
    Returns (nodes, edges) for frontend visualization.
    """
    session.execute(f"USE {space};")
    q = f"MATCH (v)-[e]->(n) RETURN v, e, n LIMIT {limit};"
    result = session.execute(q)
    nodes = set()
    edges = []
    for row in result.rows():
        v = row[0].as_node()
        n = row[2].as_node()
        e = row[1].as_relationship()
        subj = v.properties.get("name", str(v.get_id()))
        obj = n.properties.get("name", str(n.get_id()))
        verb = e.properties.get("verb", "")
        nodes.add(subj)
        nodes.add(obj)
        edges.append({"source": subj, "target": obj, "verb": verb})
    return list(nodes), edges

@router.post("/nebula/upload_triplets/")
async def nebula_upload_triplets(
    file: UploadFile = File(...),
    space: str = Form(default=NEBULA_SPACE)
):
    """
    Upload a CSV file of SVO triplets (subject, verb, object) and load into Nebula Graph.
    Returns the nodes and edges for frontend visualization.

    - CSV columns required: "subject", "verb", "object"
    - Returns: {"nodes": [...], "edges": [...]} for graph rendering.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    try:
        ensure_nebula_running()
        session = connect_nebula()
        create_space_and_schema(session, space)
        content = await file.read()
        f = io.StringIO(content.decode("utf-8"))
        reader = csv.DictReader(f)
        triplets = []
        for row in reader:
            subj = row.get("subject", "").strip()
            verb = row.get("verb", "").strip()
            obj = row.get("object", "").strip()
            if subj and verb and obj:
                triplets.append((subj, verb, obj))
        if not triplets:
            raise HTTPException(status_code=400, detail="No valid SVO triplets found in CSV.")
        load_triplets_to_nebula(session, triplets, space)
        nodes, edges = fetch_graph(session, space)
        session.release()
        return JSONResponse({
            "nodes": nodes,
            "edges": edges,
            "message": f"Graph loaded. Visualize using returned graph data."
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Nebula ingestion failed: {str(e)}")