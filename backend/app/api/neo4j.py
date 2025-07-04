"""
FastAPI Neo4j API:
- Accepts SVO triplet CSV uploads from the frontend
- Inserts into Neo4j using Cypher
- Returns a JSON of nodes/edges for direct frontend visualization
"""

import os
import csv
import io
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")  # Set your password!

router = APIRouter()

def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

@router.post("/neo4j/upload_triplets/")
async def neo4j_upload_triplets(file: UploadFile = File(...)):
    """
    Accepts a CSV of triplets, loads into Neo4j, and returns nodes/edges as JSON for direct frontend visualization.
    """
    if file.filename and not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    try:
        # Parse triplets from uploaded CSV
        content = await file.read()
        f = io.StringIO(content.decode("utf-8"))
        reader = csv.DictReader(f)
        triplets = [
            (row["subject"].strip(), row["verb"].strip(), row["object"].strip())
            for row in reader if row.get("subject") and row.get("verb") and row.get("object")
        ]
        if not triplets:
            raise HTTPException(status_code=400, detail="No valid SVO triplets found in CSV.")

        driver = get_driver()
        with driver.session() as session:
            # Insert data
            for subj, verb, obj in triplets:
                session.run(
                    """
                    MERGE (a:Entity {name: $s})
                    MERGE (b:Entity {name: $o})
                    MERGE (a)-[r:REL {verb: $v}]->(b)
                    """, s=subj, v=verb, o=obj
                )

            # Query graph for visualization (limit for performance)
            result = session.run(
                """
                MATCH (a:Entity)-[r:REL]->(b:Entity)
                RETURN a.name AS source, r.verb AS verb, b.name AS target
                LIMIT 100
                """
            )
            edges = []
            nodes_set = set()
            for record in result:
                source = record["source"]
                target = record["target"]
                verb = record["verb"]
                edges.append({"source": source, "target": target, "verb": verb})
                nodes_set.update([source, target])
            nodes = [{"id": n} for n in nodes_set]
        driver.close()
        return JSONResponse({"nodes": nodes, "edges": edges})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Neo4j ingestion failed: {str(e)}")

from fastapi import Body

@router.post("/neo4j/query/")
async def neo4j_query(query: dict = Body(...)):
    """
    Accepts {"query": "<cypher>"} and returns {nodes, edges} for frontend visualization.
    Only allows read (MATCH ... RETURN ...) queries!
    """
    cypher = query.get("query")
    if not cypher:
        raise HTTPException(status_code=400, detail="No query provided.")

    try:
        driver = get_driver()
        with driver.session() as session:
            result = session.run(cypher)
            edges = []
            nodes_set = set()
            # Try to extract source, target, verb from result
            for record in result:
                # The record is a dict-like object
                source = record.get("source")
                target = record.get("target")
                verb = record.get("verb")
                # Only add edges if both source and target present
                if source and target:
                    edges.append({"source": source, "target": target, "verb": verb})
                    nodes_set.update([source, target])
                # If only a node is returned (e.g., single column query)
                elif source and not target:
                    nodes_set.add(source)
                elif target and not source:
                    nodes_set.add(target)
            nodes = [{"id": n} for n in nodes_set]
        driver.close()
        return JSONResponse({"nodes": nodes, "edges": edges})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cypher query failed: {str(e)}")