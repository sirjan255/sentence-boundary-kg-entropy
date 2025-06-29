import os
from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
user = os.getenv("NEO4J_USER", "neo4j")
password = os.getenv("NEO4J_PASSWORD")

if password is None:
    raise RuntimeError("Set the NEO4J_PASSWORD environment variable before running this script.")

driver = GraphDatabase.driver(uri, auth=(user, password))

def create_graph(tx, s, v, o):
    tx.run(
        "MERGE (a:Entity {name: $s}) "
        "MERGE (b:Entity {name: $o}) "
        "MERGE (a)-[:REL {verb: $v}]->(b)",
        s=s, v=v, o=o
    )

with driver.session() as session:
    for s, v, o in SAMPLE_TRIPLETS:
        session.write_transaction(create_graph, s, v, o)
print("Graph loaded. Open Neo4j Desktop and explore!")

driver.close()