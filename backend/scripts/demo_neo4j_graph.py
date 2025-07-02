import os
from neo4j import GraphDatabase

# Configuration
uri = "bolt://localhost:7687"
user = os.getenv("NEO4J_USER", "neo4j")
password = os.getenv("NEO4J_PASSWORD")

if password is None:
    raise RuntimeError("Set the NEO4J_PASSWORD environment variable before running this script.")

# Sample triplets (Subject-Verb-Object)
SAMPLE_TRIPLETS = [
    ("John", "loves", "apples"),
    ("Mary", "eats", "oranges"),
    ("dog", "chased", "cat"),
    ("sun", "rises", "east"),
    ("Alice", "works_at", "Google"),
    ("Bob", "knows", "Alice"),
    ("London", "capital_of", "UK")
]

def create_graph(tx, s, v, o):
    """Create nodes and relationship in Neo4j"""
    tx.run(
        "MERGE (a:Entity {name: $s}) "
        "MERGE (b:Entity {name: $o}) "
        "MERGE (a)-[:REL {verb: $v}]->(b)",
        s=s, v=v, o=o
    )

def main():
    # Initialize driver
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
        with driver.session() as session:
            # Load sample triplets
            for s, v, o in SAMPLE_TRIPLETS:
                session.write_transaction(create_graph, s, v, o)
                
            # Verify creation
            result = session.read_transaction(
                lambda tx: tx.run("MATCH (n) RETURN count(n) AS node_count").single()
            )
            print(f"\nCreated graph with {result['node_count']} nodes")
            
            # Print sample query results
            print("\nSample relationships created:")
            result = session.read_transaction(
                lambda tx: tx.run(
                    "MATCH (a)-[r:REL]->(b) "
                    "RETURN a.name, r.verb, b.name "
                    "LIMIT 5"
                ).values()
            )
            for row in result:
                print(f"{row[0]} → {row[1]} → {row[2]}")
                
    finally:
        driver.close()
    
    print("\nGraph loaded successfully!")
    print("Open Neo4j Browser at http://localhost:7474 to explore your graph.")
    print("Try these queries:")
    print("  MATCH (n) RETURN n LIMIT 25")
    print("  MATCH (a)-[r]->(b) RETURN a,r,b LIMIT 10")

if __name__ == "__main__":
    main()