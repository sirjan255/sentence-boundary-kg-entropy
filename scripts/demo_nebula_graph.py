"""
Demo: Generate a Nebula Graph from sample sentences, insert SVO triplets, and instructions to visualize in Nebula Studio.
No Docker required—assumes Nebula Graph and Nebula Studio are already running natively.

REQUIREMENTS:
    - nebula3-python (pip install nebula3-python)
    - Nebula Graph DB running on localhost:9669 (native/WSL/Linux install)
    - Nebula Studio running on localhost:7000 or 7001

USAGE:
    1. Start Nebula Graph (graphd/metad/storaged) and Studio manually beforehand.
    2. Run: python scripts/demo_nebula_graph.py
    3. Open http://localhost:7000 (or :7001) in your browser, login root/nebula, choose space 'sentence_kg'

NOTE:
    - This script shows how to load a small demo graph and interact with it.
    - See Nebula docs for native installation: https://docs.nebula-graph.io/3.6.0/
"""

import os
import time
import csv
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

SAMPLE_TEXT = """
John loves apples. Mary eats oranges. The dog chased the cat. The sun rises in the east.
"""

SAMPLE_TRIPLETS = [
    ("John", "loves", "apples"),
    ("Mary", "eats", "oranges"),
    ("dog", "chased", "cat"),
    ("sun", "rises", "east"),
]
TRIPLETS_CSV = "outputs/sample_triplets.csv"
NEBULA_USER = "root"
NEBULA_PWD = "nebula"
NEBULA_HOST = "127.0.0.1"
NEBULA_PORT = 9669
NEBULA_SPACE = "sentence_kg"


def write_sample_triplets():
    os.makedirs("outputs", exist_ok=True)
    with open(TRIPLETS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "verb", "object"])
        for row in SAMPLE_TRIPLETS:
            writer.writerow(row)
    print(f"Sample triplets written to {TRIPLETS_CSV}")


def ensure_nebula_running():
    import socket
    s = socket.socket()
    try:
        s.connect((NEBULA_HOST, NEBULA_PORT))
        print(f"Nebula Graph is running on {NEBULA_HOST}:{NEBULA_PORT}")
    except Exception:
        print("ERROR: Nebula Graph is not running on 127.0.0.1:9669.")
        print("Start Nebula Graph (graphd/metad/storaged) manually before running this script.")
        exit(1)
    s.close()


def connect_nebula():
    config = Config()
    config.max_connection_pool_size = 10
    pool = ConnectionPool()
    ok = pool.init([(NEBULA_HOST, NEBULA_PORT)], config)
    if not ok:
        raise RuntimeError("Failed to connect to Nebula Graph.")
    session = pool.get_session(NEBULA_USER, NEBULA_PWD)
    return session


def create_space_and_schema(session):
    print("Creating Nebula space and schema...")
    session.execute(f"CREATE SPACE IF NOT EXISTS {NEBULA_SPACE} (partition_num=1, replica_factor=1);")
    session.execute(f"USE {NEBULA_SPACE};")
    session.execute("CREATE TAG IF NOT EXISTS Entity(name string);")
    session.execute("CREATE EDGE IF NOT EXISTS Rel(verb string);")
    # Sleep for schema to propagate
    time.sleep(3)


def load_triplets_to_nebula(session, triplet_csv):
    print("Loading triplets into Nebula Graph...")
    session.execute(f"USE {NEBULA_SPACE};")
    entities = set()
    with open(triplet_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subj = row["subject"].strip()
            obj = row["object"].strip()
            verb = row["verb"].strip()
            # Insert nodes if not already present
            if subj not in entities:
                session.execute(f'INSERT VERTEX Entity(name) VALUES "{subj}":("{subj}");')
                entities.add(subj)
            if obj not in entities:
                session.execute(f'INSERT VERTEX Entity(name) VALUES "{obj}":("{obj}");')
                entities.add(obj)
            # Insert edge
            session.execute(f'INSERT EDGE Rel(verb) VALUES "{subj}"->"{obj}":("{verb}");')


def main():
    print("Demo: Nebula Graph Generation & Visualization\n")
    print("Sample Text:\n" + SAMPLE_TEXT)
    write_sample_triplets()
    ensure_nebula_running()
    session = connect_nebula()
    create_space_and_schema(session)
    load_triplets_to_nebula(session, TRIPLETS_CSV)
    print("\nSuccess! Your demo graph is loaded.")
    print("Open Nebula Studio at: http://localhost:7000 (or http://localhost:7001)")
    print("Login as: root / nebula")
    print(f"Switch to space: {NEBULA_SPACE}")
    print("\nTry these in the query window to see your graph:")
    print("  SHOW TAGS;")
    print("  SHOW EDGES;")
    print("  MATCH (v)-[e]->(n) RETURN v,e,n LIMIT 10;")
    print("\nIf you don't see Studio on :7000, try :7001 or check your Studio install.")
    session.release()


if __name__ == "__main__":
    main()