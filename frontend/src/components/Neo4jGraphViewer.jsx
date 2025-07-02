import React, { useState } from "react";
import ForceGraph2D from "react-force-graph-2d";

export default function Neo4jGraphViewer() {
  const [graph, setGraph] = useState({ nodes: [], links: [] });
  const [loading, setLoading] = useState(false);

  async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/api/neo4j/upload_triplets/", {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      alert("Failed to upload or parse file.");
      setLoading(false);
      return;
    }
    const data = await res.json();
    setGraph({
      nodes: data.nodes,
      links: data.edges.map((e) => ({
        source: e.source,
        target: e.target,
        label: e.verb,
      })),
    });
    setLoading(false);
  }

  return (
    <div>
      <h2>Upload SVO Triplets CSV to Visualize Neo4j Graph</h2>
      <input type="file" accept=".csv" onChange={handleUpload} disabled={loading} />
      {loading && <div>Loading...</div>}
      {graph.nodes.length > 0 && (
        <ForceGraph2D
          graphData={graph}
          nodeAutoColorBy="id"
          nodeLabel="id"
          linkDirectionalArrowLength={8}
          linkDirectionalArrowRelPos={1}
          linkLabel={(link) => link.label}
          linkColor={() => "rgba(100,100,255,0.7)"}
          width={800}
          height={600}
        />
      )}
    </div>
  );
}