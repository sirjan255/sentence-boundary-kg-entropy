"use client";

import React, { useState } from "react";
import {
  Upload,
  Button,
  Card as AntCard,
  Alert,
  Spin,
  Typography,
  Input,
  Space,
  message,
  Divider,
  Tooltip,
  Modal,
  Tag,
} from "antd";
import {
  UploadOutlined,
  FileExcelOutlined,
  CheckCircleTwoTone,
  EyeOutlined,
  InfoCircleOutlined,
  ProjectOutlined,
  PlayCircleOutlined,
  CloudServerOutlined,
  BulbOutlined,
} from "@ant-design/icons";
import axios from "axios";

// Dynamic import for react-force-graph (npm install react-force-graph)
import dynamic from "next/dynamic";
const ForceGraph2D = dynamic(
  () => import("react-force-graph").then((mod) => mod.ForceGraph2D),
  { ssr: false }
);

const BACKEND = process.env.REACT_APP_BACKEND || "/api";
const { Title, Paragraph, Text } = Typography;

type GraphNode = { id: string };
type GraphEdge = { source: string; target: string; verb?: string };

// Example Cypher queries for user to click and view
const EXAMPLE_QUERIES = [
  {
    label: "Show all subject-verb-object relationships",
    query:
      "MATCH (a:Entity)-[r:REL]->(b:Entity) RETURN a.name AS source, r.verb AS verb, b.name AS target LIMIT 100",
  },
  {
    label: "Show all nodes with more than one outgoing edge",
    query:
      "MATCH (a:Entity)-[r:REL]->(b:Entity) WITH a, COUNT(r) AS outdeg WHERE outdeg > 1 MATCH (a)-[r2:REL]->(b2:Entity) RETURN a.name AS source, r2.verb AS verb, b2.name AS target LIMIT 100",
  },
  {
    label: "Show all incoming edges for a node",
    query:
      "MATCH (a:Entity)-[r:REL]->(b:Entity) WHERE b.name = 'YOUR_OBJECT' RETURN a.name AS source, r.verb AS verb, b.name AS target LIMIT 100",
  },
  {
    label: "Show all nodes",
    query:
      "MATCH (n:Entity) RETURN n.name AS source, '' AS verb, '' AS target LIMIT 100",
  },
];

export function Neo4jGraphComponent() {
  // File and cypher query state
  const [tripletsFile, setTripletsFile] = useState<File | null>(null);
  const [cypherQuery, setCypherQuery] = useState<string>(EXAMPLE_QUERIES[0].query);
  // Graph state
  const [graphData, setGraphData] = useState<{ nodes: GraphNode[]; edges: GraphEdge[] } | null>(null);
  // UI state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Upload CSV and visualize
  const handleFileUpload = (file: File) => {
    setTripletsFile(file);
    setError(null);
    setGraphData(null);
    return false;
  };

  const handleUploadAndVisualize = async () => {
    setError(null);
    setGraphData(null);
    if (!tripletsFile) {
      setError("Please upload a CSV file of SVO triplets.");
      return;
    }
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", tripletsFile);
      // POST to backend
      const resp = await axios.post(`${BACKEND}/neo4j/upload_triplets/`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      if (resp.data && resp.data.nodes && resp.data.edges) {
        setGraphData({
          nodes: resp.data.nodes,
          edges: resp.data.edges,
        });
        message.success("CSV uploaded and graph loaded from Neo4j!");
      } else {
        throw new Error("Backend did not return graph data.");
      }
    } catch (err: any) {
      setError(
        String(
          err?.response?.data?.detail ||
            err?.message ||
            "Neo4j ingestion or visualization failed. Check backend and CSV format."
        )
      );
    }
    setLoading(false);
  };

  // Run arbitrary Cypher query and visualize
  const handleRunQuery = async () => {
    setError(null);
    setGraphData(null);
    setLoading(true);
    try {
      const resp = await axios.post(
        `${BACKEND}/neo4j/query/`,
        { query: cypherQuery },
        { headers: { "Content-Type": "application/json" } }
      );
      if (resp.data && resp.data.nodes && resp.data.edges) {
        setGraphData({
          nodes: resp.data.nodes,
          edges: resp.data.edges,
        });
        message.success("Cypher query executed!");
      } else {
        throw new Error("Backend did not return graph data.");
      }
    } catch (err: any) {
      setError(
        String(
          err?.response?.data?.detail ||
            err?.message ||
            "Cypher query failed. Please check backend and query syntax."
        )
      );
    }
    setLoading(false);
  };

  // Insert example query into the input
  const handleExampleClick = (query: string) => {
    setCypherQuery(query);
    setGraphData(null);
    message.info("Example query loaded! Click 'Run Query and Visualize' to see results.");
  };

  // UI rendering
  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: 24 }}>
      <Title level={2}>
        <CloudServerOutlined style={{ color: "#1890ff", marginRight: 12 }} />
        Neo4j SVO Graph Visualizer
      </Title>
      <Paragraph>
        Upload an SVO triplet CSV, or run your own Cypher query, and see your knowledge graph instantly — <b>no Neo4j Browser or Studio required</b>!
      </Paragraph>
      <Divider />
      <AntCard bordered style={{ marginBottom: 32 }}>
        {/* CSV Upload */}
        <Upload
          accept=".csv"
          beforeUpload={handleFileUpload}
          showUploadList={tripletsFile ? { showRemoveIcon: true } : false}
          maxCount={1}
          onRemove={() => setTripletsFile(null)}
          fileList={tripletsFile ? [{ uid: "-1", name: tripletsFile.name }] : []}
        >
          <Space>
            <Button icon={<UploadOutlined />}>Upload Triplets CSV</Button>
            <Tooltip title="CSV columns: subject, verb, object">
              <InfoCircleOutlined />
            </Tooltip>
          </Space>
        </Upload>
        <div style={{ marginTop: 12 }}>
          <Button
            type="primary"
            icon={<CheckCircleTwoTone twoToneColor="#52c41a" />}
            onClick={handleUploadAndVisualize}
            loading={loading}
            disabled={loading || !tripletsFile}
            style={{ width: 220, fontWeight: "bold" }}
          >
            Upload and Visualize
          </Button>
        </div>
      </AntCard>
      <AntCard bordered style={{ marginBottom: 32 }}>
        <Paragraph>
          <b>Run Custom or Example Cypher Query:</b>
        </Paragraph>
        <div style={{ marginBottom: 10 }}>
          <Space>
            <BulbOutlined style={{ color: "#faad14" }} />
            {EXAMPLE_QUERIES.map((q, idx) => (
              <Button
                key={idx}
                size="small"
                type="dashed"
                onClick={() => handleExampleClick(q.query)}
                style={{ marginRight: 4 }}
              >
                {q.label}
              </Button>
            ))}
          </Space>
        </div>
        <Input.TextArea
          value={cypherQuery}
          onChange={e => setCypherQuery(e.target.value)}
          autoSize={{ minRows: 3, maxRows: 6 }}
          style={{ fontFamily: "monospace", marginBottom: 12 }}
          placeholder="Cypher query (e.g. MATCH (a:Entity)-[r:REL]->(b:Entity) RETURN ...)"
        />
        <Button
          icon={<PlayCircleOutlined />}
          type="default"
          style={{ marginBottom: 0, width: 200 }}
          loading={loading}
          disabled={loading || !cypherQuery}
          onClick={handleRunQuery}
        >
          Run Query and Visualize
        </Button>
      </AntCard>
      {/* Error alert if any */}
      {error && (
        <Alert
          type="error"
          showIcon
          message={String(error)}
          style={{ marginBottom: 24 }}
        />
      )}
      {/* Graph visualization */}
      {graphData && (
        <AntCard
          bordered
          title={
            <span>
              <EyeOutlined style={{ color: "#52c41a" }} /> Graph Visualization
            </span>
          }
          style={{ marginBottom: 32, background: "#f8fff8" }}
        >
          <div style={{ width: "100%", height: 600, background: "#fff", border: "1px solid #eee", borderRadius: 10 }}>
            <ForceGraph2D
              graphData={{
                nodes: graphData.nodes,
                links: graphData.edges,
              }}
              nodeLabel="id"
              linkLabel="verb"
              linkDirectionalArrowLength={8}
              linkDirectionalArrowRelPos={1}
              linkCurvature={0.1}
              width={1000}
              height={600}
              nodeAutoColorBy="id"
              linkColor={() => "#888"}
              linkWidth={2}
              linkDirectionalParticles={2}
              linkDirectionalParticleWidth={2}
              linkDirectionalParticleColor={() => "#faad14"}
            />
          </div>
        </AntCard>
      )}
      {/* How to use */}
      <AntCard type="inner" title="How to use this tool" style={{ marginTop: 16 }}>
        <ul>
          <li>
            <b>Prepare your CSV:</b> Columns: <code>subject</code>, <code>verb</code>, <code>object</code>.
          </li>
          <li>
            <b>Upload</b> your CSV file and click "Upload and Visualize" to push the data to Neo4j and see the graph.
          </li>
          <li>
            <b>Or run a custom Cypher query</b> (read-only) to visualize any part of your graph, or click an example above.
          </li>
          <li>
            <b>No local Neo4j Browser or Studio required:</b> All visualization happens here!
          </li>
        </ul>
        <Divider />
        <Paragraph>
          <Text type="secondary">
            <InfoCircleOutlined /> All computation is in-memory and secure. No files are saved on server. Frontend never connects directly to Neo4j — backend does all DB access.
          </Text>
        </Paragraph>
      </AntCard>
    </div>
  );
}

/*
1. User uploads a CSV of SVO triplets (columns: subject, verb, object).
2. Or run an arbitrary Cypher query to visualize any subgraph. Click an example to auto-fill the input!
3. All visualization is interactive and in-browser. No Neo4j Desktop/Browser required.
4. Requires: npm install react-force-graph @ant-design/icons antd axios
*/