"use client";

import React, { useState } from "react";
import {
  Upload,
  Button,
  Card as AntCard,
  Alert,
  Spin,
  Typography,
  Table,
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
  SyncOutlined,
} from "@ant-design/icons";
import axios from "axios";

// Backend API base URL
const BACKEND = process.env.REACT_APP_BACKEND || "/api";
const { Title, Paragraph, Text } = Typography;

// Only allow CSV files for triplets input
const ACCEPTED_FILE_TYPES = [".csv"];

// d3-force-graph is a good zero-config choice
import dynamic from "next/dynamic";
const ForceGraph2D = dynamic(() => import("react-force-graph").then(mod => mod.ForceGraph2D), {
  ssr: false,
});

// Graph node/edge types for TypeScript
type GraphNode = { id: string };
type GraphEdge = { source: string; target: string; verb?: string };

export function NebulaGraphUploadAndVisualizeComponent() {
  // File state and Nebula space
  const [tripletsFile, setTripletsFile] = useState<File | null>(null);
  const [space, setSpace] = useState<string>("sentence_kg");

  // Graph data
  const [graphData, setGraphData] = useState<{ nodes: GraphNode[]; edges: GraphEdge[] } | null>(null);

  // UI state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showPreviewModal, setShowPreviewModal] = useState(false);

  // For modal edge preview
  const [modalEdgeIdx, setModalEdgeIdx] = useState<number | null>(null);

  // Handler for file upload
  const handleFileUpload = (file: File) => {
    setTripletsFile(file);
    setGraphData(null);
    setError(null);
    return false;
  };

  // Handler for submit: POST to backend, receive graph JSON
  const handleSubmit = async () => {
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
      formData.append("space", space);

      // POST to backend /nebula/upload_triplets/
      const resp = await axios.post(`${BACKEND}/nebula/upload_triplets/`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (resp.data && resp.data.nodes && resp.data.edges) {
        // Adapt nodes to {id: ...} for graph library
        setGraphData({
          nodes: resp.data.nodes.map((n: string) => ({ id: n })),
          edges: resp.data.edges,
        });
        message.success("Graph uploaded and loaded from Nebula!");
      } else {
        throw new Error("Backend did not return graph data.");
      }
    } catch (err: any) {
      setError(
        String(
          err?.response?.data?.detail ||
          err?.message ||
          "Ingestion or visualization failed. Please check Nebula server and CSV formatting."
        )
      );
    }
    setLoading(false);
  };

  // Helper: Create a table for nodes
  const nodeColumns = [
    { title: "Node", dataIndex: "id", key: "id" },
  ];
  // Helper: Create a table for edges
  const edgeColumns = [
    {
      title: "Source",
      dataIndex: "source",
      key: "source",
      render: (v: string) => <Tag color="blue">{v}</Tag>,
    },
    {
      title: "Verb",
      dataIndex: "verb",
      key: "verb",
      render: (v: string) => <Tag color="purple">{v}</Tag>,
    },
    {
      title: "Target",
      dataIndex: "target",
      key: "target",
      render: (v: string) => <Tag color="green">{v}</Tag>,
    },
  ];

  // ---- UI Rendering ----
  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: 24 }}>
      <Title level={2}>
        <ProjectOutlined style={{ color: "#1890ff", marginRight: 12 }} />
        Nebula Graph: SVO Triplet Ingestion & Visualization
      </Title>
      <Paragraph>
        Upload a CSV file of SVO triplets (subject, verb, object) and instantly see your knowledge graph rendered, powered by a live Nebula Graph database.<br />
        <Text type="secondary">
          (No need for Nebula Studio: all ingestion and visualization is handled here!)
        </Text>
      </Paragraph>
      <Divider />
      <AntCard bordered style={{ marginBottom: 32 }}>
        <Upload
          accept={ACCEPTED_FILE_TYPES.join(",")}
          beforeUpload={handleFileUpload}
          showUploadList={tripletsFile ? { showRemoveIcon: true } : false}
          maxCount={1}
          onRemove={() => setTripletsFile(null)}
          fileList={tripletsFile ? [{ uid: "-1", name: tripletsFile.name }] : []}
        >
          <Space>
            <Button icon={<UploadOutlined />}>Upload CSV (SVO Triplets)</Button>
            <Tooltip title="CSV columns: subject, verb, object">
              <InfoCircleOutlined />
            </Tooltip>
          </Space>
        </Upload>
        <div style={{ marginTop: 16 }}>
          <Button
            type="primary"
            icon={<CheckCircleTwoTone twoToneColor="#52c41a" />}
            onClick={handleSubmit}
            loading={loading}
            disabled={loading || !tripletsFile}
            style={{ width: 200, fontWeight: "bold" }}
          >
            Ingest and Visualize
          </Button>
          {loading && <Spin style={{ marginLeft: 18 }} />}
        </div>
        {/* Error alert if any */}
        {error && (
          <Alert
            type="error"
            showIcon
            message={String(error)}
            style={{ marginTop: 24 }}
          />
        )}
      </AntCard>
      {/* Visualization */}
      {graphData && (
        <AntCard
          bordered
          title={
            <span>
              <EyeOutlined style={{ color: "#52c41a" }} /> Graph Visualization
            </span>
          }
          style={{ marginBottom: 32, background: "#f8fff8" }}
          extra={
            <Button
              icon={<SyncOutlined />}
              onClick={() => setShowPreviewModal(true)}
              type="default"
              size="small"
            >
              Preview Nodes/Edges
            </Button>
          }
        >
          <div style={{ width: "100%", height: 600, background: "#fff", border: "1px solid #eee", borderRadius: 10 }}>
            {/* Interactive Graph */}
            <ForceGraph2D
              graphData={{
                nodes: graphData.nodes,
                links: graphData.edges.map((e) => ({
                  source: e.source,
                  target: e.target,
                  verb: e.verb,
                })),
              }}
              nodeLabel="id"
              linkLabel="verb"
              linkDirectionalArrowLength={6}
              linkDirectionalArrowRelPos={1}
              linkCurvature={0.1}
              width={1000}
              height={600}
              nodeAutoColorBy="id"
              linkColor={() => "#888"}
              linkWidth={2}
              linkDirectionalParticles={2}
              linkDirectionalParticleWidth={2}
              linkDirectionalParticleColor={() => "#f5222d"}
              onLinkClick={(_: any, link: any) => {
                const idx = graphData.edges.findIndex(
                  (e) =>
                    e.source === link.source && e.target === link.target && e.verb === link.verb
                );
                if (idx >= 0) setModalEdgeIdx(idx);
              }}
            />
          </div>
        </AntCard>
      )}
      {/* Modal for node/edge preview */}
      <Modal
        open={showPreviewModal}
        onCancel={() => setShowPreviewModal(false)}
        footer={null}
        title="Graph Nodes and Edges"
        width={800}
      >
        {graphData && (
          <>
            <Paragraph>
              <b>Nodes:</b> {graphData.nodes.length}, <b>Edges:</b> {graphData.edges.length}
            </Paragraph>
            <Divider>Nodes</Divider>
            <Table
              dataSource={graphData.nodes.map((node, idx) => ({ ...node, key: idx }))}
              columns={nodeColumns}
              size="small"
              pagination={false}
              scroll={{ y: 120 }}
              bordered
            />
            <Divider>Edges</Divider>
            <Table
              dataSource={graphData.edges.map((e, idx) => ({ ...e, key: idx }))}
              columns={edgeColumns}
              size="small"
              pagination={false}
              scroll={{ y: 180 }}
              bordered
            />
          </>
        )}
      </Modal>
      {/* Modal for edge details (on link click) */}
      <Modal
        open={modalEdgeIdx !== null}
        onCancel={() => setModalEdgeIdx(null)}
        footer={null}
        title="Edge Details"
      >
        {modalEdgeIdx !== null && graphData && (
          <>
            <Paragraph>
              <Tag color="blue">{graphData.edges[modalEdgeIdx].source}</Tag>
              <Tag color="purple">{graphData.edges[modalEdgeIdx].verb}</Tag>
              <Tag color="green">{graphData.edges[modalEdgeIdx].target}</Tag>
            </Paragraph>
          </>
        )}
      </Modal>
      {/* How to use */}
      <AntCard type="inner" title="How to use this tool" style={{ marginTop: 16 }}>
        <ul>
          <li>
            <b>Prepare your CSV:</b> Columns: <code>subject</code>, <code>verb</code>, <code>object</code>.
          </li>
          <li>
            <b>Upload</b> your CSV file using the button above.
          </li>
          <li>
            <b>Ingest and Visualize</b> to upload to Nebula Graph and instantly view your knowledge graph here.
          </li>
          <li>
            <b>Preview</b> all nodes and edges, or click any edge in the graph for details.
          </li>
        </ul>
        <Divider />
        <Paragraph>
          <Text type="secondary">
            <InfoCircleOutlined /> All computation is in-memory and secure. No files are saved on server.
          </Text>
        </Paragraph>
        <Divider />
        <Paragraph>
          <b>Troubleshooting:</b> If you see a connection error, ensure Nebula Graph is running and API server can connect.<br />
          No need to open Nebula Studio -- everything is handled here!
        </Paragraph>
      </AntCard>
    </div>
  );
}

/*
1. User uploads a CSV of SVO triplets (columns: subject, verb, object).
2. "Ingest and Visualize" will POST to backend, store in Nebula, and instantly show a graph here.
3. Interactive force-directed visualization and table previews included.
4. Requires: npm install react-force-graph @ant-design/icons antd axios
*/