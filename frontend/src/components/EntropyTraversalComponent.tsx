"use client";

import React, { useState } from "react";
import {
  Upload,
  Button,
  InputNumber,
  Select,
  Form,
  message,
  Spin,
  Card as AntCard,
  Table,
  Alert,
  Typography,
  Collapse,
} from "antd";
import { UploadOutlined, PlayCircleOutlined } from "@ant-design/icons";
import axios from "axios";

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL || "/api";
const { Title, Paragraph } = Typography;
const { Panel } = Collapse;

const ENTROPY_METHODS = [
  { label: "BLT", value: "blt" },
  { label: "Sum", value: "sum" },
  { label: "Mean", value: "mean" },
];

// Main frontend React component for entropy-based traversal
export function EntropyTraversalComponent() {
  // --- File upload state ---
  const [kgFile, setKgFile] = useState<File | null>(null);
  const [embeddingsFile, setEmbeddingsFile] = useState<File | null>(null);
  const [nodesFile, setNodesFile] = useState<File | null>(null);
  const [startNodesFile, setStartNodesFile] = useState<File | null>(null);

  // --- Parameter state ---
  const [entropyThreshold, setEntropyThreshold] = useState<number>(0.8);
  const [maxDepth, setMaxDepth] = useState<number>(10);
  const [entropyMethod, setEntropyMethod] = useState<string>("blt");
  const [temperature, setTemperature] = useState<number>(1.0);

  // --- UI / API state ---
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  // Helper for file upload (Ant Design Upload: beforeUpload)
  const handleFileUpload = (setter: (f: File) => void) => (file: File) => {
    setter(file);
    return false;
  };

  // UI/UX: InputNumber requires onChange to accept number|null. Ignore null safely.
  const handleEntropyThresholdChange = (value: number | null) => {
    if (value !== null) setEntropyThreshold(value);
  };
  const handleMaxDepthChange = (value: number | null) => {
    if (value !== null) setMaxDepth(value);
  };
  const handleTemperatureChange = (value: number | null) => {
    if (value !== null) setTemperature(value);
  };

  // SUBMIT HANDLER: Called when form is submitted
  const handleSubmit = async () => {
    setError(null);
    setResults(null);

    if (!kgFile || !embeddingsFile || !nodesFile || !startNodesFile) {
      setError(
        "Please upload all required files: KG, embeddings, nodes, and start nodes."
      );
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("kg", kgFile);
      formData.append("embeddings", embeddingsFile);
      formData.append("nodes", nodesFile);
      formData.append("start_nodes", startNodesFile);
      formData.append("entropy_threshold", entropyThreshold.toString());
      formData.append("max_depth", maxDepth.toString());
      formData.append("entropy_method", entropyMethod);
      formData.append("temperature", temperature.toString());

      const resp = await axios.post(`${BACKEND}/entropy_traversal/`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setResults(resp.data);
      message.success("Traversal complete!");
    } catch (err: any) {
      setError(
        String(
          err?.response?.data?.detail ||
            err?.message ||
            "Entropy traversal failed: check your files and parameters."
        )
      );
    }
    setLoading(false);
  };

  // --- Type guards for result objects ---
  // Check if the value is a valid result object
  function isNodeResultObject(
    data: any
  ): data is {
    nodes_in_sentence: string[];
    entropies: Record<string, number>;
  } {
    return (
      data &&
      typeof data === "object" &&
      Array.isArray((data as any).nodes_in_sentence) &&
      typeof (data as any).entropies === "object" &&
      (data as any).entropies !== null
    );
  }

  // Check if the value is an error object
  function isNodeErrorObject(data: any): data is { error: string } {
    return data && typeof data === "object" && typeof data.error === "string";
  }

  // --- UI: Render results for each start node as collapsible panels with tables ---
  const renderResultPanels = (resultObj: any) => {
    if (!resultObj) return null;

    return (
      <Collapse accordion>
        {Object.entries(resultObj).map(([startNode, data]) => (
          <Panel
            header={
              isNodeErrorObject(data)
                ? `Start Node: ${startNode} (Error)`
                : `Start Node: ${startNode}`
            }
            key={String(startNode)}
            style={
              isNodeErrorObject(data)
                ? { background: "#fff5f5", border: "1px solid #ffa39e" }
                : {}
            }
          >
            {/* Error display */}
            {isNodeErrorObject(data) ? (
              <Alert type="error" message={String(data.error)} />
            ) : isNodeResultObject(data) ? (
              <>
                {/* List of nodes in sentence */}
                <Paragraph>
                  <b>Nodes in Sentence:</b>{" "}
                  {data.nodes_in_sentence.length > 0 ? (
                    data.nodes_in_sentence.join(", ")
                  ) : (
                    <i>None found</i>
                  )}
                </Paragraph>
                {/* Table of entropies */}
                <Table
                  dataSource={
                    data.entropies
                      ? Object.entries(data.entropies).map(
                          ([node, entropy], idx) => ({
                            key: idx,
                            node,
                            entropy,
                          })
                        )
                      : []
                  }
                  columns={[
                    { title: "Node", dataIndex: "node", key: "node" },
                    {
                      title: "Entropy",
                      dataIndex: "entropy",
                      key: "entropy",
                      render: (v: number) => v.toFixed(4),
                    },
                  ]}
                  pagination={false}
                  size="small"
                  style={{ marginTop: 10 }}
                />
              </>
            ) : (
              <Alert
                type="warning"
                message="No valid result for this start node."
              />
            )}
          </Panel>
        ))}
      </Collapse>
    );
  };

  // ------------- Main UI rendering starts here -------------
  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: 24 }}>
      {/* Title and instructions */}
      <Title level={2}>
        Entropy-Based Traversal for Sentence Boundary Detection
      </Title>
      <Paragraph>
        <b>
          Upload your Knowledge Graph and related files below to run
          entropy-based sentence boundary detection.
        </b>
      </Paragraph>
      <Paragraph type="secondary">
        <ul>
          <li>
            <b>Knowledge Graph (.pkl)</b>: A pickled NetworkX{" "}
            <code>DiGraph</code>.
          </li>
          <li>
            <b>Node Embeddings (.npy)</b>: Numpy array of node embeddings.
          </li>
          <li>
            <b>Node Names (.txt)</b>: Plain text, one node name per line (order
            must match embeddings).
          </li>
          <li>
            <b>Start Nodes (.txt)</b>: Plain text, one starting node per line.
          </li>
        </ul>
        Optional: Adjust entropy threshold, max depth, entropy method, or
        temperature.
      </Paragraph>

      {/* Input form for files and params */}
      <AntCard bordered style={{ marginBottom: 32 }}>
        <Form layout="vertical" onFinish={handleSubmit}>
          {/* Knowledge Graph */}
          <Form.Item label="Knowledge Graph (.pkl)">
            <Upload
              beforeUpload={handleFileUpload(setKgFile)}
              showUploadList={kgFile ? { showRemoveIcon: false } : false}
              maxCount={1}
              fileList={kgFile ? [{ uid: "-1", name: kgFile.name }] : []}
            >
              <Button icon={<UploadOutlined />}>Upload KG</Button>
            </Upload>
          </Form.Item>
          {/* Node Embeddings */}
          <Form.Item label="Node Embeddings (.npy)">
            <Upload
              beforeUpload={handleFileUpload(setEmbeddingsFile)}
              showUploadList={
                embeddingsFile ? { showRemoveIcon: false } : false
              }
              maxCount={1}
              fileList={
                embeddingsFile ? [{ uid: "-2", name: embeddingsFile.name }] : []
              }
            >
              <Button icon={<UploadOutlined />}>Upload Embeddings</Button>
            </Upload>
          </Form.Item>
          {/* Node Names */}
          <Form.Item label="Node Names (.txt)">
            <Upload
              beforeUpload={handleFileUpload(setNodesFile)}
              showUploadList={nodesFile ? { showRemoveIcon: false } : false}
              maxCount={1}
              fileList={nodesFile ? [{ uid: "-3", name: nodesFile.name }] : []}
            >
              <Button icon={<UploadOutlined />}>Upload Node Names</Button>
            </Upload>
          </Form.Item>
          {/* Start Nodes */}
          <Form.Item label="Start Nodes (.txt)">
            <Upload
              beforeUpload={handleFileUpload(setStartNodesFile)}
              showUploadList={
                startNodesFile ? { showRemoveIcon: false } : false
              }
              maxCount={1}
              fileList={
                startNodesFile ? [{ uid: "-4", name: startNodesFile.name }] : []
              }
            >
              <Button icon={<UploadOutlined />}>Upload Start Nodes</Button>
            </Upload>
          </Form.Item>

          {/* Parameters */}
          <Form.Item label="Entropy Threshold">
            <InputNumber
              value={entropyThreshold}
              min={0}
              max={5}
              step={0.01}
              onChange={handleEntropyThresholdChange}
              style={{ width: 120 }}
            />
          </Form.Item>
          <Form.Item label="Max Depth">
            <InputNumber
              value={maxDepth}
              min={1}
              max={1000}
              step={1}
              onChange={handleMaxDepthChange}
              style={{ width: 120 }}
            />
          </Form.Item>
          <Form.Item label="Entropy Method">
            <Select
              value={entropyMethod}
              onChange={setEntropyMethod}
              style={{ width: 160 }}
            >
              {ENTROPY_METHODS.map((m) => (
                <Select.Option value={m.value} key={m.value}>
                  {m.label}
                </Select.Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item label="Temperature">
            <InputNumber
              value={temperature}
              min={0.01}
              max={10}
              step={0.01}
              onChange={handleTemperatureChange}
              style={{ width: 120 }}
            />
          </Form.Item>
          {/* Submit button */}
          <Form.Item>
            <Button
              type="primary"
              htmlType="submit"
              icon={<PlayCircleOutlined />}
              loading={loading}
              disabled={loading}
            >
              Run Entropy Traversal
            </Button>
          </Form.Item>
        </Form>
        {/* Show error if any */}
        {error && (
          <Alert
            type="error"
            message={String(error)}
            style={{ marginBottom: 16 }}
          />
        )}
        {/* Show spinner if loading */}
        {loading && <Spin />}
      </AntCard>

      {/* Output display: result panels */}
      {results && (
        <AntCard bordered title="Traversal Results">
          <Paragraph>
            For each start node, you will see the traversed nodes (in the
            detected sentence) and the entropy value at each step.
          </Paragraph>
          {renderResultPanels(results)}
        </AntCard>
      )}
    </div>
  );
}
