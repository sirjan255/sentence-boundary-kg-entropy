"use client";

import React, { useState } from "react";
import { Upload, Button, InputNumber, Select, Form, message, Spin, Card as AntCard, Table, Alert } from "antd";
import { UploadOutlined, SearchOutlined } from "@ant-design/icons";
import axios from "axios";

// Backend endpoint; can be configured using REACT_APP_BACKEND or defaults to /api
const BACKEND = process.env.REACT_APP_BACKEND || "/api";

// Entropy method options for the dropdown
const ENTROPY_METHODS = [
  { label: "BLT", value: "blt" },
  { label: "Sum", value: "sum" },
  { label: "Mean", value: "mean" }
];

// Main component for detecting sentence boundaries
export function DetectSentenceBoundaryComponent() {
  // State hooks for all file uploads
  const [kgFile, setKgFile] = useState<File | null>(null); // Knowledge graph (.pkl)
  const [embeddingsFile, setEmbeddingsFile] = useState<File | null>(null); // Node embeddings (.npy)
  const [nodesFile, setNodesFile] = useState<File | null>(null); // Node names (.txt)
  const [startsFile, setStartsFile] = useState<File | null>(null); // Start nodes (.txt)

  // State hooks for all the numeric and string parameters
  const [entropyThreshold, setEntropyThreshold] = useState<number>(0.8); // Entropy threshold
  const [entropyMethod, setEntropyMethod] = useState<string>("blt"); // Method for entropy
  const [temperature, setTemperature] = useState<number>(1.0); // Temperature parameter
  const [maxNodes, setMaxNodes] = useState<number>(30); // Max nodes to traverse

  // State for loading spinner, results, and errors
  const [loading, setLoading] = useState(false); // Indicates if request is ongoing
  const [results, setResults] = useState<any>(null); // Stores API results
  const [error, setError] = useState<string | null>(null); // Stores any error message

  // Helper for file upload: set file in state and prevent default upload behavior
  const handleFileUpload = (setter: (f: File) => void) => (file: File) => {
    setter(file);
    return false; // Prevent default upload
  };

  // --- InputNumber onChange handlers must accept number | null ---
  // If null, don't change the value
  const handleEntropyThresholdChange = (value: number | null) => {
    if (value !== null) setEntropyThreshold(value);
  };
  const handleTemperatureChange = (value: number | null) => {
    if (value !== null) setTemperature(value);
  };
  const handleMaxNodesChange = (value: number | null) => {
    if (value !== null) setMaxNodes(value);
  };

  // Handle form submission: validates inputs, sends POST request, parses results
  const handleSubmit = async () => {
    setError(null);     // Clear previous errors
    setResults(null);   // Clear previous results

    // Ensure all input files are provided
    if (!kgFile || !embeddingsFile || !nodesFile || !startsFile) {
      setError("Please provide all required files (kg, embeddings, nodes, starts).");
      return;
    }
    setLoading(true); // Show spinner

    try {
      // Build FormData with all input files and parameters
      const formData = new FormData();
      formData.append("kg", kgFile);
      formData.append("embeddings", embeddingsFile);
      formData.append("nodes", nodesFile);
      formData.append("starts", startsFile);
      formData.append("entropy_threshold", entropyThreshold.toString());
      formData.append("entropy_method", entropyMethod);
      formData.append("temperature", temperature.toString());
      formData.append("max_nodes", maxNodes.toString());

      // POST request to backend endpoint
      const resp = await axios.post(`${BACKEND}/detect_sentence_boundary/`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });

      setResults(resp.data); // Save results for display
      message.success("Detection complete!"); // Show success message
    } catch (err: any) {
      setError(
        String(
          err?.response?.data?.detail ||
          err?.message ||
          "Detection failed. Please make sure all files and parameters are correct."
        )
      );
    }
    setLoading(false); // Hide spinner
  };

  // Render a table for each start node's results
  const renderResultTables = (resultsObj: any) => {
    if (!resultsObj) return null;
    // Each start node gives a list of nodes with entropy, or an error
    return Object.entries(resultsObj).map(([startNode, data]) => (
      <AntCard
        key={startNode}
        title={`Start Node: ${startNode}`}
        style={{ marginBottom: 24, background: "#f9fafb" }}
        bordered
      >
        {Array.isArray(data) ? (
          // If valid results, show as a table
          <Table
            dataSource={data.map((row: any, idx: number) => ({
              key: idx,
              node: row.node,
              entropy: row.entropy
            }))}
            columns={[
              { title: "Node", dataIndex: "node", key: "node" },
              { title: "Entropy", dataIndex: "entropy", key: "entropy", render: (v: number) => v.toFixed(4) }
            ]}
            pagination={false}
            size="small"
          />
        ) : (data && typeof data === "object" && "error" in data) ? (
          // Fix: Alert.message must be ReactNode, so always cast to string
          <Alert type="error" message={String(data.error)} />
        ) : (
          <div>No data</div>
        )}
      </AntCard>
    ));
  };

  // Main render
  return (
    <div style={{ maxWidth: 800, margin: "0 auto", padding: 24 }}>
      <AntCard title="Detect Sentence Boundaries (Entropy Traversal)" bordered>
        <Form layout="vertical" onFinish={handleSubmit}>
          {/* Upload KG (.pkl) */}
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
          {/* Upload embeddings (.npy) */}
          <Form.Item label="Node Embeddings (.npy)">
            <Upload
              beforeUpload={handleFileUpload(setEmbeddingsFile)}
              showUploadList={embeddingsFile ? { showRemoveIcon: false } : false}
              maxCount={1}
              fileList={embeddingsFile ? [{ uid: "-2", name: embeddingsFile.name }] : []}
            >
              <Button icon={<UploadOutlined />}>Upload Embeddings</Button>
            </Upload>
          </Form.Item>
          {/* Upload node names (.txt) */}
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
          {/* Upload start nodes (.txt) */}
          <Form.Item label="Start Nodes (.txt)">
            <Upload
              beforeUpload={handleFileUpload(setStartsFile)}
              showUploadList={startsFile ? { showRemoveIcon: false } : false}
              maxCount={1}
              fileList={startsFile ? [{ uid: "-4", name: startsFile.name }] : []}
            >
              <Button icon={<UploadOutlined />}>Upload Start Nodes</Button>
            </Upload>
          </Form.Item>
          {/* Entropy threshold parameter */}
          <Form.Item label="Entropy Threshold">
            <InputNumber value={entropyThreshold} min={0} max={5} step={0.01} onChange={handleEntropyThresholdChange} />
          </Form.Item>
          {/* Entropy method dropdown */}
          <Form.Item label="Entropy Method">
            <Select value={entropyMethod} onChange={setEntropyMethod} style={{ width: 160 }}>
              {ENTROPY_METHODS.map(m => (
                <Select.Option value={m.value} key={m.value}>{m.label}</Select.Option>
              ))}
            </Select>
          </Form.Item>
          {/* Temperature parameter */}
          <Form.Item label="Temperature">
            <InputNumber value={temperature} min={0.01} max={10} step={0.01} onChange={handleTemperatureChange} />
          </Form.Item>
          {/* Max nodes parameter */}
          <Form.Item label="Max Nodes to Traverse">
            <InputNumber value={maxNodes} min={1} max={1000} step={1} onChange={handleMaxNodesChange} />
          </Form.Item>
          {/* Submit button */}
          <Form.Item>
            <Button
              type="primary"
              htmlType="submit"
              icon={<SearchOutlined />}
              loading={loading}
              disabled={loading}
            >
              Detect Boundaries
            </Button>
          </Form.Item>
        </Form>
        {/* Show error if any */}
        {error && <Alert type="error" message={String(error)} style={{ marginBottom: 16 }} />}
        {/* Show spinner if loading */}
        {loading && <Spin />}
        {/* Show results ("tables") after successful API call */}
        {results && <div style={{ marginTop: 24 }}>{renderResultTables(results)}</div>}
      </AntCard>
    </div>
  );
}