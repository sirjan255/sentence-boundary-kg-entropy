"use client";

import React, { useState } from "react";
import {
  Upload,
  Button,
  Form,
  Card as AntCard,
  Alert,
  Spin,
  Typography,
  Table,
  Input,
  InputNumber,
  Space,
  message,
  Slider,
  Collapse,
  Tag,
  Divider,
  Tooltip,
  Descriptions,
  Statistic,
} from "antd";
import {
  UploadOutlined,
  SearchOutlined,
  SettingOutlined,
  FileTextOutlined,
  FileImageOutlined,
  FilePdfOutlined,
} from "@ant-design/icons";
import axios from "axios";

// Backend API URL (set via .env or fallback to /api)
const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL || "/api";
const { Title, Paragraph } = Typography;
const { Panel } = Collapse;

const ACCEPTED_FILE_TYPES = [".txt", ".json", ".pkl", ".npy", ".pdf"];

export function GNNBoundaryExperimentComponent() {
  // --- State for file uploads ---
  const [kgFile, setKgFile] = useState<File | null>(null);
  const [embeddingsFile, setEmbeddingsFile] = useState<File | null>(null);
  const [nodesFile, setNodesFile] = useState<File | null>(null);
  const [goldFile, setGoldFile] = useState<File | null>(null);

  // --- Hyperparameter state ---
  const [epochs, setEpochs] = useState<number>(30);
  const [lr, setLr] = useState<number>(0.01);
  const [testRatios, setTestRatios] = useState<string>("0.1,0.2,0.3,0.5");
  const [hiddenDims, setHiddenDims] = useState<string>("32,64,128");
  const [seeds, setSeeds] = useState<string>("42,2024,7");
  const [runsPerSetting, setRunsPerSetting] = useState<number>(3);

  // --- UI, loading, error, result ---
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

  // --- File upload handler ---
  const handleFileUpload = (setter: (f: File) => void) => (file: File) => {
    setter(file);
    return false;
  };

  // --- Form submit: Run experiment ---
  const handleSubmit = async () => {
    setError(null);
    setResult(null);

    // Validate files
    if (!kgFile || !embeddingsFile || !nodesFile || !goldFile) {
      setError(
        "Please upload all required files: KG, embeddings, nodes, gold."
      );
      return;
    }
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("kg", kgFile);
      formData.append("embeddings", embeddingsFile);
      formData.append("nodes", nodesFile);
      formData.append("gold", goldFile);
      formData.append("epochs", epochs.toString());
      formData.append("lr", lr.toString());
      formData.append("test_ratios", testRatios);
      formData.append("hidden_dims", hiddenDims);
      formData.append("seeds", seeds);
      formData.append("runs_per_setting", runsPerSetting.toString());

      const resp = await axios.post(
        `${BACKEND}/gnn_boundary_experiment/`,
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
          timeout: 1000 * 60 * 10, // 10 min timeout for big experiments
        }
      );
      setResult(resp.data);
      message.success("Experiment completed!");
    } catch (err: any) {
      setError(
        String(
          err?.response?.data?.detail ||
            err?.message ||
            "Experiment failed. Please check your files and parameters."
        )
      );
    }
    setLoading(false);
  };

  // --- Data for summary table ---
  const summaryColumns = [
    {
      title: "Test Ratio",
      dataIndex: "test_ratio",
      key: "test_ratio",
      render: (v: number) => v.toFixed(2),
    },
    { title: "Hidden Dim", dataIndex: "hidden_dim", key: "hidden_dim" },
    {
      title: "Avg Precision",
      dataIndex: "avg_precision",
      key: "avg_precision",
      render: (v: number) => <Statistic value={v} precision={3} />,
    },
    {
      title: "Avg Recall",
      dataIndex: "avg_recall",
      key: "avg_recall",
      render: (v: number) => <Statistic value={v} precision={3} />,
    },
    {
      title: "Avg F1",
      dataIndex: "avg_f1",
      key: "avg_f1",
      render: (v: number) => (
        <Statistic value={v} precision={3} valueStyle={{ color: "#52c41a" }} />
      ),
    },
    {
      title: "Avg Test Size",
      dataIndex: "avg_test_size",
      key: "avg_test_size",
    },
    {
      title: "Avg Gold Test Count",
      dataIndex: "avg_gold_test_count",
      key: "avg_gold_test_count",
    },
    { title: "Runs", dataIndex: "runs", key: "runs" },
  ];

  // --- Data for per-run details table ---
  const perRunColumns = [
    {
      title: "Test Ratio",
      dataIndex: "test_ratio",
      key: "test_ratio",
      render: (v: number) => v.toFixed(2),
    },
    { title: "Hidden Dim", dataIndex: "hidden_dim", key: "hidden_dim" },
    { title: "Seed", dataIndex: "seed", key: "seed" },
    {
      title: "Precision",
      dataIndex: "precision",
      key: "precision",
      render: (v: number) => v.toFixed(3),
    },
    {
      title: "Recall",
      dataIndex: "recall",
      key: "recall",
      render: (v: number) => v.toFixed(3),
    },
    {
      title: "F1",
      dataIndex: "f1",
      key: "f1",
      render: (v: number) => <Tag color="green">{v.toFixed(3)}</Tag>,
    },
    { title: "Test Size", dataIndex: "test_size", key: "test_size" },
    {
      title: "Gold Test Count",
      dataIndex: "gold_test_count",
      key: "gold_test_count",
    },
  ];

  // --- UI Rendering starts here ---
  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: 24 }}>
      <Title level={2}>GNN Boundary Classifier Experiments</Title>
      <Paragraph>
        <b>
          Upload your Knowledge Graph data and gold boundaries, set
          hyperparameters, and run GNN boundary classification experiments using
          PyTorch Geometric.
        </b>
      </Paragraph>
      <Paragraph type="secondary">
        <ul>
          <li>
            <b>Knowledge Graph (.pkl):</b> Pickled NetworkX DiGraph.
          </li>
          <li>
            <b>Node Embeddings (.npy):</b> Numpy array.
          </li>
          <li>
            <b>Node Names (.txt):</b> One per line (Python literal if needed).
          </li>
          <li>
            <b>Gold Boundaries (.json):</b> Boundary nodes in JSON format.
          </li>
        </ul>
        All computation is done in-memory - no file I/O!
      </Paragraph>

      <AntCard bordered style={{ marginBottom: 32 }}>
        <Form layout="vertical" onFinish={handleSubmit}>
          {/* File Uploads */}
          <Form.Item label="Knowledge Graph (.pkl)">
            <Upload
              accept=".pkl"
              beforeUpload={handleFileUpload(setKgFile)}
              showUploadList={kgFile ? { showRemoveIcon: false } : false}
              maxCount={1}
              fileList={kgFile ? [{ uid: "-1", name: kgFile.name }] : []}
            >
              <Button icon={<UploadOutlined />}>Upload KG</Button>
            </Upload>
          </Form.Item>
          <Form.Item label="Node Embeddings (.npy)">
            <Upload
              accept=".npy"
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
          <Form.Item label="Node Names (.txt)">
            <Upload
              accept=".txt"
              beforeUpload={handleFileUpload(setNodesFile)}
              showUploadList={nodesFile ? { showRemoveIcon: false } : false}
              maxCount={1}
              fileList={nodesFile ? [{ uid: "-3", name: nodesFile.name }] : []}
            >
              <Button icon={<UploadOutlined />}>Upload Node Names</Button>
            </Upload>
          </Form.Item>
          <Form.Item label="Gold Boundaries (.json)">
            <Upload
              accept=".json"
              beforeUpload={handleFileUpload(setGoldFile)}
              showUploadList={goldFile ? { showRemoveIcon: false } : false}
              maxCount={1}
              fileList={goldFile ? [{ uid: "-4", name: goldFile.name }] : []}
            >
              <Button icon={<UploadOutlined />}>Upload Gold Boundaries</Button>
            </Upload>
          </Form.Item>

          {/* Hyperparameters */}
          <Divider orientation="left">
            <SettingOutlined /> Hyperparameters
          </Divider>
          <Form.Item label="Epochs">
            <Slider
              min={10}
              max={100}
              marks={{ 10: "10", 30: "30", 50: "50", 100: "100" }}
              step={1}
              value={epochs}
              onChange={setEpochs}
              style={{ maxWidth: 400 }}
            />
          </Form.Item>
          <Form.Item label="Learning Rate">
            <InputNumber
              min={0.0001}
              max={1}
              step={0.0001}
              value={lr}
              onChange={(v) => v !== null && setLr(Number(v))}
              style={{ width: 120 }}
            />
          </Form.Item>
          <Form.Item
            label="Test Ratios (comma-separated, e.g. 0.1,0.2)"
            tooltip="Comma-separated list of test set ratios"
          >
            <Input
              value={testRatios}
              onChange={(e) => setTestRatios(e.target.value)}
              style={{ width: 250 }}
            />
          </Form.Item>
          <Form.Item
            label="Hidden Dims (comma-separated, e.g. 32,64,128)"
            tooltip="Comma-separated list of hidden layer sizes"
          >
            <Input
              value={hiddenDims}
              onChange={(e) => setHiddenDims(e.target.value)}
              style={{ width: 250 }}
            />
          </Form.Item>
          <Form.Item
            label="Seeds (comma-separated, e.g. 42,2024,7)"
            tooltip="Comma-separated list of random seeds"
          >
            <Input
              value={seeds}
              onChange={(e) => setSeeds(e.target.value)}
              style={{ width: 250 }}
            />
          </Form.Item>
          <Form.Item
            label="Runs Per Setting"
            tooltip="How many runs per hyperparameter setting"
          >
            <InputNumber
              min={1}
              max={10}
              value={runsPerSetting}
              onChange={(v) => v !== null && setRunsPerSetting(Number(v))}
              style={{ width: 120 }}
            />
          </Form.Item>
          <Form.Item>
            <Button
              type="primary"
              htmlType="submit"
              icon={<SearchOutlined />}
              loading={loading}
              disabled={loading}
            >
              Run Experiment
            </Button>
            {loading && <Spin style={{ marginLeft: 16 }} />}
          </Form.Item>
        </Form>
        {/* Error Alert */}
        {error && (
          <Alert
            type="error"
            message={String(error)}
            showIcon
            style={{ marginBottom: 16 }}
          />
        )}
      </AntCard>

      {/* Results display */}
      {result && (
        <AntCard
          bordered
          title="Experiment Results"
          style={{ marginBottom: 32, background: "#f8fff8" }}
        >
          <Collapse defaultActiveKey={["summary", "perrun"]}>
            <Panel header="Summary (Averaged per Setting)" key="summary">
              <Table
                dataSource={
                  Array.isArray(result.summary)
                    ? result.summary.map((row: any, i: number) => ({
                        ...row,
                        key: i,
                      }))
                    : []
                }
                columns={summaryColumns}
                pagination={false}
                bordered
                size="middle"
              />
            </Panel>
            <Panel header="Per-run Details" key="perrun">
              <Table
                dataSource={
                  Array.isArray(result.runs)
                    ? result.runs.map((row: any, i: number) => ({
                        ...row,
                        key: i,
                      }))
                    : []
                }
                columns={perRunColumns}
                pagination={{ pageSize: 20, showSizeChanger: true }}
                bordered
                size="small"
                scroll={{ x: 1000 }}
              />
            </Panel>
          </Collapse>
          <Divider />
          <Descriptions
            title="Experiment Settings"
            bordered
            column={1}
            size="small"
          >
            <Descriptions.Item label="Epochs">{epochs}</Descriptions.Item>
            <Descriptions.Item label="Learning Rate">{lr}</Descriptions.Item>
            <Descriptions.Item label="Test Ratios">
              {testRatios}
            </Descriptions.Item>
            <Descriptions.Item label="Hidden Dims">
              {hiddenDims}
            </Descriptions.Item>
            <Descriptions.Item label="Seeds">{seeds}</Descriptions.Item>
            <Descriptions.Item label="Runs/Setting">
              {runsPerSetting}
            </Descriptions.Item>
          </Descriptions>
        </AntCard>
      )}

      {/* Help Section */}
      <AntCard
        type="inner"
        title="How to use this tool"
        style={{ marginTop: 16 }}
      >
        <ul>
          <li>
            <b>Upload your files:</b> Upload KG (.pkl), embeddings (.npy), node
            names (.txt), and gold boundaries (.json).
          </li>
          <li>
            <b>Set hyperparameters:</b> Adjust epochs, learning rate, test
            ratios, hidden dims, seeds, and runs per setting.
          </li>
          <li>
            <b>Run Experiment:</b> Click the button to launch GNN training and
            evaluation. <br />
            Each setting will be run multiple times for reliable metrics.
          </li>
          <li>
            <b>View Results:</b> See a summary table (mean across runs/settings)
            and per-run details for all hyperparameter combinations.
          </li>
        </ul>
      </AntCard>
    </div>
  );
}

/*
1. Upload your KG, embeddings, node names, and gold boundaries.
2. Adjust hyperparameters as desired.
3. Click "Run Experiment" and see results (summary and per-run) in a clean UI.
*/
