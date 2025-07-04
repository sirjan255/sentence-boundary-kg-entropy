"use client";

import React, { useState } from "react";
import {
  Upload,
  Button,
  Card as AntCard,
  Alert,
  Spin,
  Typography,
  InputNumber,
  Form,
  Select,
  Input,
  Space,
  message,
  Divider,
  Tooltip,
  Tag,
  Collapse,
  theme,
} from "antd";
import {
  UploadOutlined,
  CheckCircleTwoTone,
  SettingOutlined,
  InfoCircleOutlined,
  BulbOutlined,
  CloudDownloadOutlined,
} from "@ant-design/icons";
import axios from "axios";

const BACKEND = "http://localhost:8000/api";
const { Title, Paragraph, Text } = Typography;
const { Panel } = Collapse;
const { useToken } = theme;

export function SelectStartingNodesComponent() {
  const { token } = useToken();
  const [kgFile, setKgFile] = useState<File | null>(null);
  const [embeddingsFile, setEmbeddingsFile] = useState<File | null>(null);
  const [nodesFile, setNodesFile] = useState<File | null>(null);
  const [strategy, setStrategy] = useState<
    "degree" | "random" | "entropy" | "all"
  >("degree");
  const [num, setNum] = useState<number>(3);
  const [entropyMethod, setEntropyMethod] = useState<string>("blt");
  const [temperature, setTemperature] = useState<number>(1.0);
  const [seed, setSeed] = useState<number>(42);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleKgUpload = (file: File) => {
    setKgFile(file);
    setResult(null);
    setError(null);
    return false;
  };

  const handleEmbeddingsUpload = (file: File) => {
    setEmbeddingsFile(file);
    setResult(null);
    setError(null);
    return false;
  };

  const handleNodesUpload = (file: File) => {
    setNodesFile(file);
    setResult(null);
    setError(null);
    return false;
  };

  const handleSubmit = async () => {
    setError(null);
    setResult(null);

    if (!kgFile) {
      setError("Please upload your Knowledge Graph pickle (.pkl) file.");
      return;
    }
    if (
      (strategy === "entropy" || strategy === "all") &&
      (!embeddingsFile || !nodesFile)
    ) {
      setError(
        "Embeddings (.npy) and node list (.txt) are required for entropy-based selection."
      );
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("kg", kgFile);
      formData.append("strategy", strategy);
      formData.append("num", String(num));
      formData.append("entropy_method", entropyMethod);
      formData.append("temperature", String(temperature));
      formData.append("seed", String(seed));
      if (embeddingsFile) formData.append("embeddings", embeddingsFile);
      if (nodesFile) formData.append("nodes", nodesFile);

      const resp = await axios.post(
        `${BACKEND}/select_starting_nodes/`,
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      setResult(resp.data);
      message.success("Node selection complete!");
    } catch (err: any) {
      setError(
        String(
          err?.response?.data?.detail ||
            err?.message ||
            "Node selection failed. Please check your files and parameters."
        )
      );
    }
    setLoading(false);
  };

  const renderNodeList = (nodes: any) => {
    if (!nodes) return null;
    if (Array.isArray(nodes)) {
      return (
        <Space wrap>
          {nodes.map((n: string, i: number) => (
            <Tag
              key={i}
              color="geekblue"
              style={{ fontSize: 16, padding: "3px 12px" }}
            >
              {n}
            </Tag>
          ))}
        </Space>
      );
    }
    if (typeof nodes === "object" && nodes.error)
      return <Alert message={nodes.error} type="error" showIcon />;
    return <span>{String(nodes)}</span>;
  };

  return (
    <div style={{ maxWidth: 880, margin: "0 auto", padding: 24, background: token.colorBgContainer }}>
      <Title level={2} style={{ color: token.colorText }}>
        <BulbOutlined style={{ color: token.colorPrimary, marginRight: 12 }} />
        Select Starting Nodes from Knowledge Graph
      </Title>
      <Paragraph style={{ color: token.colorText }}>
        Upload a <b>pickled NetworkX graph</b> and select a node selection
        strategy.
        <br />
        <Text type="secondary" style={{ color: token.colorTextSecondary }}>
          Strategies: top degree, random, entropy-based (needs embeddings), or
          all combined.
        </Text>
      </Paragraph>
      <Divider style={{ borderColor: token.colorBorder }} />
      <AntCard bordered style={{ marginBottom: 32, background: token.colorBgContainer }}>
        <Form layout="vertical" onFinish={handleSubmit}>
          <Form.Item label="Knowledge Graph (.pkl)">
            <Upload
              accept=".pkl"
              beforeUpload={handleKgUpload}
              showUploadList={kgFile ? { showRemoveIcon: true } : false}
              maxCount={1}
              onRemove={() => setKgFile(null)}
              fileList={kgFile ? [{ uid: "-1", name: kgFile.name }] : []}
            >
              <Button icon={<UploadOutlined />}>Upload KG</Button>
            </Upload>
          </Form.Item>
          <Form.Item
            label="Node Embeddings (.npy) [required for entropy]"
            extra="Required for entropy-based selection"
          >
            <Upload
              accept=".npy"
              beforeUpload={handleEmbeddingsUpload}
              showUploadList={embeddingsFile ? { showRemoveIcon: true } : false}
              maxCount={1}
              onRemove={() => setEmbeddingsFile(null)}
              fileList={
                embeddingsFile ? [{ uid: "-2", name: embeddingsFile.name }] : []
              }
            >
              <Button icon={<UploadOutlined />}>Upload Embeddings</Button>
            </Upload>
          </Form.Item>
          <Form.Item
            label="Node Names (.txt) [required for entropy]"
            extra="Required for entropy-based selection"
          >
            <Upload
              accept=".txt"
              beforeUpload={handleNodesUpload}
              showUploadList={nodesFile ? { showRemoveIcon: true } : false}
              maxCount={1}
              onRemove={() => setNodesFile(null)}
              fileList={nodesFile ? [{ uid: "-3", name: nodesFile.name }] : []}
            >
              <Button icon={<UploadOutlined />}>Upload Node Names</Button>
            </Upload>
          </Form.Item>
          <Form.Item label="Node Selection Strategy" required>
            <Select
              value={strategy}
              onChange={(v) => setStrategy(v)}
              style={{ width: 220 }}
            >
              <Select.Option value="degree">Top N by Degree</Select.Option>
              <Select.Option value="random">Random N Nodes</Select.Option>
              <Select.Option value="entropy">Top N by Entropy</Select.Option>
              <Select.Option value="all">All (combined)</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item label="Number of Nodes (per strategy)">
            <InputNumber
              min={1}
              max={100}
              value={num}
              onChange={(value) => {
                if (value !== null) setNum(value);
              }}
              style={{ width: 100 }}
            />
          </Form.Item>
          {(strategy === "entropy" || strategy === "all") && (
            <>
              <Form.Item label="Entropy Method">
                <Input
                  value={entropyMethod}
                  onChange={(e) => setEntropyMethod(e.target.value)}
                  style={{ width: 180 }}
                  placeholder="blt"
                />
              </Form.Item>
              <Form.Item label="Temperature">
                <InputNumber
                  min={0.01}
                  max={10}
                  step={0.01}
                  value={temperature}
                  onChange={(value) => {
                    if (value !== null) setTemperature(value);
                  }}
                  style={{ width: 100 }}
                />
              </Form.Item>
            </>
          )}
          <Form.Item label="Random Seed">
            <InputNumber
              min={0}
              max={1000000}
              step={1}
              value={seed}
              onChange={(value) => {
                if (value !== null) setSeed(value);
              }}
              style={{ width: 120 }}
            />
          </Form.Item>
          <Form.Item>
            <Button
              type="primary"
              htmlType="submit"
              icon={<CheckCircleTwoTone twoToneColor="#52c41a" />}
              loading={loading}
              disabled={loading}
              style={{ width: 220 }}
            >
              Select Starting Nodes
            </Button>
            {loading && <Spin style={{ marginLeft: 16 }} />}
          </Form.Item>
        </Form>
        {error && (
          <Alert
            type="error"
            showIcon
            message={String(error)}
            style={{ marginTop: 16 }}
          />
        )}
      </AntCard>
      {result && (
        <AntCard
          bordered
          style={{ marginBottom: 32, background: token.colorBgContainer }}
          title={<span style={{ color: token.colorText }}>Selected Starting Nodes</span>}
        >
          <Collapse defaultActiveKey={Object.keys(result)}>
            {Object.entries(result).map(([strategy, nodes]) => (
              <Panel
                header={
                  <span style={{ color: token.colorText }}>
                    <Tag
                      color="blue"
                      style={{ fontWeight: "bold", fontSize: 16 }}
                    >
                      {strategy.toUpperCase()}
                    </Tag>
                  </span>
                }
                key={strategy}
              >
                {renderNodeList(nodes)}
              </Panel>
            ))}
          </Collapse>
        </AntCard>
      )}
      <AntCard
        type="inner"
        title={<span style={{ color: token.colorText }}>How to use this tool</span>}
        style={{ marginTop: 16, background: token.colorBgContainer }}
      >
        <ul style={{ color: token.colorText }}>
          <li>
            <b>Upload your Knowledge Graph:</b> Pickled NetworkX graph (
            <code> .pkl</code>).
          </li>
          <li>
            <b>Optionally:</b> For entropy-based selection, also upload node
            embeddings (<code>.npy</code>) and node list (<code>.txt</code>).
          </li>
          <li>
            <b>Pick a strategy:</b> Degree, random, entropy, or all.
          </li>
          <li>
            <b>Adjust parameters</b> as needed and run.
          </li>
          <li>
            <b>View selected nodes</b> for each strategy below.
          </li>
        </ul>
        <Divider style={{ borderColor: token.colorBorder }} />
        <Paragraph style={{ color: token.colorText }}>
          <Text type="secondary" style={{ color: token.colorTextSecondary }}>
            <InfoCircleOutlined /> No files are saved on server. All computation
            is in-memory.
          </Text>
        </Paragraph>
      </AntCard>
    </div>
  );
}