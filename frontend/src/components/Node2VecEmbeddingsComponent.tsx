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
  Switch,
  Input,
  Space,
  message,
  Divider,
  Tooltip,
  Modal,
  Radio,
  theme,
} from "antd";
import {
  UploadOutlined,
  FileZipOutlined,
  DownloadOutlined,
  SettingOutlined,
  InfoCircleOutlined,
  CheckCircleTwoTone,
  CloudDownloadOutlined,
} from "@ant-design/icons";
import axios from "axios";

const BACKEND = "http://localhost:8000/api";
const { Title, Paragraph, Text } = Typography;
const { useToken } = theme;

export function Node2VecEmbeddingsComponent() {
  const { token } = useToken();
  const [kgFile, setKgFile] = useState<File | null>(null);
  const [formVals, setFormVals] = useState({
    dimensions: 64,
    walk_length: 30,
    num_walks: 200,
    workers: 2,
    seed: 42,
    directed: true,
    undirected: false,
    p: 1.0,
    q: 1.0,
    weighted: false,
    normalize: false,
    output_format: "zip",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [jsonResult, setJsonResult] = useState<any>(null);
  const [showJsonModal, setShowJsonModal] = useState(false);

  const handleFileUpload = (file: File) => {
    setKgFile(file);
    setDownloadUrl(null);
    setJsonResult(null);
    setError(null);
    return false;
  };

  const handleFormChange = (changedVals: any, allVals: any) => {
    setFormVals(allVals);
  };

  const handleSubmit = async () => {
    setError(null);
    setDownloadUrl(null);
    setJsonResult(null);

    if (!kgFile) {
      setError("Please upload a pickled NetworkX KG (.pkl) file.");
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("kg", kgFile);
      Object.entries(formVals).forEach(([k, v]) =>
        formData.append(k, String(v))
      );

      const resp = await axios.post(
        `${BACKEND}/node2vec_embeddings/`,
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
          responseType: formVals.output_format === "zip" ? "blob" : "json",
        }
      );

      if (formVals.output_format === "zip") {
        const url = window.URL.createObjectURL(new Blob([resp.data]));
        setDownloadUrl(url);
        message.success("Embeddings generated! Download ready below.");
      } else {
        setJsonResult(resp.data);
        setShowJsonModal(true);
        message.success("Embeddings generated! See preview.");
      }
    } catch (err: any) {
      setError(
        String(
          err?.response?.data?.detail ||
            err?.message ||
            "Node2vec embedding failed. Check your KG file and parameters."
        )
      );
    }
    setLoading(false);
  };

  const previewTable = (emb: any, nodes: string[]) => {
    if (!emb || !nodes) return null;
    const previewRows = nodes.slice(0, 10).map((n: string, i: number) => ({
      node: n,
      embedding:
        emb[i]
          .slice(0, 5)
          .map((x: number) => Number(x).toFixed(3))
          .join(", ") + (emb[i].length > 5 ? " ..." : ""),
      key: n,
    }));
    return (
      <table
        style={{ width: "100%", borderCollapse: "collapse", marginTop: 10 }}
      >
        <thead>
          <tr>
            <th style={{ border: "1px solid #eee", padding: 4 }}>Node</th>
            <th style={{ border: "1px solid #eee", padding: 4 }}>
              Embedding (first 5 dims)
            </th>
          </tr>
        </thead>
        <tbody>
          {previewRows.map((row) => (
            <tr key={row.node}>
              <td style={{ border: "1px solid #eee", padding: 4 }}>
                {row.node}
              </td>
              <td
                style={{
                  border: "1px solid #eee",
                  padding: 4,
                  fontFamily: "monospace",
                }}
              >
                {row.embedding}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  };

  return (
    <div style={{ maxWidth: 800, margin: "0 auto", padding: 24, background: token.colorBgContainer }}>
      <Title level={2} style={{ color: token.colorText }}>
        <FileZipOutlined style={{ color: token.colorPrimary, marginRight: 12 }} />
        Generate node2vec Embeddings from Knowledge Graph
      </Title>
      <Paragraph style={{ color: token.colorText }}>
        Upload a <b>pickled NetworkX graph (.pkl)</b> and set node2vec
        parameters to generate node embeddings.
        <br />
        <Text type="secondary" style={{ color: token.colorTextSecondary }}>
          No files are saved on server. All computation is in-memory.
        </Text>
      </Paragraph>
      <Divider style={{ borderColor: token.colorBorder }} />
      <AntCard bordered style={{ marginBottom: 32, background: token.colorBgContainer }}>
        <Form
          layout="vertical"
          initialValues={formVals}
          onValuesChange={handleFormChange}
          onFinish={handleSubmit}
        >
          <Form.Item label="Knowledge Graph (.pkl)">
            <Upload
              accept=".pkl"
              beforeUpload={handleFileUpload}
              showUploadList={kgFile ? { showRemoveIcon: true } : false}
              maxCount={1}
              onRemove={() => setKgFile(null)}
              fileList={kgFile ? [{ uid: "-1", name: kgFile.name }] : []}
            >
              <Button icon={<UploadOutlined />}>Upload KG</Button>
            </Upload>
          </Form.Item>
          <Divider orientation="left">
            <SettingOutlined /> node2vec Parameters
          </Divider>
          <Form.Item label="Dimensions" name="dimensions">
            <InputNumber min={2} max={1024} step={1} style={{ width: 100 }} />
          </Form.Item>
          <Form.Item label="Walk Length" name="walk_length">
            <InputNumber min={1} max={500} step={1} style={{ width: 100 }} />
          </Form.Item>
          <Form.Item label="Num Walks" name="num_walks">
            <InputNumber min={1} max={1000} step={1} style={{ width: 100 }} />
          </Form.Item>
          <Form.Item label="Workers" name="workers">
            <InputNumber min={1} max={16} step={1} style={{ width: 100 }} />
          </Form.Item>
          <Form.Item label="Seed" name="seed">
            <InputNumber
              min={0}
              max={1000000}
              step={1}
              style={{ width: 120 }}
            />
          </Form.Item>
          <Form.Item
            label="p (return param)"
            name="p"
            tooltip="Return parameter (p)"
          >
            <InputNumber
              min={0.01}
              max={10}
              step={0.01}
              style={{ width: 100 }}
            />
          </Form.Item>
          <Form.Item
            label="q (inout param)"
            name="q"
            tooltip="Inout parameter (q)"
          >
            <InputNumber
              min={0.01}
              max={10}
              step={0.01}
              style={{ width: 100 }}
            />
          </Form.Item>
          <Form.Item label="Directed" name="directed" valuePropName="checked">
            <Switch />
          </Form.Item>
          <Form.Item
            label="Undirected"
            name="undirected"
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>
          <Form.Item label="Weighted" name="weighted" valuePropName="checked">
            <Switch />
          </Form.Item>
          <Form.Item
            label="Normalize Embeddings"
            name="normalize"
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>
          <Form.Item label="Output Format" name="output_format">
            <Radio.Group>
              <Radio value="zip">Download (.zip of npy+txt)</Radio>
              <Radio value="json">Preview (JSON)</Radio>
            </Radio.Group>
          </Form.Item>
          <Form.Item>
            <Button
              type="primary"
              htmlType="submit"
              icon={<CloudDownloadOutlined />}
              loading={loading}
              disabled={loading}
              style={{ width: 220 }}
            >
              Generate Embeddings
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
      {downloadUrl && (
        <AntCard
          bordered
          style={{ marginBottom: 32, background: token.colorBgContainer }}
          title={
            <span style={{ color: token.colorText }}>
              <DownloadOutlined style={{ color: token.colorSuccess }} /> Download
              Embeddings
            </span>
          }
        >
          <Button
            type="primary"
            icon={<DownloadOutlined />}
            size="large"
            href={downloadUrl}
            download="embeddings.zip"
            style={{ marginBottom: 12 }}
          >
            Download Embeddings (.zip)
          </Button>
          <Paragraph style={{ color: token.colorText }}>
            Contains <code>embeddings.npy</code> (numpy array) and{" "}
            <code>embeddings_nodes.txt</code> (node names).
          </Paragraph>
        </AntCard>
      )}
      <Modal
        open={showJsonModal}
        onCancel={() => setShowJsonModal(false)}
        footer={null}
        title={<span style={{ color: token.colorText }}>Embeddings Preview</span>}
        width={700}
        styles={{
          header: { 
            background: token.colorBgContainer,
            borderBottom: `1px solid ${token.colorBorder}`
          },
          content: { background: token.colorBgContainer },
          body: { 
            background: token.colorBgContainer,
            color: token.colorText
          }
        }}
      >
        {jsonResult && (
          <>
            <Paragraph style={{ color: token.colorText }}>
              <b>Nodes:</b> {jsonResult.nodes.length}, <b>Dimensions:</b>{" "}
              {jsonResult.embeddings[0]?.length}
            </Paragraph>
            <Divider style={{ borderColor: token.colorBorder }}>Sample (first 10 nodes)</Divider>
            {previewTable(jsonResult.embeddings, jsonResult.nodes)}
          </>
        )}
      </Modal>
      <AntCard
        type="inner"
        title={<span style={{ color: token.colorText }}>How to use this tool</span>}
        style={{ marginTop: 16, background: token.colorBgContainer }}
      >
        <ul style={{ color: token.colorText }}>
          <li>
            <b>Prepare your KG:</b> Upload a pickled NetworkX graph (.pkl).
          </li>
          <li>
            <b>Set node2vec parameters:</b> Adjust dimensions, walk length, p/q,
            etc.
          </li>
          <li>
            <b>Choose output:</b> Download as .zip or preview as JSON table.
          </li>
          <li>
            <b>Download</b> the embeddings for further ML or graph analysis.
          </li>
        </ul>
        <Divider style={{ borderColor: token.colorBorder }} />
        <Paragraph style={{ color: token.colorText }}>
          <Text type="secondary" style={{ color: token.colorTextSecondary }}>
            <InfoCircleOutlined /> All computation is in-memory. No files are
            saved on the server.
          </Text>
        </Paragraph>
      </AntCard>
    </div>
  );
}