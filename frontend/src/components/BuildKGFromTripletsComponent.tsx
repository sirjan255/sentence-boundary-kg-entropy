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
} from "antd";
import {
  UploadOutlined,
  FileExcelOutlined,
  CheckCircleTwoTone,
  DownloadOutlined,
  EyeOutlined,
  InfoCircleOutlined,
} from "@ant-design/icons";
import axios from "axios";

// Backend API base URL
const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL || "/api";
const { Title, Paragraph, Text } = Typography;

// Only allow CSV files for triplets input
const ACCEPTED_FILE_TYPES = [".csv"];

export function BuildKGFromTripletsComponent() {
  // State for file upload, preview, loading, error, download
  const [tripletsFile, setTripletsFile] = useState<File | null>(null);
  const [previewData, setPreviewData] = useState<any>(null);
  const [kgDownloadUrl, setKgDownloadUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [showPreviewModal, setShowPreviewModal] = useState(false);

  // File upload handler (Ant Design Upload)
  const handleFileUpload = (file: File) => {
    setTripletsFile(file);
    setPreviewData(null);
    setKgDownloadUrl(null);
    setError(null);
    setPreviewError(null);
    return false;
  };

  // Handle building KG (download .pkl)
  const handleBuildKG = async () => {
    setError(null);
    setKgDownloadUrl(null);

    if (!tripletsFile) {
      setError("Please upload a CSV file of SVO triplets.");
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("triplets", tripletsFile);

      // POST to backend /build_kg/
      const resp = await axios.post(`${BACKEND}/build_kg/`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        responseType: "blob",
      });

      // Prepare blob for download
      const url = window.URL.createObjectURL(new Blob([resp.data]));
      setKgDownloadUrl(url);

      message.success("Knowledge Graph built! Download available below.");
    } catch (err: any) {
      setError(
        String(
          err?.response?.data?.detail ||
            err?.message ||
            "KG construction failed. Please check your CSV file."
        )
      );
    }
    setLoading(false);
  };

  // Handle preview (lightweight)
  const handlePreview = async () => {
    setPreviewError(null);
    setPreviewData(null);

    if (!tripletsFile) {
      setPreviewError("Please upload a CSV file of SVO triplets.");
      return;
    }

    setPreviewLoading(true);

    try {
      const formData = new FormData();
      formData.append("triplets", tripletsFile);

      // POST to backend /build_kg/preview/
      const resp = await axios.post(`${BACKEND}/build_kg/preview/`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setPreviewData(resp.data);
      setShowPreviewModal(true);
      message.success("Preview generated!");
    } catch (err: any) {
      setPreviewError(
        String(
          err?.response?.data?.detail ||
            err?.message ||
            "Preview failed. Please check your CSV file."
        )
      );
    }
    setPreviewLoading(false);
  };

  // Table columns for preview
  const nodeColumns = [{ title: "Node", dataIndex: "node", key: "node" }];
  const edgeColumns = [
    { title: "Source", dataIndex: "source", key: "source" },
    { title: "Verb", dataIndex: "verb", key: "verb" },
    { title: "Target", dataIndex: "target", key: "target" },
  ];

  // UI rendering
  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: 24 }}>
      <Title level={2}>
        <FileExcelOutlined style={{ color: "#1890ff", marginRight: 12 }} />
        Build Knowledge Graph from SVO Triplets (CSV)
      </Title>
      <Paragraph>
        Upload a <b>CSV file of SVO triplets</b> (subject, verb, object[,
        sentence]) and generate a <b>pickled Knowledge Graph</b> for downstream
        tasks. You can also preview the graph structure!
      </Paragraph>
      <Divider />
      <AntCard bordered style={{ marginBottom: 32 }}>
        <Upload
          accept={ACCEPTED_FILE_TYPES.join(",")}
          beforeUpload={handleFileUpload}
          showUploadList={tripletsFile ? { showRemoveIcon: true } : false}
          maxCount={1}
          onRemove={() => setTripletsFile(null)}
          fileList={
            tripletsFile ? [{ uid: "-1", name: tripletsFile.name }] : []
          }
        >
          <Space>
            <Button icon={<UploadOutlined />}>Upload Triplets CSV</Button>
            <Tooltip title="CSV columns: subject, verb, object (and optional sentence)">
              <InfoCircleOutlined />
            </Tooltip>
          </Space>
        </Upload>
        <div style={{ marginTop: 16 }}>
          <Space>
            <Button
              type="primary"
              icon={<CheckCircleTwoTone twoToneColor="#52c41a" />}
              onClick={handleBuildKG}
              loading={loading}
              disabled={loading || !tripletsFile}
            >
              Build KG (.pkl)
            </Button>
            <Button
              icon={<EyeOutlined />}
              onClick={handlePreview}
              loading={previewLoading}
              disabled={previewLoading || !tripletsFile}
            >
              Preview Graph
            </Button>
            {loading && <Spin />}
          </Space>
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
        {previewError && (
          <Alert
            type="error"
            showIcon
            message={String(previewError)}
            style={{ marginTop: 24 }}
          />
        )}
      </AntCard>
      {/* Download link for KG */}
      {kgDownloadUrl && (
        <AntCard
          bordered
          style={{ marginBottom: 32, background: "#f8fff8" }}
          title={
            <span>
              <DownloadOutlined style={{ color: "#52c41a" }} /> Download KG
            </span>
          }
        >
          <Button
            type="primary"
            icon={<DownloadOutlined />}
            size="large"
            href={kgDownloadUrl}
            download="kg.pkl"
            style={{ marginBottom: 12 }}
          >
            Download Pickled KG (.pkl)
          </Button>
          <Paragraph>
            Use this .pkl file in your downstream graph experiments,
            visualizations, or traversal tools.
          </Paragraph>
        </AntCard>
      )}
      {/* Preview Modal */}
      <Modal
        open={showPreviewModal}
        onCancel={() => setShowPreviewModal(false)}
        footer={null}
        title="KG Preview"
        width={700}
      >
        {previewData && (
          <>
            <Paragraph>
              <b>Nodes:</b> {previewData.num_nodes}, <b>Edges:</b>{" "}
              {previewData.num_edges}
            </Paragraph>
            <Divider>Nodes</Divider>
            <Table
              dataSource={previewData.nodes.map(
                (node: string, idx: number) => ({ node, key: idx })
              )}
              columns={nodeColumns}
              size="small"
              pagination={false}
              scroll={{ y: 160 }}
              bordered
            />
            <Divider>Edges</Divider>
            <Table
              dataSource={previewData.edges.map((e: any, idx: number) => ({
                ...e,
                key: idx,
              }))}
              columns={edgeColumns}
              size="small"
              pagination={false}
              scroll={{ y: 200 }}
              bordered
            />
          </>
        )}
      </Modal>
      {/* How to use */}
      <AntCard
        type="inner"
        title="How to use this tool"
        style={{ marginTop: 16 }}
      >
        <ul>
          <li>
            <b>Prepare your CSV:</b> Columns: <code>subject</code>,{" "}
            <code>verb</code>, <code>object</code> (optionally{" "}
            <code>sentence</code>).
          </li>
          <li>
            <b>Upload</b> your CSV file using the button above.
          </li>
          <li>
            <b>Preview Graph</b> to see nodes and edges before building.
          </li>
          <li>
            <b>Build KG</b> to generate a pickled NetworkX DiGraph (.pkl) ready
            for any KG tool.
          </li>
          <li>
            <b>Download</b> the .pkl file for use in other apps or analysis.
          </li>
        </ul>
        <Divider />
        <Paragraph>
          <Text type="secondary">
            <InfoCircleOutlined /> All computation is in-memory and secure. No
            files are saved on server.
          </Text>
        </Paragraph>
      </AntCard>
    </div>
  );
}

/*
1. User uploads a CSV of SVO triplets (columns: subject, verb, object[, sentence]).
2. Preview the KG (nodes/edges) or generate a .pkl for download.
3. Fully integrated with backend endpoints:
   - POST /build_kg/ for .pkl file
   - POST /build_kg/preview/ for JSON preview
*/
