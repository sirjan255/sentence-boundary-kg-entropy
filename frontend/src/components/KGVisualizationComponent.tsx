"use client";

import React, { useState } from "react";
import {
  Upload,
  Button,
  Card as AntCard,
  Alert,
  Spin,
  Typography,
  Space,
  message,
  Tooltip,
  Divider,
} from "antd";
import {
  UploadOutlined,
  FilePptOutlined,
  PictureOutlined,
  DownloadOutlined,
  InfoCircleOutlined,
  EyeOutlined,
} from "@ant-design/icons";
import axios from "axios";

// Backend API base URL
const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL || "/api";
const { Title, Paragraph, Text } = Typography;

// Only allow pickle files for KG upload
const ACCEPTED_FILE_TYPES = [".pkl"];

export function KGVisualizationComponent() {
  // State for file upload, image result, loading, error
  const [kgFile, setKgFile] = useState<File | null>(null);
  const [imageB64, setImageB64] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Handler for file upload (AntD's beforeUpload returns false to prevent auto-upload)
  const handleFileUpload = (file: File) => {
    setKgFile(file);
    setImageB64(null);
    setError(null);
    return false;
  };

  // Main submit handler: POST the KG file to backend and handle the image response
  const handleSubmit = async () => {
    setError(null);
    setImageB64(null);

    // Validate input
    if (!kgFile) {
      setError("Please upload a pickled NetworkX KG file (.pkl).");
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("kg", kgFile);

      // POST to backend /visualize_kg/
      const resp = await axios.post(`${BACKEND}/visualize_kg/`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      // Check if backend responded with image_b64
      if (resp.data && resp.data.image_b64) {
        setImageB64(resp.data.image_b64);
        message.success("Visualization generated!");
      } else {
        throw new Error("Backend did not return an image.");
      }
    } catch (err: any) {
      setError(
        String(
          err?.response?.data?.detail ||
            err?.message ||
            "Visualization failed. Please check your file format and try again."
        )
      );
    }
    setLoading(false);
  };

  // Download the PNG as a file
  const handleDownload = () => {
    if (!imageB64) return;
    const link = document.createElement("a");
    link.href = `data:image/png;base64,${imageB64}`;
    link.download = "kg_visualization.png";
    link.click();
  };

  // UI rendering starts here
  return (
    <div style={{ maxWidth: 880, margin: "0 auto", padding: 24 }}>
      <Title level={2}>
        <PictureOutlined style={{ color: "#1890ff", marginRight: 12 }} />
        Knowledge Graph Visualization
      </Title>
      <Paragraph>
        Upload a <b>pickled NetworkX graph</b> (<code>.pkl</code>) and generate
        a <b>PNG visualization</b> with node/edge/label info.
        <br />
        <Text type="secondary">
          (Edge labels use <code>verb</code> or <code>label</code> attributes if
          present.)
        </Text>
      </Paragraph>
      <Divider />
      <AntCard bordered style={{ marginBottom: 32 }}>
        {/* File upload */}
        <Upload
          accept={ACCEPTED_FILE_TYPES.join(",")}
          beforeUpload={handleFileUpload}
          showUploadList={kgFile ? { showRemoveIcon: true } : false}
          maxCount={1}
          onRemove={() => setKgFile(null)}
          fileList={kgFile ? [{ uid: "-1", name: kgFile.name }] : []}
        >
          <Space>
            <Button icon={<UploadOutlined />}>Upload KG (.pkl)</Button>
            <Tooltip title="Pickle file exported from NetworkX (Graph/DiGraph)">
              <InfoCircleOutlined />
            </Tooltip>
          </Space>
        </Upload>
        <div style={{ marginTop: 16 }}>
          <Button
            type="primary"
            icon={<EyeOutlined />}
            onClick={handleSubmit}
            loading={loading}
            disabled={loading || !kgFile}
            style={{ width: 180, fontWeight: "bold" }}
          >
            Generate Visualization
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
      {/* Visualization result */}
      {imageB64 && (
        <AntCard
          bordered
          title={
            <span>
              <PictureOutlined style={{ color: "#52c41a" }} /> Graph PNG
              Visualization
            </span>
          }
          style={{ marginBottom: 32, background: "#f8fff8" }}
          extra={
            <Button
              icon={<DownloadOutlined />}
              onClick={handleDownload}
              type="default"
              size="small"
            >
              Download PNG
            </Button>
          }
        >
          <div style={{ textAlign: "center", margin: "24px 0" }}>
            <img
              src={`data:image/png;base64,${imageB64}`}
              alt="KG Visualization"
              style={{
                maxWidth: "100%",
                maxHeight: "600px",
                border: "1px solid #d9d9d9",
                borderRadius: 8,
                background: "#fff",
                boxShadow: "0 0 16px #eee",
              }}
            />
          </div>
        </AntCard>
      )}
      {/* Help/How to use */}
      <AntCard
        type="inner"
        title="How to use this tool"
        style={{ marginTop: 16 }}
      >
        <ul>
          <li>
            <b>Prepare a pickled NetworkX graph</b> (<code>Graph</code> or{" "}
            <code>DiGraph</code>) using Python's <code>pickle.dump()</code>.
          </li>
          <li>
            <b>Upload</b> your <code>.pkl</code> file using the button above.
          </li>
          <li>
            <b>Click "Generate Visualization"</b> to render the full graph with
            labels.
          </li>
          <li>
            <b>Download</b> the PNG or use it inline.
          </li>
          <li>
            Edge labels will be shown if your graph has <code>verb</code> or{" "}
            <code>label</code> attributes.
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
1. User uploads a pickled NetworkX graph (.pkl file).
2. On "Generate Visualization", the PNG is fetched and displayed (and can be downloaded).
3. Backend endpoint: POST /visualize_kg/ with FormData field 'kg'.
*/
