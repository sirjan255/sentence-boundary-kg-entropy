"use client";

import React, { useState } from "react";
import {
  Upload,
  Button,
  Form,
  Alert,
  Card as AntCard,
  Spin,
  Typography,
  Table,
  Space,
  message,
} from "antd";
import { UploadOutlined, CheckCircleTwoTone } from "@ant-design/icons";
import axios from "axios";

const BACKEND = process.env.REACT_APP_BACKEND || "/api";
const { Title, Paragraph } = Typography;

// Main component for evaluating normalized sentence boundaries
export function EvaluateBoundariesNormalizedComponent() {
  // State to store the selected files (predicted and actual)
  const [predictedFile, setPredictedFile] = useState<File | null>(null);
  const [actualFile, setActualFile] = useState<File | null>(null);

  // State for the evaluation result and UI feedback
  const [result, setResult] = useState<{
    macro_precision: number;
    macro_recall: number;
    macro_f1: number;
  } | null>(null);

  const [loading, setLoading] = useState(false); // For spinner
  const [error, setError] = useState<string | null>(null); // For error message

  // Helper: handle file selection for Upload component
  const handleFileUpload = (setter: (f: File) => void) => (file: File) => {
    setter(file);
    return false; // Prevent auto-upload, we handle with form submit
  };

  // Handle form submission: send files to backend, handle response
  const handleSubmit = async () => {
    setError(null);
    setResult(null);

    // Validate both files present
    if (!predictedFile || !actualFile) {
      setError("Please upload both the predicted and actual JSON files.");
      return;
    }
    setLoading(true);

    try {
      // Build multipart/form-data payload
      const formData = new FormData();
      formData.append("predicted", predictedFile);
      formData.append("actual", actualFile);

      // POST to backend
      const resp = await axios.post(
        `${BACKEND}/evaluate_boundaries_normalized/`,
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      setResult(resp.data);
      message.success("Evaluation completed successfully!");
    } catch (err: any) {
      setError(
        String(
          err?.response?.data?.detail ||
          err?.message ||
          "Evaluation failed. Please check your files and try again."
        )
      );
    }
    setLoading(false);
  };

  // Table columns for displaying metrics
  const columns = [
    {
      title: "Metric",
      dataIndex: "metric",
      key: "metric",
    },
    {
      title: "Score",
      dataIndex: "score",
      key: "score",
      render: (val: number) => (val !== undefined ? val.toFixed(4) : "-"),
    },
  ];

  // Build table data from result
  const tableData = result
    ? [
        {
          key: "1",
          metric: "Macro Precision",
          score: result.macro_precision,
        },
        {
          key: "2",
          metric: "Macro Recall",
          score: result.macro_recall,
        },
        {
          key: "3",
          metric: "Macro F1",
          score: result.macro_f1,
        },
      ]
    : [];

  // ----------- UI Rendering starts here -----------
  return (
    <div style={{ maxWidth: 600, margin: "0 auto", padding: 24 }}>
      {/* Title and instructions */}
      <Title level={3}>Evaluate Sentence Boundary Predictions (Normalized)</Title>
      <Paragraph>
        Upload your <b>predicted</b> and <b>actual</b> sentence boundary files (in JSON format),
        then click <b>Evaluate</b> to compute macro precision, recall, and F1-score. This tool
        automatically normalizes noisy or variant keys/nodes.
      </Paragraph>

      {/* Main form card - updated with proper dark mode styling */}
      <AntCard 
        style={{ 
          marginBottom: 32,
          background: 'transparent'
        }}
        styles={{
        body: {
          padding: 24,
          background: 'var(--ant-component-background)',
          borderRadius: 8
        }
      }}
      >
        <Form layout="vertical" onFinish={handleSubmit}>
          {/* Upload: Predicted */}
          <Form.Item
            label="Predicted Boundaries (JSON)"
            required
            tooltip="Upload your predicted boundaries file (JSON)"
          >
            <Upload
              accept=".json"
              beforeUpload={handleFileUpload(setPredictedFile)}
              showUploadList={predictedFile ? { showRemoveIcon: false } : false}
              maxCount={1}
              fileList={predictedFile ? [{ uid: "-1", name: predictedFile.name }] : []}
            >
              <Button icon={<UploadOutlined />}>Upload Predicted</Button>
            </Upload>
          </Form.Item>

          {/* Upload: Actual */}
          <Form.Item
            label="Actual Boundaries (Ground Truth, JSON)"
            required
            tooltip="Upload the ground truth boundaries file (JSON)"
          >
            <Upload
              accept=".json"
              beforeUpload={handleFileUpload(setActualFile)}
              showUploadList={actualFile ? { showRemoveIcon: false } : false}
              maxCount={1}
              fileList={actualFile ? [{ uid: "-2", name: actualFile.name }] : []}
            >
              <Button icon={<UploadOutlined />}>Upload Actual</Button>
            </Upload>
          </Form.Item>

          {/* Submit button */}
          <Form.Item>
            <Space>
              <Button
                type="primary"
                htmlType="submit"
                icon={<CheckCircleTwoTone twoToneColor="#52c41a" />}
                loading={loading}
                disabled={loading}
              >
                Evaluate
              </Button>
              {/* Loading spinner */}
              {loading && <Spin />}
            </Space>
          </Form.Item>
        </Form>

        {/* Show error if any */}
        {error && (
          <Alert
            type="error"
            message={String(error)}
            showIcon
            style={{ marginBottom: 16 }}
          />
        )}
      </AntCard>

      {/* Output metrics table */}
      {result && (
        <AntCard 
          title="Evaluation Metrics" 
          style={{ marginTop: 16 }}
          styles={{
          body: {
            padding: 0
          }
        }}
        >
          <Table
            dataSource={tableData}
            columns={columns}
            pagination={false}
            size="middle"
            bordered
          />
        </AntCard>
      )}

      {/* Help: where to get your files */}
      <AntCard
        title="Where do I get the JSON files?"
        style={{ marginTop: 16 }}
        styles={{
        body: {
          padding: 24,
          background: 'var(--ant-component-background)',
          borderRadius: 8
        }
      }}
      >
        <ul>
          <li>
            <b>Predicted Boundaries</b>: This is the output from your boundary prediction model or algorithm, saved as a JSON file.
          </li>
          <li>
            <b>Actual Boundaries (Ground Truth)</b>: This is the labeled/ground-truth data you wish to compare against, also as a JSON file.
          </li>
          <li>
            Both files should be in <code>{"{ ... }"}</code> JSON object format, with keys as sentence/group identifiers and values as lists of nodes/strings or <code>[{"{"}node: ...{"}"}]</code> objects.
          </li>
        </ul>
      </AntCard>
    </div>
  );
}


/*
1. Upload two JSON files:
   - Predicted: output of your model (JSON)
   - Actual: ground truth labels (JSON)
2. Click "Evaluate" to submit to backend endpoint /evaluate_boundaries_normalized/.
3. Results (macro precision, recall, F1) will be shown in a table below.
4. Errors are shown as alerts.
5. You can get the JSON files from your model output, labeling tools, or previous steps in the pipeline.
*/