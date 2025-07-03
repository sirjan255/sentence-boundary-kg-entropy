import React, { useState } from "react";
import {
  Tabs,
  Upload,
  Button,
  Input,
  Select,
  Form,
  message,
  Spin,
  Table,
  Modal,
  Typography,
  Card,
  Alert,
} from "antd";
import {
  UploadOutlined,
  DownloadOutlined,
  SearchOutlined,
  BarChartOutlined,
  ProjectOutlined,
  InfoCircleOutlined,
} from "@ant-design/icons";
import Papa from "papaparse";
import axios from "axios";
import PatchEmbeddingApp from "./components/PatchEmbedding";

const { TabPane } = Tabs;
const { TextArea } = Input;
const { Title, Paragraph } = Typography;

function arrayBufferToBase64(buffer: ArrayBuffer) {
  let binary = "";
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < bytes.length; i++)
    binary += String.fromCharCode(bytes[i]);
  return window.btoa(binary);
}

const BACKEND = process.env.REACT_APP_BACKEND || "/api";

function App() {
  // STATE
  const [patchFile, setPatchFile] = useState<File | null>(null);
  const [patchPaste, setPatchPaste] = useState<string>("");
  const [kgFile, setKgFile] = useState<File | null>(null);
  const [embedResult, setEmbedResult] = useState<any>(null);
  const [trainLoading, setTrainLoading] = useState(false);
  const [kgImg, setKgImg] = useState<string | null>(null);
  const [kgStats, setKgStats] = useState<any>(null);
  const [explainResult, setExplainResult] = useState<any>(null);
  const [similarResult, setSimilarResult] = useState<any>(null);
  const [queryEmbedding, setQueryEmbedding] = useState<string>("");
  const [embeddingsFile, setEmbeddingsFile] = useState<File | null>(null);
  const [selectedPatch, setSelectedPatch] = useState<string | null>(null);
  const [modalVisible, setModalVisible] = useState(false);
  const [modalTokens, setModalTokens] = useState<any[]>([]);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [kgError, setKgError] = useState<string | null>(null);
  const [kgLoading, setKgLoading] = useState(false);
  const [explainError, setExplainError] = useState<string | null>(null);
  const [explainLoading, setExplainLoading] = useState(false);
  const [similarError, setSimilarError] = useState<string | null>(null);
  const [similarLoading, setSimilarLoading] = useState(false);
  const [patchTaskId, setPatchTaskId] = useState<string | null>(null);
  const [patchPolling, setPatchPolling] = useState(false);
  const [patchPollingError, setPatchPollingError] = useState<string | null>(
    null
  );
  const [lossCurveUrl, setLossCurveUrl] = useState<string | null>(null);
  const [umapUrl, setUmapUrl] = useState<string | null>(null);
  const [encoderUrl, setEncoderUrl] = useState<string | null>(null);
  const [embeddingsUrl, setEmbeddingsUrl] = useState<string | null>(null);
  const [explainTaskId, setExplainTaskId] = useState<string | null>(null);
  const [explainPolling, setExplainPolling] = useState(false);
  const [explainPollingError, setExplainPollingError] = useState<string | null>(
    null
  );

  // UTILS
  const getFormData = (fields: { [k: string]: any }) => {
    const fd = new FormData();
    for (let k in fields) {
      if (fields[k] != null) fd.append(k, fields[k]);
    }
    return fd;
  };

  // HANDLERS

  // 1. Train Patch Embedding
  const handleTrain = async (values: any) => {
    setTrainLoading(true);
    setTrainError(null);
    setPatchTaskId(null);
    setPatchPolling(false);
    setPatchPollingError(null);
    setEmbedResult(null);
    try {
      const fd = getFormData({
        data: patchFile,
        pasted: patchPaste,
        encoder_model: values.encoder_model,
        embedding_dim: values.embedding_dim,
        epochs: values.epochs,
        batch_size: values.batch_size,
        max_length: values.max_length,
      });
      const res = await axios.post(`${BACKEND}/train_patch_embedding/`, fd, {
        headers: { "Content-Type": "multipart/form-data" },
        validateStatus: () => true,
      });
      if (res.status === 202 && res.data.task_id) {
        setPatchTaskId(res.data.task_id);
        setPatchPolling(true);
        pollPatchEmbeddingStatus(res.data.task_id);
      } else {
        setTrainError(
          "Failed to start training: " + (res.data?.detail || res.status)
        );
      }
    } catch (err: any) {
      setTrainError(
        "Training failed: " + (err.response?.data?.detail || err.message)
      );
    }
    setTrainLoading(false);
  };

  const pollPatchEmbeddingStatus = async (taskId: string) => {
    let attempts = 0;
    const poll = async () => {
      try {
        const res = await axios.get(
          `${BACKEND}/train_patch_embedding_status/${taskId}/`
        );
        if (res.data.status === "pending") {
          if (attempts < 120) {
            // up to 10 minutes
            attempts++;
            setTimeout(poll, 5000);
          } else {
            setPatchPolling(false);
            setPatchPollingError("Training timed out. Please try again.");
          }
        } else if (res.data.status === "done") {
          setEmbedResult(res.data);
          setPatchPolling(false);
          setPatchPollingError(null);
          // Set URLs for images and downloads
          setLossCurveUrl(res.data.loss_curve_url);
          setUmapUrl(res.data.umap_url);
          setEncoderUrl(res.data.encoder_url);
          setEmbeddingsUrl(res.data.embeddings_url);
        } else if (res.data.status === "error") {
          setPatchPolling(false);
          setPatchPollingError("Training failed: " + res.data.error);
        } else {
          setPatchPolling(false);
          setPatchPollingError("Unknown job status.");
        }
      } catch (err: any) {
        setPatchPolling(false);
        setPatchPollingError(
          "Failed to poll job status: " + (err.message || "Unknown error")
        );
      }
    };
    poll();
  };

  // 2. Visualize KG
  const handleKgVisualize = async () => {
    if (!kgFile) {
      setKgError("Upload a KG file.");
      return;
    }
    setKgLoading(true);
    setKgError(null);
    const fd = new FormData();
    fd.append("kg", kgFile);
    try {
      const res = await axios.post(`${BACKEND}/visualize_kg/`, fd, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setKgImg(res.data.image_b64);
      setKgStats(res.data.stats);
    } catch (err: any) {
      setKgError(
        "KG visualization failed: " +
          (err.response?.data?.detail || err.message)
      );
      message.error(
        "KG visualization failed: " +
          (err.response?.data?.detail || err.message)
      );
    }
    setKgLoading(false);
  };

  // 3. Patch Explain (token saliency)
  const handleExplain = async () => {
    // Use model_task_id if available
    if (patchTaskId && embedResult?.summary?.encoder_model) {
      if (!patchFile && !patchPaste) {
        setExplainError("Provide patch data.");
        message.error("Provide patch data.");
        return;
      }
      setExplainLoading(true);
      setExplainError(null);
      setExplainTaskId(null);
      setExplainPolling(false);
      setExplainPollingError(null);
      setExplainResult(null);
      try {
        const fd = getFormData({
          model_task_id: patchTaskId,
          model_name: embedResult.summary.encoder_model,
          pasted: patchPaste,
          patches: patchFile,
        });
        const res = await axios.post(`${BACKEND}/patch_explain/`, fd, {
          headers: { "Content-Type": "multipart/form-data" },
          validateStatus: () => true,
        });
        if (res.status === 202 && res.data.task_id) {
          setExplainTaskId(res.data.task_id);
          setExplainPolling(true);
          pollPatchExplainStatus(res.data.task_id);
        } else {
          setExplainError(
            "Failed to start explainability: " +
              (res.data?.detail || res.status)
          );
        }
      } catch (err: any) {
        setExplainError(
          "Explainability failed: " + (err.message || "Unknown error")
        );
      }
      setExplainLoading(false);
      return;
    }
    // Fallback: legacy file upload (should rarely be needed)
    if (!encoderUrl) {
      setExplainError("Train a model first.");
      message.error("Train a model first.");
      return;
    }
    if (!patchFile && !patchPaste) {
      setExplainError("Provide patch data.");
      message.error("Provide patch data.");
      return;
    }
    setExplainLoading(true);
    setExplainError(null);
    setExplainTaskId(null);
    setExplainPolling(false);
    setExplainPollingError(null);
    setExplainResult(null);
    try {
      const encoderRes = await fetch(BACKEND + encoderUrl);
      if (!encoderRes.ok) throw new Error("Failed to fetch encoder file");
      const encoderBlob = await encoderRes.blob();
      const encoderFile = new File([encoderBlob], "encoder.pt", {
        type: "application/octet-stream",
      });
      const fd = getFormData({
        model_file: encoderFile,
        model_name: embedResult?.summary?.encoder_model || "bert-base-uncased",
        pasted: patchPaste,
        patches: patchFile,
      });
      const res = await axios.post(`${BACKEND}/patch_explain/`, fd, {
        headers: { "Content-Type": "multipart/form-data" },
        validateStatus: () => true,
      });
      if (res.status === 202 && res.data.task_id) {
        setExplainTaskId(res.data.task_id);
        setExplainPolling(true);
        pollPatchExplainStatus(res.data.task_id);
      } else {
        setExplainError(
          "Failed to start explainability: " + (res.data?.detail || res.status)
        );
      }
    } catch (err: any) {
      setExplainError(
        "Explainability failed: " + (err.message || "Unknown error")
      );
    }
    setExplainLoading(false);
  };

  const pollPatchExplainStatus = async (taskId: string) => {
    let attempts = 0;
    const poll = async () => {
      try {
        const res = await axios.get(
          `${BACKEND}/patch_explain_status/${taskId}/`
        );
        if (res.data.status === "pending") {
          if (attempts < 120) {
            attempts++;
            setTimeout(poll, 5000);
          } else {
            setExplainPolling(false);
            setExplainPollingError(
              "Explainability timed out. Please try again."
            );
          }
        } else if (res.data.status === "done" && res.data.result_url) {
          // Robustly handle /api/ prefix
          let url = res.data.result_url;
          if (!url.startsWith("/api/")) url = `/api${url}`;
          const resultRes = await fetch(url);
          if (!resultRes.ok) throw new Error("Failed to fetch explanations");
          const resultJson = await resultRes.json();
          setExplainResult(resultJson.explanations);
          setExplainPolling(false);
          setExplainPollingError(null);
        } else if (res.data.status === "error") {
          setExplainPolling(false);
          setExplainPollingError("Explainability failed: " + res.data.error);
        } else {
          setExplainPolling(false);
          setExplainPollingError("Unknown job status.");
        }
      } catch (err: any) {
        setExplainPolling(false);
        setExplainPollingError(
          "Failed to poll job status: " + (err.message || "Unknown error")
        );
      }
    };
    poll();
  };

  // 4. Similar Patch Search
  const handleSimilar = async () => {
    if (!embeddingsFile || !queryEmbedding) {
      setSimilarError("Provide embeddings file and query embedding.");
      message.error("Provide embeddings file and query embedding.");
      return;
    }
    setSimilarLoading(true);
    setSimilarError(null);
    const fd = getFormData({
      embeddings_file: embeddingsFile,
      query_embedding: queryEmbedding,
    });
    try {
      const res = await axios.post(`${BACKEND}/find_similar_patches/`, fd, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setSimilarResult(res.data);
      message.success("Similarity search complete.");
    } catch (err: any) {
      setSimilarError(
        "Similarity search failed: " +
          (err.response?.data?.detail || err.message)
      );
      message.error(
        "Similarity search failed: " +
          (err.response?.data?.detail || err.message)
      );
    }
    setSimilarLoading(false);
  };

  // 5. Embedding/Model download
  const handleDownload = (b64: string, fname: string) => {
    const blob = b64toBlob(b64, "application/octet-stream", fname);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = fname;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 500);
  };

  const handleDownloadUrl = (url: string, fname: string) => {
    fetch(url)
      .then((res) => res.blob())
      .then((blob) => {
        const urlObj = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = urlObj;
        a.download = fname;
        a.click();
        setTimeout(() => URL.revokeObjectURL(urlObj), 500);
      })
      .catch(() => message.error("Failed to download file."));
  };

  // 6. Patch File Upload
  const handlePatchFile = (file: File) => {
    setPatchFile(file);
    return false;
  };
  const handleEmbeddingsFile = (file: File) => {
    setEmbeddingsFile(file);
    return false;
  };
  const handleKgFile = (file: File) => {
    setKgFile(file);
    return false;
  };

  // Helper: base64 to Blob
  function b64toBlob(b64: string, mime: string, fname: string) {
    const byteStr = atob(b64);
    const ab = new ArrayBuffer(byteStr.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteStr.length; i++) ia[i] = byteStr.charCodeAt(i);
    return new File([ab], fname, { type: mime });
  }

  // TABLE COLUMNS
  const explainColumns = [
    { title: "Patch #", dataIndex: "idx", key: "idx", width: 80 },
    {
      title: "Top Tokens",
      dataIndex: "tokens",
      key: "tokens",
      render: (_: any, row: any) => (
        <Button
          size="small"
          onClick={() => {
            setModalTokens(
              row.tokens.map((t: string, i: number) => ({
                token: t,
                importance: row.importances[i],
              }))
            );
            setModalVisible(true);
          }}
        >
          Show
        </Button>
      ),
    },
  ];

  // UI

  // Fix for antd Upload showUploadList prop:
  // Instead of false | [{ name }], use: showUploadList={patchFile ? { showRemoveIcon: false } : false}
  // To show only the file name without remove icon (since you handle state manually).
  // Similar for other Upload components.

  return (
    <div style={{ maxWidth: 1100, margin: "auto", padding: 20 }}>
      <Title>
        Interactive Analytics Dashboard: Patches & Knowledge Graphs{" "}
        <BarChartOutlined />
      </Title>
      <Tabs defaultActiveKey="1" type="card">
        {/* TAB 1: PATCH EMBEDDING */}
        <TabPane
          tab={
            <span>
              <ProjectOutlined />
              &nbsp; Patch Embedding
            </span>
          }
          key="1"
        >
          {trainError && (
            <Alert
              type="error"
              message={trainError}
              style={{ marginBottom: 16 }}
            />
          )}
          {patchPollingError && (
            <Alert
              type="error"
              message={patchPollingError}
              style={{ marginBottom: 16 }}
            />
          )}
          <Spin
            spinning={trainLoading || patchPolling}
            tip={patchPolling ? "Training in progress..." : "Training model..."}
          >
            <Form
              layout="vertical"
              onFinish={handleTrain}
              initialValues={{
                encoder_model: "bert-base-uncased",
                embedding_dim: 128,
                epochs: 5,
                batch_size: 8,
                max_length: 128,
              }}
            >
              <Form.Item
                label="Upload Patch Data (.csv/.json)"
                valuePropName="file"
              >
                <Upload
                  beforeUpload={handlePatchFile}
                  showUploadList={patchFile ? { showRemoveIcon: false } : false}
                  maxCount={1}
                  fileList={
                    patchFile ? [{ uid: "-1", name: patchFile.name }] : []
                  }
                >
                  <Button icon={<UploadOutlined />}>Upload Patch File</Button>
                </Upload>
              </Form.Item>
              <Form.Item label="Or Paste/Edit Patches (CSV/JSON/TSV)">
                <TextArea
                  rows={4}
                  value={patchPaste}
                  onChange={(e) => setPatchPaste(e.target.value)}
                  placeholder="Paste CSV, JSON, or TSV here"
                />
              </Form.Item>
              <Form.Item label="Encoder Model" name="encoder_model">
                <Select>
                  <Select.Option value="bert-base-uncased">
                    bert-base-uncased
                  </Select.Option>
                  <Select.Option value="roberta-base">
                    roberta-base
                  </Select.Option>
                  <Select.Option value="distilbert-base-uncased">
                    distilbert-base-uncased
                  </Select.Option>
                </Select>
              </Form.Item>
              <Form.Item label="Embedding Dimension" name="embedding_dim">
                <Input type="number" min={16} max={768} />
              </Form.Item>
              <Form.Item label="Epochs" name="epochs">
                <Input type="number" min={1} max={25} />
              </Form.Item>
              <Form.Item label="Batch Size" name="batch_size">
                <Input type="number" min={1} max={64} />
              </Form.Item>
              <Form.Item label="Max Patch Length" name="max_length">
                <Input type="number" min={32} max={512} />
              </Form.Item>
              <Form.Item>
                <Button type="primary" htmlType="submit" loading={trainLoading}>
                  Train Model
                </Button>
              </Form.Item>
            </Form>
            {patchPolling && (
              <Alert
                type="info"
                message="Training in progress. This may take several minutes. Please do not close the tab."
                showIcon
                style={{ marginTop: 16 }}
              />
            )}
            {embedResult && (
              <div style={{ marginTop: 24 }}>
                <Card
                  title="Training Summary"
                  bordered
                  style={{ marginBottom: 24 }}
                >
                  <p>
                    <b>Model:</b> {embedResult.summary.encoder_model}
                  </p>
                  <p>
                    <b>Embedding Dim:</b> {embedResult.summary.embedding_dim}
                  </p>
                  <p>
                    <b>Epochs:</b> {embedResult.summary.epochs}
                  </p>
                  <p>
                    <b>Batch Size:</b> {embedResult.summary.batch_size}
                  </p>
                  <p>
                    <b>Max Length:</b> {embedResult.summary.max_length}
                  </p>
                  <p>
                    <b>Final Loss:</b>{" "}
                    {embedResult.summary.final_loss?.toFixed(4)}
                  </p>
                  <p>
                    <b>Outlier Patch Indices:</b>{" "}
                    {embedResult.summary.outliers?.join(", ") || "None"}
                  </p>
                </Card>
                <div
                  style={{ display: "flex", flexDirection: "column", gap: 32 }}
                >
                  <Card
                    title="Loss Curve"
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    {lossCurveUrl && (
                      <img
                        src={BACKEND + lossCurveUrl}
                        alt="Loss Curve"
                        style={{ border: "1px solid #ddd" }}
                      />
                    )}
                  </Card>
                  <Card
                    title="UMAP Embedding"
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    {umapUrl && (
                      <img
                        src={BACKEND + umapUrl}
                        alt="UMAP"
                        style={{ border: "1px solid #ddd" }}
                      />
                    )}
                  </Card>
                </div>
                <Card
                  title="Downloads"
                  style={{
                    marginTop: 32,
                    display: "flex",
                    flexDirection: "column",
                    gap: 16,
                  }}
                >
                  <Button
                    icon={<DownloadOutlined />}
                    style={{ marginBottom: 8, width: "100%" }}
                    onClick={() =>
                      encoderUrl &&
                      handleDownloadUrl(BACKEND + encoderUrl, "encoder.pt")
                    }
                    disabled={!encoderUrl}
                  >
                    Download Encoder
                  </Button>
                  <Button
                    icon={<DownloadOutlined />}
                    style={{ width: "100%" }}
                    onClick={() =>
                      embeddingsUrl &&
                      handleDownloadUrl(
                        BACKEND + embeddingsUrl,
                        "patch_embeddings.npy"
                      )
                    }
                    disabled={!embeddingsUrl}
                  >
                    Download Embeddings
                  </Button>
                </Card>
              </div>
            )}
          </Spin>
        </TabPane>

        {/* TAB 2: PATCH EXPLAINABILITY */}
        <TabPane
          tab={
            <span>
              <BarChartOutlined />
              &nbsp; Explainability
            </span>
          }
          key="2"
        >
          {explainError && (
            <Alert
              type="error"
              message={explainError}
              style={{ marginBottom: 16 }}
            />
          )}
          {explainPollingError && (
            <Alert
              type="error"
              message={explainPollingError}
              style={{ marginBottom: 16 }}
            />
          )}
          <Spin
            spinning={explainLoading || explainPolling}
            tip={explainPolling ? "Computing token saliency..." : ""}
          >
            <Paragraph>
              Highlight which tokens in a patch embedding contributed most.
            </Paragraph>
            <Button
              onClick={handleExplain}
              type="primary"
              style={{ marginBottom: 16 }}
            >
              Compute Token Saliency
            </Button>
            {explainPolling && (
              <Alert
                type="info"
                message="Explainability in progress. This may take several minutes. Please do not close the tab."
                showIcon
                style={{ marginTop: 16 }}
              />
            )}
            {explainResult && (
              <Card title="Token Saliency Explanations" bordered>
                <Table
                  dataSource={explainResult.map((row: any, idx: number) => ({
                    ...row,
                    idx,
                  }))}
                  columns={explainColumns}
                  rowKey="idx"
                  pagination={false}
                  style={{ marginTop: 16, maxWidth: 700 }}
                />
              </Card>
            )}
            <Modal
              open={modalVisible}
              title="Token Importances"
              onCancel={() => setModalVisible(false)}
              footer={null}
            >
              <div style={{ fontFamily: "monospace", fontSize: 16 }}>
                {modalTokens.map((t, i) => (
                  <span
                    key={i}
                    style={{
                      background: `rgba(255,0,0,${Math.min(
                        0.8,
                        t.importance /
                          Math.max(...modalTokens.map((x) => x.importance))
                      )})`,
                      marginRight: 4,
                      padding: "2px 4px",
                      borderRadius: 3,
                      color: "#222",
                    }}
                  >
                    {t.token}
                  </span>
                ))}
              </div>
            </Modal>
          </Spin>
        </TabPane>

        {/* TAB 3: PATCH SIMILARITY */}
        <TabPane
          tab={
            <span>
              <SearchOutlined />
              &nbsp; Patch Similarity
            </span>
          }
          key="3"
        >
          {similarError && (
            <Alert
              type="error"
              message={similarError}
              style={{ marginBottom: 16 }}
            />
          )}
          <Spin spinning={similarLoading} tip="Finding similar patches...">
            <Paragraph>
              Find similar patches using an embedding vector.
            </Paragraph>
            <Upload
              beforeUpload={handleEmbeddingsFile}
              showUploadList={
                embeddingsFile ? { showRemoveIcon: false } : false
              }
              maxCount={1}
              fileList={
                embeddingsFile ? [{ uid: "-1", name: embeddingsFile.name }] : []
              }
            >
              <Button icon={<UploadOutlined />}>
                Upload Embeddings (.npy)
              </Button>
            </Upload>
            <Input.TextArea
              rows={1}
              value={queryEmbedding}
              onChange={(e) => setQueryEmbedding(e.target.value)}
              placeholder="Paste query embedding (comma separated floats)"
              style={{ margin: "12px 0" }}
            />
            <Button
              type="primary"
              onClick={handleSimilar}
              style={{ marginBottom: 16 }}
            >
              Find Similar
            </Button>
            {similarResult && (
              <Card title="Similarity Results" bordered>
                <Table
                  dataSource={similarResult.indices.map(
                    (idx: number, i: number) => ({
                      idx,
                      similarity: similarResult.similarities[i],
                    })
                  )}
                  columns={[
                    { title: "Patch Index", dataIndex: "idx", key: "idx" },
                    {
                      title: "Cosine Similarity",
                      dataIndex: "similarity",
                      key: "similarity",
                      render: (val: number) => val.toFixed(4),
                    },
                  ]}
                  rowKey="idx"
                  pagination={false}
                  style={{ maxWidth: 400, marginTop: 16 }}
                />
              </Card>
            )}
          </Spin>
        </TabPane>

        {/* TAB 4: KNOWLEDGE GRAPH */}
        <TabPane
          tab={
            <span>
              <BarChartOutlined />
              &nbsp; Knowledge Graph
            </span>
          }
          key="4"
        >
          {kgError && (
            <Alert
              type="error"
              message={kgError}
              style={{ marginBottom: 16 }}
            />
          )}
          <Spin spinning={kgLoading} tip="Visualizing knowledge graph...">
            <Paragraph>
              Visualize a knowledge graph (pickled NetworkX .pkl file).
            </Paragraph>
            <Upload
              beforeUpload={handleKgFile}
              showUploadList={kgFile ? { showRemoveIcon: false } : false}
              maxCount={1}
              fileList={kgFile ? [{ uid: "-1", name: kgFile.name }] : []}
            >
              <Button icon={<UploadOutlined />}>Upload KG (.pkl)</Button>
            </Upload>
            <Button
              type="primary"
              onClick={handleKgVisualize}
              style={{ marginLeft: 14, marginBottom: 16 }}
            >
              Visualize
            </Button>
            {kgImg && (
              <Card
                title="Knowledge Graph Visualization"
                bordered
                style={{ marginTop: 16 }}
              >
                <img
                  src={`data:image/png;base64,${kgImg}`}
                  alt="KG"
                  style={{ width: 600, border: "1px solid #ddd" }}
                />
              </Card>
            )}
            {kgStats && (
              <Card title="Graph Stats" bordered style={{ marginTop: 12 }}>
                <pre style={{ background: "#f5f5f5", padding: 8 }}>
                  {JSON.stringify(kgStats, null, 2)}
                </pre>
              </Card>
            )}
          </Spin>
        </TabPane>

        {/* TAB 5: ABOUT */}
        <TabPane
          tab={
            <span>
              <InfoCircleOutlined />
              &nbsp; About
            </span>
          }
          key="5"
        >
          <Paragraph>
            <b>Interactive Analytics Dashboard</b> for patch and knowledge graph
            data. Upload your patch/graph files, train embeddings, analyze, and
            explore with explainability and interactive visualizations.
          </Paragraph>
          <Paragraph>
            <b>Features:</b> Patch embedding training, loss/UMAP visualization,
            patch similarity, anomaly/outlier detection, token-level
            explainability, and knowledge graph stats/visuals.
          </Paragraph>
          <Paragraph>
            <b>Backend:</b> HuggingFace Transformers, NetworkX, UMAP,
            IsolationForest, FastAPI.
            <br />
            <b>Frontend:</b> React + Ant Design.
          </Paragraph>
        </TabPane>
      </Tabs>
    </div>
  );
}

export default App;
