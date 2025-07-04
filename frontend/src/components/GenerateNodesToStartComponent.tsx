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
  Space,
  message,
  Modal,
  Tag,
  Tooltip,
  Divider,
} from "antd";
import {
  UploadOutlined,
  FileTextOutlined,
  FilePdfOutlined,
  FileImageOutlined,
  SearchOutlined,
  CopyOutlined,
  DownloadOutlined,
} from "@ant-design/icons";
import axios from "axios";

// Backend API URL (set via .env or fallback to /api)

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL || "/api";
console.log(BACKEND);
const { Title, Paragraph } = Typography;

// Supported file types for upload
const ACCEPTED_FILE_TYPES = [
  ".txt",
  ".pdf",
  ".jpg",
  ".jpeg",
  ".png",
  ".bmp",
  ".tiff",
  ".gif",
];

// CSV export utility
function toCSV(rows: string[]): string {
  // Just one column, each row is a node
  return (
    "subject\n" +
    rows.map((node) => `"${(node ?? "").replace(/"/g, '""')}"`).join("\n")
  );
}

export function GenerateNodesToStartComponent() {
  // --- State for file upload and raw text input ---
  const [file, setFile] = useState<File | null>(null);
  const [rawText, setRawText] = useState<string>("");

  // --- Result, loading, error, CSV modal ---
  const [result, setResult] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [csvModalVisible, setCsvModalVisible] = useState(false);

  // --- Helper: Convert PDF or image to text ---
  const handleFileRead = async (file: File): Promise<string> => {
    // TXT: just read as text
    if (file.type === "text/plain" || file.name.endsWith(".txt")) {
      return await file.text();
    }
    // PDF: Use pdfjs-dist/legacy/build/pdf for compatibility
    if (file.type === "application/pdf" || file.name.endsWith(".pdf")) {
      // @ts-ignore
      const pdfjsLib = await import("pdfjs-dist/legacy/build/pdf");
      // @ts-ignore
      pdfjsLib.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.js`;
      const arrayBuffer = await file.arrayBuffer();
      const data = new Uint8Array(arrayBuffer);
      const pdf = await pdfjsLib.getDocument({ data }).promise;
      let text = "";
      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const content = await page.getTextContent();
        text += content.items.map((item: any) => item.str).join(" ") + "\n";
      }
      return text;
    }
    // Images: OCR via Tesseract.js
    if (
      [
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/bmp",
        "image/tiff",
        "image/gif",
      ].includes(file.type) ||
      ACCEPTED_FILE_TYPES.some((ext) => file.name.endsWith(ext))
    ) {
      const Tesseract = (await import("tesseract.js")).default;
      const blobUrl = URL.createObjectURL(file);
      const {
        data: { text },
      } = await Tesseract.recognize(blobUrl, "eng");
      URL.revokeObjectURL(blobUrl);
      return text;
    }
    throw new Error("Unsupported file type for extraction.");
  };

  // --- Handlers for upload and text input ---
  const handleFileUpload = (file: File) => {
    setFile(file);
    setRawText(""); // Clear textarea if uploading a file
    return false;
  };

  const handleRawTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setRawText(e.target.value);
    setFile(null); // Clear file if typing text
  };

  // --- Main Extraction Handler: Calls the backend API ---
  const handleExtract = async () => {
    setError(null);
    setResult([]);
    setLoading(true);

    try {
      let formData: FormData | null = null;
      let fileText = "";

      if (file) {
        // If it's a txt file, send as-is; otherwise, convert to text and send as raw_text
        if (file.type === "text/plain" || file.name.endsWith(".txt")) {
          formData = new FormData();
          formData.append("text", file, file.name);
        } else {
          fileText = await handleFileRead(file);
          if (!fileText.trim()) throw new Error("No text detected in file.");
        }
      }

      // If using raw text input or fileText (from PDF/image conversion)
      if ((!file && rawText) || (file && fileText)) {
        formData = new FormData();
        formData.append("raw_text", file ? fileText : rawText);
      }

      // POST to backend
      const resp = await axios.post(
        `${BACKEND}/generate_nodes_to_start/`,
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      if (!Array.isArray(resp.data))
        throw new Error("Malformed backend response.");
      setResult(resp.data);
      message.success("Starting nodes generated!");
    } catch (err: any) {
      setError(
        String(
          err?.response?.data?.detail ||
            err?.message ||
            "Extraction failed. Please check your file/text and try again."
        )
      );
    }
    setLoading(false);
  };

  // --- Copy-to-clipboard handler for result ---
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(result.join("\n"));
      message.success("Copied to clipboard!");
    } catch {
      message.error("Copy failed.");
    }
  };

  // --- Download CSV handler ---
  const handleDownloadCSV = () => {
    const csv = toCSV(result);
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "starting_nodes.csv";
    a.click();
    URL.revokeObjectURL(url);
    setCsvModalVisible(false);
  };

    // --- Download TXT handler ---
  const handleDownloadTxt = () => {
    const textContent = result.join("\n");
    const blob = new Blob([textContent], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "start_nodes.txt";
    a.click();
    URL.revokeObjectURL(url);
  };

  // --- Table columns for displaying extracted nodes ---
  const columns = [
    {
      title: "Sentence #",
      dataIndex: "index",
      key: "index",
      width: 100,
      align: "center" as const,
    },
    {
      title: "Subject (Starting Node)",
      dataIndex: "subject",
      key: "subject",
      render: (v: string) => (
        <Tag color="geekblue" style={{ fontSize: 16, padding: "4px 12px" }}>
          {v}
        </Tag>
      ),
    },
  ];

  // --- CURRENT BACKEND LOGIC  ---
  // The backend logic is correct for extracting the subject of the first SVO in each sentence.
  // However, if a sentence does not have any SVO triplets, it will be skipped. This is usually desired.
  // If we want to always emit a subject per sentence (even if no SVO is found), we could try a fallback, but for node boundary tasks, our current logic is the best.

  // --- UI Rendering ---
  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: 24 }}>
      <Title level={2}>Generate Starting Nodes from Text (SVO Subjects)</Title>
      <Paragraph>
        Upload a <b>text, PDF, or image</b> file, or paste your text below. This
        tool extracts the <b>subject</b> of the first SVO (subject-verb-object)
        triplet in each sentence to generate a list of starting nodes. <br />
        Use these nodes as entry points for graph traversal or sentence boundary
        evaluation!
      </Paragraph>
      <AntCard variant="borderless" style={{ marginBottom: 24 }}>
        <Form layout="vertical">
          {/* File upload (txt/pdf/image) */}
          <Form.Item
            label="Upload Text, PDF, or Image"
            extra="Accepted: .txt, .pdf, .jpg, .jpeg, .png, .bmp, .tiff, .gif. For PDF and images, OCR will be performed."
          >
            <Upload
              accept={ACCEPTED_FILE_TYPES.join(",")}
              beforeUpload={handleFileUpload}
              showUploadList={file ? { showRemoveIcon: true } : false}
              maxCount={1}
              onRemove={() => setFile(null)}
              fileList={file ? [{ uid: "-1", name: file.name }] : []}
            >
              <Space>
                <Button icon={<UploadOutlined />}>Upload</Button>
                <Button icon={<FilePdfOutlined />} />
                <Button icon={<FileImageOutlined />} />
                <Button icon={<FileTextOutlined />} />
              </Space>
            </Upload>
            <div style={{ marginTop: 8 }}>
              Or just <b>paste or type</b> any text below:
            </div>
          </Form.Item>
          {/* Raw text input */}
          <Form.Item label="Paste or type text">
            <Input.TextArea
              value={rawText}
              onChange={handleRawTextChange}
              autoSize={{ minRows: 5, maxRows: 12 }}
              allowClear
              placeholder="Paste or type any amount of text here. This input will override file upload if both are used."
              style={{ fontFamily: "monospace" }}
              maxLength={30000}
            />
          </Form.Item>
          {/* Extract Button */}
          <Form.Item>
            <Space>
              <Button
                type="primary"
                icon={<SearchOutlined />}
                onClick={handleExtract}
                loading={loading}
                disabled={loading || (!file && !rawText)}
              >
                Generate Nodes
              </Button>
              {/* Download CSV button, if results present */}
              {result.length > 0 && (
                <Button
                  icon={<DownloadOutlined />}
                  onClick={() => setCsvModalVisible(true)}
                >
                  Download CSV
                </Button>
              )}
              {result.length > 0 && (
                <Tooltip title="Download as text file">
                  <Button 
                    icon={<FileTextOutlined />} 
                    onClick={handleDownloadTxt}
                  >
                    Download TXT
                  </Button>
                </Tooltip>
              )}
              {/* Copy to clipboard */}
              {result.length > 0 && (
                <Tooltip title="Copy all nodes to clipboard">
                  <Button icon={<CopyOutlined />} onClick={handleCopy}>
                    Copy
                  </Button>
                </Tooltip>
              )}
              {loading && <Spin />}
            </Space>
          </Form.Item>
        </Form>
        {/* Error alert */}
        {error && (
          <Alert
            type="error"
            message={String(error)}
            showIcon
            style={{ marginBottom: 16 }}
          />
        )}
      </AntCard>
      {/* Results: Table of starting nodes */}
      {result.length > 0 && (
        <AntCard bordered title="Generated Starting Nodes">
          <Table
            dataSource={result.map((subject, i) => ({
              index: i + 1,
              subject,
              key: i,
            }))}
            columns={columns}
            pagination={{ pageSize: 10, showSizeChanger: true }}
            bordered
          />
          <Divider />
          <Paragraph>
            <b>Total nodes found:</b> <Tag color="green">{result.length}</Tag>
          </Paragraph>
        </AntCard>
      )}
      {/* CSV Download Modal */}
      <Modal
        title="Download Nodes as CSV"
        open={csvModalVisible}
        onOk={handleDownloadCSV}
        onCancel={() => setCsvModalVisible(false)}
        okText="Download"
        cancelText="Cancel"
      >
        <Paragraph>
          Click "Download" to save the starting nodes as a CSV file.
        </Paragraph>
      </Modal>
      {/* Help Section */}
      <AntCard
        type="inner"
        title="How to use this tool"
        style={{ marginTop: 16 }}
      >
        <ul>
          <li>
            <b>Upload a file</b>: Accepts plain text (.txt), PDF, or image
            files. For images/PDFs, OCR is applied automatically.
          </li>
          <li>
            <b>Paste/type text</b>: You can also paste or type any raw text,
            which will override the uploaded file if both are present.
          </li>
          <li>
            <b>Generate Nodes</b>: Click the button to extract SVO subjects from
            each sentence.
          </li>
          <li>
            <b>Download CSV</b>: Save your node list as a CSV file.
          </li>
          <li>
            <b>Copy</b>: Copy all nodes to your clipboard for use in other
            tools.
          </li>
        </ul>
      </AntCard>
    </div>
  );
}

/*
Frontend Usage:
1. Supports .txt, .pdf, images, or direct text input.
2. PDF/image files are converted to text via OCR (Tesseract.js/pdfjs-dist/legacy/build/pdf).
3. Calls /generate_nodes_to_start/ on backend with FormData (file or raw_text).
4. Subjects (starting nodes) are displayed in a table and can be downloaded/copied.
*/

// To use PDF/image OCR, install dependencies:
// npm install pdfjs-dist tesseract.js
