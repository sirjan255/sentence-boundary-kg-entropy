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
} from "antd";
import {
  UploadOutlined,
  FilePdfOutlined,
  FileImageOutlined,
  FileTextOutlined,
  CopyOutlined,
  DownloadOutlined,
  SearchOutlined,
} from "@ant-design/icons";
import axios from "axios";

// Backend URL
const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL || "/api";
const { Title, Paragraph } = Typography;

// File types supported for upload
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

// CSV export utility function
function toCSV(rows: any[]) {
  if (!rows.length) return "";
  const keys = Object.keys(rows[0]);
  const csvRows = [keys.join(",")];
  for (const row of rows) {
    csvRows.push(
      keys.map((k) => `"${(row[k] ?? "").replace(/"/g, '""')}"`).join(",")
    );
  }
  return csvRows.join("\n");
}

export function ExtractSVOComponent() {
  // File state
  const [file, setFile] = useState<File | null>(null);
  // For pasted or typed text
  const [rawText, setRawText] = useState<string>("");
  // Loading, error, and result
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any[]>([]);
  // For CSV download
  const [csvModalVisible, setCsvModalVisible] = useState(false);

  // Helper: Convert various file types to text, with fallback for images/pdf
  const handleFileRead = async (file: File): Promise<string> => {
    // TXT: just read as text
    if (file.type === "text/plain" || file.name.endsWith(".txt")) {
      return await file.text();
    }
    // PDF: Use pdfjs-dist/legacy/build/pdf to avoid Buffer error and allow dynamic import
    if (file.type === "application/pdf" || file.name.endsWith(".pdf")) {
      // @ts-ignore
      const pdfjsLib = await import("pdfjs-dist/legacy/build/pdf");
      // @ts-ignore
      pdfjsLib.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.js`;
      const arrayBuffer = await file.arrayBuffer();
      // Convert ArrayBuffer to Uint8Array for pdfjs
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
    // Images: Use Tesseract.js for OCR
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
      // Tesseract expects base64, URL, or HTMLImageElement, so create a blob URL
      const blobUrl = URL.createObjectURL(file);
      const {
        data: { text },
      } = await Tesseract.recognize(blobUrl, "eng");
      URL.revokeObjectURL(blobUrl);
      return text;
    }
    throw new Error("Unsupported file type for SVO extraction.");
  };

  // File upload handler (prevents auto-upload)
  const handleFileUpload = (file: File) => {
    setFile(file);
    setRawText(""); // clear any textarea input
    return false;
  };

  // Handler for text input (typed/pasted)
  const handleRawTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setRawText(e.target.value);
    setFile(null); // clear file if typing
  };

  // Handler for SVO extraction
  const handleExtract = async () => {
    setError(null);
    setResult([]);
    setLoading(true);

    try {
      let formData: FormData | null = null;
      let dataSource: "file" | "raw" = "raw";
      let fileText = "";

      if (file) {
        // Convert file to text if needed (images/pdf)
        if (file.type === "text/plain" || file.name.endsWith(".txt")) {
          formData = new FormData();
          formData.append("text", file, file.name);
          dataSource = "file";
        } else {
          // Convert image/pdf to text, then use raw_text
          fileText = await handleFileRead(file);
          if (!fileText.trim()) throw new Error("No text detected in file.");
        }
      }

      // If using raw text input or image/pdf conversion
      if ((!file && rawText) || (file && !formData)) {
        formData = new FormData();
        formData.append("raw_text", file ? fileText : rawText);
        dataSource = "raw";
      }

      // Backend API call
      const resp = await axios.post(`${BACKEND}/extract_svo/`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (!Array.isArray(resp.data))
        throw new Error("Malformed backend response.");
      setResult(resp.data);
      message.success("SVO extraction complete!");
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

  // SVO Table columns
  const columns = [
    { title: "Sentence", dataIndex: "sentence", key: "sentence", width: 250 },
    { title: "Subject", dataIndex: "subject", key: "subject", width: 120 },
    { title: "Verb", dataIndex: "verb", key: "verb", width: 120 },
    { title: "Object", dataIndex: "object", key: "object", width: 120 },
  ];

  // Download CSV handler
  const handleDownloadCSV = () => {
    const csv = toCSV(result);
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "svo_triplets.csv";
    a.click();
    URL.revokeObjectURL(url);
    setCsvModalVisible(false);
  };

  // UI rendering
  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: 24 }}>
      <Title level={2}>Extract SVO Triplets from Any Text</Title>
      <Paragraph>
        Upload <b>any text, PDF, or image</b> file, or paste text below. SVO
        (Subject-Verb-Object) triplets will be extracted using spaCy on the
        backend. You can also download the results as CSV.
      </Paragraph>
      <AntCard bordered style={{ marginBottom: 24 }}>
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
                Extract SVO
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
      {/* Results: Table of triplets */}
      {result.length > 0 && (
        <AntCard bordered title="Extracted SVO Triplets">
          <Table
            dataSource={result.map((row, i) => ({ ...row, key: i }))}
            columns={columns}
            pagination={{ pageSize: 10, showSizeChanger: true }}
            scroll={{ x: 800 }}
            bordered
          />
        </AntCard>
      )}
      {/* CSV Download Modal */}
      <Modal
        title="Download SVO Triplets as CSV"
        open={csvModalVisible}
        onOk={handleDownloadCSV}
        onCancel={() => setCsvModalVisible(false)}
        okText="Download"
        cancelText="Cancel"
      >
        <Paragraph>
          Click "Download" to save the extracted SVO triplets as a CSV file.
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
            <b>Extract</b>: Click "Extract SVO" to run extraction via the
            backend API.
          </li>
          <li>
            <b>Download CSV</b>: After extraction, download your SVO results for
            further analysis.
          </li>
        </ul>
      </AntCard>
    </div>
  );
}

/*
How to use this component:
1. User can upload .txt, .pdf, or image files (jpg/png/gif/bmp/tiff), or paste/type raw text.
2. PDF/image files are converted to text using OCR (Tesseract.js/pdfjs-dist/legacy/build/pdf).
3. On "Extract SVO", the backend /extract_svo/ endpoint is called with either the file or raw text.
4. Results are displayed in a table and can be downloaded as CSV.
*/

// NPM dependencies you must install for this to work:
// npm install pdfjs-dist tesseract.js
