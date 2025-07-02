"use client";

import React, { useState } from "react";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "./ui/card";
import { Progress } from "./ui/progress";
import { Upload, Button, Alert, Spin, Table, Tag } from "antd";
import { Target, Zap, TrendingUp } from "lucide-react";
import { UploadOutlined } from "@ant-design/icons";
import axios from "axios";


const BACKEND = process.env.REACT_APP_BACKEND || "/api";

export function BoundaryEvaluationComponent() {
  const [goldFile, setGoldFile] = useState<File | null>(null);
  const [outputFile, setOutputFile] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleEvaluate() {
    if (!goldFile || !outputFile) {
      setError("Please select both gold and output files.");
      return;
    }
    setError(null);
    setLoading(true);
    setResult(null);
    try {
      const formData = new FormData();
      formData.append("gold", goldFile);
      formData.append("output", outputFile);
      const resp = await axios.post(`${BACKEND}/evaluate/`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(resp.data);
    } catch (err: any) {
      setError(
        err?.response?.data?.detail ||
          err?.message ||
          "Failed to evaluate. Please check your files and try again."
      );
    } finally {
      setLoading(false);
    }
  }

  // Create metrics for pretty cards if result is available
  const metrics = result
    ? [
        {
          title: "F1 Score",
          value: result.f1 != null ? result.f1.toFixed(4) : "-",
          icon: Target,
          description: "Harmonic mean of precision and recall",
          progress: result.f1 != null ? result.f1 * 100 : 0,
        },
        {
          title: "Precision",
          value: result.precision != null ? result.precision.toFixed(4) : "-",
          icon: Zap,
          description: "True positives / (True positives + False positives)",
          progress: result.precision != null ? result.precision * 100 : 0,
        },
        {
          title: "Recall",
          value: result.recall != null ? result.recall.toFixed(4) : "-",
          icon: TrendingUp,
          description: "True positives / (True positives + False negatives)",
          progress: result.recall != null ? result.recall * 100 : 0,
        },
      ]
    : [];

  // Helper: Show arrays as tables or tags
  function renderArrayTable(title: string, arr: any[], color = "blue") {
    return (
      <Card title={title} bordered={false} style={{ marginBottom: 24 }}>
        {arr && arr.length > 0 ? (
          <Table
            dataSource={arr.map((n: string, idx: number) => ({
              key: idx,
              value: n,
            }))}
            columns={[{ title: "Node", dataIndex: "value", key: "value" }]}
            size="small"
            pagination={false}
          />
        ) : (
          <Tag color={color}>None</Tag>
        )}
      </Card>
    );
  }

  // Dynamic: Show all other (non-array) fields from result
  function renderOtherFields(obj: any) {
    if (!obj) return null;
    const skip = ["f1", "precision", "recall", "missing", "extra"];
    return Object.entries(obj)
      .filter(([k, v]) => !skip.includes(k) && !Array.isArray(v))
      .map(([k, v]) => (
        <div key={k} style={{ fontWeight: 600, marginBottom: 8 }}>
          {k.charAt(0).toUpperCase() + k.slice(1).replace(/_/g, " ")}:{" "}
          <span style={{ fontWeight: 400 }}>{String(v)}</span>
        </div>
      ));
  }

  return (
    <div style={{ maxWidth: 700, margin: "0 auto", padding: 24 }}>
      <Card title="Boundary Prediction Evaluation" bordered>
        <div style={{ display: "flex", gap: 16, marginBottom: 16 }}>
          <Upload
            accept=".json"
            beforeUpload={file => {
              setGoldFile(file);
              return false;
            }}
            showUploadList={false}
          >
            <Button icon={<UploadOutlined />}>
              {goldFile ? `Gold: ${goldFile.name}` : "Select Gold File"}
            </Button>
          </Upload>
          <Upload
            accept=".json"
            beforeUpload={file => {
              setOutputFile(file);
              return false;
            }}
            showUploadList={false}
          >
            <Button icon={<UploadOutlined />}>
              {outputFile ? `Output: ${outputFile.name}` : "Select Output File"}
            </Button>
          </Upload>
          <Button
            type="primary"
            onClick={handleEvaluate}
            disabled={!goldFile || !outputFile}
            loading={loading}
          >
            Evaluate
          </Button>
        </div>

        {error && (
          <Alert type="error" message={error} style={{ marginBottom: 16 }} />
        )}

        {loading && <Spin style={{ marginBottom: 16 }} />}

        {result && (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
              {metrics.map((metric) => {
                const Icon = metric.icon;
                return (
                  <Card key={metric.title}>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">
                        {metric.title}
                      </CardTitle>
                      <Icon className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold mb-1">
                        {metric.value}
                      </div>
                      <CardDescription className="text-xs">
                        {metric.description}
                      </CardDescription>
                      <Progress value={metric.progress} className="mb-2" />
                    </CardContent>
                  </Card>
                );
              })}
            </div>
            <Card style={{ marginTop: 24 }} bordered={false}>
              {renderOtherFields(result)}
            </Card>
            {renderArrayTable("Missing Predictions", result.missing, "red")}
            {renderArrayTable("Extra Predictions", result.extra, "orange")}
          </>
        )}
      </Card>
    </div>
  );
}