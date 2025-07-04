"use client";

import React, { useState } from 'react';
import { Upload, Button, Form, Card, Alert, Spin, Typography, Table, Input, Space, message, Modal, Tag, Tooltip, Divider } from 'antd';
import { UploadOutlined, FileTextOutlined, FileExcelOutlined, BarChartOutlined, DownloadOutlined } from '@ant-design/icons';
import axios from 'axios';

const { Title, Text } = Typography;

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL || "/api";

interface TrainingResult {
  accuracy: number;
  plot_b64: string;
  model_id: string;
}

interface PredictionResult {
  predictions: number[];
}

const PatchGATComponent = () => {
  const [form] = Form.useForm();
  const [trainingResult, setTrainingResult] = useState<TrainingResult | null>(null);
  const [predictions, setPredictions] = useState<number[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [nodeFile, setNodeFile] = useState<File | null>(null);
  const [edgeFile, setEdgeFile] = useState<File | null>(null);
  const [predictionNodeFile, setPredictionNodeFile] = useState<File | null>(null);
  const [predictionEdgeFile, setPredictionEdgeFile] = useState<File | null>(null);

  const handleTrain = async (values: any) => {
    setLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      if (nodeFile) formData.append('node_file', nodeFile);
      if (edgeFile) formData.append('edge_file', edgeFile);
      formData.append('feature_cols', JSON.stringify(values.feature_cols.split(',').map(s => s.trim()));
      formData.append('label_col', values.label_col);
      formData.append('window_size', values.window_size.toString());
      formData.append('epochs', values.epochs.toString());
      formData.append('batch_size', values.batch_size.toString());
      formData.append('hidden_dim', values.hidden_dim.toString());

      const response = await axios.post(`${BACKEND}/train_patch_gat`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setTrainingResult(response.data);
      message.success('Model trained successfully!');
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Training failed');
      message.error('Training failed');
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async () => {
    if (!trainingResult?.model_id || !predictionNodeFile || !predictionEdgeFile) {
      message.error('Please train a model and upload prediction files first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('model_id', trainingResult.model_id);
      if (predictionNodeFile) formData.append('node_file', predictionNodeFile);
      if (predictionEdgeFile) formData.append('edge_file', predictionEdgeFile);
      formData.append('feature_cols', JSON.stringify(form.getFieldValue('feature_cols').split(',').map(s => s.trim()));
      formData.append('window_size', form.getFieldValue('window_size').toString());

      const response = await axios.post<PredictionResult>(`${BACKEND}/predict_patch_gat`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setPredictions(response.data.predictions);
      message.success('Predictions generated successfully!');
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Prediction failed');
      message.error('Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: 24 }}>
      <Title level={2}>Graph Patch Classification with GAT</Title>
      <Text type="secondary">
        Train and predict using Graph Attention Networks on graph subgraphs/patches
      </Text>

      <Divider />

      <Card title="Train GAT Model" style={{ marginBottom: 24 }}>
        <Form
          form={form}
          layout="vertical"
          onFinish={handleTrain}
          initialValues={{
            feature_cols: 'feature1,feature2,feature3',
            label_col: 'label',
            window_size: 3,
            epochs: 5,
            batch_size: 16,
            hidden_dim: 32,
          }}
        >
          <Form.Item label="Node Features CSV" name="node_file" rules={[{ required: true }]}>
            <Upload
              accept=".csv"
              beforeUpload={(file) => {
                setNodeFile(file);
                return false;
              }}
              maxCount={1}
              fileList={nodeFile ? [{ uid: 'node', name: nodeFile.name }] : []}
              onRemove={() => setNodeFile(null)}
            >
              <Button icon={<UploadOutlined />}>Upload Node File</Button>
            </Upload>
          </Form.Item>

          <Form.Item label="Edge List CSV" name="edge_file" rules={[{ required: true }]}>
            <Upload
              accept=".csv"
              beforeUpload={(file) => {
                setEdgeFile(file);
                return false;
              }}
              maxCount={1}
              fileList={edgeFile ? [{ uid: 'edge', name: edgeFile.name }] : []}
              onRemove={() => setEdgeFile(null)}
            >
              <Button icon={<UploadOutlined />}>Upload Edge File</Button>
            </Upload>
          </Form.Item>

          <Form.Item
            label="Feature Columns (comma separated)"
            name="feature_cols"
            rules={[{ required: true }]}
          >
            <Input placeholder="feature1,feature2,feature3" />
          </Form.Item>

          <Form.Item label="Label Column" name="label_col" rules={[{ required: true }]}>
            <Input placeholder="label" />
          </Form.Item>

          <Form.Item label="Window Size" name="window_size" rules={[{ required: true }]}>
            <Input type="number" min={2} max={10} />
          </Form.Item>

          <Form.Item label="Epochs" name="epochs" rules={[{ required: true }]}>
            <Input type="number" min={1} max={50} />
          </Form.Item>

          <Form.Item label="Batch Size" name="batch_size" rules={[{ required: true }]}>
            <Input type="number" min={1} max={128} />
          </Form.Item>

          <Form.Item label="Hidden Dimension" name="hidden_dim" rules={[{ required: true }]}>
            <Input type="number" min={8} max={256} />
          </Form.Item>

          <Form.Item>
            <Button type="primary" htmlType="submit" loading={loading}>
              Train Model
            </Button>
          </Form.Item>
        </Form>

        {error && <Alert message={error} type="error" showIcon style={{ marginTop: 16 }} />}

        {trainingResult && (
          <div style={{ marginTop: 24 }}>
            <Card title="Training Results" bordered>
              <p>Accuracy: {(trainingResult.accuracy * 100).toFixed(2)}%</p>
              <p>Model ID: {trainingResult.model_id}</p>
              <img 
                src={`data:image/png;base64,${trainingResult.plot_b64}`} 
                alt="Accuracy Plot"
                style={{ maxWidth: '100%' }}
              />
            </Card>
          </div>
        )}
      </Card>

      <Card title="Make Predictions">
        <Space direction="vertical" style={{ width: '100%' }}>
          <div>
            <Text strong>Prediction Node File:</Text>
            <Upload
              accept=".csv"
              beforeUpload={(file) => {
                setPredictionNodeFile(file);
                return false;
              }}
              maxCount={1}
              fileList={predictionNodeFile ? [{ uid: 'pred_node', name: predictionNodeFile.name }] : []}
              onRemove={() => setPredictionNodeFile(null)}
            >
              <Button icon={<UploadOutlined />}>Upload Node File</Button>
            </Upload>
          </div>

          <div>
            <Text strong>Prediction Edge File:</Text>
            <Upload
              accept=".csv"
              beforeUpload={(file) => {
                setPredictionEdgeFile(file);
                return false;
              }}
              maxCount={1}
              fileList={predictionEdgeFile ? [{ uid: 'pred_edge', name: predictionEdgeFile.name }] : []}
              onRemove={() => setPredictionEdgeFile(null)}
            >
              <Button icon={<UploadOutlined />}>Upload Edge File</Button>
            </Upload>
          </div>

          <Button
            type="primary"
            onClick={handlePredict}
            loading={loading}
            disabled={!trainingResult?.model_id || !predictionNodeFile || !predictionEdgeFile}
            icon={<BarChartOutlined />}
          >
            Generate Predictions
          </Button>

          {predictions.length > 0 && (
            <Card title="Prediction Results" bordered style={{ marginTop: 16 }}>
              <Table
                dataSource={predictions.map((pred, idx) => ({
                  key: idx,
                  patch: `Patch ${idx + 1}`,
                  prediction: pred,
                }))}
                columns={[
                  { title: 'Patch', dataIndex: 'patch', key: 'patch' },
                  { 
                    title: 'Prediction', 
                    dataIndex: 'prediction', 
                    key: 'prediction',
                    render: (val) => (
                      <Tag color={val === 1 ? 'green' : 'red'}>
                        {val === 1 ? 'Boundary' : 'Non-boundary'}
                      </Tag>
                    )
                  },
                ]}
                pagination={{ pageSize: 10 }}
              />
            </Card>
          )}
        </Space>
      </Card>
    </div>
  );
};

export default PatchGATComponent;