import React, { useState } from "react";
import axios, { AxiosError } from "axios";

const API_URL = "http://127.0.0.1:8000/api";

type PatchEmbeddingResponse = {
  summary: any;
  loss_curve_b64: string;
  umap_b64: string;
  encoder_file: string;
  encoder_file_name: string;
  embeddings_file: string;
  embeddings_file_name: string;
};

export default function PatchEmbeddingApp() {
  const [pastedText, setPastedText] = useState("");
  const [encoderModel, setEncoderModel] = useState("bert-base-uncased");
  const [embeddingDim, setEmbeddingDim] = useState(128);
  const [epochs, setEpochs] = useState(5);
  const [batchSize, setBatchSize] = useState(8);
  const [maxLength, setMaxLength] = useState(128);
  const [response, setResponse] = useState<PatchEmbeddingResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setResponse(null);
    try {
      const formData = new FormData();
      formData.append("pasted", pastedText);
      formData.append("encoder_model", encoderModel);
      formData.append("embedding_dim", embeddingDim.toString());
      formData.append("epochs", epochs.toString());
      formData.append("batch_size", batchSize.toString());
      formData.append("max_length", maxLength.toString());

      const res = await axios.post(
        `${API_URL}/train_patch_embedding/`,
        formData
      );
      setResponse(res.data);
    } catch (err: AxiosError | Error | any) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const downloadFile = (base64: string, fileName: string) => {
    const link = document.createElement("a");
    link.href = `data:application/octet-stream;base64,${base64}`;
    link.download = fileName;
    link.click();
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Patch Embedding Trainer</h1>

      <textarea
        className="w-full border p-2 rounded mb-4 h-40"
        value={pastedText}
        onChange={(e) => setPastedText(e.target.value)}
        placeholder="Paste JSON patch array here"
      />

      <div className="grid grid-cols-2 gap-4 mb-4">
        <input
          type="text"
          className="border p-2"
          placeholder="Encoder Model"
          value={encoderModel}
          onChange={(e) => setEncoderModel(e.target.value)}
        />

        <input
          type="number"
          className="border p-2"
          placeholder="Embedding Dim"
          value={embeddingDim}
          onChange={(e) => setEmbeddingDim(Number(e.target.value))}
        />

        <input
          type="number"
          className="border p-2"
          placeholder="Epochs"
          value={epochs}
          onChange={(e) => setEpochs(Number(e.target.value))}
        />

        <input
          type="number"
          className="border p-2"
          placeholder="Batch Size"
          value={batchSize}
          onChange={(e) => setBatchSize(Number(e.target.value))}
        />

        <input
          type="number"
          className="border p-2"
          placeholder="Max Length"
          value={maxLength}
          onChange={(e) => setMaxLength(Number(e.target.value))}
        />
      </div>

      <button
        className="bg-blue-600 text-white px-4 py-2 rounded"
        onClick={handleSubmit}
        disabled={loading}
      >
        {loading ? "Training..." : "Train Model"}
      </button>

      {error && <div className="text-red-600 mt-4">Error: {error}</div>}

      {response && (
        <div className="mt-6">
          <h2 className="text-xl font-semibold mb-2">Training Summary</h2>
          <pre className="bg-gray-100 p-4 rounded text-sm whitespace-pre-wrap">
            {JSON.stringify(response.summary, null, 2)}
          </pre>

          <h3 className="mt-4 font-semibold">Loss Curve</h3>
          <img
            src={`data:image/png;base64,${response.loss_curve_b64}`}
            alt="Loss Curve"
            className="mb-4"
          />

          <h3 className="font-semibold">UMAP Plot</h3>
          <img
            src={`data:image/png;base64,${response.umap_b64}`}
            alt="UMAP"
            className="mb-4"
          />

          <div className="flex gap-4">
            <button
              className="bg-green-600 text-white px-4 py-2 rounded"
              onClick={() =>
                downloadFile(response.encoder_file, response.encoder_file_name)
              }
            >
              Download Encoder
            </button>

            <button
              className="bg-purple-600 text-white px-4 py-2 rounded"
              onClick={() =>
                downloadFile(
                  response.embeddings_file,
                  response.embeddings_file_name
                )
              }
            >
              Download Embeddings
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
