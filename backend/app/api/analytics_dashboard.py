"""
FastAPI API: Interactive Graph & Patch Analytics Dashboard with Explainable AI

Features:
- Upload patch data (CSV/JSON/paste) and/or pickled NetworkX graphs.
- Train patch embedding models (autoencoder, user-selectable transformer backbone).
- Visualize patch embedding (UMAP/t-SNE), show loss curves.
- Upload and visualize knowledge graphs, show graph statistics (degree, centrality, clusters).
- Patch similarity search, anomaly detection (outlier detection in embedding space).
- Explainable AI: token saliency for patches, show most influential tokens for embedding.
- Interactive endpoints: user can submit patches, find similar, see explanations, explore graph.
- No dependency on git history or local files—everything is user-provided.

Frontend: Calls the endpoints below.
"""
import io
import os
import tempfile
import pickle
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import umap
import networkx as nx
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

router = APIRouter()

# DATA INGESTION

def ingest_patches(data_file: UploadFile = None, pasted: str = None):
    if pasted:
        try:
            if pasted.strip().startswith("["):
                df = pd.read_json(io.StringIO(pasted))
            elif "\t" in pasted:
                df = pd.read_csv(io.StringIO(pasted), sep="\t")
            elif pasted.strip().startswith("{"):
                df = pd.read_json(io.StringIO(pasted), lines=True)
            else:
                df = pd.read_csv(io.StringIO(pasted))
        except Exception as e:
            raise ValueError(f"Could not parse pasted input: {e}")
    elif data_file:
        fn = data_file.filename.lower()
        content = data_file.file.read()
        if fn.endswith(".json"):
            df = pd.read_json(io.BytesIO(content))
        elif fn.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        else:
            raise ValueError("Only .csv, .json supported for file upload.")
    else:
        raise ValueError("No patch data provided.")
    if "patch" not in df:
        raise ValueError("Your data must contain a 'patch' column.")
    if "id" not in df:
        df["id"] = np.arange(len(df))
    return df

# PATCH DATASET

class PatchDataset(Dataset):
    def __init__(self, patches, tokenizer, max_length=128):
        self.encodings = tokenizer(
            list(patches),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}
    def __len__(self):
        return self.encodings['input_ids'].size(0)

# AUTOENCODER MODEL

class PatchAutoencoder(nn.Module):
    def __init__(self, encoder_model_name='bert-base-uncased', embedding_dim=128):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.proj = nn.Linear(hidden_size, embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.encoder.config.vocab_size)
        )
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = outputs.last_hidden_state[:,0]  # [CLS]
        z = self.proj(h)
        recon_logits = self.decoder(z)
        return recon_logits, z

# TRAINING AND EVALUATION

def train_patch_autoencoder(model, train_loader, epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    loss_curve = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = input_ids[:, 0]
            optimizer.zero_grad()
            recon_logits, _ = model(input_ids, attention_mask)
            loss = criterion(recon_logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        loss_curve.append(avg_loss)
    return loss_curve

def plot_loss_curve(loss_curve):
    plt.figure()
    plt.plot(np.arange(1, len(loss_curve)+1), loss_curve, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title("Training Loss Curve")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def compute_umap(embeddings, n_neighbors=10, min_dist=0.1):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    return reducer.fit_transform(embeddings)

def plot_umap(umap_coords, ids):
    plt.figure(figsize=(6,6))
    plt.scatter(umap_coords[:,0], umap_coords[:,1], alpha=0.7)
    for i, pt in enumerate(umap_coords):
        if i < 30:
            plt.text(pt[0], pt[1], str(ids[i]), fontsize=7)
    plt.title("Patch Embeddings (UMAP projection)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def get_embeddings(model, dataset, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=1)
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            _, z = model(input_ids, attention_mask)
            embeddings.append(z.cpu().numpy()[0])
    return np.stack(embeddings)

# EXPLAINABILITY ( TOKEN SALIENCY )

def patch_token_saliency(model, tokenizer, patch_texts, device="cpu"):
    model.eval()
    results = []
    for text in patch_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        for k in inputs:
            inputs[k].requires_grad = True if k == "input_ids" else False
        outputs = model.encoder(**inputs)
        h = outputs.last_hidden_state[:,0]  # [CLS]
        z = model.proj(h)
        out = model.decoder(z)
        pred = out.argmax().unsqueeze(0)
        loss = nn.CrossEntropyLoss()(out, pred)
        loss.backward()
        grads = inputs["input_ids"].grad
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        importances = grads.abs().sum(dim=1).detach().cpu().numpy().tolist() if grads is not None else [0]*len(tokens)
        results.append({"tokens": tokens, "importances": importances})
    return results

# OUTLIER DETECTION ( ANOMALY DETECTION )

def find_outliers(embeddings):
    clf = IsolationForest(random_state=42, contamination=0.05)
    outlier_flags = clf.fit_predict(embeddings)
    return np.where(outlier_flags == -1)[0].tolist()

# GRAPH VISUALIZATION AND STATISTICS

def draw_kg(G):
    plt.figure(figsize=(20, 15))
    pos = nx.spring_layout(G, seed=42, k=0.5)
    nx.draw(
        G, pos, with_labels=True, node_color='skyblue',
        edge_color='gray', node_size=2000, font_size=9, alpha=0.85
    )
    edge_attr = None
    if all('verb' in d for _,_,d in G.edges(data=True)):
        edge_attr = 'verb'
    elif all('label' in d for _,_,d in G.edges(data=True)):
        edge_attr = 'label'
    if edge_attr:
        edge_labels = nx.get_edge_attributes(G, edge_attr)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
    plt.title("Full Knowledge Graph")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_bytes = buf.read()
    return img_bytes

def compute_graph_stats(G):
    stats = {
        "type": "DiGraph" if isinstance(G, nx.DiGraph) else "Graph",
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "degree_centrality": sorted(nx.degree_centrality(G).items(), key=lambda x: -x[1])[:5],
        "clustering_coeff": nx.average_clustering(G) if not isinstance(G, nx.DiGraph) else None,
        "components": nx.number_connected_components(G) if not isinstance(G, nx.DiGraph) else None,
        "density": nx.density(G),
    }
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G.to_undirected())
        stats["communities"] = len(set(partition.values()))
    except Exception:
        stats["communities"] = None
    return stats

# API ENDPOINTS

@router.post("/train_patch_embedding/")
async def train_patch_embedding(
    data: UploadFile = File(None),
    pasted: str = Form(None),
    encoder_model: str = Form("bert-base-uncased"),
    embedding_dim: int = Form(128),
    epochs: int = Form(5),
    batch_size: int = Form(8),
    max_length: int = Form(128),
):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        df = ingest_patches(data, pasted)
        tokenizer = AutoTokenizer.from_pretrained(encoder_model)
        patches = list(df["patch"])
        dataset = PatchDataset(patches, tokenizer, max_length)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = PatchAutoencoder(encoder_model, embedding_dim).to(device)
        loss_curve = train_patch_autoencoder(model, train_loader, epochs, device)
        loss_curve_b64 = plot_loss_curve(loss_curve)
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder_path = os.path.join(tmpdir, "encoder.pt")
            torch.save(model.encoder.state_dict(), encoder_path)
            with open(encoder_path, "rb") as f:
                encoder_bytes = f.read()
        embeddings = get_embeddings(model, dataset, device)
        umap_coords = compute_umap(embeddings)
        umap_b64 = plot_umap(umap_coords, df["id"].tolist())
        with tempfile.TemporaryDirectory() as tmpdir:
            npy_path = os.path.join(tmpdir, "patch_embeddings.npy")
            np.save(npy_path, embeddings)
            with open(npy_path, "rb") as f:
                embeddings_bytes = f.read()
        outlier_indices = find_outliers(embeddings)
        summary = {
            "encoder_model": encoder_model,
            "embedding_dim": embedding_dim,
            "epochs": epochs,
            "batch_size": batch_size,
            "max_length": max_length,
            "final_loss": float(loss_curve[-1]),
            "loss_curve": loss_curve,
            "outliers": outlier_indices
        }
        return JSONResponse({
            "summary": summary,
            "loss_curve_b64": loss_curve_b64,
            "umap_b64": umap_b64,
            "encoder_file": base64.b64encode(encoder_bytes).decode(),
            "encoder_file_name": "encoder.pt",
            "embeddings_file": base64.b64encode(embeddings_bytes).decode(),
            "embeddings_file_name": "patch_embeddings.npy"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Patch embedding training failed: {str(e)}")

@router.post("/patch_explain/")
async def patch_explain(
    encoder_file: UploadFile = File(...),
    encoder_model: str = Form("bert-base-uncased"),
    patches: UploadFile = File(None),
    pasted: str = Form(None),
    max_length: int = Form(128),
    embedding_dim: int = Form(128),
):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with tempfile.TemporaryDirectory() as tmpdir:
            enc_path = os.path.join(tmpdir, "encoder.pt")
            with open(enc_path, "wb") as f:
                f.write(encoder_file.file.read())
            model = PatchAutoencoder(encoder_model, embedding_dim).to(device)
            model.encoder.load_state_dict(torch.load(enc_path, map_location=device))
        tokenizer = AutoTokenizer.from_pretrained(encoder_model)
        df = ingest_patches(patches, pasted)
        patch_texts = list(df["patch"])
        explain = patch_token_saliency(model, tokenizer, patch_texts, device)
        return JSONResponse({"explanations": explain})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explainability failed: {str(e)}")

@router.post("/find_similar_patches/")
async def find_similar_patches(
    embeddings_file: UploadFile = File(...),
    query_embedding: str = Form(...),  # comma-separated floats
    top_k: int = Form(5)
):
    try:
        emb_np = np.load(io.BytesIO(embeddings_file.file.read()))
        query_emb = np.array([float(x) for x in query_embedding.strip().split(",")])
        def cosine(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        sims = [cosine(query_emb, v) for v in emb_np]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return JSONResponse({
            "indices": top_idx.tolist(),
            "similarities": [float(sims[i]) for i in top_idx]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {str(e)}")

@router.post("/visualize_kg/")
async def visualize_kg(
    kg: UploadFile = File(...)
):
    try:
        kg_bytes = await kg.read()
        G = pickle.loads(kg_bytes)
        if not isinstance(G, nx.Graph) and not isinstance(G, nx.DiGraph):
            raise ValueError("Uploaded object is not a NetworkX Graph or DiGraph.")
        img_bytes = draw_kg(G)
        b64 = base64.b64encode(img_bytes).decode()
        stats = compute_graph_stats(G)
        return JSONResponse({
            "image_b64": b64,
            "stats": stats
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"KG visualization failed: {str(e)}")