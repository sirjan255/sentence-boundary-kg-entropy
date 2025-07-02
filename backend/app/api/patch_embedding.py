"""
FastAPI API: Patch Embedding Model Training, Visualization, Playground, and Comparison

Features:
- Accepts code patch data as CSV, JSON, or pasted/edited text (no git history needed)
- Lets user specify encoder model, embedding dimension, and training hyperparameters
- Trains a patch autoencoder (using HuggingFace Transformer backbone)
- Returns:
    - Loss curve (training loss vs. epochs, as base64 PNG)
    - Trained encoder (downloadable)
    - Patch embeddings (downloadable)
    - 2D projection (UMAP/t-SNE) plot of embeddings (as base64 PNG)
    - Training log/summary (JSON)
- Provides playground: embed new patches, search for similar patches
- Allows user to compare two embedding model configs (side-by-side loss and UMAP/t-SNE plots)
All in-memory, no git/commit required.

Frontend: POST patch data & params to these endpoints.
"""

import os
import io
import tempfile
import base64
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import umap
from typing import List

router = APIRouter()

# DATA INGESTION

def ingest_patches(data_file: UploadFile = None, pasted: str = None):
    if pasted:
        try:
            # Accept pasted TSV, CSV, or JSON lines (parse liberally)
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
    # Require "patch" column
    if "patch" not in df:
        raise ValueError("Your data must contain a 'patch' column.")
    # Using id as identifier
    if "id" not in df:
        df["id"] = np.arange(len(df))
    return df

# ---- PATCH DATASET ----

class PatchDataset(Dataset):
    def __init__(self, patches: List[str], tokenizer, max_length=128):
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

# ---- AUTOENCODER MODEL ----

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

# ---- TRAINING LOOP ----

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
        if i < 30:  # Label first 30 points
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

# API FOR TRAINING PATCH AUTOENCODER

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
    """
    Train a patch autoencoder and return loss curve, trained encoder, embeddings, UMAP plot, and summary.
    """
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
        # Save encoder only (for downstream use)
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder_path = os.path.join(tmpdir, "encoder.pt")
            torch.save(model.encoder.state_dict(), encoder_path)
            with open(encoder_path, "rb") as f:
                encoder_bytes = f.read()
        # Compute embeddings and UMAP
        embeddings = get_embeddings(model, dataset, device)
        umap_coords = compute_umap(embeddings)
        umap_b64 = plot_umap(umap_coords, df["id"].tolist())
        # Save embeddings as npy
        with tempfile.TemporaryDirectory() as tmpdir:
            npy_path = os.path.join(tmpdir, "patch_embeddings.npy")
            np.save(npy_path, embeddings)
            with open(npy_path, "rb") as f:
                embeddings_bytes = f.read()
        # Training summary
        summary = {
            "encoder_model": encoder_model,
            "embedding_dim": embedding_dim,
            "epochs": epochs,
            "batch_size": batch_size,
            "max_length": max_length,
            "final_loss": float(loss_curve[-1]),
            "loss_curve": loss_curve,
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

# API ENDPOINT FOR EMBEDDING NEW PATCHES

@router.post("/embed_patches/")
async def embed_patches(
    encoder_file: UploadFile = File(...),
    encoder_model: str = Form("bert-base-uncased"),
    patches: UploadFile = File(None),
    pasted: str = Form(None),
    max_length: int = Form(128),
    embedding_dim: int = Form(128),
):
    """
    Return embeddings for new patches using the trained encoder.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with tempfile.TemporaryDirectory() as tmpdir:
            enc_path = os.path.join(tmpdir, "encoder.pt")
            with open(enc_path, "wb") as f:
                f.write(encoder_file.file.read())
            encoder = AutoModel.from_pretrained(encoder_model)
            encoder.load_state_dict(torch.load(enc_path, map_location=device))
        tokenizer = AutoTokenizer.from_pretrained(encoder_model)
        df = ingest_patches(patches, pasted)
        dataset = PatchDataset(list(df["patch"]), tokenizer, max_length)
        embeddings = []
        encoder.eval()
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=1):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
                h = outputs.last_hidden_state[:,0]
                embeddings.append(h.cpu().numpy()[0].tolist())
        return JSONResponse({
            "ids": df["id"].tolist(),
            "embeddings": embeddings
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Patch embedding failed: {str(e)}")

# API ENDPOINT FOR FINDING SIMILAR PATCHES

@router.post("/find_similar_patches/")
async def find_similar_patches(
    embeddings_file: UploadFile = File(...),
    query_embedding: str = Form(...),  # comma-separated floats
    top_k: int = Form(5)
):
    """
    Find top-K similar patches by cosine similarity.
    """
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

# API ENDPOINT FOR COMPARING TWO PATCH EMBEDDING MODELS

@router.post("/compare_patch_embeddings/")
async def compare_patch_embeddings(
    data: UploadFile = File(None),
    pasted: str = Form(None),
    encoder_model1: str = Form("bert-base-uncased"),
    encoder_model2: str = Form("roberta-base"),
    embedding_dim1: int = Form(128),
    embedding_dim2: int = Form(128),
    epochs: int = Form(5),
    batch_size: int = Form(8),
    max_length: int = Form(128)
):
    """
    Train two patch embedding models, return side-by-side loss curves and UMAPs.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        df = ingest_patches(data, pasted)
        tokenizer1 = AutoTokenizer.from_pretrained(encoder_model1)
        tokenizer2 = AutoTokenizer.from_pretrained(encoder_model2)
        patches = list(df["patch"])
        dataset1 = PatchDataset(patches, tokenizer1, max_length)
        dataset2 = PatchDataset(patches, tokenizer2, max_length)
        train_loader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=True)
        train_loader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=True)
        # Model 1
        model1 = PatchAutoencoder(encoder_model1, embedding_dim1).to(device)
        loss_curve1 = train_patch_autoencoder(model1, train_loader1, epochs, device)
        embeddings1 = get_embeddings(model1, dataset1, device)
        umap_coords1 = compute_umap(embeddings1)
        umap_b64_1 = plot_umap(umap_coords1, df["id"].tolist())
        loss_curve_b64_1 = plot_loss_curve(loss_curve1)
        # Model 2
        model2 = PatchAutoencoder(encoder_model2, embedding_dim2).to(device)
        loss_curve2 = train_patch_autoencoder(model2, train_loader2, epochs, device)
        embeddings2 = get_embeddings(model2, dataset2, device)
        umap_coords2 = compute_umap(embeddings2)
        umap_b64_2 = plot_umap(umap_coords2, df["id"].tolist())
        loss_curve_b64_2 = plot_loss_curve(loss_curve2)
        # Return side by side
        return JSONResponse({
            "model1": {
                "encoder_model": encoder_model1,
                "embedding_dim": embedding_dim1,
                "loss_curve_b64": loss_curve_b64_1,
                "umap_b64": umap_b64_1,
                "final_loss": float(loss_curve1[-1])
            },
            "model2": {
                "encoder_model": encoder_model2,
                "embedding_dim": embedding_dim2,
                "loss_curve_b64": loss_curve_b64_2,
                "umap_b64": umap_b64_2,
                "final_loss": float(loss_curve2[-1])
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Patch embedding comparison failed: {str(e)}")