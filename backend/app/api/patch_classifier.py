"""
FastAPI API: Train and Evaluate Patch Classifiers/Embeddings

Features:
- Accept patch data as CSV or JSON from frontend (no git history needed)
- Train regression or classification models (HuggingFace Transformers)
- Return evaluation metrics (regression/classification), including regression scatter plot (base64 PNG)
- Model explainability: return per-token saliency for input patches
- Allow user to download the trained model and a training summary file
- Compare two models: train and evaluate two configs, return side-by-side metrics and plots
- All in-memory; allows user to run multiple experiments, download results, view explanations

Frontend:
- POST patch data & params to endpoints below
"""

import io
import base64
import pandas as pd
import tempfile
import torch
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import uuid
import logging
import threading
import os
import json

router = APIRouter()

# UTILS


def read_table(upload: UploadFile):
    filename = upload.filename or ""
    content = upload.file.read()
    if filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
    elif filename.endswith(".json"):
        df = pd.read_json(io.BytesIO(content))
    else:
        raise ValueError("Only .csv or .json supported for patch data.")
    return df


def make_regression_plot(y_true, y_pred):
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot(
        [min(y_true), max(y_true)],
        [min(y_true), max(y_true)],
        color="red",
        linestyle="--",
        lw=2,
    )
    plt.xlabel("True label")
    plt.ylabel("Predicted")
    plt.title("Regression Scatter Plot")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def patch_token_saliency(model, tokenizer, patch_texts, device="cpu"):
    """
    Use gradient-based saliency for regression/classification:
    Returns: List of dicts per patch: {tokens, importances}
    """
    model.eval()
    results = []
    for text in patch_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["labels"] = torch.zeros(
            1,
            dtype=(
                torch.float if model.config.problem_type == "regression" else torch.long
            ),
        ).to(device)
        for k in inputs:
            inputs[k].requires_grad = True if k == "input_ids" else False
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        # Get gradients w.r.t. input_ids
        grads = inputs["input_ids"].grad
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        # Use abs(grad) as importance
        importances = (
            grads.abs().sum(dim=1).detach().cpu().numpy().tolist()
            if grads is not None
            else [0] * len(tokens)
        )
        results.append({"tokens": tokens, "importances": importances})
    return results


# DATASET CLASS
# This class handles tokenization and prepares the dataset for training


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# MAIN TRAINING ENDPOINT


@router.post("/train_patch_classifier/")
async def train_patch_classifier(
    data: UploadFile = File(...),
    model_name: str = Form("bert-base-uncased"),
    label_column: str = Form("label"),
    patch_column: str = Form("patch"),
    task_type: str = Form("regression"),
    epochs: int = Form(2),
    batch_size: int = Form(8),
    learning_rate: float = Form(5e-5),
    max_length: int = Form(256),
):
    """
    Train a patch classifier/regressor on user-uploaded CSV/JSON.
    Returns: metrics, scatter plot (regression), downloadable model, logs.
    """
    try:
        # Data
        df = read_table(data)
        if patch_column not in df or label_column not in df:
            raise ValueError(
                "Patch and label columns must be present in the uploaded file."
            )
        texts = list(df[patch_column])
        labels = list(df[label_column])
        if task_type == "regression":
            labels = [float(x) for x in labels]
        else:
            # Try to infer integer labels if not already
            try:
                labels = [int(x) for x in labels]
            except:
                unique = sorted(list(set(labels)))
                lookup = {v: i for i, v in enumerate(unique)}
                labels = [lookup[x] for x in labels]

        # Split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_dataset = PatchDataset(train_texts, train_labels, tokenizer, max_length)
        val_dataset = PatchDataset(val_texts, val_labels, tokenizer, max_length)

        # Model
        is_reg = task_type == "regression"
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1 if is_reg else len(set(labels)),
            problem_type="regression" if is_reg else "single_label_classification",
        )

        # Training Arguments
        run_id = uuid.uuid4().hex
        output_dir = f"tmp_model_{run_id}"
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            save_total_limit=1,
            report_to="none",
            eval_strategy="epoch",
            remove_unused_columns=False,
        )

        # Metrics
        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            if is_reg:
                preds = preds.squeeze()
                mse = ((preds - labels) ** 2).mean()
                mae = (abs(preds - labels)).mean()
                return {"mse": mse, "mae": mae}
            else:
                preds = preds.argmax(axis=-1)
                acc = (preds == labels).mean()
                return {"accuracy": acc}

        # Training
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset if len(val_dataset) > 0 else None,
            compute_metrics=compute_metrics,
        )
        trainer.train()

        # Save Model (as bytes)
        final_model_dir = tempfile.mkdtemp()
        trainer.save_model(final_model_dir)
        # Zip the model directory
        import shutil

        model_zip_path = os.path.join(final_model_dir, "model.zip")
        shutil.make_archive(
            base_name=model_zip_path[:-4], format="zip", root_dir=final_model_dir
        )
        with open(model_zip_path, "rb") as f:
            model_zip_bytes = f.read()

        # Evaluation
        eval_result = trainer.evaluate()
        logs = trainer.state.log_history

        # Regression Plot
        plot_b64 = None
        if is_reg and len(val_dataset) > 0:
            preds = trainer.predict(val_dataset)
            # Fix: handle tuple predictions and squeeze properly
            predictions = preds.predictions
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            predictions = predictions.squeeze()
            plot_b64 = make_regression_plot(val_labels, predictions)

        summary = {
            "model_name": model_name,
            "task_type": task_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "metrics": eval_result,
            "plot_b64": plot_b64,
            "log_history": logs,
        }

        # Return response (model as downloadable, summary, plot)
        response = {
            "summary": summary,
            "model_file": base64.b64encode(model_zip_bytes).decode("utf-8"),
            "model_file_name": "model.zip",
        }
        return JSONResponse(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# EXPLAINABILITY ENDPOINT

JOBS_DIR = os.path.join(tempfile.gettempdir(), "patch_explain_jobs")
os.makedirs(JOBS_DIR, exist_ok=True)
job_results = {}


def run_patch_explain_job(
    task_id, model_file_path, model_name, patches_file_path, task_type
):
    try:
        import shutil
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Load model
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "model.zip")
            shutil.copy(model_file_path, zip_path)
            shutil.unpack_archive(zip_path, tmpdir)
            model = AutoModelForSequenceClassification.from_pretrained(tmpdir)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load patches
        import pandas as pd

        df = (
            pd.read_csv(patches_file_path)
            if patches_file_path.endswith(".csv")
            else pd.read_json(patches_file_path)
        )
        patch_texts = list(df["patch"])
        # Compute saliency
        from .patch_classifier import patch_token_saliency

        explain = patch_token_saliency(model, tokenizer, patch_texts)
        # Save result as JSON
        job_dir = os.path.join(JOBS_DIR, task_id)
        os.makedirs(job_dir, exist_ok=True)
        result_path = os.path.join(job_dir, "explanations.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({"explanations": explain}, f)
        job_results[task_id] = {
            "status": "done",
            "result_url": f"/api/download_patch_explain_result/{task_id}/",
        }
    except Exception as e:
        job_results[task_id] = {"status": "error", "error": str(e)}


@router.post("/patch_explain/")
async def patch_explain(
    background_tasks: BackgroundTasks,
    model_file: UploadFile = File(None),
    model_task_id: str = Form(None),
    model_name: str = Form("bert-base-uncased"),
    patches: UploadFile = File(...),
    pasted: str = Form(None),
    task_type: str = Form("regression"),
):
    """
    Start patch explainability as a background job. Returns a task_id immediately.
    Accepts either model_file (UploadFile) or model_task_id (str) to load the model.
    """
    task_id = uuid.uuid4().hex
    # Save patches to disk
    # Handle case where patches.filename may be None
    if patches.filename and "." in patches.filename:
        ext = patches.filename.rsplit(".", 1)[-1]
    else:
        ext = "bin"
    patches_file_path = os.path.join(
        tempfile.gettempdir(),
        f"patch_explain_patches_{task_id}.{ext}",
    )
    with open(patches_file_path, "wb") as f:
        f.write(await patches.read())
    # Save model file to disk if provided
    model_file_path = None
    if model_file is not None:
        model_file_path = os.path.join(
            tempfile.gettempdir(), f"patch_explain_model_{task_id}.zip"
        )
        with open(model_file_path, "wb") as f:
            f.write(await model_file.read())

    def job_wrapper():
        try:
            if model_task_id:
                # Load encoder from patch embedding job dir
                from backend.app.api.patch_embedding import JOBS_DIR as EMBED_JOBS_DIR

                encoder_path = os.path.join(EMBED_JOBS_DIR, model_task_id, "encoder.pt")
                if not os.path.exists(encoder_path):
                    raise Exception(
                        f"Encoder file not found for model_task_id {model_task_id}"
                    )
                # Create a temp zip for transformers
                import zipfile

                with tempfile.TemporaryDirectory() as tmpdir:
                    zip_path = os.path.join(tmpdir, "encoder.zip")
                    with zipfile.ZipFile(zip_path, "w") as zipf:
                        zipf.write(encoder_path, arcname="pytorch_model.bin")
                    run_patch_explain_job(
                        task_id, zip_path, model_name, patches_file_path, task_type
                    )
            elif model_file_path:
                run_patch_explain_job(
                    task_id, model_file_path, model_name, patches_file_path, task_type
                )
            else:
                raise Exception("No model_file or model_task_id provided.")
        finally:
            if model_file_path and os.path.exists(model_file_path):
                os.remove(model_file_path)
            if os.path.exists(patches_file_path):
                os.remove(patches_file_path)

    background_tasks.add_task(threading.Thread(target=job_wrapper).start)
    job_results[task_id] = {"status": "pending"}
    return JSONResponse({"task_id": task_id, "status": "pending"}, status_code=202)


@router.get("/patch_explain_status/{task_id}/")
async def patch_explain_status(task_id: str):
    job = job_results.get(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="Task ID not found")
    if job["status"] == "pending":
        return {"status": "pending"}
    elif job["status"] == "done":
        return {"status": "done", "result_url": job["result_url"]}
    elif job["status"] == "error":
        return {"status": "error", "error": job["error"]}
    else:
        return {"status": "unknown"}


@router.get("/download_patch_explain_result/{task_id}/")
def download_patch_explain_result(task_id: str):
    job_dir = os.path.join(JOBS_DIR, task_id)
    result_path = os.path.join(job_dir, "explanations.json")
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result file not found")
    return FileResponse(
        result_path, media_type="application/json", filename="explanations.json"
    )


# PREDICTION ENDPOINT


@router.post("/patch_predict/")
async def patch_predict(
    model_file: UploadFile = File(...),
    model_name: str = Form("bert-base-uncased"),
    patches: UploadFile = File(...),
    task_type: str = Form("regression"),
):
    """
    Predict label for given patches using the uploaded model.
    """
    try:
        # Load model
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "model.zip")
            with open(zip_path, "wb") as f:
                f.write(model_file.file.read())
            import shutil

            shutil.unpack_archive(zip_path, tmpdir)
            model = AutoModelForSequenceClassification.from_pretrained(tmpdir)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        df = read_table(patches)
        patch_texts = list(df["patch"])
        inputs = tokenizer(
            patch_texts,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**{k: v for k, v in inputs.items()})
            if task_type == "regression":
                preds = outputs.logits.squeeze().cpu().numpy().tolist()
            else:
                preds = outputs.logits.argmax(dim=-1).cpu().numpy().tolist()
        return JSONResponse({"predictions": preds})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# MODEL COMPARISON
# Allows training and evaluating two patch classifiers with different configs, returning comparison


@router.post("/compare_patch_classifiers/")
async def compare_patch_classifiers(
    data: UploadFile = File(...),
    config1: str = Form(...),
    config2: str = Form(...),
    label_column: str = Form("label"),
    patch_column: str = Form("patch"),
):
    """
    Train and evaluate two patch classifiers with different configs, return comparison.
    config1, config2: JSON string with keys as in train_patch_classifier.
    """
    try:
        import json

        df = read_table(data)
        results = []
        plot_b64s = []
        for config_str in [config1, config2]:
            config = json.loads(config_str)
            task_type = config.get("task_type", "regression")
            model_name = config.get("model_name", "bert-base-uncased")
            epochs = int(config.get("epochs", 2))
            batch_size = int(config.get("batch_size", 8))
            learning_rate = float(config.get("learning_rate", 5e-5))
            max_length = int(config.get("max_length", 256))

            # Prepare data
            texts = list(df[patch_column])
            labels = list(df[label_column])
            if task_type == "regression":
                labels = [float(x) for x in labels]
            else:
                try:
                    labels = [int(x) for x in labels]
                except:
                    unique = sorted(list(set(labels)))
                    lookup = {v: i for i, v in enumerate(unique)}
                    labels = [lookup[x] for x in labels]
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=0.2, random_state=42
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            train_dataset = PatchDataset(
                train_texts, train_labels, tokenizer, max_length
            )
            val_dataset = PatchDataset(val_texts, val_labels, tokenizer, max_length)
            is_reg = task_type == "regression"
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=1 if is_reg else len(set(labels)),
                problem_type="regression" if is_reg else "single_label_classification",
            )
            run_id = uuid.uuid4().hex
            output_dir = f"tmp_model_{run_id}"
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                logging_dir=f"{output_dir}/logs",
                logging_steps=10,
                save_total_limit=1,
                report_to="none",
                eval_strategy="epoch",
                remove_unused_columns=False,
            )

            def compute_metrics(eval_pred):
                preds, labels = eval_pred
                if is_reg:
                    preds = preds.squeeze()
                    mse = ((preds - labels) ** 2).mean()
                    mae = (abs(preds - labels)).mean()
                    return {"mse": mse, "mae": mae}
                else:
                    preds = preds.argmax(axis=-1)
                    acc = (preds == labels).mean()
                    return {"accuracy": acc}

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset if len(val_dataset) > 0 else None,
                compute_metrics=compute_metrics,
            )
            trainer.train()
            eval_result = trainer.evaluate()
            logs = trainer.state.log_history
            plot_b64 = None
            if is_reg and len(val_dataset) > 0:
                preds = trainer.predict(val_dataset)
                # Fix: handle tuple predictions and squeeze properly
                predictions = preds.predictions
                if isinstance(predictions, tuple):
                    predictions = predictions[0]
                predictions = predictions.squeeze()
                plot_b64 = make_regression_plot(val_labels, predictions)

            summary = {
                "model_name": model_name,
                "task_type": task_type,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "metrics": eval_result,
                "plot_b64": plot_b64,
                "log_history": logs,
            }
            results.append(summary)
            plot_b64s.append(plot_b64)
        return JSONResponse(
            {
                "comparison": results,
                "plot_b64_1": plot_b64s[0],
                "plot_b64_2": plot_b64s[1],
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")
