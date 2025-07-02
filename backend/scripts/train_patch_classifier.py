import os
import git
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


def extract_patches(repo_path, max_commits=1000):
    repo = git.Repo(repo_path)
    patches = []
    for commit in repo.iter_commits("main", max_count=max_commits):
        if commit.parents:
            diff = commit.diff(commit.parents[0], create_patch=True)
            for d in diff:
                if d.diff:
                    patches.append(
                        {
                            "patch": d.diff.decode(errors="ignore"),
                            "message": commit.message.strip(),
                        }
                    )
    return pd.DataFrame(patches)


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.encodings = tokenizer(
            list(df["patch"]),
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = list(df["label"])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def main():
    repo_path = (
        r"C:\Users\kaurp\sentence-boundary-kg-entropy"  # Change to your repo path
    )
    model_name = "bert-base-uncased"
    output_dir = "./patch_classifier_output"
    max_length = 256
    batch_size = 4
    epochs = 2

    print("Extracting patches from repo...")
    df = extract_patches(repo_path)

    # Label = number of words in the commit message
    df["label"] = df["message"].apply(lambda x: len(x.split()))
    print(df[["message", "label"]].head())

    # For regression, labels should be float
    df["label"] = df["label"].astype(float)

    # Split data (no stratify in regression)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = PatchDataset(train_df, tokenizer, max_length)
    val_dataset = PatchDataset(val_df, tokenizer, max_length)

    # Use AutoModelForSequenceClassification with regression head
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, problem_type="regression"
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_total_limit=2,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = preds.squeeze()
        mse = ((preds - labels) ** 2).mean()
        mae = (abs(preds - labels)).mean()
        return {"mse": mse, "mae": mae}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_dataset) > 0 else None,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

    if len(val_dataset) > 0:
        eval_result = trainer.evaluate()
        print("Evaluation:", eval_result)
    else:
        print("No validation set to evaluate.")


if __name__ == "__main__":
    main()
