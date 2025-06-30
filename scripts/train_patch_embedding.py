import os
import git
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split

# 1. Data extraction (same as classifier)
def extract_patches(repo_path, max_commits=1000):
    repo = git.Repo(repo_path)
    patches = []
    for commit in repo.iter_commits('main', max_count=max_commits):
        if commit.parents:
            diff = commit.diff(commit.parents[0], create_patch=True)
            for d in diff:
                if d.diff:
                    patches.append({
                        "patch": d.diff.decode(errors="ignore"),
                        "commit_hash": commit.hexsha
                    })
    return pd.DataFrame(patches)

# 2. Dataset for patch text
class PatchDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.encodings = tokenizer(
            list(df['patch']),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}
    def __len__(self):
        return self.encodings['input_ids'].size(0)

# 3. Autoencoder model
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
        # Encode
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = outputs.last_hidden_state[:,0]  # [CLS] token
        z = self.proj(h)
        # Decode
        recon_logits = self.decoder(z)
        return recon_logits, z

def main():
    repo_path = r"C:\Users\kaurp\sentence-boundary-kg-entropy"  # Change as needed
    model_name = "bert-base-uncased"
    output_dir = "./patch_embedding_output"
    max_length = 128
    batch_size = 4
    epochs = 5
    embedding_dim = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Extracting patches from repo...")
    df = extract_patches(repo_path)
    if len(df) < 10:
        print("Not enough patches found.")
        return

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = PatchDataset(train_df, tokenizer, max_length)
    val_dataset = PatchDataset(val_df, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = PatchAutoencoder(model_name, embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # Target is next token (language modeling style) on [CLS] token
            targets = input_ids[:, 0]  # We'll reconstruct the [CLS] token id
            optimizer.zero_grad()
            recon_logits, _ = model(input_ids, attention_mask)
            loss = criterion(recon_logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}")

    # Save encoder for generating embeddings
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "patch_autoencoder.pt"))
    print(f"Model saved to {output_dir}/patch_autoencoder.pt")

    # Generate and save embeddings for all patches
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for batch in DataLoader(PatchDataset(df, tokenizer, max_length), batch_size=1):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            _, z = model(input_ids, attention_mask)
            all_embeddings.append(z.cpu().numpy()[0])
    emb_df = df.copy()
    emb_df['embedding'] = all_embeddings
    emb_df.to_pickle(os.path.join(output_dir, "patch_embeddings.pkl"))
    print(f"Patch embeddings saved to {output_dir}/patch_embeddings.pkl")

if __name__ == "__main__":
    main()