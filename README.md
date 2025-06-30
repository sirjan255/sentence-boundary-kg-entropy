# Sentence Boundary Detection in Knowledge Graphs via Entropy

This project detects sentence boundaries in a knowledge graph (KG) derived from text, using entropy-based traversal and supervised GNN classification. Everything is implemented in Python 3.11. Below is a comprehensive guide for Windows users, including installation, environment setup, and detailed command walkthroughs with explanations.

---

## 1. Python 3.11 Installation (Windows)

> **Why Python 3.11?**  
> The project uses libraries (e.g., torch-geometric, node2vec) that are best supported and tested with Python 3.11.  
> Using other versions may cause dependency issues or incompatibilities.

**Step-by-step (Run as Administrator in Windows PowerShell):**

```powershell
winget install Python.Python.3.11
```
- After installation, click **"Add to PATH"** in the installer GUI.
- Confirm installation:
    ```powershell
    python --version
    py -3.11 --version
    ```

---

## 2. Virtual Environment Setup

```powershell
py -3.11 -m venv venv
.\venv\Scripts\activate
```

- Upgrade pip:
    ```powershell
    python -m pip install --upgrade pip
    ```

---

## 3. Install Required Packages

Install all packages **inside** the activated virtual environment:

```powershell
pip install numpy scipy --prefer-binary
pip install scikit-learn pandas --prefer-binary
pip install spacy networkx --prefer-binary
pip install node2vec matplotlib --prefer-binary
```

**Test all dependencies:**
```powershell
python -c "import numpy, sklearn, spacy, networkx, node2vec, pandas, scipy, matplotlib; print('All required packages are installed successfully!')"
```

---

## 4. Knowledge Graph Construction and Preprocessing

### 4.1. Generate Starting Nodes

Extracts main subject from each sentence and saves starting nodes for traversal.

```powershell
python scripts/generate_nodes_to_start.py
```

- **Output:** `data/nodes_to_start.txt` containing one starting node per sentence.

### 4.2. Extract SVO Triplets

Extracts Subject-Verb-Object triplets from sample text.

```powershell
python scripts/extract_svo.py --input data/sample_text.txt --output outputs/triplets.csv
```

- **Output:** `outputs/triplets.csv` with SVO triplets.

### 4.3. Build the Knowledge Graph

Builds a knowledge graph (KG) from extracted triplets.

```powershell
python scripts/build_kg.py --triplets outputs/triplets.csv --output outputs/kg.pkl
```

- **Output:** `outputs/kg.pkl` (pickled KG).

### 4.4. Visualize the Full Knowledge Graph

Visualizes entities and relationships in the KG.

```powershell
python scripts/visualize_full_kg.py
```

---

## 5. Embedding Generation and Validation

### 5.1. Generate Node2Vec Embeddings

```powershell
python scripts/node2vec_embeddings.py --kg outputs/kg.pkl --output outputs/embeddings --normalize
```

- **Output:** `outputs/embeddings.npy` and `outputs/embeddings_nodes.txt`

### 5.2. Validate Embeddings

```powershell
python scripts/check_embeddings.py
```

---

## 6. Node Selection Strategies

Run different strategies to select starting nodes for boundary detection:

```powershell
python scripts/select_starting_nodes.py --strategy degree --num 3
python scripts/select_starting_nodes.py --strategy random --num 3
python scripts/select_starting_nodes.py --strategy entropy --num 3 --embeddings outputs/embeddings.npy --nodes outputs/embeddings_nodes.txt
python scripts/select_starting_nodes.py --strategy all --num 3 --embeddings outputs/embeddings.npy --nodes outputs/embeddings_nodes.txt
```

- **Explanation:** These commands select nodes based on degree, randomly, by entropy, or all nodes, for use as traversal starting points.

---

## 7. Entropy-Based Sentence Boundary Detection

**Detect sentence boundaries using entropy-based traversal (BLT-style entropy):**

```powershell
python scripts/detect_sentence_boundary.py --kg outputs/kg.pkl --embeddings outputs/embeddings.npy --nodes outputs/embeddings_nodes.txt --starts data/nodes_to_start.txt --entropy_threshold 0.8 --debug
```

- **Explanation:**  
  Traverses the KG from each start node, computes entropy at each step, and stops when entropy exceeds the threshold (indicating a likely sentence boundary). Debug mode prints detailed traversal info.

---

## 8. Post-processing and Evaluation

### 8.1. Convert Predicted Boundaries to Node Lists

```powershell
python scripts/convert_predicted_to_strings.py --predicted outputs/predicted_boundaries.json --output outputs/predicted_boundaries_nodes.json
```

- **Explanation:** Converts the predicted boundaries with entropy to plain node lists for evaluation.

### 8.2. Evaluate Raw (Unnormalized) Boundaries

```powershell
python scripts/evaluate_boundaries.py --predicted outputs/predicted_boundaries_nodes.json --actual data/actual_boundaries.json
```

- **Explanation:** Evaluates predictions without normalizing node names. Raw evaluation may give low or zero scores due to string mismatches.

### 8.3. Evaluate with Normalized Boundaries

```powershell
python scripts/evaluate_boundaries_normalized.py --predicted outputs/predicted_boundaries.json --actual data/actual_boundaries.json
```

- **Explanation:** Evaluates after normalizing node names (handles quotes, whitespace, etc), giving more realistic F1/precision/recall.

---

## 9. GNN-based Classifier: Training and Evaluation

### 9.1. Install PyTorch and Torch-Geometric

```powershell
pip install torch torchvision torchaudio
pip install torch-geometric
```

### 9.2. Train the GNN Model and Generate Predictions

```powershell
python scripts/gnn_boundary_classifier.py --kg outputs/kg.pkl --embeddings outputs/embeddings.npy --nodes outputs/embeddings_nodes.txt --gold data/actual_boundaries.json --epochs 60 --output outputs/predicted_gnn_boundaries.json
```

- **Explanation:** Trains the GNN boundary classifier and outputs predicted boundary nodes.

### 9.3. Evaluate GNN Predictions

```powershell
python scripts/evaluate.py --gold data/actual_boundaries_strings.json --output outputs/predicted_gnn_boundaries.json
```

- **Explanation:** Evaluates the GNN’s predicted boundaries against the gold standard.

### 9.4. Run Experiment Script with Multiple Settings

```powershell
python scripts/gnn_boundary_classifier_experiments.py --kg outputs/kg.pkl --embeddings outputs/embeddings.npy --nodes outputs/embeddings_nodes.txt --gold data/actual_boundaries.json --epochs 60 --test_ratios 0.1,0.2,0.3,0.5 --hidden_dims 32,64,128 --seeds 42,2024,7 --runs_per_setting 3 --output_dir gnn_experiment_results --verbose
```

- **Explanation:** Runs the GNN classifier with various hyperparameters and test/train splits, saving all results for comparison.

---

## 10. Knowledge Graph Demo with Neo4j

This section guides you through running a demo knowledge graph using Neo4j and Python.

### Steps:

1. **Install Neo4j Desktop**

   - Download Neo4j Desktop from [https://neo4j.com/download/](https://neo4j.com/download/)
   - Install and launch Neo4j Desktop on your system.
   - Create and start a new database instance.
   - Set an initial password when prompted and remember it.

2. **Configure Environment Variables**

   - Set up environment variables for Neo4j credentials:
     - `NEO4J_USER` (default is `neo4j`)
     - `NEO4J_PASSWORD` (your password set in Neo4j Desktop)
   - In your terminal, set the password as follows (for PowerShell/VSCode Terminal on Windows):

     ```powershell
     $env:NEO4J_PASSWORD="your_actual_password"
     ```

   - Or, for Command Prompt (CMD):

     ```cmd
     set NEO4J_PASSWORD=your_actual_password
     ```

3. **Install Python Dependencies**

   - Activate your virtual environment and install the Neo4j Python driver:

     ```sh
     pip install neo4j
     ```

4. **Run the Demo Script**

   - Execute the demo script to load the sample SVO triplets into Neo4j:

     ```powershell
     python scripts/demo_neo4j_graph.py
     ```

5. **Visualize the Graph**

   - Open your browser and go to [http://localhost:7474](http://localhost:7474)
   - Log in using the username (`neo4j`) and your password.
   - Use Cypher queries such as:

     ```
     MATCH (n) RETURN n LIMIT 25;
     MATCH p=()-[:REL]->() RETURN p LIMIT 25;
     MATCH (a)-[r:REL]->(b) RETURN a.name, r.verb, b.name;
     ```

   - Explore and visualize your knowledge graph!

---

## 11. Patch Classifier Training Demo

This section explains how to train the patch classifier model (regression on commit message word count) in your virtual environment. Follow the steps below:

### 11.1 Create and Activate a Virtual Environment

```sh
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 11.2 Install Required Packages

Install the necessary dependencies inside your virtual environment:

```sh
pip install torch transformers scikit-learn pandas gitpython
```

### 11.3 Install `accelerate` (Required for HuggingFace Trainer)

Install the `accelerate` package:

```sh
pip install accelerate
```

> **What does this command do?**  
> `pip install accelerate` installs HuggingFace's Accelerate library, which is required to handle device placement (CPU/GPU) when training models using the HuggingFace Trainer API. Without it, you will get an ImportError.

### 11.4 Run the Patch Classifier Training Script

```sh
python scripts/train_patch_classifier.py
```

---

### 11.5 Expected Warnings and Logs (and why to ignore them)

During training, you may see the following warnings and logs:

- **Some weights of BertForSequenceClassification were not initialized...**  
  _Reason:_ The classification layer is always randomly initialized; this is expected whenever you fine-tune BERT for a new task.  
  _**Ignore this warning.**_

- **You should probably TRAIN this model on a down-stream task...**  
  _Reason:_ Just a reminder that you need to fine-tune the classifier head.  
  _**Ignore this warning.**_

- **UserWarning: 'pin_memory' argument is set as true but no accelerator is found...**  
  _Reason:_ Pin memory is used for faster data transfer to GPU. If you are training on CPU, this has no effect and is safe to ignore.  
  _**Ignore this warning.**_

- **Progress bars (0% ... 100%)**  
  _Reason:_ These are normal progress bars from the training process.  
  _**Ignore/monitor as desired.**_

---


## 12. Summary

- **This project provides both entropy-based unsupervised and GNN-based supervised approaches to sentence boundary detection in KGs.**
- **Make sure to always activate your virtual environment before running any scripts.**
- **Follow the commands in order for a smooth setup, training, and evaluation pipeline.**

---

## Troubleshooting

- If you encounter import errors, double-check your Python version and ensure your virtual environment is activated.
- For best results, use the exact package versions and Python 3.11 as described above.

---