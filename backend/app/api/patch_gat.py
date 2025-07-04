import io
import os
import json
import uuid
import tempfile
import base64
import torch
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Create a new FastAPI router for patch-based GAT API endpoints
router = APIRouter()

# --- Define the Graph Attention Network model for classifying subgraphs/patches ---
class PatchGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
        """
        Initialize the GAT model.

        Args:
            in_channels: Number of input features per node.
            hidden_channels: Number of hidden units in the GAT layer.
            out_channels: Number of output classes (for classification).
            heads: Number of attention heads in GAT layers.
        """
        super().__init__()
        # First GAT layer with multi-head attention
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        # Second GAT layer that reduces to out_channels with single head
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1)
        # Non-linear activation function
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index):
        """
        Forward pass through the network.

        Args:
            x: Node feature matrix [num_nodes, in_channels].
            edge_index: Graph connectivity [2, num_edges].

        Returns:
            Output logits for each node.
        """
        # Apply first GATConv layer + ReLU
        x = self.gat1(x, edge_index)
        x = self.relu(x)
        # Apply second GATConv layer to get output logits
        x = self.gat2(x, edge_index)
        return x


# --- Utility function to load graph data from CSV files ---
def load_graph_from_csv(node_file, edge_file, feature_cols, label_col):
    """
    Load node features, edges, and labels from CSV files.

    Args:
        node_file: CSV file uploaded containing node features and labels.
        edge_file: CSV file uploaded containing edge list (source,target).
        feature_cols: List of column names in node_file to use as features.
        label_col: Column name for node labels (e.g. boundary/not-boundary).

    Returns:
        x: Tensor of node features [num_nodes, num_features].
        edge_index: Tensor of edge indices [2, num_edges].
        labels: Tensor of node labels [num_nodes].
    """
    # Read nodes CSV into DataFrame
    nodes_df = pd.read_csv(io.BytesIO(node_file.file.read()))
    # Read edges CSV into DataFrame
    edges_df = pd.read_csv(io.BytesIO(edge_file.file.read()))
    
    # Extract feature columns as float tensor
    x = torch.tensor(nodes_df[feature_cols].values, dtype=torch.float)
    
    # Convert edges to tensor of shape [2, num_edges]
    edge_index = torch.tensor(edges_df[['source', 'target']].values.T, dtype=torch.long)
    
    # Extract labels as tensor
    labels = torch.tensor(nodes_df[label_col].values, dtype=torch.long)
    
    return x, edge_index, labels


# --- Generate sliding-window subgraphs from graph nodes ---
def generate_subgraphs(edge_index, window_size=3):
    """
    Create overlapping subgraphs by taking consecutive windows of nodes.

    Args:
        edge_index: Tensor with edges [2, num_edges].
        window_size: Number of nodes per subgraph.

    Returns:
        List of node index lists, each representing a subgraph.
    """
    num_nodes = edge_index.max().item() + 1  # total number of nodes
    subgraphs = []
    # Slide window over all nodes to create subgraphs
    for start in range(num_nodes - window_size + 1):
        subgraphs.append(list(range(start, start + window_size)))
    return subgraphs


# --- Create a PyG Data object for a given subgraph ---
def create_subgraph_data(x, edge_index, node_indices, labels):
    """
    Extract subgraph data (features, edges, label) for given node indices.

    Args:
        x: Full node feature tensor.
        edge_index: Full edge index tensor.
        node_indices: List of node indices in this subgraph.
        labels: Tensor of all node labels.

    Returns:
        PyG Data object representing the subgraph.
    """
    # Boolean mask selecting nodes in the subgraph
    mask = torch.zeros(x.size(0), dtype=torch.bool)
    mask[node_indices] = True
    
    # Filter edges that connect only nodes in the subgraph
    edge_mask = (mask[edge_index[0]] & mask[edge_index[1]])
    sub_edge_index = edge_index[:, edge_mask]
    
    # Relabel node indices in subgraph to range 0..N-1 for PyG
    node_id_map = {nid: i for i, nid in enumerate(node_indices)}
    sub_edge_index = torch.tensor(
        [[node_id_map[int(n)] for n in sub_edge_index[0].tolist()],
         [node_id_map[int(n)] for n in sub_edge_index[1].tolist()]],
        dtype=torch.long,
    )
    
    # Select features only for nodes in subgraph
    sub_x = x[node_indices]
    
    # Assign label to subgraph — here majority label of included nodes
    sub_label = torch.mode(labels[node_indices])[0].item()
    
    # Create PyG Data object with node features, edges, and label
    data = Data(x=sub_x, edge_index=sub_edge_index, y=torch.tensor([sub_label]))
    return data


# --- Helper to create an accuracy scatter plot encoded as base64 ---
def make_accuracy_plot(y_true, y_pred):
    """
    Plot predicted vs true labels and encode plot image as base64 string.

    Args:
        y_true: List/array of true labels.
        y_pred: List/array of predicted labels.

    Returns:
        Base64 encoded PNG image of the scatter plot.
    """
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    plt.title("GAT Subgraph Classification Results")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# --- API endpoint to train the GAT model on subgraphs ---
@router.post("/train_patch_gat/")
async def train_patch_gat(
    node_file: UploadFile = File(...),  # CSV of nodes
    edge_file: UploadFile = File(...),  # CSV of edges
    feature_cols: str = Form(...),      # JSON string of feature column names
    label_col: str = Form(...),         # Label column name
    window_size: int = Form(3),         # Size of sliding window for subgraphs
    epochs: int = Form(5),              # Number of training epochs
    batch_size: int = Form(16),         # Batch size for training
    hidden_dim: int = Form(32),         # Hidden dimension size for GAT
):
    """
    Train GAT on subgraphs extracted from the uploaded graph.

    Returns training accuracy, loss plot, and model ID for later predictions.
    """
    try:
        # Parse JSON string of feature columns into list
        feature_cols = json.loads(feature_cols)
        
        # Load graph data from CSVs
        x, edge_index, labels = load_graph_from_csv(node_file, edge_file, feature_cols, label_col)
        
        # Generate overlapping subgraphs as node index windows
        subgraphs = generate_subgraphs(edge_index, window_size)
        
        # Create PyG Data objects for each subgraph with features, edges, and labels
        data_list = [create_subgraph_data(x, edge_index, nodes, labels) for nodes in subgraphs]
        
        # Split data into train and validation sets (80/20)
        train_data, val_data = train_test_split(data_list, test_size=0.2, random_state=42)
        
        # Data loaders to batch subgraphs during training
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)
        
        # Initialize GAT model
        model = PatchGAT(in_channels=x.size(1), hidden_channels=hidden_dim, out_channels=len(labels.unique()))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)
                
                # Aggregate node logits into graph-level logits by averaging
                out_graph = out.mean(dim=0, keepdim=True)  # shape (1, num_classes)
                
                # Calculate loss between prediction and true label
                loss = criterion(out_graph, batch.y.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
        
        # Evaluate on validation set
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
                out_graph = out.mean(dim=0, keepdim=True)
                pred = out_graph.argmax(dim=-1).cpu().item()
                y_pred.append(pred)
                y_true.append(batch.y.cpu().item())
        
        # Calculate accuracy on validation data
        acc = (np.array(y_true) == np.array(y_pred)).mean()
        
        # Generate plot of predicted vs true labels as base64 PNG image
        plot_b64 = make_accuracy_plot(y_true, y_pred)
        
        # Save trained model with unique ID for future predictions
        model_id = uuid.uuid4().hex
        model_path = os.path.join(tempfile.gettempdir(), f"patch_gat_{model_id}.pt")
        torch.save(model.state_dict(), model_path)
        
        # Return accuracy, plot, and model ID to the client
        response = {
            "accuracy": acc,
            "plot_b64": plot_b64,
            "model_id": model_id,
        }
        return JSONResponse(response)
    
    except Exception as e:
        # Return HTTP 500 error if training fails
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# --- API endpoint to predict with a saved GAT model on new graph patches ---
@router.post("/predict_patch_gat/")
async def predict_patch_gat(
    model_id: str = Form(...),          # Unique ID of saved model to load
    node_file: UploadFile = File(...),  # CSV of nodes with features
    edge_file: UploadFile = File(...),  # CSV of edges
    feature_cols: str = Form(...),       # JSON list of feature column names
    window_size: int = Form(3),          # Window size for subgraphs
):
    """
    Load a saved GAT model and predict on new graph patches.

    Returns list of predicted class labels for each patch.
    """
    try:
        feature_cols = json.loads(feature_cols)
        
        # Load graph data from CSVs (labels can be dummy here)
        x, edge_index, _ = load_graph_from_csv(node_file, edge_file, feature_cols, label_col=None)
        
        # Generate subgraphs using sliding window approach
        subgraphs = generate_subgraphs(edge_index, window_size)
        
        # Create PyG Data objects for each subgraph (dummy labels)
        data_list = [create_subgraph_data(x, edge_index, nodes, torch.zeros(x.size(0))) for nodes in subgraphs]

        # Load model from disk
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # NOTE: adapt out_channels if needed
        model = PatchGAT(in_channels=x.size(1), hidden_channels=32, out_channels=2)
        model_path = os.path.join(tempfile.gettempdir(), f"patch_gat_{model_id}.pt")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        preds = []
        # Predict on each subgraph independently
        with torch.no_grad():
            for data in data_list:
                data = data.to(device)
                out = model(data.x, data.edge_index)
                out_graph = out.mean(dim=0, keepdim=True)
                pred = out_graph.argmax(dim=-1).cpu().item()
                preds.append(pred)
        
        # Return list of predictions to client
        return JSONResponse({"predictions": preds})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")