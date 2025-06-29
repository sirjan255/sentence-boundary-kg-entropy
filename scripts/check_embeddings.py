import numpy as np

# Load the embeddings from the .npy file
embeddings = np.load("outputs/embeddings.npy")

# Print the shape and data type of the embeddings array
print("Shape:", embeddings.shape)
print("Dtype:", embeddings.dtype)