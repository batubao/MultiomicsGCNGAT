import os
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data

# Define folder path
folder_path = 'ROSMAP'

# Load data function
def load_data(file_name):
    return pd.read_csv(os.path.join(folder_path, file_name))

# Convert pandas DataFrames to torch tensors
def df_to_tensor(df):
    return torch.tensor(df.values, dtype=torch.float)

# Apply PCA for dimensionality reduction
def apply_pca(train_data, test_data, n_components=200):
    pca = PCA(n_components=n_components)
    train_data_reduced = pca.fit_transform(train_data)
    test_data_reduced = pca.transform(test_data)
    return torch.tensor(train_data_reduced, dtype=torch.float), torch.tensor(test_data_reduced, dtype=torch.float)

# Create edge index based on cosine similarity
def create_edge_index_with_cosine_similarity(data_tensor, threshold=0.5):
    cosine_sim = cosine_similarity(data_tensor)
    adjacency_matrix = (cosine_sim >= threshold).astype(int)
    edge_index = [[i, j] for i in range(adjacency_matrix.shape[0]) for j in range(adjacency_matrix.shape[1])
                  if adjacency_matrix[i, j] == 1 and i != j]
    return torch.tensor(edge_index).t().contiguous()
