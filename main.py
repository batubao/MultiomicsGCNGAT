import torch
import numpy as np
from utils import load_data, apply_pca, create_edge_index_with_cosine_similarity
from model import GCN, MultimodalFusionGAT
from training_testing import train, test
from torch_geometric.data import Data

# Load and preprocess data
data_1_tr = load_data('1_tr.csv')
data_2_tr = load_data('2_tr.csv')
data_3_tr = load_data('3_tr.csv')
labels_tr = load_data('labels_tr.csv')

data_1_te = load_data('1_te.csv')
data_2_te = load_data('2_te.csv')
data_3_te = load_data('3_te.csv')
labels_te = load_data('labels_te.csv')

# Convert to tensors and apply PCA
data_1_tr_tensor, data_1_te_tensor = apply_pca(data_1_tr.values, data_1_te.values)
data_2_tr_tensor, data_2_te_tensor = apply_pca(data_2_tr.values, data_2_te.values)
data_3_tr_tensor, data_3_te_tensor = apply_pca(data_3_tr.values, data_3_te.values)
labels_tr_tensor = torch.tensor(labels_tr.values, dtype=torch.long).squeeze()
labels_te_tensor = torch.tensor(labels_te.values, dtype=torch.long).squeeze()

# Create edge indices based on cosine similarity for each modality
edge_index_1 = create_edge_index_with_cosine_similarity(data_1_tr_tensor, threshold=0.5)
edge_index_2 = create_edge_index_with_cosine_similarity(data_2_tr_tensor, threshold=0.5)
edge_index_3 = create_edge_index_with_cosine_similarity(data_3_tr_tensor, threshold=0.5)

# Construct Graph Data objects for each modality
train_data_1 = Data(x=data_1_tr_tensor, edge_index=edge_index_1)
train_data_2 = Data(x=data_2_tr_tensor, edge_index=edge_index_2)
train_data_3 = Data(x=data_3_tr_tensor, edge_index=edge_index_3)

# Similarly, create edge indices for test datasets
edge_index_1_te = create_edge_index_with_cosine_similarity(data_1_te_tensor, threshold=0.5)
edge_index_2_te = create_edge_index_with_cosine_similarity(data_2_te_tensor, threshold=0.5)
edge_index_3_te = create_edge_index_with_cosine_similarity(data_3_te_tensor, threshold=0.5)

test_data_1 = Data(x=data_1_te_tensor, edge_index=edge_index_1_te)
test_data_2 = Data(x=data_2_te_tensor, edge_index=edge_index_2_te)
test_data_3 = Data(x=data_3_te_tensor, edge_index=edge_index_3_te)

# Combine training and testing data for testing phase
combined_data_1 = torch.cat([data_1_tr_tensor, data_1_te_tensor], dim=0)
combined_data_2 = torch.cat([data_2_tr_tensor, data_2_te_tensor], dim=0)
combined_data_3 = torch.cat([data_3_tr_tensor, data_3_te_tensor], dim=0)
combined_labels = torch.cat([labels_tr_tensor, labels_te_tensor], dim=0)

# Create edge indices for the combined dataset
edge_index_combined_1 = create_edge_index_with_cosine_similarity(combined_data_1, threshold=0.5)
edge_index_combined_2 = create_edge_index_with_cosine_similarity(combined_data_2, threshold=0.5)
edge_index_combined_3 = create_edge_index_with_cosine_similarity(combined_data_3, threshold=0.5)

combined_data_1 = Data(x=combined_data_1, edge_index=edge_index_combined_1)
combined_data_2 = Data(x=combined_data_2, edge_index=edge_index_combined_2)
combined_data_3 = Data(x=combined_data_3, edge_index=edge_index_combined_3)


# Model parameters
in_channels = data_1_tr_tensor.shape[1]
hidden_channels = 64
out_channels = 32
num_classes = len(labels_tr_tensor.unique())

# Initialize models and optimizer
gcn1 = GCN(in_channels, hidden_channels, out_channels)
gcn2 = GCN(in_channels, hidden_channels, out_channels)
gcn3 = GCN(in_channels, hidden_channels, out_channels)
fusion_model = MultimodalFusionGAT(out_channels, hidden_channels, out_channels, num_classes)

optimizer = torch.optim.Adam(
    list(gcn1.parameters()) + list(gcn2.parameters()) + list(gcn3.parameters()) + list(fusion_model.parameters()),
    lr=0.0001,
    weight_decay=1e-4
)

# Training loop with metrics storage
num_epochs = 5000
accuracy_list, f1_list, auc_list = [], [], []

for epoch in range(1, num_epochs + 1):
    loss = train(gcn1, gcn2, gcn3, fusion_model, optimizer, combined_data_1, combined_data_2, combined_data_3, labels_tr_tensor)
    if epoch % 10 == 0:
        accuracy, f1, auc = test(gcn1, gcn2, gcn3, fusion_model, combined_data_1, combined_data_2, combined_data_3, combined_labels)
        accuracy_list.append(accuracy)
        f1_list.append(f1)
        auc_list.append(auc)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Accuracy: {accuracy:.3f}')

# Final results
def format_metric(values):
    return f"{np.mean(values):.3f} ± {np.std(values):.3f
