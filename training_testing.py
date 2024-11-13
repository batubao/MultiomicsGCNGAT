import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# Training function
def train(gcn1, gcn2, gcn3, fusion_model, optimizer, train_data_1, train_data_2, train_data_3, labels_tr_tensor):
    gcn1.train()
    gcn2.train()
    gcn3.train()
    fusion_model.train()
    optimizer.zero_grad()
    
    x1 = gcn1(train_data_1.x, train_data_1.edge_index)
    x2 = gcn2(train_data_2.x, train_data_2.edge_index)
    x3 = gcn3(train_data_3.x, train_data_3.edge_index)
    
    out = fusion_model(x1, x2, x3, train_data_1.edge_index)
    loss = F.nll_loss(out, labels_tr_tensor)
    loss.backward()
    optimizer.step()
    return loss.item()

# Testing function
def test(gcn1, gcn2, gcn3, fusion_model, combined_data_1, combined_data_2, combined_data_3, combined_labels):
    gcn1.eval()
    gcn2.eval()
    gcn3.eval()
    fusion_model.eval()
    
    x1 = gcn1(combined_data_1.x, combined_data_1.edge_index)
    x2 = gcn2(combined_data_2.x, combined_data_2.edge_index)
    x3 = gcn3(combined_data_3.x, combined_data_3.edge_index)
    
    out = fusion_model(x1, x2, x3, combined_data_1.edge_index)
    pred = out.argmax(dim=1)
    
    accuracy = accuracy_score(combined_labels.cpu().numpy(), pred.cpu().numpy())
    f1 = f1_score(combined_labels.cpu().numpy(), pred.cpu().numpy(), average='macro')
    auc = roc_auc_score(combined_labels.cpu().numpy(), F.softmax(out, dim=1).cpu().numpy()[:, 1], multi_class='ovo')
    
    return accuracy, f1, auc
