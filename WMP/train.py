import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from utils import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, precision_recall_curve, auc, matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from model import MoGCN
import argparse

if __name__ == "__main__":

    # create argument parser
    parser = argparse.ArgumentParser(description='Multi-omics Graph Convolutional Network')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--adj_thresh', type=float, default=0.1, help='Adjacency threshold')
    parser.add_argument('--bias', type=bool, default=True, help='Bias')
    parser.add_argument('--k', type=float, default=0.8, help='Message Passing Weight')
    parser.add_argument('--mRNA_dir', type=str, default='sample_data/mRNA/', help='Direcotry of mRNA data')
    parser.add_argument('--miRNA_dir', type=str, default='sample_data/miRNA/', help='Direcotry of miRNA data')
    parser.add_argument('--label_dir', type=str, default='sample_data/labels/', help='Direcotry of label data')
    parser.add_argument('--bip_path', type=str, default='sample_data/bip/bip_mRNA_miRNA.pkl', help='Path of bipartite data')
    args = parser.parse_args()

    num_layers = args.num_layers
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    hidden_dim = args.hidden_dim
    adj_thresh = args.adj_thresh
    bias = args.bias
    k = args.k
    mRNA_dir = args.mRNA_dir
    miRNA_dir = args.miRNA_dir
    label_dir = args.label_dir
    bip_path = args.bip_path

    # Check if GPU is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    # --------------------------------- X_u Preparation --------------------------------- #
    x_u, x_u_val, x_u_test = get_data(mRNA_dir)
    x_u = torch.tensor(x_u, dtype=torch.float32)
    x_u_val = torch.tensor(x_u_val, dtype=torch.float32)
    x_u_test = torch.tensor(x_u_test, dtype=torch.float32)

    scaler = StandardScaler()
    x_u = scaler.fit_transform(x_u)
    x_u = torch.tensor(x_u, dtype=torch.float32)

    scaler = StandardScaler()
    x_u_val = scaler.fit_transform(x_u_val)
    x_u_val = torch.tensor(x_u_val, dtype=torch.float32)

    scaler = StandardScaler()
    x_u_test = scaler.fit_transform(x_u_test)
    x_u_test = torch.tensor(x_u_test, dtype=torch.float32)

    x_u = x_u.t()               
    x_u_val = x_u_val.t()       
    x_u_test = x_u_test.t()     
    
    A_u = get_adjacency_matrix(x_u, threshold=adj_thresh, metric='cosine')
    A_u = torch.tensor(A_u, dtype=torch.float32)
    A_u = A_u + torch.eye(A_u.size(0), device=A_u.device)
    degree_matrix = torch.diag(torch.sum(A_u, dim=1))
    degree_matrix_inv_sqrt = torch.pow(degree_matrix, -0.5)
    degree_matrix_inv_sqrt[torch.isinf(degree_matrix_inv_sqrt)] = 0. # handle division by zero
    A_u = torch.matmul(torch.matmul(degree_matrix_inv_sqrt, A_u), degree_matrix_inv_sqrt)       # n x n

    # Move data to GPU
    x_u = x_u.to(device)
    x_u_val = x_u_val.to(device)
    x_u_test = x_u_test.to(device)

    # --------------------------------- X_v Preparation --------------------------------- #
    x_v, x_v_val, x_v_test = get_data(miRNA_dir)
    x_v = torch.tensor(x_v, dtype=torch.float32)
    x_v_val = torch.tensor(x_v_val, dtype=torch.float32)
    x_v_test = torch.tensor(x_v_test, dtype=torch.float32)

    scaler = StandardScaler()
    x_v = scaler.fit_transform(x_v)
    x_v = torch.tensor(x_v, dtype=torch.float32)
    
    scaler = StandardScaler()
    x_v_val = scaler.fit_transform(x_v_val)
    x_v_val = torch.tensor(x_v_val, dtype=torch.float32)

    scaler = StandardScaler()
    x_v_test = scaler.fit_transform(x_v_test)
    x_v_test = torch.tensor(x_v_test, dtype=torch.float32)

    x_v = x_v.t()               
    x_v_val = x_v_val.t()       
    x_v_test = x_v_test.t()     
    
    A_v = get_adjacency_matrix(x_v, threshold=adj_thresh)
    A_v = torch.tensor(A_v, dtype=torch.float32)
    A_v = A_v + torch.eye(A_v.size(0), device=A_v.device)
    degree_matrix = torch.diag(torch.sum(A_v, dim=1))
    degree_matrix_inv_sqrt = torch.pow(degree_matrix, -0.5)
    degree_matrix_inv_sqrt[torch.isinf(degree_matrix_inv_sqrt)] = 0. # handle division by zero
    A_v = torch.matmul(torch.matmul(degree_matrix_inv_sqrt, A_v), degree_matrix_inv_sqrt)       # m x m

    # Move data to GPU
    x_v = x_v.to(device)
    x_v_val = x_v_val.to(device)   
    x_v_test = x_v_test.to(device)

    # --------------------------------- Y Preparation --------------------------------- #
    y, y_val, y_test = get_data(label_dir)
    y = torch.tensor(y, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    y = y.view(-1, 1)
    y_val = y_val.view(-1, 1)
    y_test = y_test.view(-1, 1)

    # Move data to GPU
    y = y.to(device)
    y_val = y_val.to(device)
    y_test = y_test.to(device)

    # bipartite adjacency matrix
    bip = pickle.load(open(bip_path, 'rb'))
    bip = torch.tensor(bip, dtype=torch.float32)
    B_u, B_v = normalized_adjacency_bipartite(bip)  
    B_u = torch.tensor(B_u, dtype=torch.float32)
    B_v = torch.tensor(B_v, dtype=torch.float32)

    A_u = A_u.to(device)
    A_v = A_v.to(device)
    B_u = B_u.to(device)
    B_v = B_v.to(device)



    # --------------------------------- Model Training --------------------------------- #
    
    n = x_u.size(0)
    m = x_v.size(0)
    d = x_u.size(1)

    # Initialize the model
    model = MoGCN(input_features_u=n, input_features_v=m, num_layers=num_layers, hidden_dim=hidden_dim, bias=bias, k=k).to(device)

    # training the model end to end
    data = Data(x_u=x_u, x_v=x_v, y=y).to(device)
    loader = DataLoader([data], batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    # criterion = FocalLoss(gamma=2.0, alpha=0.25)

    # Training loop
    print('\nTraining the model...')
    model.train()

    # Variables for early stopping
    best_val_loss = float('inf')  
    patience = 10  
    epochs_without_improvement = 0  

    for epoch in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            out, _, _ = model(batch.x_u, batch.x_v, A_u, A_v, B_u)
            loss = criterion(out, batch.y)

            loss.backward()
            optimizer.step()

            # Calculate validation loss
            model.eval()
            with torch.no_grad():
                out_val, _, _ = model(x_u_val, x_v_val, A_u, A_v, B_u)
                val_loss = criterion(out_val, y_val)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} due to no improvement in validation loss.")
                break

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}', f'Val Loss: {val_loss.item():.4f}')
        
        if epochs_without_improvement >= patience:
            break


    # Evaluation
    print("\nTraining set metrics...")
    model.eval()
    with torch.no_grad():
        preds, h_u_train, h_v_train = model(data.x_u, data.x_v, A_u, A_v, B_u)
        preds = preds.cpu()  
        fpr, tpr, thresholds = precision_recall_curve(data.y.cpu(), preds, pos_label=1)
        auprc = auc(tpr, fpr)
        fpr, tpr, thresholds = roc_curve(data.y.cpu(), preds, pos_label=1)
        roc_auc = auc(fpr, tpr)
        best_threshold = get_best_threshold(fpr, tpr, thresholds)
        preds = (preds > best_threshold).float()
        acc = accuracy_score(data.y.cpu(), preds)
        f1 = f1_score(data.y.cpu(), preds)
        precision = precision_score(data.y.cpu(), preds)
        recall = recall_score(data.y.cpu(), preds)
        mcc = matthews_corrcoef(data.y.cpu(), preds)

        print("Accuracy: ", acc)
        print("F1 Score: ", f1)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("ROC AUC: ", roc_auc)
        print("AUPRC: ", auprc)
        print("MCC: ", mcc)

    print("\nValidation set metrics...")
    model.eval()
    with torch.no_grad():
        x_u_val.to(device)
        y_val = y_val.cpu().numpy()
        output, h_u_val, h_v_val = model(x_u_val, x_v_val, A_u, A_v, B_u)
        output = output.cpu()
        fpr, tpr, thresholds = precision_recall_curve(y_val, output, pos_label=1)
        val_auprc = auc(tpr, fpr)
        fpr, tpr, thresholds = roc_curve(y_val, output, pos_label=1)
        val_roc_auc = auc(fpr, tpr)
        best_threshold = get_best_threshold(fpr, tpr, thresholds)
        output = (output > best_threshold).float()
        val_acc = accuracy_score(y_val, output)
        val_f1 = f1_score(y_val, output)
        val_precision = precision_score(y_val, output)
        val_recall = recall_score(y_val, output)
        val_mcc = matthews_corrcoef(y_val, output)

        print("Accuracy: ", val_acc)
        print("F1 Score: ", val_f1)
        print("Precision: ", val_precision)
        print("Recall: ", val_recall)
        print("ROC AUC: ", val_roc_auc)
        print("AUPRC: ", val_auprc)
        print("MCC: ", val_mcc)

    model.eval()
    print("\nTesting set metrics...")
    with torch.no_grad():
        x_u_test.to(device)
        output, h_u_test, h_v_test = model(x_u_test, x_v_test, A_u, A_v, B_u)
        output = output.cpu()
        fpr, tpr, thresholds = precision_recall_curve(y_test.cpu(), output, pos_label=1)
        test_auprc = auc(tpr, fpr)
        fpr, tpr, thresholds = roc_curve(y_test.cpu(), output, pos_label=1)
        test_roc_auc = auc(fpr, tpr)
        best_threshold = get_best_threshold(fpr, tpr, thresholds)
        output = (output > best_threshold).float()
        test_acc = accuracy_score(y_test.cpu(), output)
        test_f1 = f1_score(y_test.cpu(), output)
        test_precision = precision_score(y_test.cpu(), output)
        test_recall = recall_score(y_test.cpu(), output)
        test_mcc = matthews_corrcoef(y_test.cpu(), output)
        
        print("Accuracy: ", test_acc)
        print("F1 Score: ", test_f1)
        print("Precision: ", test_precision)
        print("Recall: ", test_recall)
        print("ROC AUC: ", test_roc_auc)
        print("AUPRC: ", test_auprc)
        print("MCC: ", test_mcc)