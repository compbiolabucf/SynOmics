import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from utils import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, precision_recall_curve, auc, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from model import SynOmics
import argparse
import pandas as pd

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
    parser.add_argument('--k1', type=float, default=0.3, help='Message Passing Weight k1')
    parser.add_argument('--k2', type=float, default=0.4, help='Message Passing Weight k2')
    parser.add_argument('--mRNA_path', type=str, default='sample_data/mRNA.csv', help='Path of mRNA data')
    parser.add_argument('--miRNA_path', type=str, default='sample_data/miRNA.csv', help='Path of miRNA data')
    parser.add_argument('--DNA_Meth_path', type=str, default='sample_data/DNA_Meth.csv', help='Path of DNA_Meth data')
    parser.add_argument('--label_path', type=str, default='sample_data/label.csv', help='Path of label data')
    parser.add_argument('--label_mask', type=str, default=None, help='Path of label mask data')
    parser.add_argument('--bip12_path', type=str, default='sample_data/bip/bip_mRNA_miRNA.csv', help='Path of mRNA-miRNA bipartite data')
    parser.add_argument('--bip23_path', type=str, default='sample_data/bip/bip_miRNA_DNA_Meth.csv', help='Path of miRNA-DNA_Meth bipartite data')
    parser.add_argument('--bip31_path', type=str, default='sample_data/bip/bip_DNA_Meth_mRNA.csv', help='Path of DNA_Meth-mRNA bipartite data')
    parser.add_argument('--task', type=str, default='class', help='Task: classification or embedding generation') 
    args = parser.parse_args()

    num_layers = args.num_layers
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    hidden_dim = args.hidden_dim
    adj_thresh = args.adj_thresh
    bias = args.bias
    k1 = args.k1
    k2 = args.k2
    mRNA_path = args.mRNA_path
    miRNA_path = args.miRNA_path
    DNA_Meth_path = args.DNA_Meth_path
    label_path = args.label_path
    bip12_path = args.bip12_path
    bip23_path = args.bip23_path
    bip31_path = args.bip31_path
    task = args.task

    # Check if GPU is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # --------------------------------- Label Preparation --------------------------------- #
    y_orig = pd.read_csv(label_path, index_col=0, header=0)

    if args.label_mask:
        masks = pd.read_csv(args.label_mask, index_col=0, header=0)
        y = y_orig.loc[masks['mask']==1]
        y_val = y_orig.loc[masks['mask']==2]
        y_test = y_orig.loc[masks['mask']==3]
    else:
        y, y_test = train_test_split(y_orig, test_size=0.2, random_state=42, stratify=y_orig)
        y, y_val = train_test_split(y, test_size=0.2, random_state=42, stratify=y)

    train_samples = y.index
    val_samples = y_val.index
    test_samples = y_test.index

    y = y.values
    y_val = y_val.values
    y_test = y_test.values

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


    # --------------------------------- X_u Preparation --------------------------------- #
    x_u_orig = pd.read_csv(mRNA_path, index_col=0, header=0)

    x_u = x_u_orig.loc[train_samples].values
    x_u_val = x_u_orig.loc[val_samples].values
    x_u_test = x_u_orig.loc[test_samples].values

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
    x_v_orig = pd.read_csv(miRNA_path, index_col=0, header=0)

    x_v = x_v_orig.loc[train_samples].values
    x_v_val = x_v_orig.loc[val_samples].values
    x_v_test = x_v_orig.loc[test_samples].values

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

    # --------------------------------- X_w Preparation --------------------------------- #
    x_w_orig = pd.read_csv(DNA_Meth_path, index_col=0, header=0)

    x_w = x_w_orig.loc[train_samples].values
    x_w_val = x_w_orig.loc[val_samples].values
    x_w_test = x_w_orig.loc[test_samples].values

    x_w = torch.tensor(x_w, dtype=torch.float32)
    x_w_val = torch.tensor(x_w_val, dtype=torch.float32)
    x_w_test = torch.tensor(x_w_test, dtype=torch.float32)

    scaler = StandardScaler()
    x_w = scaler.fit_transform(x_w)
    x_w = torch.tensor(x_w, dtype=torch.float32)
    
    scaler = StandardScaler()
    x_w_val = scaler.fit_transform(x_w_val)
    x_w_val = torch.tensor(x_w_val, dtype=torch.float32)

    scaler = StandardScaler()
    x_w_test = scaler.fit_transform(x_w_test)
    x_w_test = torch.tensor(x_w_test, dtype=torch.float32)

    x_w = x_w.t()               
    x_w_val = x_w_val.t()       
    x_w_test = x_w_test.t()     
    
    A_w = get_adjacency_matrix(x_w, threshold=adj_thresh)
    A_w = torch.tensor(A_w, dtype=torch.float32)
    A_w = A_w + torch.eye(A_w.size(0), device=A_w.device)
    degree_matrix = torch.diag(torch.sum(A_w, dim=1))
    degree_matrix_inv_sqrt = torch.pow(degree_matrix, -0.5)
    degree_matrix_inv_sqrt[torch.isinf(degree_matrix_inv_sqrt)] = 0. # handle division by zero
    A_w = torch.matmul(torch.matmul(degree_matrix_inv_sqrt, A_w), degree_matrix_inv_sqrt)       # m x m

    # Move data to GPU
    x_w = x_w.to(device)
    x_w_val = x_w_val.to(device)   
    x_w_test = x_w_test.to(device)

    # ---------------------------------------------------------------------------------- #

    # bipartite adjacency matrix
    bip = pd.read_csv(bip12_path, index_col=0, header=0).values
    bip = torch.tensor(bip, dtype=torch.float32)
    B_uv, B_vu = normalized_adjacency_bipartite(bip)  
    B_uv = torch.tensor(B_uv, dtype=torch.float32)        
    B_vu = torch.tensor(B_vu, dtype=torch.float32)

    bip = pd.read_csv(bip23_path, index_col=0, header=0).values
    bip = torch.tensor(bip, dtype=torch.float32)
    B_vw, B_wv = normalized_adjacency_bipartite(bip)  
    B_vw = torch.tensor(B_vw, dtype=torch.float32)        
    B_wv = torch.tensor(B_wv, dtype=torch.float32)

    bip = pd.read_csv(bip31_path, index_col=0, header=0).values
    bip = torch.tensor(bip, dtype=torch.float32)
    B_wu, B_uw = normalized_adjacency_bipartite(bip)  
    B_wu = torch.tensor(B_wu, dtype=torch.float32)        
    B_uw = torch.tensor(B_uw, dtype=torch.float32)

    A_u = A_u.to(device)
    A_v = A_v.to(device)
    A_w = A_w.to(device)

    B_uv = B_uv.to(device)
    B_vu = B_vu.to(device)
    B_vw = B_vw.to(device)
    B_wv = B_wv.to(device)
    B_wu = B_wu.to(device)
    B_uw = B_uw.to(device)



    # --------------------------------- Model Training --------------------------------- #
    
    p = x_u.size(0)
    q = x_v.size(0)
    r = x_w.size(0)

    # Initialize the model
    model = SynOmics(input_features_u=p, input_features_v=q, input_features_w=r, 
                  num_layers=num_layers, hidden_dim=hidden_dim, bias=bias, k1=k1, k2=k2).to(device)

    # training the model end to end
    data = Data(x_u=x_u, x_v=x_v, x_w=x_w, y=y).to(device)
    loader = DataLoader([data], batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    # criterion = FocalLoss(gamma=2.0, alpha=0.25)

    # Training loop
    print('\nTraining the model...')
    model.train()

    # Variables for early stopping
    best_val_loss = float('inf')  # Initialize to a large value
    patience = 10  # Number of epochs to wait before stopping if no improvement
    epochs_without_improvement = 0  # Counter for epochs without improvement

    for epoch in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            out, _, _, _ = model(batch.x_u, batch.x_v, batch.x_w,
                        A_u, A_v, A_w,
                        B_uv, B_vw, B_wu)
            loss = criterion(out, batch.y)

            loss.backward()
            optimizer.step()

            # Calculate validation loss
            model.eval()
            with torch.no_grad():
                out_val, _, _, _ = model(x_u_val, x_v_val, x_w_val,
                                A_u, A_v, A_w,
                                B_uv, B_vw, B_wu)
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


    if task == 'class':
        print("\nTraining set metrics...")
        model.eval()
        with torch.no_grad():
            preds, _, _, _ = model(data.x_u, data.x_v, data.x_w, 
                        A_u, A_v, A_w,
                        B_uv, B_vw, B_wu)
            preds = preds.cpu()  
            fpr, tpr, thresholds = roc_curve(data.y.cpu(), preds, pos_label=1)
            roc_auc = auc(fpr, tpr)
            best_threshold = get_best_threshold(fpr, tpr, thresholds)
            preds = (preds > best_threshold).float()
            acc = accuracy_score(data.y.cpu(), preds)
            f1 = f1_score(data.y.cpu(), preds)
            mcc = matthews_corrcoef(data.y.cpu(), preds)

            print(f"Accuracy: {acc:.4f}")
            print(f"F1: {f1:.4f}")
            print(f"ROC: {roc_auc:.4f}")
            print(f"MCC: {mcc:.4f}")

        print("\nValidation set metrics...")
        model.eval()
        with torch.no_grad():
            x_u_val.to(device)
            y_val = y_val.cpu().numpy()
            output, _, _, _ = model(x_u_val, x_v_val, x_w_val,
                        A_u, A_v, A_w,
                        B_uv, B_vw, B_wu)
            output = output.cpu()
            fpr, tpr, thresholds = roc_curve(y_val, output, pos_label=1)
            val_roc_auc = auc(fpr, tpr)
            best_threshold = get_best_threshold(fpr, tpr, thresholds)
            output = (output > best_threshold).float()
            val_acc = accuracy_score(y_val, output)
            val_f1 = f1_score(y_val, output)
            val_mcc = matthews_corrcoef(y_val, output)

            print(f"Accuracy: {val_acc:.4f}")
            print(f"F1: {val_f1:.4f}")
            print(f"ROC: {val_roc_auc:.4f}")
            print(f"MCC: {val_mcc:.4f}")

        # Testing
        model.eval()
        print("\nTesting set metrics...")
        with torch.no_grad():
            x_u_test.to(device)
            output, _, _, _ = model(x_u_test, x_v_test, x_w_test,
                            A_u, A_v, A_w,
                            B_uv, B_vw, B_wu)
            output = output.cpu()
            fpr, tpr, thresholds = roc_curve(y_test.cpu(), output, pos_label=1)
            test_roc_auc = auc(fpr, tpr)
            best_threshold = get_best_threshold(fpr, tpr, thresholds)
            output = (output > best_threshold).float()
            test_acc = accuracy_score(y_test.cpu(), output)
            test_f1 = f1_score(y_test.cpu(), output)
            test_mcc = matthews_corrcoef(y_test.cpu(), output)
            
            print(f"Accuracy: {test_acc:.4f}")
            print(f"F1: {test_f1:.4f}")
            print(f"ROC: {test_roc_auc:.4f}")
            print(f"MCC: {test_mcc:.4f}")

    elif task == 'emb':
        # Generate embeddings
        model.eval()
        with torch.no_grad():
            _, h_u_train, h_v_train, h_w_train = model(data.x_u, data.x_v, data.x_w, 
                        A_u, A_v, A_w,
                        B_uv, B_vw, B_wu)
            _, h_u_val, h_v_val, h_w_val = model(x_u_val, x_v_val, x_w_val,
                        A_u, A_v, A_w,
                        B_uv, B_vw, B_wu)
            _, h_u_test, h_v_test, h_w_test = model(x_u_test, x_v_test, x_w_test,
                            A_u, A_v, A_w,
                            B_uv, B_vw, B_wu)
            h_u_train = h_u_train.cpu().numpy()
            h_v_train = h_v_train.cpu().numpy()
            h_w_train = h_w_train.cpu().numpy()
            h_u_val = h_u_val.cpu().numpy()
            h_v_val = h_v_val.cpu().numpy()
            h_w_val = h_w_val.cpu().numpy()
            h_u_test = h_u_test.cpu().numpy()
            h_v_test = h_v_test.cpu().numpy()
            h_w_test = h_w_test.cpu().numpy()

        h_train = np.concatenate([h_u_train, h_v_train, h_w_train], axis=0)
        h_val = np.concatenate([h_u_val, h_v_val, h_w_val], axis=0)
        h_test = np.concatenate([h_u_test, h_v_test, h_w_test], axis=0)
        
        h = np.concatenate([h_train.T, h_val.T, h_test.T], axis=0)
        np.save('emb/embeddings.npy', h)

        print("Embeddings saved successfully!")
        print("You can find the embeddings in the 'emb' folder.")

    else:
        raise ValueError("Task must be either 'class' or 'emb'.")
