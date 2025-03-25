# GCN-Omics

This repository contains the code for GCN-Omics, a multi-omics integration model that utilizes Graph Convolution Networks for the downstream task of disease classification. The model uses separate and parallel GCNs to learn the intra-omics and inter-omics interactions.

## Overall Framework
<img src="GCN-Omics/Arch.jpg">

## How To Run

First, install the dependencies using the requirements.txt file. It is recommended to create a python virtual environment first.
```bash
pip install -r requirements.txt
```

### Input Data
The model primarily utilizes two types of omics data: Gene Expression and miRNA Expression. It can also be extended to incorporate any number of omics data type. Here, we use DNA Methylation as the third omics type. There is a set of sample data from the TCGA Breast Cancer (BRCA) for the Estrogen Receptor (ER) label in the "sample_data" directory. The directory contains <br>
1. Separate train, validation and test files for each of the omics in specific directories (i.e. mRNA, miRNA, DNA_Meth). The files are organized into *(sample x value)* manner. <br>
2. Separate label files for train, validation and test data in "labels" directory. The labels are 0-indexed and structured in a 1-D array of samples. <br>
3. The bipartite graphs stored in "bip_{omic1}_{omic2}.pkl" files in "bip" directory, which are structured into *(omic1 features x omic2 features)* manner. The omics are named "mRNA", "miRNA" and "DNA_Meth" respectively. <br>

To run successfully, the number of samples in each train, validation or test data must be the same. The number of samples in the label data must also be the same as the number of samples in the omics data. The feature dimensions of the bipartite graphs must match the number of features in the respective omics data.

### Dual Alignment
To run the model with Dual Alignment module with the sample data, run the following command:
```bash
python DA/train.py --mRNA_dir=sample_data/mRNA --miRNA_dir=sample_data/miRNA --label_dir=sample_data/labels --bip_path=sample_data/bip/bip_mRNA_miRNA.pkl
```

The model is flexible to be trained with different hyper-parameter settings. The hyper-parameters are:

| **Parameter**       | **Description**                                                                                                      |
|---------------------|----------------------------------------------------------------------------------------------------------------------|
| `num_layers`        | Number of GCN-Omics layers                                                                                           |
| `batch_size`        | Batch size for the input data                                                                                        |
| `gcn_epochs`        | Number of epochs for the pretraining of the GCN-Omics layers                                                          |
| `epochs`            | Number of layers to train the model end-to-end                                                                       |
| `lr`                | Learning rate                                                                                                        |
| `hidden_dim`        | Hidden representation dimension                                                                                      |
| `adj_thresh`        | Adjacency matrix threshold for the intra-omics graph (must be within the range [0, 1]. A higher value makes the graph sparser) |
| `alpha`             | Weight for the reconstruction loss (must be within the range [0, 1])                                                 |
| `bias`              | Enable training with bias (True or False)                                                                             |
| `mRNA_dir`          | Directory containing the mRNA data                                                                                   |
| `miRNA_dir`         | Directory containing the miRNA data                                                                                  |
| `label_dir`         | Directory containing the label data                                                                                  |
| `bip_path`          | Path to the bipartite graph file                                                                                      |


To train with different hyper-parameter settings, train using
```bash
python train.py --<hyper-parameter1>=<value> --<hyper-parameter2>=<value> ...
```

### Weighted Message Passing
To run the model with Weighted Message Passing module with the sample data, run the following command:
```bash
python WMP/train.py --mRNA_dir=sample_data/mRNA --miRNA_dir=sample_data/miRNA --label_dir=sample_data/labels --bip_path=sample_data/bip/bip_mRNA_miRNA.pkl
```

The model is flexible to be trained with different hyper-parameter settings. The hyper-parameters are:

| **Parameter**       | **Description**                                                                                                      |
|---------------------|----------------------------------------------------------------------------------------------------------------------|
| `num_layers`        | Number of GCN-Omics layers                                                                                           |
| `batch_size`        | Batch size for the input data                                                                                        |
| `k`                 | Weight for the intra-omics data (must be within the range [0, 1])                                                     |
| `epochs`            | Number of layers to train the model                                                                                  |
| `lr`                | Learning rate                                                                                                        |
| `hidden_dim`        | Hidden representation dimension                                                                                      |
| `adj_thresh`        | Adjacency matrix threshold for the intra-omics graph (must be within the range [0, 1]. A higher value makes the graph sparser) |
| `bias`              | Enable training with bias (True or False)                                                                             |
| `mRNA_dir`          | Directory containing the mRNA data                                                                                   |
| `miRNA_dir`         | Directory containing the miRNA data                                                                                  |
| `label_dir`         | Directory containing the label data                                                                                  |
| `bip_path`          | Path to the bipartite graph file                                                                                      |


To train with different hyper-parameter settings, train using
```bash
python train.py --<hyper-parameter1>=<value> --<hyper-parameter2>=<value> ...
```

### Extended Model of 3 Omics
To run the model with 3 omics data on the sample data, run the following command:
```bash
python 3_Omics/train.py --mRNA_dir=sample_data/mRNA --miRNA_dir=sample_data/miRNA --DNA_Meth_dir=sample_data/DNA_Meth --label_dir=sample_data/labels --bip1_path=sample_data/bip/bip_mRNA_miRNA.pkl --bip2_path=sample_data/bip/bip_miRNA_DNA_Meth.pkl --bip3_path=sample_data/bip/bip_DNA_Meth_mRNA.pkl
```

The model is flexible to be trained with different hyper-parameter settings. The hyper-parameters are:

| **Parameter**       | **Description**                                                                                                      |
|---------------------|----------------------------------------------------------------------------------------------------------------------|
| `num_layers`        | Number of GCN-Omics layers                                                                                           |
| `batch_size`        | Batch size for the input data                                                                                        |
| `k1`                | Weight for the contribution of the first inter-omics data (must be within the range [0, 1])                           |
| `k2`                | Weight for the contribution of the second inter-omics data (must be within the range [0, 1])                          |
| `epochs`            | Number of layers to train the model                                                                                  |
| `lr`                | Learning rate                                                                                                        |
| `hidden_dim`        | Hidden representation dimension                                                                                      |
| `adj_thresh`        | Adjacency matrix threshold for the intra-omics graph (must be within the range [0, 1]. A higher value makes the graph sparser) |
| `bias`              | Enable training with bias (True or False)                                                                             |
| `mRNA_dir`          | Directory containing the mRNA data                                                                                   |
| `miRNA_dir`         | Directory containing the miRNA data                                                                                  |
| `DNA_Meth_dir`      | Directory containing the DNA Methylation data                                                                         |
| `label_dir`         | Directory containing the label data                                                                                  |
| `bip1_path`         | Path to the first bipartite graph file                                                                                |
| `bip2_path`         | Path to the second bipartite graph file                                                                               |
| `bip3_path`         | Path to the third bipartite graph file                                                                                |


To train with different hyper-parameter settings, train using
```bash
python train.py --<hyper-parameter1>=<value> --<hyper-parameter2>=<value> ...
```
