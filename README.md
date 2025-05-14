# MuLAAIP: Multi-Modality Representation Learning for Antibody-Antigen Interaction Prediction

This repository contains the official implementation of **MuLAAIP**, a novel deep learning framework for predicting antibody-antigen interactions (AAI) by integrating **3D structural** and **1D sequence** data. Our approach addresses critical challenges in AAI prediction, including structural data scarcity, sequence-structure dependency modeling, and imbalanced label distributions.

## Key Features
- **Multi-Modality Framework**:  
  - Captures hierarchical relationships at residue/backbone/side-chain levels via **3D geometric graphs**.
  - Incorporates sequence information using protein language models (e.g., ESM2, ProtTrans).
- **Innovative Architecture**:  
  - **Graph Attention Networks**: Extracts structural features from 3D atomic coordinates.
  - **Normalized Adaptive Graph Convolution**: Models inter-protein sequence associations.
  - Hybrid fusion of structural/sequence representations for interaction prediction.
- **Comprehensive Benchmark**:  
  - Includes 4 datasets with structural/sequence labels:
    - **Wild-type/Mutant-type Affinity**: 1,191/1,742 antibody-antigen pairs.
    - **Alphaseq**: 248k antibodies with SARS-CoV-2 mutations.
    - **SARS-CoV-2 Neutralization**: 310 labeled pairs (228 positive/82 negative).
  - All structures predicted via ESMFold when experimental data is unavailable.

## Performance Highlights
- **SOTA Results**:  
  - Achieves **75.7% ROC-AUC** on SARS-CoV-2 neutralization prediction (vs. 69.6% for PIPR).
  - Reduces MAE by **15-20%** on binding affinity tasks across wild-type/mutant datasets.
- **Robustness**:  
  - Handles imbalanced data effectively (e.g., 54.9% MCC on SARS-CoV-2 neutralization vs. 27.8% for ProtTrans).
  - Maintains accuracy even with predicted (ESMFold) structures.

## Getting Started
```bash
# Clone the repo
git clone https://github.com/trashTian/MuLAAIP.git 
cd MuLAAIP

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py --config configs/mulaaip.yaml

# Evaluate on benchmarks
python evaluate.py --checkpoint saved_models/best.pth

Obtain benchmark: https://pan.baidu.com/s/1HqXfAUIjGp6h1gh3M2Pa8Q. Extract code: iuqs 
