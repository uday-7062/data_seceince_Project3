# CS 6010 Project 3: Graph Classification on MUTAG Dataset
## Comprehensive Project Summary

---

## 📋 Project Overview

This project implements and compares multiple graph classification approaches on the **MUTAG dataset**, a benchmark dataset for graph classification tasks. The project explores two fundamentally different paradigms:

1. **Traditional Machine Learning Approach**: Frequent Subgraph Mining + Classic ML Models
2. **Deep Learning Approach**: Graph Neural Networks (GNNs)

The project includes comprehensive evaluation, comparison, and explainability analysis of both approaches.

---

## 🎯 Project Objectives

1. **Q-1**: Implement frequent subgraph mining with classic ML classifiers (Random Forest, SVM)
2. **Q-2**: Implement multiple GNN architectures (GCN, GIN, GraphSAGE, GAT) for graph classification
3. **Q-3**: Compare both approaches across multiple metrics (accuracy, F1, precision, recall, runtime)
4. **Q-4**: Analyze explainability using GNNExplainer for GNNs and feature importance for classic ML

---

## 📊 Dataset: MUTAG

- **Source**: Hugging Face Datasets (`graphs-datasets/MUTAG`)
- **Size**: 188 chemical compound graphs
- **Task**: Binary classification (mutagenic vs non-mutagenic compounds)
- **Average Graph Size**: ~18 nodes, ~20 edges per graph
- **Domain**: Chemistry/Bioinformatics
- **Significance**: Standard benchmark for graph classification algorithms

---

## 🏗️ Project Architecture

### Component 1: Q-1 - Frequent Subgraph Mining + Classic ML

**Approach:**
- **Frequent Subgraph Mining**: DFS-based pattern extraction (simplified gSpan algorithm)
- **Feature Engineering**: Convert graphs to feature vectors based on frequent subgraph patterns
- **Classifiers**: Random Forest and Support Vector Machine (SVM)

**Key Features:**
- Extracts frequent subgraphs up to a maximum size (default: 4 nodes)
- Filters patterns by minimum support threshold (default: 30%)
- Builds feature vectors where each dimension represents the count of a frequent pattern
- Includes ablation studies on:
  - Minimum support threshold (0.1 to 0.5)
  - Maximum subgraph size (3 to 6 nodes)
  - Model hyperparameters

**Implementation Details:**
- `SimpleGraphMiner` class: Implements DFS-based subgraph extraction
- Pattern mining from each class separately
- Feature vector construction based on pattern counts
- StandardScaler for SVM preprocessing

---

### Component 2: Q-2 - Graph Neural Networks

**Approach:**
- **Four GNN Architectures**:
  1. **GCN (Graph Convolutional Network)**: Spectral-based convolution
  2. **GIN (Graph Isomorphism Network)**: Provably powerful for graph classification
  3. **GraphSAGE**: Inductive learning with neighbor sampling
  4. **GAT (Graph Attention Network)**: Attention mechanism for edge weighting

**Key Features:**
- End-to-end learning from raw graph structure
- Batch normalization and dropout for regularization
- Global pooling (mean pooling) for graph-level representation
- Hyperparameter tuning and ablation studies:
  - Hidden dimensions: [32, 64, 128]
  - Number of layers: [2, 3, 4, 5]
  - Learning rate scheduling with ReduceLROnPlateau

**Training Details:**
- Train/Validation/Test split: 60%/20%/20%
- Batch size: 32
- Optimizer: Adam with weight decay (5e-4)
- Loss: Cross-entropy
- Early stopping based on validation accuracy

---

### Component 3: Q-3 - Comprehensive Comparison

**Comparison Metrics:**

1. **Quality Metrics:**
   - Accuracy
   - F1-Score
   - Precision
   - Recall

2. **Efficiency Metrics:**
   - Training time
   - Testing time
   - Total runtime

**Analysis:**
- Side-by-side comparison tables
- Visualization plots (bar charts, scatter plots)
- Quality vs Efficiency tradeoff analysis
- Best method identification for each metric

**Outputs:**
- Comparison tables (CSV)
- Quality metrics plots
- Efficiency metrics plots
- Tradeoff analysis visualization

---

### Component 4: Q-4 - Explainability Analysis

**GNN Explainability:**
- **GNNExplainer**: Post-hoc explanation method
  - Optimizes edge masks to identify important edges
  - Metrics:
    - **Fidelity+**: How well important edges preserve prediction
    - **Fidelity-**: How prediction changes when removing important edges
    - **Sparsity**: Fraction of edges considered important
  - Visualizes edge importance on graphs

**Classic ML Explainability:**
- **Feature Importance**: Direct interpretability
  - Random Forest: Built-in feature importance scores
  - SVM: Coefficient analysis (for linear kernels)
  - Pattern-level explanations (which subgraph patterns matter)

**Comparison:**
- GNNExplainer provides edge-level explanations
- Classic ML provides pattern-level explanations
- Different interpretability paradigms

---

## 📁 Project Structure

```
Data_Sceince_Project3/
├── README.md                    # Project documentation
├── requirements.txt              # Python dependencies
├── main.py                       # Main execution script
├── data_loader.py                # Dataset loading utilities
├── q1_frequent_subgraph.py       # Q-1 implementation
├── q2_gnn.py                     # Q-2 implementation
├── q3_comparison.py              # Q-3 implementation
├── q4_explainability.py          # Q-4 implementation
├── utils.py                      # Utility functions
├── check_dataset.py              # Dataset verification
├── verify_setup.py               # Environment verification
├── results/                       # Output directory
│   ├── q1_classic_ml_results.json
│   ├── q1_ablation_results.json
│   ├── q2_gnn_results.json
│   ├── q2_gnn_ablation_results.json
│   ├── q3_comparison_results.json
│   ├── q3_comparison_table.csv
│   ├── q4_explainability_results.json
│   └── *.pt (saved model files)
└── data/                         # Dataset cache
```

---

## 🔧 Technical Implementation

### Technologies Used:
- **PyTorch Geometric**: GNN implementations and graph data structures
- **scikit-learn**: Classic ML models (Random Forest, SVM)
- **NetworkX**: Graph manipulation and visualization
- **NumPy/Pandas**: Data processing
- **Matplotlib/Seaborn**: Visualization

### Key Design Decisions:

1. **Frequent Subgraph Mining**: Simplified DFS-based approach (not full gSpan) for efficiency
2. **GNN Architectures**: Implemented from scratch using PyTorch Geometric primitives
3. **Evaluation**: Comprehensive metrics for fair comparison
4. **Explainability**: Simplified GNNExplainer implementation focusing on edge importance

---

## 📈 Expected Results & Insights

### Performance Comparison:
- **GNNs** typically achieve higher accuracy due to end-to-end learning
- **Classic ML** is faster but may have lower accuracy
- **Tradeoff**: Quality vs Efficiency

### Key Findings:
1. GNNs capture complex graph structures better
2. Classic ML is more interpretable and faster
3. Different approaches suit different use cases
4. Explainability methods provide complementary insights

---

## 🚀 Usage

### Installation:
```bash
pip install -r requirements.txt
```

### Run All Components:
```bash
python main.py
```

### Run Individual Components:
```bash
python q1_frequent_subgraph.py  # Q-1
python q2_gnn.py                # Q-2
python q3_comparison.py         # Q-3
python q4_explainability.py     # Q-4
```

---

## 📚 References

1. Kong, X., & Yu, P. S. (2010). Multi-Label Feature Selection for Graph Classification. ICDM.
2. Dwivedi, V. P., et al. (2022). Benchmarking Graph Neural Networks. JMLR.
3. Ying, R., et al. (2019). GNNExplainer: Generating Explanations for Graph Neural Networks. NeurIPS.
4. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR.
5. Xu, K., et al. (2019). How Powerful are Graph Neural Networks? ICLR.

---

## 🎓 Learning Outcomes

This project demonstrates:
- Understanding of graph representation learning
- Implementation of both traditional and deep learning approaches
- Comprehensive experimental evaluation
- Explainability analysis for both paradigms
- Tradeoff analysis between quality and efficiency

---

## 📝 Notes

- Results are saved in JSON format for reproducibility
- Models are saved for explainability analysis
- Ablation studies help understand parameter sensitivity
- All visualizations are saved as high-resolution PNG files
