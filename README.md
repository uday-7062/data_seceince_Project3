# CS 6010 Project 3: Graph Classification on MUTAG Dataset

## Project Overview

This project implements and compares multiple graph classification approaches on the MUTAG dataset:
1. **Frequent Subgraph Mining + Classic ML**: Using gSpan algorithm with Random Forest and SVM
2. **Graph Neural Networks**: GCN, GIN, GraphSAGE, and GAT architectures
3. **Comparison**: Experimental evaluation of both approaches
4. **Explainability**: GNNExplainer for GNNs and feature importance for classic ML

## Dataset

The MUTAG dataset contains 188 chemical compound graphs with binary classification (mutagenic/non-mutagenic). Average graph size: 18 nodes, 20 edges.

Source: https://huggingface.co/datasets/graphs-datasets/MUTAG

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── README.md
├── requirements.txt
├── main.py                    # Main execution script
├── data_loader.py            # Dataset loading utilities
├── q1_frequent_subgraph.py   # Q-1: Frequent subgraph mining + Classic ML
├── q2_gnn.py                 # Q-2: Graph Neural Networks
├── q3_comparison.py          # Q-3: Comparison framework
├── q4_explainability.py      # Q-4: Explainability analysis
├── utils.py                  # Utility functions
└── results/                  # Output directory for results
```

## Usage

### Run all experiments:
```bash
python main.py
```

### Run individual components:
```bash
# Q-1: Frequent Subgraph Mining + Classic ML
python q1_frequent_subgraph.py

# Q-2: Graph Neural Networks
python q2_gnn.py

# Q-3: Comparison
python q3_comparison.py

# Q-4: Explainability
python q4_explainability.py
```

## Key Features

- **Q-1**: Frequent subgraph mining (DFS-based pattern extraction) with Random Forest and SVM classifiers, including ablation studies on mining thresholds and model parameters
- **Q-2**: Multiple GNN architectures (GCN, GIN, GraphSAGE, GAT) with hyperparameter tuning and ablation studies
- **Q-3**: Comprehensive comparison between classic ML and GNN approaches with multiple metrics (accuracy, F1, precision, recall, runtime)
- **Q-4**: GNNExplainer implementation for GNN models with fidelity+, fidelity-, sparsity metrics, and comparison with classic ML feature importance

## Implementation Notes

- **Frequent Subgraph Mining**: We implement a simplified DFS-based frequent subgraph mining approach. While not the full gSpan algorithm, it effectively extracts frequent patterns for feature construction.
- **GNN Models**: All GNN architectures are implemented from scratch using PyTorch Geometric.
- **Explainability**: GNNExplainer is implemented as a simplified version that optimizes edge masks to explain predictions.

## Results

Results are saved in the `results/` directory, including:
- Model performance metrics
- Ablation study results
- Comparison plots and tables
- Explanation visualizations

## Dependencies

See `requirements.txt` for full list. Key libraries:
- PyTorch Geometric for GNN implementations
- scikit-learn for classic ML models
- gspan-mining for frequent subgraph mining
- NetworkX for graph utilities

## Authors

[Your Group Name/Names]

## References

- Kong, X., & Yu, P. S. (2010). Multi-Label Feature Selection for Graph Classification. ICDM.
- Dwivedi, V. P., et al. (2022). Benchmarking Graph Neural Networks. JMLR.
- Ying, R., et al. (2019). GNNExplainer: Generating Explanations for Graph Neural Networks. NeurIPS.

