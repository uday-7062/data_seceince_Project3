# Quick Reference Guide

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all components
python main.py

# Run individual components
python q1_frequent_subgraph.py
python q2_gnn.py
python q3_comparison.py
python q4_explainability.py
```

---

## 📊 Project Overview

**Goal**: Compare Frequent Subgraph Mining + Classic ML vs Graph Neural Networks for graph classification

**Dataset**: MUTAG (188 chemical compounds, binary classification)

**Components**:
- **Q1**: Frequent subgraph mining + Random Forest/SVM
- **Q2**: GNN architectures (GCN, GIN, GraphSAGE, GAT)
- **Q3**: Comprehensive comparison
- **Q4**: Explainability analysis

---

## 🎯 Key Results Summary

| Method | Accuracy | F1-Score | Training Time | Best For |
|--------|----------|----------|---------------|----------|
| Random Forest | ~75-80% | ~0.75 | ~1-5s | Speed, Interpretability |
| SVM | ~70-75% | ~0.70 | ~2-10s | Speed |
| GCN | ~85-90% | ~0.85 | ~30-60s | Balanced |
| GIN | ~88-92% | ~0.88 | ~40-80s | Accuracy |
| GraphSAGE | ~85-90% | ~0.85 | ~35-70s | Large graphs |
| GAT | ~88-92% | ~0.88 | ~50-100s | Accuracy, Attention |

---

## 📁 File Structure

```
├── main.py                    # Main execution
├── q1_frequent_subgraph.py    # Q-1: Classic ML
├── q2_gnn.py                  # Q-2: GNNs
├── q3_comparison.py           # Q-3: Comparison
├── q4_explainability.py       # Q-4: Explainability
├── data_loader.py             # Data loading
├── utils.py                   # Utilities
├── results/                   # Output directory
│   ├── *.json                 # Results files
│   ├── *.csv                  # Comparison tables
│   ├── *.png                  # Visualizations
│   └── *.pt                   # Saved models
└── data/                      # Dataset cache
```

---

## 🔑 Key Concepts

### Frequent Subgraph Mining
- Extract common patterns from graphs
- Use patterns as features
- Train classic ML models

### Graph Neural Networks
- Learn graph representations end-to-end
- Message passing between nodes
- Graph-level pooling for classification

### Explainability
- **GNNExplainer**: Edge importance scores
- **Feature Importance**: Pattern importance scores

---

## 📈 Metrics Explained

- **Accuracy**: % of correct predictions
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: Of predicted positives, how many are correct?
- **Recall**: Of actual positives, how many did we find?
- **Fidelity+**: Important edges preserve prediction
- **Fidelity-**: Removing important edges changes prediction
- **Sparsity**: Fraction of edges considered important

---

## 💡 When to Use What?

**Use Classic ML when:**
- ✅ Speed is critical
- ✅ Interpretability needed
- ✅ Limited resources
- ✅ Small datasets

**Use GNNs when:**
- ✅ Accuracy is critical
- ✅ Complex patterns needed
- ✅ GPU available
- ✅ Large datasets

---

## 📚 Documentation Files

- **PROJECT_SUMMARY.md**: Comprehensive project overview
- **PRESENTATION_NOTES.md**: Slide-by-slide presentation guide
- **PROJECT_EXPLANATION.md**: Detailed explanation of concepts
- **QUICK_REFERENCE.md**: This file (quick lookup)

---

## 🐛 Common Issues

**Issue**: Dataset not downloading
- **Solution**: Check internet connection, PyTorch Geometric will download automatically

**Issue**: Out of memory
- **Solution**: Reduce batch size in q2_gnn.py, or use smaller max_size in q1

**Issue**: GPU not found
- **Solution**: GNNs will use CPU (slower but works)

---

## 📞 Quick Commands

```bash
# Check dataset
python check_dataset.py

# Verify setup
python verify_setup.py

# Run with specific component
python main.py --q1    # Only Q-1
python main.py --q2    # Only Q-2
python main.py --all   # All components
```

---

## 🎓 Key Takeaways

1. **GNNs achieve better accuracy** (10-15% improvement)
2. **Classic ML is faster** (5-10x speedup)
3. **Both have their place** depending on requirements
4. **Explainability methods** provide valuable insights
5. **No one-size-fits-all** solution

---

**For detailed information, see PROJECT_EXPLANATION.md**
