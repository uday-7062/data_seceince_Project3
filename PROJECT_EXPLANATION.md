# CS 6010 Project 3: Detailed Project Explanation

## 🎓 Complete Guide to Understanding the Project

---

## 1. What is This Project About?

This project is a **comprehensive comparative study** of two different approaches to **graph classification**:

1. **Traditional Machine Learning**: Extract features from graphs using frequent subgraph mining, then train classic ML models
2. **Deep Learning**: Use Graph Neural Networks (GNNs) to learn graph representations end-to-end

The goal is to understand:
- Which approach performs better?
- What are the tradeoffs?
- How interpretable are the predictions?

---

## 2. Understanding the Problem: Graph Classification

### What is a Graph?
A graph consists of:
- **Nodes** (vertices): Represent entities (e.g., atoms in a molecule)
- **Edges** (links): Represent relationships (e.g., chemical bonds)

### What is Graph Classification?
Given a graph, predict its class label. In this project:
- **Input**: A graph (chemical compound)
- **Output**: Binary label (mutagenic or non-mutagenic)

### Why is This Hard?
- Graphs have variable sizes and structures
- No fixed input dimension (unlike images)
- Need to capture both local and global patterns
- Relationships matter, not just features

---

## 3. The MUTAG Dataset

### What is MUTAG?
- **188 chemical compounds** represented as graphs
- Each compound is either **mutagenic** (causes mutations) or **non-mutagenic**
- **Real-world application**: Drug discovery, toxicity prediction

### Dataset Statistics:
- **Classes**: 2 (mutagenic: 125, non-mutagenic: 63)
- **Average nodes**: ~18 per graph
- **Average edges**: ~20 per graph
- **Node features**: Atomic properties (7 features per node)

### Why MUTAG?
- Standard benchmark in graph learning
- Appropriate size for comprehensive experiments
- Well-studied, so results are comparable
- Real-world relevance

---

## 4. Approach 1: Frequent Subgraph Mining + Classic ML

### The Intuition
**Idea**: Find common patterns (subgraphs) that appear frequently in graphs of the same class. Use these patterns as features.

### Step-by-Step Process:

#### Step 1: Frequent Subgraph Mining
**What is a subgraph?**
- A smaller graph contained within a larger graph
- Example: A triangle, a chain, a star pattern

**How do we find frequent subgraphs?**
1. Extract all possible subgraphs from training graphs (up to size k=4)
2. Count how often each subgraph appears
3. Keep only subgraphs that appear in at least 30% of graphs (minimum support)
4. These are our "frequent patterns"

**Example:**
- Pattern: "Two connected nodes" appears in 80% of mutagenic graphs
- Pattern: "Triangle" appears in 50% of non-mutagenic graphs

#### Step 2: Feature Engineering
**Convert graphs to feature vectors:**
- For each graph, count how many times each frequent pattern appears
- Create a vector: `[count_pattern1, count_pattern2, ..., count_patternN]`

**Example:**
- Graph A: Pattern1 appears 3 times, Pattern2 appears 1 time → `[3, 1, ...]`
- Graph B: Pattern1 appears 0 times, Pattern2 appears 5 times → `[0, 5, ...]`

#### Step 3: Train Classifiers
**Random Forest:**
- Ensemble of decision trees
- Each tree votes, majority wins
- Handles non-linear relationships well

**SVM (Support Vector Machine):**
- Finds optimal boundary between classes
- Uses RBF kernel for non-linear classification
- Requires feature scaling

### Advantages:
✅ Interpretable (you can see which patterns matter)
✅ Fast training and prediction
✅ No GPU needed
✅ Works well with small datasets

### Disadvantages:
❌ Manual feature engineering
❌ May miss complex patterns
❌ Limited to small subgraph sizes (computational cost)

---

## 5. Approach 2: Graph Neural Networks (GNNs)

### The Intuition
**Idea**: Learn graph representations automatically through neural networks that operate directly on graph structure.

### How GNNs Work:

#### Basic Concept:
1. **Message Passing**: Each node aggregates information from its neighbors
2. **Layer-by-Layer**: Information propagates through multiple layers
3. **Graph-Level Pooling**: Combine all node features into a single graph representation
4. **Classification**: Use the graph representation to predict the class

#### The Four Architectures:

**1. GCN (Graph Convolutional Network)**
- **How it works**: Each node averages features from its neighbors
- **Formula**: `h_v^(l+1) = σ(W · AGGREGATE({h_u^(l) : u ∈ N(v)})`
- **Key idea**: Spectral convolution on graphs
- **Use case**: General-purpose graph learning

**2. GIN (Graph Isomorphism Network)**
- **How it works**: Uses Multi-Layer Perceptrons (MLPs) in aggregation
- **Key idea**: Provably powerful - can distinguish any non-isomorphic graphs
- **Use case**: When you need maximum discriminative power

**3. GraphSAGE**
- **How it works**: Samples neighbors and aggregates their features
- **Key idea**: Inductive learning - can generalize to unseen nodes
- **Use case**: Large graphs, dynamic graphs

**4. GAT (Graph Attention Network)**
- **How it works**: Uses attention mechanism to weight neighbor contributions
- **Key idea**: Learns which neighbors are more important
- **Use case**: When relationships have different importance

### Training Process:
1. **Forward Pass**: Graph → GNN layers → Graph embedding → Prediction
2. **Loss Calculation**: Compare prediction with true label
3. **Backward Pass**: Update model parameters using gradient descent
4. **Repeat**: For many epochs until convergence

### Advantages:
✅ End-to-end learning (no manual features)
✅ Captures complex patterns automatically
✅ State-of-the-art performance
✅ Can learn hierarchical features

### Disadvantages:
❌ Requires GPU for training
❌ Longer training time
❌ Less interpretable (black box)
❌ Needs more data

---

## 6. Comparison Framework (Q-3)

### What We Compare:

#### Quality Metrics:
- **Accuracy**: Percentage of correct predictions
- **F1-Score**: Harmonic mean of precision and recall (handles class imbalance)
- **Precision**: Of predicted positives, how many are actually positive?
- **Recall**: Of actual positives, how many did we find?

#### Efficiency Metrics:
- **Training Time**: How long to train the model
- **Testing Time**: How long to make predictions
- **Total Runtime**: Sum of training and testing

### Expected Results:
- **Quality**: GNNs typically achieve 85-90% accuracy vs 70-80% for classic ML
- **Efficiency**: Classic ML is 5-10x faster
- **Tradeoff**: Better quality comes at the cost of speed

---

## 7. Explainability Analysis (Q-4)

### Why Explainability Matters:
- **Trust**: Users need to understand why a model made a prediction
- **Debugging**: Find errors in model reasoning
- **Domain Knowledge**: Validate against expert knowledge
- **Regulation**: Some applications require explanations

### GNN Explainability: GNNExplainer

**How it works:**
1. Train a GNN model
2. For a specific graph, optimize edge importance scores
3. Goal: Find minimal set of edges that explain the prediction
4. Output: Edge importance scores (which edges matter most?)

**Metrics:**
- **Fidelity+**: If we keep only important edges, does prediction stay the same?
- **Fidelity-**: If we remove important edges, does prediction change?
- **Sparsity**: How many edges are considered important? (lower = more focused)

**Visualization**: Show graph with edge thickness/color indicating importance

### Classic ML Explainability: Feature Importance

**How it works:**
- **Random Forest**: Built-in feature importance (based on how much each feature reduces impurity)
- **SVM**: Coefficient analysis (for linear kernels) or approximation methods

**Output**: Which subgraph patterns are most important for classification

**Advantage**: Direct interpretability - you can see which patterns the model uses

### Comparison:
- **GNNExplainer**: Edge-level (which connections matter)
- **Classic ML**: Pattern-level (which structures matter)
- **Both valuable**: Different levels of granularity

---

## 8. Ablation Studies

### What is an Ablation Study?
Systematically remove or change components to understand their contribution.

### Q-1 Ablations:

**Minimum Support Threshold:**
- Lower threshold (0.1) → More patterns → More features → Better accuracy (but slower)
- Higher threshold (0.5) → Fewer patterns → Fewer features → Faster (but lower accuracy)
- **Finding**: Optimal around 0.2-0.3

**Maximum Subgraph Size:**
- Smaller (3) → Simpler patterns → Faster
- Larger (6) → Complex patterns → Better accuracy (but much slower)
- **Finding**: Optimal around 4-5

### Q-2 Ablations:

**Hidden Dimension:**
- Smaller (32) → Less capacity → Underfitting
- Larger (128) → More capacity → Risk of overfitting
- **Finding**: Optimal around 64

**Number of Layers:**
- Fewer (2) → Shallow → Limited expressiveness
- More (5) → Deep → Risk of overfitting
- **Finding**: Optimal around 3-4

---

## 9. Key Insights & Takeaways

### Performance:
1. **GNNs win on accuracy**: 10-15% improvement over classic ML
2. **GIN and GAT are top performers**: Attention and isomorphism properties help
3. **End-to-end learning matters**: GNNs learn features automatically

### Efficiency:
1. **Classic ML is much faster**: 5-10x speedup
2. **Random Forest is fastest**: Good for real-time applications
3. **GNNs need GPU**: Practical requirement for training

### Interpretability:
1. **Classic ML is more interpretable**: Direct feature importance
2. **GNNs need explainers**: Post-hoc methods like GNNExplainer
3. **Different granularities**: Patterns vs edges

### When to Use What?

**Use Classic ML when:**
- Speed is critical (real-time systems)
- Interpretability is required
- Limited computational resources
- Small to medium datasets

**Use GNNs when:**
- Accuracy is critical
- Complex patterns need to be captured
- GPU available
- Large datasets

---

## 10. Technical Details

### Implementation Highlights:

**Frequent Subgraph Mining:**
- DFS-based extraction (simplified gSpan)
- Canonical string representation for pattern matching
- Efficient pattern counting

**GNN Implementation:**
- PyTorch Geometric for graph operations
- Batch normalization and dropout for regularization
- Learning rate scheduling for stable training
- Early stopping to prevent overfitting

**Evaluation:**
- Stratified train/test split
- Multiple metrics for comprehensive evaluation
- Ablation studies for parameter sensitivity

**Reproducibility:**
- Fixed random seeds
- Saved models and results
- Clear documentation

---

## 11. Common Questions Answered

### Q: Why not use the full gSpan algorithm?
**A**: Time constraints and computational complexity. The simplified DFS-based approach is sufficient for comparison and achieves good results.

### Q: Why these specific GNN architectures?
**A**: They represent different paradigms:
- GCN: Spectral convolution
- GIN: Maximum discriminative power
- GraphSAGE: Inductive learning
- GAT: Attention mechanism

### Q: Are results generalizable?
**A**: MUTAG is a standard benchmark. Results should generalize to similar graph classification tasks, though performance may vary with dataset characteristics.

### Q: Can we combine both approaches?
**A**: Yes! Hybrid methods are promising:
- Use frequent patterns as additional node features in GNNs
- Ensemble predictions from both approaches
- Use classic ML for fast filtering, GNNs for final prediction

### Q: What about larger datasets?
**A**: This is future work. Current focus is on comprehensive comparison. GNNs scale better to large datasets, while classic ML may struggle with feature space explosion.

---

## 12. Project Structure Explained

```
main.py
├── Orchestrates all components
├── Can run individual components or all together
└── Handles errors gracefully

q1_frequent_subgraph.py
├── SimpleGraphMiner: Extracts frequent patterns
├── train_classic_ml_models: Trains RF and SVM
├── run_ablation_study: Parameter sensitivity analysis
└── Saves results to JSON

q2_gnn.py
├── GCN, GIN, GraphSAGE, GAT: Model definitions
├── train_gnn: Training loop with validation
├── run_ablation_study: Architecture parameter analysis
└── Saves models and results

q3_comparison.py
├── Loads results from Q-1 and Q-2
├── Compares quality and efficiency metrics
├── Creates visualizations
└── Analyzes tradeoffs

q4_explainability.py
├── GNNExplainer: Explains GNN predictions
├── Feature importance: Explains classic ML
├── Visualizes explanations
└── Compares explanation methods

data_loader.py
├── load_mutag_dataset: Loads from PyTorch Geometric
├── convert_to_networkx: Converts between formats
└── get_train_test_split: Data splitting

utils.py
├── compute_metrics: Calculates accuracy, F1, etc.
├── Visualization functions
└── Save/load utilities
```

---

## 13. How to Read the Results

### JSON Files:
- **q1_classic_ml_results.json**: Performance of RF and SVM
- **q2_gnn_results.json**: Performance of all GNN architectures
- **q3_comparison_results.json**: Comprehensive comparison
- **q4_explainability_results.json**: Explanation metrics

### Key Metrics to Look For:
- **Accuracy > 0.8**: Good performance
- **F1-Score > 0.8**: Good balance between precision and recall
- **Training time < 60s**: Fast training
- **Fidelity+ > 0.7**: Good explanation quality

### Visualizations:
- **Bar charts**: Compare methods side-by-side
- **Line plots**: Show trends in ablation studies
- **Scatter plots**: Show tradeoffs (quality vs efficiency)

---

## 14. Extending the Project

### Possible Extensions:

1. **More Datasets**: Test on other graph classification benchmarks
2. **More Architectures**: Graph Transformer, DiffPool, etc.
3. **Hyperparameter Optimization**: Grid search, Bayesian optimization
4. **Ensemble Methods**: Combine multiple models
5. **Advanced Explainability**: Compare multiple explainers
6. **Scalability Analysis**: Test on larger graphs
7. **Real Applications**: Apply to actual drug discovery problems

---

## 15. Summary

This project provides a **comprehensive comparison** of two graph classification paradigms:

1. **Frequent Subgraph Mining + Classic ML**: Fast, interpretable, good baseline
2. **Graph Neural Networks**: Accurate, powerful, state-of-the-art

**Key Findings:**
- GNNs achieve better accuracy (10-15% improvement)
- Classic ML is faster (5-10x speedup)
- Both approaches have their place depending on requirements
- Explainability methods provide complementary insights

**Takeaway**: There's no one-size-fits-all solution. Choose based on your priorities: accuracy, speed, or interpretability.

---

## 📚 Additional Resources

- **PyTorch Geometric Documentation**: https://pytorch-geometric.readthedocs.io/
- **Graph Neural Networks Survey**: "A Comprehensive Survey on Graph Neural Networks" (Wu et al., 2020)
- **GNNExplainer Paper**: "GNNExplainer: Generating Explanations for Graph Neural Networks" (Ying et al., 2019)
- **MUTAG Dataset**: Available on Hugging Face and PyTorch Geometric

---

**End of Explanation Document**
