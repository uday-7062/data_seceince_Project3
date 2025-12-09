"""
Q-4: Explainability Analysis
Implements GNNExplainer for GNNs and feature importance for classic ML
"""
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
import json
from utils import save_results
from data_loader import load_mutag_dataset, get_train_test_split, convert_to_networkx


class GNNExplainer:
    """
    Simplified GNNExplainer implementation
    """
    def __init__(self, model, num_hops: int = 2, epochs: int = 200, lr: float = 0.01):
        self.model = model
        self.num_hops = num_hops
        self.epochs = epochs
        self.lr = lr
    
    def explain(self, data, target_class: int = None, device='cpu'):
        """
        Generate explanation for a graph
        """
        self.model.eval()
        data = data.to(device)
        
        # Get prediction
        with torch.no_grad():
            logits = self.model(data.x, data.edge_index, data.batch)
            pred_class = logits.argmax(dim=1).item()
            if target_class is None:
                target_class = pred_class
        
        # Initialize edge mask
        edge_mask = torch.ones(data.edge_index.shape[1], requires_grad=True, device=device)
        optimizer = torch.optim.Adam([edge_mask], lr=self.lr)
        
        # Optimize edge mask
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Apply mask to edges by filtering
            edge_weights = torch.sigmoid(edge_mask)
            # Use top-k edges based on importance
            k = max(1, int(edge_weights.sum().item()))
            _, top_indices = torch.topk(edge_weights, k)
            masked_edge_index = data.edge_index[:, top_indices]
            
            # Forward pass with masked edges
            logits = self.model(data.x, masked_edge_index, data.batch)
            
            # Loss: maximize target class probability + sparsity
            target_loss = -F.log_softmax(logits, dim=1)[0, target_class]
            sparsity_loss = edge_mask.mean()
            loss = target_loss + 0.1 * sparsity_loss
            
            loss.backward()
            optimizer.step()
        
        # Get final edge importance
        edge_importance = torch.sigmoid(edge_mask).detach().cpu().numpy()
        
        return edge_importance, pred_class
    
    def compute_fidelity_plus(self, model, data, edge_importance, threshold: float = 0.5,
                            device='cpu'):
        """
        Compute fidelity+ (importance of important edges)
        """
        data = data.to(device)
        model.eval()
        
        # Original prediction
        with torch.no_grad():
            original_logits = model(data.x, data.edge_index, data.batch)
            original_pred = original_logits.argmax(dim=1).item()
            original_prob = F.softmax(original_logits, dim=1)[0, original_pred].item()
        
        # Keep only important edges
        important_edges = edge_importance >= threshold
        if important_edges.sum() == 0:
            # Use top edge if none above threshold
            top_idx = np.argmax(edge_importance)
            important_edges = np.zeros_like(edge_importance, dtype=bool)
            important_edges[top_idx] = True
        
        important_indices = np.where(important_edges)[0]
        if len(important_indices) == 0:
            return 0.0
        
        masked_edge_index = data.edge_index[:, important_indices]
        
        # Prediction with important edges only
        with torch.no_grad():
            masked_logits = model(data.x, masked_edge_index, data.batch)
            masked_pred = masked_logits.argmax(dim=1).item()
            masked_prob = F.softmax(masked_logits, dim=1)[0, masked_pred].item()
        
        # Fidelity+ = how well important edges preserve prediction
        if masked_pred == original_pred:
            fidelity_plus = masked_prob
        else:
            fidelity_plus = 0.0
        
        return fidelity_plus
    
    def compute_fidelity_minus(self, model, data, edge_importance, threshold: float = 0.5,
                              device='cpu'):
        """
        Compute fidelity- (unimportance of unimportant edges)
        """
        data = data.to(device)
        model.eval()
        
        # Original prediction
        with torch.no_grad():
            original_logits = model(data.x, data.edge_index, data.batch)
            original_pred = original_logits.argmax(dim=1).item()
        
        # Remove important edges (keep only unimportant ones)
        unimportant_edges = edge_importance < threshold
        if unimportant_edges.sum() == 0:
            # If all edges are important, remove half of them
            num_remove = len(edge_importance) // 2
            least_important = np.argsort(edge_importance)[:num_remove]
            unimportant_edges = np.zeros_like(edge_importance, dtype=bool)
            unimportant_edges[least_important] = True
        
        unimportant_indices = np.where(unimportant_edges)[0]
        if len(unimportant_indices) == 0:
            return 1.0
        
        masked_edge_index = data.edge_index[:, unimportant_indices]
        
        # Prediction without important edges
        with torch.no_grad():
            masked_logits = model(data.x, masked_edge_index, data.batch)
            masked_pred = masked_logits.argmax(dim=1).item()
        
        # Fidelity- = how prediction changes when removing important edges
        if masked_pred != original_pred:
            fidelity_minus = 1.0
        else:
            fidelity_minus = 0.0
        
        return fidelity_minus
    
    def compute_sparsity(self, edge_importance, threshold: float = 0.5):
        """
        Compute sparsity (fraction of edges considered important)
        """
        important_edges = (edge_importance >= threshold).sum()
        total_edges = len(edge_importance)
        sparsity = 1.0 - (important_edges / total_edges)
        return sparsity


class SimpleGNN(torch.nn.Module):
    """Simple GNN for explanation"""
    def __init__(self, num_node_features, hidden_dim=64, num_classes=2):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index, batch, edge_weights=None):
        if edge_weights is not None:
            # Apply edge weights (simplified - would need proper implementation)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
        else:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
        
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x


def explain_gnn_model(model, test_graphs, device='cpu', num_samples: int = 10):
    """
    Explain GNN model using GNNExplainer
    """
    explainer = GNNExplainer(model, epochs=100)
    
    results = {
        'fidelity_plus': [],
        'fidelity_minus': [],
        'sparsity': [],
        'runtime': []
    }
    
    sample_graphs = test_graphs[:num_samples]
    
    for i, graph in enumerate(sample_graphs):
        import time
        start_time = time.time()
        
        edge_importance, pred_class = explainer.explain(graph, device=device)
        
        fidelity_plus = explainer.compute_fidelity_plus(
            model, graph, edge_importance, device=device
        )
        fidelity_minus = explainer.compute_fidelity_minus(
            model, graph, edge_importance, device=device
        )
        sparsity = explainer.compute_sparsity(edge_importance)
        
        runtime = time.time() - start_time
        
        results['fidelity_plus'].append(fidelity_plus)
        results['fidelity_minus'].append(fidelity_minus)
        results['sparsity'].append(sparsity)
        results['runtime'].append(runtime)
    
    # Average results
    avg_results = {
        'fidelity_plus': np.mean(results['fidelity_plus']),
        'fidelity_minus': np.mean(results['fidelity_minus']),
        'sparsity': np.mean(results['sparsity']),
        'avg_runtime': np.mean(results['runtime'])
    }
    
    return avg_results, results


def explain_classic_ml(rf_model, svm_model, feature_importance, frequent_patterns):
    """
    Explain classic ML models using feature importance
    """
    # Random Forest feature importance
    rf_importance = {
        'top_features': [],
        'importance_scores': []
    }
    
    if feature_importance is not None and len(feature_importance) > 0:
        top_indices = np.argsort(feature_importance)[-10:][::-1]
        pattern_list = list(frequent_patterns.keys())
        
        for idx in top_indices:
            if idx < len(pattern_list):
                rf_importance['top_features'].append(pattern_list[idx])
                rf_importance['importance_scores'].append(float(feature_importance[idx]))
    
    # SVM - use absolute coefficients (for linear kernel) or feature importance approximation
    svm_importance = {
        'note': 'SVM feature importance requires linear kernel or approximation',
        'top_features': []
    }
    
    return {
        'RandomForest': rf_importance,
        'SVM': svm_importance
    }


def visualize_explanations(edge_importance, graph, save_path: str = None):
    """
    Visualize graph with edge importance
    """
    G = convert_to_networkx(graph)
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    
    # Draw edges with importance
    edges = list(G.edges())
    edge_weights = []
    for edge in edges:
        # Find corresponding edge in edge_index
        edge_idx = None
        for i in range(graph.edge_index.shape[1]):
            if (graph.edge_index[0, i].item() == edge[0] and 
                graph.edge_index[1, i].item() == edge[1]):
                edge_idx = i
                break
        if edge_idx is not None and edge_idx < len(edge_importance):
            edge_weights.append(edge_importance[edge_idx])
        else:
            edge_weights.append(0.5)
    
    # Normalize edge weights for visualization
    if len(edge_weights) > 0:
        max_weight = max(edge_weights) if max(edge_weights) > 0 else 1.0
        edge_weights = [w / max_weight for w in edge_weights]
    
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=500, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=[w * 3 for w in edge_weights],
                          alpha=0.6, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.title('Graph Explanation (Edge Importance)', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """
    Main execution for Q-4
    """
    print("=" * 60)
    print("Q-4: Explainability Analysis")
    print("=" * 60)
    
    # Load data
    print("\nLoading MUTAG dataset...")
    graphs, labels = load_mutag_dataset()
    train_graphs, test_graphs, train_labels, test_labels = get_train_test_split(
        graphs, labels, test_size=0.2, random_state=42
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load GNN model (use GCN as example)
    print("\nLoading GNN model...")
    try:
        from q2_gnn import GCN
        num_node_features = graphs[0].x.shape[1] if graphs[0].x is not None else 1
        gnn_model = GCN(num_node_features, hidden_dim=64, num_layers=3)
        
        # Try to load pretrained model
        try:
            gnn_model.load_state_dict(torch.load('results/gcn_model.pt', map_location=device))
            print("Loaded pretrained GCN model")
        except FileNotFoundError:
            print("Warning: Pretrained model not found. Training a new model...")
            from torch_geometric.data import DataLoader
            train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
            val_loader = DataLoader(test_graphs[:len(test_graphs)//2], batch_size=32, shuffle=False)
            from q2_gnn import train_gnn
            gnn_model, _ = train_gnn(gnn_model, train_loader, val_loader, 
                                    num_epochs=50, device=device)
        
        gnn_model = gnn_model.to(device)
        
        # Explain GNN model
        print("\nExplaining GNN model with GNNExplainer...")
        gnn_results, detailed_gnn_results = explain_gnn_model(
            gnn_model, test_graphs, device=device, num_samples=10
        )
        
        print("\nGNN Explanation Results:")
        print(f"  Fidelity+: {gnn_results['fidelity_plus']:.4f}")
        print(f"  Fidelity-: {gnn_results['fidelity_minus']:.4f}")
        print(f"  Sparsity: {gnn_results['sparsity']:.4f}")
        print(f"  Avg Runtime: {gnn_results['avg_runtime']:.4f}s")
        
    except Exception as e:
        print(f"Error in GNN explanation: {e}")
        gnn_results = {}
    
    # Load classic ML results
    print("\nLoading classic ML feature importance...")
    try:
        import json
        with open('results/q1_classic_ml_results.json', 'r') as f:
            q1_results = json.load(f)
        
        # Get feature importance from Random Forest
        rf_importance = q1_results.get('RandomForest', {}).get('feature_importance', [])
        
        # Load frequent patterns
        # This would need to be saved from Q-1
        frequent_patterns = {}  # Placeholder
        
        classic_ml_explanations = explain_classic_ml(
            None, None, rf_importance, frequent_patterns
        )
        
        print("\nClassic ML Explanation Results:")
        print(f"  Random Forest top features: {len(classic_ml_explanations['RandomForest']['top_features'])}")
        
    except Exception as e:
        print(f"Error in classic ML explanation: {e}")
        classic_ml_explanations = {}
    
    # Compare explanations
    print("\n" + "=" * 60)
    print("Explanation Comparison")
    print("=" * 60)
    
    comparison = {
        'GNN_Explainer': {
            'method': 'Post-hoc (GNNExplainer)',
            'fidelity_plus': gnn_results.get('fidelity_plus', 0),
            'fidelity_minus': gnn_results.get('fidelity_minus', 0),
            'sparsity': gnn_results.get('sparsity', 0),
            'runtime': gnn_results.get('avg_runtime', 0),
            'interpretability': 'Edge-level importance'
        },
        'Classic_ML': {
            'method': 'Self-explainable (Feature Importance)',
            'interpretability': 'Pattern-level importance',
            'note': 'Direct feature importance from trained models'
        }
    }
    
    print("\nKey Differences:")
    print("1. GNNExplainer provides post-hoc explanations (edge importance)")
    print("2. Classic ML models are self-explainable (feature importance)")
    print("3. GNN explanations show which edges matter for predictions")
    print("4. Classic ML explanations show which subgraph patterns matter")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    save_results({
        'gnn_explanation': gnn_results,
        'classic_ml_explanation': classic_ml_explanations,
        'comparison': comparison
    }, 'results/q4_explainability_results.json')
    
    print("\n" + "=" * 60)
    print("Explainability Analysis Complete!")
    print("=" * 60)
    print("\nResults saved to:")
    print("  - results/q4_explainability_results.json")
    
    return gnn_results, classic_ml_explanations, comparison


if __name__ == "__main__":
    main()

