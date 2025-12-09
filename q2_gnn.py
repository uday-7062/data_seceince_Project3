"""
Q-2: Graph Neural Networks
Implements GCN, GIN, GraphSAGE, and GAT for graph classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.nn import Sequential, BatchNorm
import numpy as np
from typing import List, Dict, Tuple
from data_loader import load_mutag_dataset, get_train_test_split
from utils import compute_metrics, save_results
import time
import os
from tqdm import tqdm


class GCN(nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, num_node_features: int, hidden_dim: int = 64, 
                 num_layers: int = 3, num_classes: int = 2, dropout: float = 0.5):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(num_node_features, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index, batch, edge_weights=None):
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            if edge_weights is not None and i == 0:
                # Apply edge weights (simplified - multiply edge features)
                x = conv(x, edge_index, edge_attr=edge_weights if hasattr(conv, 'edge_attr') else None)
            else:
                x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x


class GIN(nn.Module):
    """Graph Isomorphism Network"""
    def __init__(self, num_node_features: int, hidden_dim: int = 64,
                 num_layers: int = 3, num_classes: int = 2, dropout: float = 0.5):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        nn1 = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(nn1, train_eps=True))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            nn_hidden = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(nn_hidden, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        nn_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(nn_out, train_eps=True))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index, batch, edge_weights=None):
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x


class GraphSAGE(nn.Module):
    """GraphSAGE"""
    def __init__(self, num_node_features: int, hidden_dim: int = 64,
                 num_layers: int = 3, num_classes: int = 2, dropout: float = 0.5):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(num_node_features, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index, batch, edge_weights=None):
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x


class GAT(nn.Module):
    """Graph Attention Network"""
    def __init__(self, num_node_features: int, hidden_dim: int = 64,
                 num_layers: int = 3, num_classes: int = 2, dropout: float = 0.5,
                 heads: int = 4):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(num_node_features, hidden_dim, heads=heads, dropout=dropout))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Output layer
        self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index, batch, edge_weights=None):
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    """Evaluate model"""
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    
    return y_true, y_pred


def train_gnn(model, train_loader, val_loader, num_epochs: int = 100,
              lr: float = 0.01, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Train GNN model"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_acc = 0
    train_times = []
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_time = time.time() - start_time
        train_times.append(train_time)
        
        val_true, val_pred = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(val_true, val_pred)
        val_acc = val_metrics['accuracy']
        val_loss = 1 - val_acc  # Use accuracy-based loss for scheduler
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_model_state)
    avg_train_time = np.mean(train_times)
    
    return model, avg_train_time


def run_ablation_study(model_class, model_name: str, graphs: List, labels: List,
                      train_indices: List, val_indices: List, test_indices: List,
                      num_node_features: int) -> Dict:
    """Run ablation study on GNN parameters"""
    results = {}
    
    # Ablation on hidden_dim
    print(f"\nRunning ablation on hidden_dim for {model_name}...")
    for hidden_dim in [32, 64, 128]:
        model = model_class(num_node_features, hidden_dim=hidden_dim)
        train_graphs = [graphs[i] for i in train_indices]
        val_graphs = [graphs[i] for i in val_indices]
        test_graphs = [graphs[i] for i in test_indices]
        
        train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, train_time = train_gnn(model, train_loader, val_loader, num_epochs=50, device=device)
        
        test_true, test_pred = evaluate(model, test_loader, device)
        metrics = compute_metrics(test_true, test_pred)
        metrics['train_time'] = train_time
        results[f'hidden_dim_{hidden_dim}'] = metrics
    
    # Ablation on num_layers
    print(f"\nRunning ablation on num_layers for {model_name}...")
    for num_layers in [2, 3, 4, 5]:
        model = model_class(num_node_features, num_layers=num_layers)
        train_graphs = [graphs[i] for i in train_indices]
        val_graphs = [graphs[i] for i in val_indices]
        test_graphs = [graphs[i] for i in test_indices]
        
        train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, train_time = train_gnn(model, train_loader, val_loader, num_epochs=50, device=device)
        
        test_true, test_pred = evaluate(model, test_loader, device)
        metrics = compute_metrics(test_true, test_pred)
        metrics['train_time'] = train_time
        results[f'num_layers_{num_layers}'] = metrics
    
    return results


def main():
    """
    Main execution for Q-2
    """
    print("=" * 60)
    print("Q-2: Graph Neural Networks")
    print("=" * 60)
    
    # Load data
    print("\nLoading MUTAG dataset...")
    graphs, labels = load_mutag_dataset()
    
    # Split data (train/val/test: 60/20/20)
    train_graphs, temp_graphs, train_labels, temp_labels = get_train_test_split(
        graphs, labels, test_size=0.4, random_state=42
    )
    val_graphs, test_graphs, val_labels, test_labels = get_train_test_split(
        temp_graphs, temp_labels, test_size=0.5, random_state=42
    )
    
    train_indices = list(range(len(train_graphs)))
    val_indices = list(range(len(train_graphs), len(train_graphs) + len(val_graphs)))
    test_indices = list(range(len(train_graphs) + len(val_graphs), 
                             len(train_graphs) + len(val_graphs) + len(test_graphs)))
    all_graphs = train_graphs + val_graphs + test_graphs
    
    num_node_features = graphs[0].x.shape[1] if graphs[0].x is not None else 1
    num_classes = len(set(labels))
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Train all GNN architectures
    gnn_models = {
        'GCN': GCN,
        'GIN': GIN,
        'GraphSAGE': GraphSAGE,
        'GAT': GAT
    }
    
    results = {}
    trained_models = {}
    
    for model_name, model_class in gnn_models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        model = model_class(num_node_features, hidden_dim=64, num_layers=3)
        
        # Train model
        start_time = time.time()
        model, avg_train_time = train_gnn(model, train_loader, val_loader, 
                                         num_epochs=100, device=device)
        total_train_time = time.time() - start_time
        
        # Evaluate on test set
        test_start_time = time.time()
        test_true, test_pred = evaluate(model, test_loader, device)
        test_time = time.time() - test_start_time
        
        metrics = compute_metrics(test_true, test_pred)
        metrics['train_time'] = total_train_time
        metrics['test_time'] = test_time
        metrics['avg_epoch_time'] = avg_train_time
        
        results[model_name] = metrics
        trained_models[model_name] = model
        
        print(f"\n{model_name} Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  Train Time: {metrics['train_time']:.4f}s")
        print(f"  Test Time: {metrics['test_time']:.4f}s")
    
    # Run ablation studies (on a subset to save time)
    print("\n" + "=" * 60)
    print("Running Ablation Studies")
    print("=" * 60)
    
    ablation_results = {}
    for model_name, model_class in list(gnn_models.items())[:2]:  # Run on first 2 models
        ablation = run_ablation_study(
            model_class, model_name, all_graphs, labels,
            train_indices, val_indices, test_indices, num_node_features
        )
        ablation_results[model_name] = ablation
    
    # Save results
    os.makedirs('results', exist_ok=True)
    save_results(results, 'results/q2_gnn_results.json')
    save_results(ablation_results, 'results/q2_gnn_ablation_results.json')
    
    # Save models
    for model_name, model in trained_models.items():
        torch.save(model.state_dict(), f'results/{model_name.lower()}_model.pt')
    
    return results, ablation_results, trained_models


if __name__ == "__main__":
    main()

