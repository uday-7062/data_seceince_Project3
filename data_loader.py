"""
Data loading utilities for MUTAG dataset
"""
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict


def load_mutag_dataset() -> Tuple[List[Data], List[int]]:
    """
    Load MUTAG dataset from PyTorch Geometric
    
    Returns:
        graphs: List of graph data objects
        labels: List of graph labels
    """
    dataset = TUDataset(root='./data', name='MUTAG')
    graphs = []
    labels = []
    
    for data in dataset:
        graphs.append(data)
        labels.append(data.y.item())
    
    return graphs, labels


def convert_to_networkx(graph: Data) -> nx.Graph:
    """
    Convert PyTorch Geometric graph to NetworkX graph
    
    Args:
        graph: PyTorch Geometric Data object
        
    Returns:
        NetworkX graph
    """
    G = nx.Graph()
    
    # Add nodes with features
    for i in range(graph.num_nodes):
        node_features = graph.x[i].numpy() if graph.x is not None else []
        G.add_node(i, features=node_features)
    
    # Add edges
    if graph.edge_index is not None:
        edge_index = graph.edge_index.numpy()
        for j in range(edge_index.shape[1]):
            src, dst = edge_index[0, j], edge_index[1, j]
            G.add_edge(int(src), int(dst))
    
    return G


def get_train_test_split(graphs: List[Data], labels: List[int], 
                         test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Split dataset into train and test sets
    
    Args:
        graphs: List of graph data objects
        labels: List of graph labels
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        train_graphs, test_graphs, train_labels, test_labels
    """
    np.random.seed(random_state)
    indices = np.arange(len(graphs))
    np.random.shuffle(indices)
    
    split_idx = int(len(graphs) * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_graphs = [graphs[i] for i in train_indices]
    test_graphs = [graphs[i] for i in test_indices]
    train_labels = [labels[i] for i in train_indices]
    test_labels = [labels[i] for i in test_indices]
    
    return train_graphs, test_graphs, train_labels, test_labels


def get_class_statistics(graphs: List[Data], labels: List[int]) -> Dict:
    """
    Get statistics about the dataset
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_graphs': len(graphs),
        'class_distribution': {},
        'avg_nodes': 0,
        'avg_edges': 0,
        'num_node_features': graphs[0].x.shape[1] if graphs[0].x is not None else 0
    }
    
    # Class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        stats['class_distribution'][int(label)] = int(count)
    
    # Average nodes and edges
    total_nodes = sum(g.num_nodes for g in graphs)
    total_edges = sum(g.edge_index.shape[1] // 2 if g.edge_index is not None else 0 
                      for g in graphs)
    stats['avg_nodes'] = total_nodes / len(graphs)
    stats['avg_edges'] = total_edges / len(graphs)
    
    return stats

