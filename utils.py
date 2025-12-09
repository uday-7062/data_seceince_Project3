"""
Utility functions for evaluation metrics and visualization
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Compute classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    return metrics


def measure_time(func):
    """
    Decorator to measure execution time
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    return wrapper


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], 
                         title: str = "Confusion Matrix", save_path: str = None):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_ablation_results(results: Dict, metric: str = 'accuracy', 
                         save_path: str = None):
    """
    Plot ablation study results
    """
    plt.figure(figsize=(10, 6))
    for method, values in results.items():
        params = list(values.keys())
        scores = [values[p][metric] for p in params]
        plt.plot(params, scores, marker='o', label=method)
    plt.xlabel('Parameter Value')
    plt.ylabel(metric.capitalize())
    plt.title(f'Ablation Study: {metric.capitalize()} vs Parameter')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_comparison(results: Dict, save_path: str = None):
    """
    Plot comparison of different methods
    """
    methods = list(results.keys())
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        values = [results[method][metric] for method in methods]
        axes[idx].bar(methods, values)
        axes[idx].set_title(f'{metric.capitalize()}')
        axes[idx].set_ylabel('Score')
        axes[idx].set_ylim([0, 1])
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def save_results(results: Dict, filepath: str):
    """
    Save results to file
    """
    import json
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

