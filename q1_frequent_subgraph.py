"""
Q-1: Frequent Subgraph Mining + Classic ML
Implements gSpan-based frequent subgraph mining with Random Forest and SVM
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
import networkx as nx
from typing import List, Dict, Tuple, Set
from data_loader import load_mutag_dataset, convert_to_networkx, get_train_test_split
from utils import compute_metrics, measure_time, plot_ablation_results, save_results
import time
import os


class SimpleGraphMiner:
    """
    Simplified frequent subgraph mining using DFS-based pattern extraction
    """
    def __init__(self, min_support: float = 0.3, max_size: int = 5):
        self.min_support = min_support
        self.max_size = max_size
        self.frequent_patterns = []
        
    def extract_subgraphs(self, graph: nx.Graph, max_size: int) -> List[nx.Graph]:
        """
        Extract all connected subgraphs up to max_size nodes
        """
        subgraphs = []
        nodes = list(graph.nodes())
        
        # Extract single nodes
        for node in nodes:
            sg = nx.Graph()
            sg.add_node(node)
            subgraphs.append(sg)
        
        # Extract edges (2-node subgraphs)
        for edge in graph.edges():
            sg = nx.Graph()
            sg.add_edge(edge[0], edge[1])
            subgraphs.append(sg)
        
        # Extract larger subgraphs using DFS
        if max_size > 2:
            for start_node in nodes:
                visited = set()
                self._dfs_subgraphs(graph, start_node, nx.Graph(), visited, 
                                   subgraphs, max_size)
        
        return subgraphs
    
    def _dfs_subgraphs(self, graph: nx.Graph, node: int, current_sg: nx.Graph,
                      visited: Set, subgraphs: List, max_size: int):
        """DFS to extract subgraphs"""
        if len(current_sg.nodes()) >= max_size:
            return
        
        current_sg.add_node(node)
        visited.add(node)
        
        if len(current_sg.nodes()) > 1:
            subgraphs.append(current_sg.copy())
        
        neighbors = list(graph.neighbors(node))
        for neighbor in neighbors:
            if neighbor not in visited and len(current_sg.nodes()) < max_size:
                # Check if edge exists before adding
                if graph.has_edge(node, neighbor):
                    current_sg.add_edge(node, neighbor)
                    self._dfs_subgraphs(graph, neighbor, current_sg, visited,
                                      subgraphs, max_size)
                    if current_sg.has_edge(node, neighbor):
                        current_sg.remove_edge(node, neighbor)
        
        visited.remove(node)
        if node in current_sg.nodes():
            current_sg.remove_node(node)
    
    def graph_to_string(self, graph: nx.Graph) -> str:
        """Convert graph to canonical string representation"""
        edges = sorted(graph.edges())
        return str(edges)
    
    def mine_frequent_patterns(self, graphs: List[nx.Graph], labels: List[int]) -> Dict:
        """
        Mine frequent subgraphs from each class
        """
        class_graphs = defaultdict(list)
        for graph, label in zip(graphs, labels):
            class_graphs[label].append(graph)
        
        all_patterns = defaultdict(int)
        pattern_classes = defaultdict(set)
        
        # Extract patterns from each graph
        for label, class_graph_list in class_graphs.items():
            for graph in class_graph_list:
                subgraphs = self.extract_subgraphs(graph, self.max_size)
                seen_patterns = set()
                for sg in subgraphs:
                    pattern_str = self.graph_to_string(sg)
                    if pattern_str not in seen_patterns:
                        all_patterns[pattern_str] += 1
                        pattern_classes[pattern_str].add(label)
                        seen_patterns.add(pattern_str)
        
        # Filter by minimum support
        min_count = int(self.min_support * len(graphs))
        frequent_patterns = {p: count for p, count in all_patterns.items() 
                           if count >= min_count}
        
        return frequent_patterns, pattern_classes
    
    def build_feature_vectors(self, graphs: List[nx.Graph], 
                            frequent_patterns: Dict) -> np.ndarray:
        """
        Build feature vectors based on frequent pattern counts
        """
        pattern_list = list(frequent_patterns.keys())
        features = []
        
        for graph in graphs:
            subgraphs = self.extract_subgraphs(graph, self.max_size)
            pattern_counts = Counter([self.graph_to_string(sg) for sg in subgraphs])
            
            feature_vector = [pattern_counts.get(pattern, 0) for pattern in pattern_list]
            features.append(feature_vector)
        
        return np.array(features)


def run_ablation_study(graphs: List[nx.Graph], labels: List[int],
                      train_indices: List[int], test_indices: List[int]) -> Dict:
    """
    Run ablation study on mining thresholds and model parameters
    """
    results = {
        'min_support': {},
        'max_size': {},
        'rf_params': {},
        'svm_params': {}
    }
    
    # Ablation on min_support
    print("Running ablation on min_support...")
    for min_support in [0.1, 0.2, 0.3, 0.4, 0.5]:
        miner = SimpleGraphMiner(min_support=min_support, max_size=4)
        train_graphs = [graphs[i] for i in train_indices]
        test_graphs = [graphs[i] for i in test_indices]
        train_labels = [labels[i] for i in train_indices]
        test_labels = [labels[i] for i in test_indices]
        
        frequent_patterns, _ = miner.mine_frequent_patterns(train_graphs, train_labels)
        
        if len(frequent_patterns) == 0:
            continue
        
        X_train = miner.build_feature_vectors(train_graphs, frequent_patterns)
        X_test = miner.build_feature_vectors(test_graphs, frequent_patterns)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, train_labels)
        y_pred = rf.predict(X_test)
        metrics = compute_metrics(test_labels, y_pred)
        
        results['min_support'][min_support] = metrics
    
    # Ablation on max_size
    print("Running ablation on max_size...")
    for max_size in [3, 4, 5, 6]:
        miner = SimpleGraphMiner(min_support=0.3, max_size=max_size)
        train_graphs = [graphs[i] for i in train_indices]
        test_graphs = [graphs[i] for i in test_indices]
        train_labels = [labels[i] for i in train_indices]
        test_labels = [labels[i] for i in test_indices]
        
        frequent_patterns, _ = miner.mine_frequent_patterns(train_graphs, train_labels)
        
        if len(frequent_patterns) == 0:
            continue
        
        X_train = miner.build_feature_vectors(train_graphs, frequent_patterns)
        X_test = miner.build_feature_vectors(test_graphs, frequent_patterns)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, train_labels)
        y_pred = rf.predict(X_test)
        metrics = compute_metrics(test_labels, y_pred)
        
        results['max_size'][max_size] = metrics
    
    return results


def train_classic_ml_models(X_train: np.ndarray, y_train: List[int],
                           X_test: np.ndarray, y_test: List[int]) -> Dict:
    """
    Train Random Forest and SVM models
    """
    results = {}
    
    # Standardize features for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest
    print("Training Random Forest...")
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    start_time = time.time()
    y_pred_rf = rf.predict(X_test)
    test_time = time.time() - start_time
    
    metrics_rf = compute_metrics(y_test, y_pred_rf)
    metrics_rf['train_time'] = train_time
    metrics_rf['test_time'] = test_time
    metrics_rf['feature_importance'] = rf.feature_importances_.tolist()
    results['RandomForest'] = metrics_rf
    
    # SVM
    print("Training SVM...")
    start_time = time.time()
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    
    start_time = time.time()
    y_pred_svm = svm.predict(X_test_scaled)
    test_time = time.time() - start_time
    
    metrics_svm = compute_metrics(y_test, y_pred_svm)
    metrics_svm['train_time'] = train_time
    metrics_svm['test_time'] = test_time
    results['SVM'] = metrics_svm
    
    return results, rf, svm


def main():
    """
    Main execution for Q-1
    """
    print("=" * 60)
    print("Q-1: Frequent Subgraph Mining + Classic ML")
    print("=" * 60)
    
    # Load data
    print("\nLoading MUTAG dataset...")
    graphs_pyg, labels = load_mutag_dataset()
    graphs_nx = [convert_to_networkx(g) for g in graphs_pyg]
    
    # Split data
    train_graphs, test_graphs, train_labels, test_labels = get_train_test_split(
        graphs_nx, labels, test_size=0.2, random_state=42
    )
    
    # For ablation study, we need to work with the full dataset
    all_graphs = graphs_nx
    all_labels = labels
    
    # Create indices for train/test split
    np.random.seed(42)
    all_indices = np.arange(len(all_graphs))
    np.random.shuffle(all_indices)
    split_idx = int(len(all_graphs) * 0.8)
    train_indices = all_indices[:split_idx].tolist()
    test_indices = all_indices[split_idx:].tolist()
    
    # Mine frequent patterns
    print("\nMining frequent subgraphs...")
    miner = SimpleGraphMiner(min_support=0.3, max_size=4)
    frequent_patterns, pattern_classes = miner.mine_frequent_patterns(
        train_graphs, train_labels
    )
    print(f"Found {len(frequent_patterns)} frequent patterns")
    
    # Build feature vectors
    print("\nBuilding feature vectors...")
    X_train = miner.build_feature_vectors(train_graphs, frequent_patterns)
    X_test = miner.build_feature_vectors(test_graphs, frequent_patterns)
    print(f"Feature vector shape: {X_train.shape}")
    
    # Train models
    print("\nTraining classic ML models...")
    results, rf_model, svm_model = train_classic_ml_models(
        X_train, train_labels, X_test, test_labels
    )
    
    # Ablation study
    print("\nRunning ablation studies...")
    ablation_results = run_ablation_study(
        all_graphs, all_labels, train_indices, test_indices
    )
    
    # Save results
    os.makedirs('results', exist_ok=True)
    save_results(results, 'results/q1_classic_ml_results.json')
    save_results(ablation_results, 'results/q1_ablation_results.json')
    
    # Print results
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  Train Time: {metrics['train_time']:.4f}s")
        print(f"  Test Time: {metrics['test_time']:.4f}s")
    
    return results, ablation_results, rf_model, svm_model, miner, frequent_patterns


if __name__ == "__main__":
    main()

