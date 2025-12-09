"""
Quick script to check MUTAG dataset statistics
"""
from data_loader import load_mutag_dataset, get_class_statistics

def main():
    print("="*60)
    print("MUTAG Dataset Information")
    print("="*60)
    
    print("\nLoading dataset (this may download it if not already present)...")
    graphs, labels = load_mutag_dataset()
    
    print(f"\n✓ Dataset loaded successfully!")
    print(f"\nTotal number of graphs: {len(graphs)}")
    
    # Get statistics
    stats = get_class_statistics(graphs, labels)
    
    print("\n" + "="*60)
    print("Dataset Statistics:")
    print("="*60)
    print(f"Total Graphs: {stats['total_graphs']}")
    print(f"\nClass Distribution:")
    for class_label, count in stats['class_distribution'].items():
        class_name = "Mutagenic" if class_label == 1 else "Non-Mutagenic"
        print(f"  {class_name} (Class {class_label}): {count} graphs")
    
    print(f"\nAverage Graph Size:")
    print(f"  Average Nodes: {stats['avg_nodes']:.2f}")
    print(f"  Average Edges: {stats['avg_edges']:.2f}")
    print(f"  Node Features: {stats['num_node_features']}")
    
    # Sample graph details
    print(f"\nSample Graph Details (first graph):")
    sample = graphs[0]
    print(f"  Nodes: {sample.num_nodes}")
    print(f"  Edges: {sample.edge_index.shape[1] // 2}")
    print(f"  Label: {sample.y.item()} ({'Mutagenic' if sample.y.item() == 1 else 'Non-Mutagenic'})")
    if sample.x is not None:
        print(f"  Node Features Shape: {sample.x.shape}")
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"The MUTAG dataset contains {stats['total_graphs']} small graphs")
    print(f"representing chemical compounds.")
    print(f"Each graph has approximately {stats['avg_nodes']:.0f} nodes and {stats['avg_edges']:.0f} edges on average.")
    print(f"Graphs are labeled as mutagenic (1) or non-mutagenic (0).")
    print("="*60)

if __name__ == "__main__":
    main()

