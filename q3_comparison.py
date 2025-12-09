"""
Q-3: Comparison between Frequent Subgraph Mining + Classic ML vs GNNs
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import json
import os
from utils import plot_comparison, save_results


def load_results() -> Dict:
    """
    Load results from Q-1 and Q-2
    """
    results = {}
    
    # Load Q-1 results
    try:
        with open('results/q1_classic_ml_results.json', 'r') as f:
            q1_results = json.load(f)
            results.update(q1_results)
    except FileNotFoundError:
        print("Warning: Q-1 results not found. Run q1_frequent_subgraph.py first.")
    
    # Load Q-2 results
    try:
        with open('results/q2_gnn_results.json', 'r') as f:
            q2_results = json.load(f)
            results.update(q2_results)
    except FileNotFoundError:
        print("Warning: Q-2 results not found. Run q2_gnn.py first.")
    
    return results


def compare_quality_metrics(results: Dict) -> pd.DataFrame:
    """
    Compare quality metrics across methods
    """
    comparison_data = []
    
    for method, metrics in results.items():
        comparison_data.append({
            'Method': method,
            'Accuracy': metrics.get('accuracy', 0),
            'F1-Score': metrics.get('f1', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0)
        })
    
    df = pd.DataFrame(comparison_data)
    return df


def compare_efficiency(results: Dict) -> pd.DataFrame:
    """
    Compare efficiency metrics across methods
    """
    efficiency_data = []
    
    for method, metrics in results.items():
        efficiency_data.append({
            'Method': method,
            'Train Time (s)': metrics.get('train_time', 0),
            'Test Time (s)': metrics.get('test_time', 0),
            'Total Time (s)': metrics.get('train_time', 0) + metrics.get('test_time', 0)
        })
    
    df = pd.DataFrame(efficiency_data)
    return df


def create_comparison_plots(quality_df: pd.DataFrame, efficiency_df: pd.DataFrame):
    """
    Create comparison plots
    """
    os.makedirs('results', exist_ok=True)
    
    # Quality metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        ax.bar(quality_df['Method'], quality_df[metric], color='steelblue', alpha=0.7)
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=10)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/q3_quality_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Efficiency comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    time_metrics = ['Train Time (s)', 'Test Time (s)', 'Total Time (s)']
    for idx, metric in enumerate(time_metrics):
        ax = axes[idx]
        ax.bar(efficiency_df['Method'], efficiency_df[metric], color='coral', alpha=0.7)
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel('Time (seconds)', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
        ax.set_yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    plt.savefig('results/q3_efficiency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Combined comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Quality radar chart alternative - bar chart
    x = np.arange(len(quality_df))
    width = 0.2
    metrics_short = ['Acc', 'F1', 'Prec', 'Rec']
    for i, metric in enumerate(['Accuracy', 'F1-Score', 'Precision', 'Recall']):
        offset = (i - 1.5) * width
        ax1.bar(x + offset, quality_df[metric], width, label=metrics_short[i], alpha=0.8)
    ax1.set_xlabel('Method', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Quality Metrics Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(quality_df['Method'], rotation=45, ha='right')
    ax1.set_ylim([0, 1])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Efficiency
    ax2.bar(efficiency_df['Method'], efficiency_df['Total Time (s)'], 
           color='coral', alpha=0.7)
    ax2.set_xlabel('Method', fontsize=11)
    ax2.set_ylabel('Total Time (seconds)', fontsize=11)
    ax2.set_title('Efficiency Comparison', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/q3_combined_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def analyze_tradeoffs(quality_df: pd.DataFrame, efficiency_df: pd.DataFrame) -> Dict:
    """
    Analyze tradeoffs between quality and efficiency
    """
    # Merge dataframes
    merged_df = quality_df.merge(efficiency_df, on='Method')
    
    # Calculate efficiency score (inverse of time, normalized)
    max_time = merged_df['Total Time (s)'].max()
    merged_df['Efficiency Score'] = 1 - (merged_df['Total Time (s)'] / max_time)
    
    # Calculate quality score (average of all metrics)
    merged_df['Quality Score'] = (
        merged_df['Accuracy'] + merged_df['F1-Score'] + 
        merged_df['Precision'] + merged_df['Recall']
    ) / 4
    
    # Create tradeoff plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for idx, row in merged_df.iterrows():
        ax.scatter(row['Total Time (s)'], row['Quality Score'], 
                  s=200, alpha=0.6, label=row['Method'])
        ax.annotate(row['Method'], 
                   (row['Total Time (s)'], row['Quality Score']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Total Time (seconds)', fontsize=11)
    ax.set_ylabel('Quality Score (Average)', fontsize=11)
    ax.set_title('Quality vs Efficiency Tradeoff', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('results/q3_tradeoff_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary statistics
    summary = {
        'best_accuracy': merged_df.loc[merged_df['Accuracy'].idxmax(), 'Method'],
        'best_f1': merged_df.loc[merged_df['F1-Score'].idxmax(), 'Method'],
        'fastest': merged_df.loc[merged_df['Total Time (s)'].idxmin(), 'Method'],
        'best_quality_score': merged_df.loc[merged_df['Quality Score'].idxmax(), 'Method'],
        'best_efficiency_score': merged_df.loc[merged_df['Efficiency Score'].idxmax(), 'Method']
    }
    
    return summary, merged_df


def generate_comparison_table(quality_df: pd.DataFrame, efficiency_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive comparison table
    """
    merged_df = quality_df.merge(efficiency_df, on='Method')
    
    # Round values for readability
    for col in ['Accuracy', 'F1-Score', 'Precision', 'Recall']:
        merged_df[col] = merged_df[col].round(4)
    for col in ['Train Time (s)', 'Test Time (s)', 'Total Time (s)']:
        merged_df[col] = merged_df[col].round(4)
    
    return merged_df


def main():
    """
    Main execution for Q-3
    """
    print("=" * 60)
    print("Q-3: Comparison between Approaches")
    print("=" * 60)
    
    # Load results
    print("\nLoading results from Q-1 and Q-2...")
    results = load_results()
    
    if not results:
        print("Error: No results found. Please run Q-1 and Q-2 first.")
        return
    
    print(f"Found results for {len(results)} methods")
    
    # Compare quality metrics
    print("\nComparing quality metrics...")
    quality_df = compare_quality_metrics(results)
    print("\nQuality Metrics:")
    print(quality_df.to_string(index=False))
    
    # Compare efficiency
    print("\nComparing efficiency...")
    efficiency_df = compare_efficiency(results)
    print("\nEfficiency Metrics:")
    print(efficiency_df.to_string(index=False))
    
    # Create comparison plots
    print("\nGenerating comparison plots...")
    create_comparison_plots(quality_df, efficiency_df)
    
    # Analyze tradeoffs
    print("\nAnalyzing quality vs efficiency tradeoffs...")
    summary, tradeoff_df = analyze_tradeoffs(quality_df, efficiency_df)
    
    print("\nTradeoff Analysis Summary:")
    for key, value in summary.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Generate comparison table
    comparison_table = generate_comparison_table(quality_df, efficiency_df)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    comparison_table.to_csv('results/q3_comparison_table.csv', index=False)
    save_results({
        'quality_metrics': quality_df.to_dict('records'),
        'efficiency_metrics': efficiency_df.to_dict('records'),
        'summary': summary,
        'tradeoff_analysis': tradeoff_df.to_dict('records')
    }, 'results/q3_comparison_results.json')
    
    print("\n" + "=" * 60)
    print("Comparison Complete!")
    print("=" * 60)
    print("\nResults saved to:")
    print("  - results/q3_comparison_table.csv")
    print("  - results/q3_comparison_results.json")
    print("  - results/q3_quality_comparison.png")
    print("  - results/q3_efficiency_comparison.png")
    print("  - results/q3_combined_comparison.png")
    print("  - results/q3_tradeoff_analysis.png")
    
    return comparison_table, summary


if __name__ == "__main__":
    main()

