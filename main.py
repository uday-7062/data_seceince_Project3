"""
Main execution script for CS 6010 Project 3
Runs all components: Q-1, Q-2, Q-3, Q-4
"""
import os
import sys
import argparse
from datetime import datetime


def run_q1():
    """Run Q-1: Frequent Subgraph Mining + Classic ML"""
    print("\n" + "="*80)
    print("Running Q-1: Frequent Subgraph Mining + Classic ML")
    print("="*80)
    from q1_frequent_subgraph import main as q1_main
    return q1_main()


def run_q2():
    """Run Q-2: Graph Neural Networks"""
    print("\n" + "="*80)
    print("Running Q-2: Graph Neural Networks")
    print("="*80)
    from q2_gnn import main as q2_main
    return q2_main()


def run_q3():
    """Run Q-3: Comparison"""
    print("\n" + "="*80)
    print("Running Q-3: Comparison")
    print("="*80)
    from q3_comparison import main as q3_main
    return q3_main()


def run_q4():
    """Run Q-4: Explainability"""
    print("\n" + "="*80)
    print("Running Q-4: Explainability")
    print("="*80)
    from q4_explainability import main as q4_main
    return q4_main()


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='CS 6010 Project 3: Graph Classification')
    parser.add_argument('--q1', action='store_true', help='Run Q-1 only')
    parser.add_argument('--q2', action='store_true', help='Run Q-2 only')
    parser.add_argument('--q3', action='store_true', help='Run Q-3 only')
    parser.add_argument('--q4', action='store_true', help='Run Q-4 only')
    parser.add_argument('--all', action='store_true', help='Run all components')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    print("="*80)
    print("CS 6010 Project 3: Graph Classification on MUTAG Dataset")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.all or (not args.q1 and not args.q2 and not args.q3 and not args.q4):
        # Run all components
        print("\nRunning all components in sequence...")
        
        # Q-1
        try:
            q1_results = run_q1()
        except Exception as e:
            print(f"Error in Q-1: {e}")
            q1_results = None
        
        # Q-2
        try:
            q2_results = run_q2()
        except Exception as e:
            print(f"Error in Q-2: {e}")
            q2_results = None
        
        # Q-3 (requires Q-1 and Q-2)
        try:
            q3_results = run_q3()
        except Exception as e:
            print(f"Error in Q-3: {e}")
            q3_results = None
        
        # Q-4 (requires Q-1 and Q-2)
        try:
            q4_results = run_q4()
        except Exception as e:
            print(f"Error in Q-4: {e}")
            q4_results = None
    
    else:
        # Run specific components
        if args.q1:
            run_q1()
        if args.q2:
            run_q2()
        if args.q3:
            run_q3()
        if args.q4:
            run_q4()
    
    print("\n" + "="*80)
    print("Execution Complete!")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults are saved in the 'results/' directory")


if __name__ == "__main__":
    main()

