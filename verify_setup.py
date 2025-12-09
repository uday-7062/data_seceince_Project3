"""
Quick verification script to check if all dependencies are installed correctly
"""
import sys

def check_imports():
    """Check if all required packages can be imported"""
    print("Checking imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch not found: {e}")
        return False
    
    try:
        import torch_geometric
        print(f"✓ PyTorch Geometric {torch_geometric.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch Geometric not found: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ scikit-learn not found: {e}")
        return False
    
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"✗ NumPy not found: {e}")
        return False
    
    try:
        import pandas
        print(f"✓ Pandas {pandas.__version__}")
    except ImportError as e:
        print(f"✗ Pandas not found: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib not found: {e}")
        return False
    
    try:
        import networkx
        print(f"✓ NetworkX {networkx.__version__}")
    except ImportError as e:
        print(f"✗ NetworkX not found: {e}")
        return False
    
    try:
        import tqdm
        print(f"✓ tqdm {tqdm.__version__}")
    except ImportError as e:
        print(f"✗ tqdm not found: {e}")
        return False
    
    return True

def check_project_files():
    """Check if all project files exist"""
    print("\nChecking project files...")
    
    required_files = [
        'main.py',
        'data_loader.py',
        'q1_frequent_subgraph.py',
        'q2_gnn.py',
        'q3_comparison.py',
        'q4_explainability.py',
        'utils.py',
        'requirements.txt',
        'README.md'
    ]
    
    import os
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} not found")
            all_exist = False
    
    return all_exist

def check_module_imports():
    """Check if project modules can be imported"""
    print("\nChecking project modules...")
    
    try:
        from data_loader import load_mutag_dataset
        print("✓ data_loader module")
    except Exception as e:
        print(f"✗ data_loader module: {e}")
        return False
    
    try:
        from utils import compute_metrics
        print("✓ utils module")
    except Exception as e:
        print(f"✗ utils module: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("CS 6010 Project 3 - Setup Verification")
    print("="*60)
    
    success = True
    success &= check_imports()
    success &= check_project_files()
    success &= check_module_imports()
    
    print("\n" + "="*60)
    if success:
        print("✓ All checks passed! Setup is correct.")
        print("\nYou can now run:")
        print("  python main.py --all")
    else:
        print("✗ Some checks failed. Please install missing dependencies:")
        print("  pip install -r requirements.txt")
    print("="*60)
    
    sys.exit(0 if success else 1)

