"""
Environment Setup Verification Script
Tests all critical dependencies and configurations
"""
import sys

def test_imports():
    """Test all critical imports"""
    print("Testing imports...")
    try:
        import torch
        import numpy
        import pandas
        import matplotlib
        import seaborn
        import scipy
        import tensorboard
        import tqdm
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_pytorch():
    """Test PyTorch functionality"""
    print("\nTesting PyTorch...")
    try:
        import torch
        x = torch.rand(3, 3)
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
        # Test basic operations
        y = x * 2
        assert y.shape == (3, 3), "Tensor operations failed"
        print("✅ PyTorch working correctly")
        return True
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def test_numpy():
    """Test NumPy functionality"""
    print("\nTesting NumPy...")
    try:
        import numpy as np
        arr = np.random.rand(5, 5)
        mean_val = np.mean(arr)
        print(f"NumPy version: {np.__version__}")
        print(f"Random array mean: {mean_val:.4f}")
        print("✅ NumPy working correctly")
        return True
    except Exception as e:
        print(f"❌ NumPy test failed: {e}")
        return False

def test_matplotlib():
    """Test Matplotlib functionality"""
    print("\nTesting Matplotlib...")
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        plt.close(fig)
        print(f"Matplotlib version: {matplotlib.__version__}")
        print("✅ Matplotlib working correctly")
        return True
    except Exception as e:
        print(f"❌ Matplotlib test failed: {e}")
        return False

def test_project_structure():
    """Verify project folder structure exists"""
    print("\nTesting project structure...")
    import os
    
    required_dirs = [
        'src',
        'src/environment',
        'src/agents',
        'src/models',
        'src/baselines',
        'src/utils',
        'experiments',
        'tests',
        'results',
        'saved_models',
        'configs'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ All required directories exist")
        return True

if __name__ == "__main__":
    print("=" * 60)
    print("ENVIRONMENT SETUP VERIFICATION")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("PyTorch", test_pytorch()))
    results.append(("NumPy", test_numpy()))
    results.append(("Matplotlib", test_matplotlib()))
    results.append(("Project Structure", test_project_structure()))
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20s}: {status}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n🎉 All tests passed! Environment is ready.")
        print("\nNext steps:")
        print("1. Start literature review (Week 1-2)")
        print("2. Begin network environment implementation (Week 3-4)")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Fix the errors above before proceeding.")
        sys.exit(1)