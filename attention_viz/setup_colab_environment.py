#!/usr/bin/env python3
"""
Proper environment setup for medical VLM analysis in Google Colab
This handles the LLaVA installation issue properly
"""

import os
import sys
import subprocess

def setup_environment_properly():
    """Setup Colab environment with proper error handling for LLaVA"""
    
    print("Setting up environment for medical VLM analysis...")
    print("-" * 50)
    
    # Step 1: Clone repository
    if not os.path.exists('/content/medical-vlm-intepret'):
        print("1. Cloning medical-vlm-intepret repository...")
        subprocess.run([
            'git', 'clone', 
            'https://github.com/thedatasense/medical-vlm-intepret.git'
        ], cwd='/content', check=True)
    else:
        print("1. Repository already exists")
    
    # Step 2: Install core dependencies
    print("\n2. Installing core dependencies...")
    core_deps = [
        'torch>=2.0.0',
        'torchvision>=0.15.0', 
        'transformers>=4.36.0',
        'bitsandbytes>=0.41.0',
        'accelerate>=0.21.0',
        'opencv-python',
        'scipy',
        'matplotlib',
        'pillow',
        'einops'
    ]
    
    for dep in core_deps:
        print(f"   Installing {dep}...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', dep], check=False)
    
    # Step 3: Handle LLaVA setup
    print("\n3. Setting up LLaVA...")
    if not os.path.exists('/content/LLaVA'):
        try:
            # Clone LLaVA repo
            subprocess.run([
                'git', 'clone', 
                'https://github.com/haotian-liu/LLaVA.git'
            ], cwd='/content', check=True)
            
            # Install specific LLaVA dependencies without the full package
            llava_specific_deps = [
                'einops', 
                'einops-exts', 
                'timm==0.6.13',
            ]
            
            for dep in llava_specific_deps:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', '-q', dep
                ], check=False)
            
            print("   ✓ LLaVA dependencies installed")
            
        except Exception as e:
            print(f"   ⚠ Warning: LLaVA setup had issues: {e}")
            print("   Continuing without full LLaVA - core functionality should work")
    else:
        print("   LLaVA directory already exists")
    
    # Step 4: Update Python path
    print("\n4. Updating Python path...")
    sys.path.insert(0, '/content/LLaVA')
    sys.path.insert(0, '/content/medical-vlm-intepret/attention_viz')
    
    # Step 5: Verify imports
    print("\n5. Verifying imports...")
    try:
        import torch
        print(f"   ✓ PyTorch {torch.__version__}")
        
        import transformers
        print(f"   ✓ Transformers {transformers.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   ✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("   ⚠ WARNING: No GPU available - this will be slow!")
        
    except ImportError as e:
        print(f"   ✗ Import error: {e}")
        raise
    
    print("\n✓ Environment setup complete!")
    print("-" * 50)
    return True


def verify_medical_data():
    """Verify that medical imaging data is available"""
    
    print("\nVerifying medical imaging data...")
    
    # Mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
    except:
        raise RuntimeError("Failed to mount Google Drive. This script requires Colab with Drive access.")
    
    # Expected paths
    required_paths = {
        'MIMIC images': '/content/drive/MyDrive/Robust_Medical_LLM_Dataset/MIMIC_JPG/hundred_vqa',
        'Base CSV': '/content/drive/MyDrive/Robust_Medical_LLM_Dataset/attention_viz/medical-cxr-vqa-questions_sample.csv',
        'Variants CSV': '/content/drive/MyDrive/Robust_Medical_LLM_Dataset/attention_viz/medical-cxr-vqa-questions_sample_hardpositives.csv'
    }
    
    all_found = True
    for name, path in required_paths.items():
        if os.path.exists(path):
            # Count items if directory
            if os.path.isdir(path):
                count = len(os.listdir(path))
                print(f"✓ {name}: {path} ({count} items)")
            else:
                size = os.path.getsize(path) / 1024  # KB
                print(f"✓ {name}: {path} ({size:.1f} KB)")
        else:
            print(f"✗ {name}: NOT FOUND at {path}")
            all_found = False
    
    if not all_found:
        print("\nERROR: Medical imaging data not found in expected locations.")
        print("Please ensure you have:")
        print("1. The MIMIC-CXR JPG dataset in your Google Drive")
        print("2. The CSV files with medical questions")
        raise FileNotFoundError("Required medical data not found")
    
    print("\n✓ All medical data verified!")
    return True


if __name__ == "__main__":
    # Run setup
    setup_environment_properly()
    
    # Verify data
    try:
        verify_medical_data()
        print("\n✅ Ready to run medical VLM analysis!")
        print("\nNext step:")
        print("from compare_attention_colab import compare_models_on_input")
        print("# or")
        print("!python run_medical_vlm_analysis.py --n_studies 100")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\nPlease set up your medical imaging data before proceeding.")