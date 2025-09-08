#!/usr/bin/env python3
"""
Setup script for LLaVA-Rad Medical in Colab
Handles OpenCLIP and other dependency conflicts
"""

import os
import sys
import subprocess
import shutil


def run_command(cmd, check=True, cwd=None):
    """Run a command and return output"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")
    return result.stdout


def setup_dependencies():
    """Install dependencies in the correct order"""
    print("📦 Installing core dependencies...")
    
    # Core ML packages first
    packages = [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "numpy pillow opencv-python scipy matplotlib",
        "tokenizers>=0.12.1 sentencepiece>=0.1.99 protobuf",
    ]
    
    for pkg in packages:
        run_command(f"{sys.executable} -m pip install -q {pkg}")
    
    # Install transformers without optional dependencies
    print("📦 Installing transformers...")
    run_command(f"{sys.executable} -m pip install -q transformers>=4.36.0 --no-deps")
    run_command(f"{sys.executable} -m pip install -q huggingface-hub safetensors regex requests tqdm packaging filelock pyyaml")
    run_command(f"{sys.executable} -m pip install -q transformers>=4.36.0")
    
    # Other required packages
    print("📦 Installing additional packages...")
    run_command(f"{sys.executable} -m pip install -q accelerate bitsandbytes einops gradio")
    
    print("✅ Dependencies installed")


def fix_conflicts():
    """Fix known model registration conflicts"""
    print("🔧 Fixing model conflicts...")
    
    try:
        # Import after transformers is installed
        from transformers.models.auto import configuration_auto
        from transformers import AutoConfig
        
        conflicts_fixed = []
        conflicting_models = ['llava', 'open_clip', 'clip']
        
        # Fix CONFIG_MAPPING
        if hasattr(configuration_auto.CONFIG_MAPPING, '_extra_content'):
            extra = configuration_auto.CONFIG_MAPPING._extra_content
            for model in conflicting_models:
                if model in extra:
                    del extra[model]
                    conflicts_fixed.append(model)
        
        # Fix AutoConfig
        if hasattr(AutoConfig, '_config_mapping') and hasattr(AutoConfig._config_mapping, '_extra_content'):
            extra = AutoConfig._config_mapping._extra_content
            for model in conflicting_models:
                if model in extra:
                    del extra[model]
                    conflicts_fixed.append(f"AutoConfig.{model}")
        
        if conflicts_fixed:
            print(f"✅ Fixed conflicts: {', '.join(conflicts_fixed)}")
        else:
            print("ℹ️ No conflicts found")
            
    except Exception as e:
        print(f"⚠️ Could not fix conflicts: {e}")


def setup_llava_rad():
    """Clone and setup LLaVA-Rad without OpenCLIP"""
    print("\n🔧 Setting up LLaVA-Rad...")
    
    llava_path = "/content/LLaVA-Rad"
    
    # Clone if not exists
    if not os.path.exists(llava_path):
        print("📥 Cloning LLaVA-Rad...")
        run_command("git clone https://github.com/microsoft/LLaVA-Rad.git", cwd="/content")
    else:
        print("ℹ️ LLaVA-Rad already exists")
    
    # Check for requirements file
    req_file = os.path.join(llava_path, "requirements.txt")
    if os.path.exists(req_file):
        print("📝 Processing requirements...")
        
        # Read requirements and filter out problematic packages
        with open(req_file, 'r') as f:
            requirements = f.readlines()
        
        # Filter out OpenCLIP and other problematic packages
        filtered_reqs = []
        skip_packages = ['open_clip', 'open-clip', 'openclip', 'clip']
        
        for req in requirements:
            req_lower = req.lower().strip()
            if not any(skip in req_lower for skip in skip_packages):
                filtered_reqs.append(req)
        
        # Write filtered requirements
        filtered_req_file = os.path.join(llava_path, "requirements_filtered.txt")
        with open(filtered_req_file, 'w') as f:
            f.writelines(filtered_reqs)
        
        # Install filtered requirements
        print("📦 Installing LLaVA-Rad requirements (without OpenCLIP)...")
        try:
            run_command(f"{sys.executable} -m pip install -r {filtered_req_file}", check=False)
        except:
            print("⚠️ Some requirements failed, continuing...")
    
    # Install LLaVA-Rad without dependencies
    print("📦 Installing LLaVA-Rad package...")
    run_command(f"{sys.executable} -m pip install -e . --no-deps", cwd=llava_path)
    
    # Install any critical missing dependencies
    critical_deps = ["einops", "timm", "gradio", "gradio_client"]
    for dep in critical_deps:
        run_command(f"{sys.executable} -m pip install -q {dep}", check=False)
    
    print("✅ LLaVA-Rad setup complete")
    
    return llava_path


def setup_medical_vlm_repo():
    """Clone the medical VLM repository"""
    print("\n🔧 Setting up Medical VLM repository...")
    
    repo_path = "/content/medical-vlm-intepret"
    
    if not os.path.exists(repo_path):
        print("📥 Cloning medical-vlm-intepret...")
        run_command("git clone https://github.com/thedatasense/medical-vlm-intepret.git", cwd="/content")
    else:
        print("ℹ️ Repository already exists")
    
    return repo_path


def setup_python_paths():
    """Add repositories to Python path"""
    print("\n🔧 Setting up Python paths...")
    
    paths = [
        "/content/LLaVA-Rad",
        "/content/medical-vlm-intepret/attention_viz"
    ]
    
    for path in paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
            print(f"✅ Added {path} to Python path")
    
    # Create __init__.py files if needed
    for path in paths:
        init_file = os.path.join(path, "__init__.py")
        if os.path.exists(path) and not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write("# Auto-generated\n")


def test_imports():
    """Test if imports work"""
    print("\n🧪 Testing imports...")
    
    # Fix conflicts again before testing
    fix_conflicts()
    
    # Test LLaVA-Rad imports
    try:
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
        print("✅ LLaVA-Rad imports successful")
    except ImportError as e:
        print(f"❌ LLaVA-Rad import failed: {e}")
        return False
    
    # Test medical VLM imports
    try:
        from llava_rad_medical_only import LLaVARadMedical
        from medgemma_enhanced import load_medgemma
        print("✅ Medical VLM imports successful")
    except ImportError as e:
        print(f"❌ Medical VLM import failed: {e}")
        return False
    
    return True


def main():
    """Main setup function"""
    print("🚀 Starting LLaVA-Rad Medical setup for Colab\n")
    
    # Step 1: Install dependencies
    setup_dependencies()
    
    # Step 2: Fix conflicts
    fix_conflicts()
    
    # Step 3: Setup LLaVA-Rad
    llava_path = setup_llava_rad()
    
    # Step 4: Setup medical VLM repo
    repo_path = setup_medical_vlm_repo()
    
    # Step 5: Setup Python paths
    setup_python_paths()
    
    # Step 6: Test imports
    success = test_imports()
    
    if success:
        print("\n✅ Setup completed successfully!")
        print("\nYou can now use:")
        print("- from llava_rad_medical_only import LLaVARadMedical")
        print("- from medgemma_enhanced import load_medgemma")
    else:
        print("\n⚠️ Setup completed with warnings")
        print("Some imports may not work properly")
    
    return success


if __name__ == "__main__":
    main()