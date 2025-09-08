#!/usr/bin/env python3
"""
Fix for model registration conflicts in transformers
Handles both LLaVA and OpenCLIP conflicts
"""

def fix_transformers_conflicts():
    """Remove conflicting model registrations from transformers"""
    conflicts_fixed = []
    
    try:
        from transformers.models.auto import configuration_auto
        
        # List of known conflicting models
        conflicting_models = ['llava', 'open_clip', 'clip']
        
        # Check if models are in the extra content
        if hasattr(configuration_auto, 'CONFIG_MAPPING'):
            config_mapping = configuration_auto.CONFIG_MAPPING
            
            # Remove from _extra_content if it exists
            if hasattr(config_mapping, '_extra_content'):
                for model_name in conflicting_models:
                    if model_name in config_mapping._extra_content:
                        del config_mapping._extra_content[model_name]
                        conflicts_fixed.append(f"CONFIG_MAPPING._extra_content['{model_name}']")
                
        # Also try the AutoConfig approach
        from transformers import AutoConfig
        if hasattr(AutoConfig, '_config_mapping') and hasattr(AutoConfig._config_mapping, '_extra_content'):
            for model_name in conflicting_models:
                if model_name in AutoConfig._config_mapping._extra_content:
                    del AutoConfig._config_mapping._extra_content[model_name]
                    conflicts_fixed.append(f"AutoConfig._config_mapping._extra_content['{model_name}']")
        
        # Try to handle AutoModel mappings too
        try:
            from transformers.models.auto import modeling_auto
            for model_name in conflicting_models:
                # Check various model mappings
                for mapping_name in ['MODEL_MAPPING', 'MODEL_FOR_CAUSAL_LM_MAPPING', 
                                   'MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING']:
                    if hasattr(modeling_auto, mapping_name):
                        mapping = getattr(modeling_auto, mapping_name)
                        if hasattr(mapping, '_extra_content') and model_name in mapping._extra_content:
                            del mapping._extra_content[model_name]
                            conflicts_fixed.append(f"{mapping_name}._extra_content['{model_name}']")
        except:
            pass
                
        if conflicts_fixed:
            print(f"✓ Fixed {len(conflicts_fixed)} conflicts:")
            for conflict in conflicts_fixed:
                print(f"  - Removed {conflict}")
            return True
        else:
            print("No conflicts found in transformers")
            return False
            
    except Exception as e:
        print(f"Error fixing conflicts: {e}")
        return False


def install_dependencies_safely():
    """Install dependencies in the right order to avoid conflicts"""
    import subprocess
    import sys
    
    print("Installing dependencies safely...")
    
    # First, install transformers without optional deps
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '-q',
        'transformers>=4.36.0', '--no-deps'
    ])
    
    # Install core dependencies
    core_deps = [
        'torch', 'torchvision', 'numpy', 'pillow',
        'tokenizers', 'sentencepiece', 'protobuf',
        'huggingface-hub', 'safetensors', 'regex',
        'requests', 'tqdm', 'packaging', 'filelock',
        'pyyaml'
    ]
    
    for dep in core_deps:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', dep])
    
    # Now install transformers with deps
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '-q',
        'transformers>=4.36.0'
    ])
    
    # Install other required packages
    other_deps = [
        'opencv-python', 'scipy', 'matplotlib',
        'accelerate', 'bitsandbytes', 'einops'
    ]
    
    for dep in other_deps:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', dep])
    
    print("✓ Dependencies installed")


if __name__ == "__main__":
    # Run both fixes
    install_dependencies_safely()
    fix_transformers_conflicts()