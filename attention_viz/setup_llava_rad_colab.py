#!/usr/bin/env python3
"""
Setup script for LLaVA-Rad in Google Colab
Clones and sets up the official Microsoft LLaVA-Rad repository
"""

import os
import sys
import subprocess


def setup_llava_rad():
    """Clone and setup LLaVA-Rad from Microsoft's repository"""
    
    print("Setting up LLaVA-Rad from Microsoft repository...")
    
    # Clone LLaVA-Rad if not exists
    if not os.path.exists('/content/LLaVA-Rad'):
        print("1. Cloning LLaVA-Rad repository...")
        subprocess.run([
            'git', 'clone', 
            'https://github.com/microsoft/LLaVA-Rad.git'
        ], cwd='/content', check=True)
    else:
        print("1. LLaVA-Rad repository already exists")
    
    # Change to LLaVA-Rad directory
    os.chdir('/content/LLaVA-Rad')
    
    # Install LLaVA-Rad dependencies
    print("\n2. Installing LLaVA-Rad dependencies...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'], check=False)
    
    # Additional dependencies that might be needed
    additional_deps = [
        'tokenizers>=0.12.1',
        'sentencepiece>=0.1.99',
        'protobuf',
        'gradio',
        'gradio_client'
    ]
    
    for dep in additional_deps:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', dep], check=False)
    
    # Add to Python path
    if '/content/LLaVA-Rad' not in sys.path:
        sys.path.insert(0, '/content/LLaVA-Rad')
    
    print("\n3. Testing LLaVA-Rad imports...")
    try:
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token
        print("✓ LLaVA-Rad imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


if __name__ == "__main__":
    setup_llava_rad()