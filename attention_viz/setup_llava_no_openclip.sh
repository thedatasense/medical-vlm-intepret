#!/bin/bash
# Setup script for LLaVA-Rad without OpenCLIP conflicts

echo "Setting up LLaVA-Rad without OpenCLIP conflicts..."

# Clone LLaVA-Rad
git clone https://github.com/microsoft/LLaVA-Rad.git /content/LLaVA-Rad
cd /content/LLaVA-Rad

# Create a modified requirements file without OpenCLIP
grep -v "open_clip" requirements.txt > requirements_no_openclip.txt || cp requirements.txt requirements_no_openclip.txt

# Install modified requirements
pip install -r requirements_no_openclip.txt

# Install the package without dependencies
pip install -e . --no-deps

# Install any missing critical dependencies
pip install -q tokenizers sentencepiece gradio gradio_client einops timm

echo "✓ LLaVA-Rad installed without OpenCLIP"