#!/usr/bin/env python3
"""
Fixed setup for LLaVA-Rad in Google Colab
Handles OpenCLIP and other dependency conflicts
"""

# ========================================
# Cell 1: Install Dependencies Carefully
# ========================================
"""
# Install dependencies in the right order to avoid conflicts
import subprocess
import sys

# Core dependencies first (without transformers)
!pip install -q torch torchvision numpy pillow
!pip install -q tokenizers sentencepiece protobuf
!pip install -q opencv-python scipy matplotlib

# Install transformers without optional dependencies
!pip install -q transformers>=4.36.0 --no-deps

# Install remaining transformers dependencies
!pip install -q huggingface-hub safetensors regex requests tqdm packaging filelock pyyaml

# Now install transformers properly
!pip install -q transformers>=4.36.0

# Other required packages
!pip install -q accelerate bitsandbytes einops

print("✓ Dependencies installed")
"""

# ========================================
# Cell 2: Fix Conflicts and Clone Repos
# ========================================
"""
# Fix any model registration conflicts
import warnings
warnings.filterwarnings('ignore')

# Function to fix conflicts
def fix_all_conflicts():
    try:
        from transformers.models.auto import configuration_auto
        
        # Remove conflicting registrations
        conflicting = ['llava', 'open_clip', 'clip']
        
        if hasattr(configuration_auto.CONFIG_MAPPING, '_extra_content'):
            extra = configuration_auto.CONFIG_MAPPING._extra_content
            for model in conflicting:
                if model in extra:
                    del extra[model]
                    print(f"✓ Removed {model} conflict")
    except Exception as e:
        print(f"Conflict fix error: {e}")

fix_all_conflicts()

# Clone LLaVA-Rad
!git clone https://github.com/microsoft/LLaVA-Rad.git /content/LLaVA-Rad
%cd /content/LLaVA-Rad

# Install LLaVA-Rad without reinstalling transformers
!pip install -e . --no-deps
!pip install -q gradio gradio_client

# Clone medical VLM repo
%cd /content
!git clone https://github.com/thedatasense/medical-vlm-intepret.git

# Add to paths
import sys
sys.path.insert(0, '/content/LLaVA-Rad')
sys.path.insert(0, '/content/medical-vlm-intepret/attention_viz')

print("✓ Repositories cloned and paths set")
"""

# ========================================
# Cell 3: Import LLaVA-Rad Components
# ========================================
"""
# Fix conflicts again before importing
from fix_model_conflicts import fix_transformers_conflicts
fix_transformers_conflicts()

# Now try importing LLaVA-Rad
try:
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    print("✓ LLaVA-Rad imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    # Try without some imports
    try:
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
        print("✓ Basic LLaVA-Rad imports successful")
    except Exception as e2:
        print(f"✗ Basic import also failed: {e2}")
"""

# ========================================
# Cell 4: Load LLaVA-Rad Medical Model
# ========================================
"""
import torch
from PIL import Image
import numpy as np

# Create a simple loader function
def load_llava_medical(model_path="microsoft/llava-med-v1.5-mistral-7b", load_8bit=False):
    \"\"\"Load LLaVA medical model with error handling\"\"\"
    
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    
    disable_torch_init()
    
    # Model paths to try
    model_paths = [
        model_path,
        "microsoft/llava-med-1.5-mistral-7b",
        "liuhaotian/llava-v1.5-7b"  # Fallback
    ]
    
    for path in model_paths:
        try:
            print(f"Trying: {path}")
            
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path=path,
                model_base=None,
                model_name=get_model_name_from_path(path) if 'get_model_name_from_path' in globals() else "llava",
                load_8bit=load_8bit,
                load_4bit=False,
                device="cuda"
            )
            
            model.eval()
            print(f"✓ Loaded: {path}")
            return tokenizer, model, image_processor
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            continue
    
    raise RuntimeError("Could not load any LLaVA model")

# Load the model
print("Loading LLaVA-Rad medical model...")
try:
    tokenizer, model, image_processor = load_llava_medical()
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
"""

# ========================================
# Cell 5: Test Model on Medical Image
# ========================================
"""
# Mount drive
from google.colab import drive
drive.mount('/content/drive')

import os
import matplotlib.pyplot as plt

# Find medical images
data_root = '/content/drive/MyDrive/Robust_Medical_LLM_Dataset'
image_dir = f'{data_root}/MIMIC_JPG/hundred_vqa'

if os.path.exists(image_dir):
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    if images:
        test_image_path = os.path.join(image_dir, images[0])
        print(f"✓ Found {len(images)} medical images")
        
        # Load and display image
        image = Image.open(test_image_path).convert('RGB')
        
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title("Test Medical Image")
        plt.axis('off')
        plt.show()
        
        # Test generation
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        
        # Setup conversation
        conv = conv_templates["llava_v1"].copy()
        question = "Is there evidence of pneumonia in this chest X-ray?"
        
        if model.config.mm_use_im_start_end:
            prompt = DEFAULT_IMAGE_TOKEN + '\\n' + question
        else:
            prompt = DEFAULT_IMAGE_TOKEN + question
            
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        
        # Process inputs
        input_ids = tokenizer_image_token(
            prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()
        
        image_tensor = process_images([image], image_processor, model.config)[0]
        image_tensor = image_tensor.unsqueeze(0).half().cuda()
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=100,
                use_cache=True
            )
        
        # Decode
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        # Extract answer
        if conv.sep_style == SeparatorStyle.TWO:
            answer = outputs.split(conv.sep2)[-1].strip()
        else:
            answer = outputs.split(conv.sep)[-1].strip()
            
        print(f"\\nQuestion: {question}")
        print(f"Answer: {answer}")
else:
    print("✗ Medical images not found")
"""

# ========================================
# Cell 6: Load MedGemma
# ========================================
"""
# Import and load MedGemma
from medgemma_enhanced import load_medgemma, build_inputs, generate_answer

print("\\nLoading MedGemma...")
medgemma_model, medgemma_processor = load_medgemma(
    dtype=torch.float16,
    device_map="auto"
)
print("✓ MedGemma loaded")
"""

# ========================================
# Cell 7: Use Enhanced Medical-Only Module
# ========================================
"""
# Import the medical-only implementation
from llava_rad_medical_only import LLaVARadMedical, MedicalAttentionConfig

# Create instance with medical config
config = MedicalAttentionConfig(
    colormap='hot',
    attention_head_mode='mean',
    alpha=0.5
)

llava_medical = LLaVARadMedical(config=config)

# Use the already loaded model
llava_medical.model = model
llava_medical.tokenizer = tokenizer
llava_medical.image_processor = image_processor

print("✓ Medical model wrapper ready")

# Test with attention
if 'test_image_path' in locals():
    result = llava_medical.generate_with_attention(
        test_image_path,
        "What are the main findings in this chest X-ray?",
        max_new_tokens=100
    )
    
    print(f"\\nAnswer: {result['answer']}")
    print(f"Attention method: {result.get('attention_method', 'N/A')}")
    
    # Visualize
    if result.get('visual_attention') is not None:
        viz = llava_medical.visualize_attention(
            image,
            result['visual_attention'],
            title="LLaVA-Rad Medical Attention"
        )
        plt.figure(figsize=(12, 6))
        plt.imshow(viz)
        plt.axis('off')
        plt.show()
"""

# ========================================  
# Cell 8: Run Full Analysis
# ========================================
"""
# Run the analysis pipeline
%cd /content/medical-vlm-intepret/attention_viz
!python run_medical_vlm_analysis.py --n_studies 5 --output_dir /content/results
"""