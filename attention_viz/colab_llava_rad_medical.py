#!/usr/bin/env python3
"""
Complete setup for using Microsoft LLaVA-Rad (Medical) in Google Colab
This handles the transformers conflict and uses the actual medical model
"""

# ========================================
# Cell 1: Fix Config Conflict & Clone LLaVA-Rad
# ========================================
"""
# Fix the transformers conflict first
import warnings
warnings.filterwarnings('ignore')

# Remove conflicting config
try:
    from transformers.models.auto import configuration_auto
    if hasattr(configuration_auto.CONFIG_MAPPING, '_extra_content'):
        if 'llava' in configuration_auto.CONFIG_MAPPING._extra_content:
            del configuration_auto.CONFIG_MAPPING._extra_content['llava']
            print("✓ Fixed transformers config conflict")
except:
    pass

# Clone LLaVA-Rad
!git clone https://github.com/microsoft/LLaVA-Rad.git
%cd LLaVA-Rad

# Install dependencies
!pip install -e .
!pip install -U tokenizers sentencepiece protobuf

# Go back to content
%cd /content

# Add to path
import sys
sys.path.insert(0, '/content/LLaVA-Rad')
print("✓ LLaVA-Rad setup complete")
"""

# ========================================
# Cell 2: Clone Medical VLM Interpret
# ========================================
"""
# Clone the analysis repository
!git clone https://github.com/thedatasense/medical-vlm-intepret.git

# Copy the fix script
!cp /content/medical-vlm-intepret/attention_viz/fix_llava_conflict.py /content/

# Add to path
sys.path.insert(0, '/content/medical-vlm-intepret/attention_viz')
print("✓ Medical VLM interpret setup complete")
"""

# ========================================
# Cell 3: Import with Conflict Handling
# ========================================
"""
# Import the fix function and run it
from fix_llava_conflict import fix_llava_config_conflict
fix_llava_config_conflict()

# Now import LLaVA-Rad
try:
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    print("✓ LLaVA-Rad imports successful")
except Exception as e:
    print(f"Import error: {e}")
"""

# ========================================
# Cell 4: Create Simple LLaVA-Rad Wrapper
# ========================================
"""
# Create a simple wrapper for LLaVA-Rad medical model
import torch
import numpy as np
from PIL import Image

class SimpleLLaVARadMedical:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.conv_mode = "llava_v1"
        
    def load_model(self, device="cuda", load_8bit=False):
        \"\"\"Load the medical LLaVA-Rad model\"\"\"
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
        
        disable_torch_init()
        
        # Try medical model first
        model_paths = [
            "microsoft/llava-med-v1.5-mistral-7b",
            "microsoft/llava-med-1.5-mistral-7b", 
            "liuhaotian/llava-v1.5-7b"  # Fallback to base model
        ]
        
        for model_path in model_paths:
            try:
                print(f"Trying to load: {model_path}")
                self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
                    model_path=model_path,
                    model_base=None,
                    model_name=get_model_name_from_path(model_path),
                    load_8bit=load_8bit,
                    load_4bit=False,
                    device=device
                )
                print(f"✓ Successfully loaded: {model_path}")
                self.model.eval()
                
                # Set conversation mode based on model
                if "med" in model_path:
                    self.conv_mode = "llava_v1"  # Medical models use v1
                else:
                    self.conv_mode = "llava_v1"
                    
                return True
                
            except Exception as e:
                print(f"Failed to load {model_path}: {e}")
                continue
                
        return False
    
    def generate(self, image_path, question, max_new_tokens=100):
        \"\"\"Generate answer for medical image\"\"\"
        if self.model is None:
            raise RuntimeError("Model not loaded")
            
        # Load and process image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
            
        # Process image
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]
        
        # Setup conversation
        conv = conv_templates[self.conv_mode].copy()
        
        # Format prompt
        if self.model.config.mm_use_im_start_end:
            question = DEFAULT_IMAGE_TOKEN + '\\n' + question
        else:
            question = DEFAULT_IMAGE_TOKEN + question
            
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Tokenize
        input_ids = tokenizer_image_token(
            prompt, 
            self.tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        ).unsqueeze(0).cuda()
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True
            )
        
        # Decode
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # Extract answer
        if conv.sep_style == SeparatorStyle.TWO:
            answer = outputs.split(conv.sep2)[-1].strip()
        else:
            answer = outputs.split(conv.sep)[-1].strip()
            
        return answer

# Create instance
llava_medical = SimpleLLaVARadMedical()
print("✓ Simple LLaVA-Rad wrapper created")
"""

# ========================================
# Cell 5: Load Model
# ========================================
"""
# Load the model
print("Loading LLaVA-Rad medical model...")
if llava_medical.load_model(device="cuda", load_8bit=False):
    print("✓ Model loaded successfully!")
else:
    print("✗ Failed to load model")
"""

# ========================================
# Cell 6: Mount Drive and Test
# ========================================
"""
# Mount drive
from google.colab import drive
drive.mount('/content/drive')

import os
import matplotlib.pyplot as plt

# Check for medical images
data_root = '/content/drive/MyDrive/Robust_Medical_LLM_Dataset'
image_dir = f'{data_root}/MIMIC_JPG/hundred_vqa'

test_image_path = None
if os.path.exists(image_dir):
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    if images:
        test_image_path = os.path.join(image_dir, images[0])
        print(f"✓ Found {len(images)} medical images")
        
        # Test the model
        image = Image.open(test_image_path)
        
        # Display image
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title("Test Medical Image")
        plt.axis('off')
        plt.show()
        
        # Generate answer
        question = "Is there evidence of pneumonia in this chest X-ray?"
        print(f"\\nQuestion: {question}")
        
        answer = llava_medical.generate(test_image_path, question)
        print(f"Answer: {answer}")
else:
    print("✗ Medical images not found")
"""

# ========================================
# Cell 7: Load MedGemma
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
# Cell 8: Compare Both Models
# ========================================
"""
if test_image_path and 'llava_medical' in locals() and 'medgemma_model' in locals():
    # Test both models
    print("\\n" + "="*50)
    print("COMPARING MODELS")
    print("="*50)
    
    question = "Describe any abnormalities in this chest X-ray."
    print(f"Question: {question}\\n")
    
    # LLaVA-Rad Medical
    print("LLaVA-Rad Medical Response:")
    llava_answer = llava_medical.generate(test_image_path, question, max_new_tokens=150)
    print(llava_answer)
    
    # MedGemma
    print("\\nMedGemma Response:")
    inputs = build_inputs(
        medgemma_processor,
        image,
        question,
        do_pan_and_scan=False
    )
    medgemma_answer = generate_answer(
        medgemma_model,
        medgemma_processor,
        inputs,
        max_new_tokens=150
    )
    print(medgemma_answer)
"""

# ========================================
# Cell 9: Run Analysis with Enhanced Modules
# ========================================
"""
# Use the enhanced modules that support attention extraction
from llava_rad_medical_only import LLaVARadMedical, MedicalAttentionConfig

# Create visualizer
config = MedicalAttentionConfig(
    colormap='hot',
    attention_head_mode='mean',
    alpha=0.5
)

llava_vis = LLaVARadMedical(config=config)

# This will use the already loaded LLaVA-Rad
llava_vis.model = llava_medical.model
llava_vis.tokenizer = llava_medical.tokenizer
llava_vis.image_processor = llava_medical.image_processor
llava_vis.using_hf_fallback = False

print("✓ Enhanced visualizer ready")

# Test with attention extraction
if test_image_path:
    result = llava_vis.generate_with_attention(
        test_image_path,
        "What are the main findings in this chest X-ray?",
        max_new_tokens=100
    )
    
    print(f"\\nAnswer: {result['answer']}")
    print(f"Attention method: {result.get('attention_method', 'N/A')}")
    
    # Visualize attention if available
    if result.get('visual_attention') is not None:
        viz = llava_vis.visualize_attention(
            image,
            result['visual_attention'],
            title="LLaVA-Rad Medical Attention Map"
        )
        plt.figure(figsize=(10, 10))
        plt.imshow(viz)
        plt.axis('off')
        plt.show()
"""

# ========================================
# Cell 10: Run Full Analysis (Optional)
# ========================================
"""
# Run the complete analysis
%cd /content/medical-vlm-intepret/attention_viz
!python run_medical_vlm_analysis.py --n_studies 5 --output_dir /content/results
"""