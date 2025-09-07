#!/usr/bin/env python3
"""
Test script for LLaVA-Rad vs MedGemma-4b-it comparison

Note: MedGemma-4b-it capabilities need to be verified (may be text-only or multimodal).
This script tests both models with appropriate inputs.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

print("=== LLaVA-Rad vs MedGemma-4b-it Test ===")
print("Note: Testing MedGemma capabilities, LLaVA-Rad is multimodal\n")

# Setup paths
sys.path.insert(0, '/content/LLaVA')
sys.path.insert(0, '/content/medical-vlm-intepret/attention_viz')

# Import modules
try:
    from llava_rad_enhanced import EnhancedLLaVARadVisualizer, AttentionConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    print("✓ Modules imported successfully")
except Exception as e:
    print(f"✗ Import error: {e}")
    exit(1)

# Find or create test image
print("\n1. Setting up test data...")
image_dirs = [
    "/content/drive/MyDrive/Robust_Medical_LLM_Dataset/MIMIC_JPG/hundred_vqa",
    "/content/drive/MyDrive/medical_images",
    "."
]

test_image_path = None
for image_dir in image_dirs:
    if os.path.exists(image_dir):
        images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))][:1]
        if images:
            test_image_path = os.path.join(image_dir, images[0])
            break

if test_image_path is None:
    # Create synthetic test image
    print("Creating synthetic chest X-ray image...")
    test_image = Image.new('RGB', (512, 512), color=(200, 200, 200))
    pixels = np.array(test_image)
    # Add dark regions to simulate lungs
    pixels[150:350, 100:200] = 50  # Left lung
    pixels[150:350, 300:400] = 50  # Right lung
    # Add bright spot to simulate potential pneumonia
    pixels[250:280, 320:350] = 180
    test_image = Image.fromarray(pixels.astype(np.uint8))
    test_image_path = "synthetic_cxr.png"
    test_image.save(test_image_path)
else:
    test_image = Image.open(test_image_path).convert('RGB')

print(f"✓ Test image: {test_image_path}")

# Test questions
questions = [
    "Is there evidence of pneumonia in this chest X-ray?",
    "What abnormalities are visible?",
    "Is the cardiac silhouette normal?"
]

print("\n2. Testing LLaVA-Rad (Multimodal)...")
try:
    # Load LLaVA-Rad
    llava_config = AttentionConfig(
        use_medical_colormap=True,
        multi_head_mode='mean',
        percentile_clip=(5, 95)
    )
    llava_vis = EnhancedLLaVARadVisualizer(config=llava_config)
    print("Loading LLaVA-Rad...")
    llava_vis.load_model(load_in_8bit=True)
    print("✓ LLaVA-Rad loaded")
    
    # Test with image
    for i, question in enumerate(questions[:1]):  # Test first question
        print(f"\nQuestion: {question}")
        result = llava_vis.generate_with_attention(
            test_image_path,
            question,
            max_new_tokens=100,
            use_cache=False
        )
        print(f"LLaVA-Rad answer: {result['answer']}")
        
        # Check attention
        att = result.get('visual_attention')
        if att is not None:
            if isinstance(att, list):
                print(f"Attention: {len(att)} heads, shape: {att[0].shape}")
            else:
                print(f"Attention shape: {att.shape}")
            print(f"Attention method: {result.get('attention_method', 'unknown')}")
            
except Exception as e:
    print(f"✗ LLaVA-Rad error: {e}")
    import traceback
    traceback.print_exc()

print("\n3. Testing MedGemma-4b-it...")
try:
    # Load MedGemma
    print("Loading MedGemma-4b-it...")
    
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained("google/medgemma-4b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/medgemma-4b-it",
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    model.eval()
    print("✓ MedGemma loaded")
    
    # Test WITHOUT image (since Gemma is text-only)
    for i, question in enumerate(questions[:1]):
        print(f"\nQuestion: {question}")
        print("Note: Testing MedGemma response")
        
        # Format as instruction
        prompt = f"<start_of_turn>user\n{question}\n<start_of_turn>model\n"
        
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract model response
        if '<start_of_turn>model' in answer:
            answer = answer.split('<start_of_turn>model')[-1].strip()
        else:
            answer = answer.split(prompt)[-1].strip()
            
        print(f"MedGemma answer: {answer}")
        
except Exception as e:
    print(f"✗ MedGemma error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test Complete ===")
print("\nKey Observations:")
print("- LLaVA-Rad: Analyzes the actual image and provides visual attention")
print("- MedGemma-4b-it: Response capabilities depend on whether it supports vision")
print("- For true multimodal comparison, both models need image processing capability")
print("\nConsider using two multimodal models for meaningful comparison!")