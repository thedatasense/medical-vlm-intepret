#!/usr/bin/env python3
"""
Fixed test script for Medical VLM Analysis with all corrections applied

Run this in Google Colab to verify the fixes work correctly.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

print("=== Fixed Medical VLM Test Script ===")
print("This includes all fixes for attention extraction and answer decoding\n")

# 1. Setup paths
print("1. Setting up paths...")
sys.path.insert(0, '/content/LLaVA')
sys.path.insert(0, '/content/medical-vlm-intepret/attention_viz')

# 2. Import modules
print("\n2. Importing modules...")
try:
    from llava_rad_enhanced import EnhancedLLaVARadVisualizer, AttentionConfig
    from medgemma_enhanced import load_model_enhanced, EnhancedAttentionExtractor, AttentionExtractionConfig
    print("✓ Modules imported successfully")
except Exception as e:
    print(f"✗ Import error: {e}")
    exit(1)

# 3. Find test image
print("\n3. Finding test image...")
image_dir = "/content/drive/MyDrive/Robust_Medical_LLM_Dataset/MIMIC_JPG/hundred_vqa"
test_image_path = None
test_image = None

if os.path.exists(image_dir):
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')][:5]
    if images:
        test_image_path = os.path.join(image_dir, images[0])
        test_image = Image.open(test_image_path).convert('RGB')
        print(f"✓ Using image: {images[0]} (size: {test_image.size})")
else:
    print(f"✗ Image directory not found: {image_dir}")
    exit(1)

test_question = "Is there evidence of pneumonia?"

# 4. Test LLaVA-Rad
print("\n4. Testing LLaVA-Rad...")
try:
    llava_config = AttentionConfig(
        use_medical_colormap=True,
        multi_head_mode='mean',  # Using mean aggregation
        percentile_clip=(5, 95)
    )
    llava_vis = EnhancedLLaVARadVisualizer(config=llava_config)
    print("  Loading model (8-bit quantization)...")
    llava_vis.load_model(load_in_8bit=True)
    print("✓ LLaVA-Rad loaded successfully")
    
    # Test inference
    print(f"  Testing with question: {test_question}")
    result = llava_vis.generate_with_attention(
        test_image_path,
        test_question,
        max_new_tokens=50,
        use_cache=False
    )
    
    print(f"✓ Answer: {result['answer']}")
    print(f"  Attention method: {result.get('attention_method', 'unknown')}")
    
    # Check attention
    att = result.get('visual_attention')
    if att is not None:
        if isinstance(att, list):
            print(f"  Attention: list of {len(att)} heads, first shape: {att[0].shape}")
        else:
            print(f"  Attention shape: {att.shape}")
            print(f"  Focus score: {result.get('metrics', {}).get('focus', 'N/A'):.3f}")
    
except Exception as e:
    print(f"✗ LLaVA-Rad error: {e}")
    import traceback
    traceback.print_exc()

# 5. Test MedGemma/PaliGemma
print("\n5. Testing MedGemma/PaliGemma...")
try:
    print("  Loading model with eager attention...")
    medgemma_model, medgemma_processor = load_model_enhanced(
        model_id="google/paligemma-3b-mix-224",
        load_in_8bit=True
    )
    print("✓ Model loaded successfully")
    
    # Create extractor
    extractor = EnhancedAttentionExtractor(
        AttentionExtractionConfig(
            attention_head_reduction='mean',
            fallback_chain=['cross_attention', 'gradcam', 'uniform']
        )
    )
    
    # Test inference with proper prompt format
    print(f"  Testing with question: {test_question}")
    
    # Use chat template for proper formatting
    if hasattr(medgemma_processor, 'apply_chat_template'):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": test_question}
            ]
        }]
        prompt = medgemma_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = f"<image>{test_question}"
    
    print(f"  Prompt format: {repr(prompt[:50])}...")
    
    # Prepare inputs
    inputs = medgemma_processor(
        text=prompt,
        images=test_image,
        return_tensors="pt"
    )
    
    device = next(medgemma_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate with attention
    with torch.no_grad():
        outputs = medgemma_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True
        )
    
    # Decode answer with Gemma format handling
    raw_answer = medgemma_processor.tokenizer.decode(
        outputs.sequences[0], skip_special_tokens=True
    )
    
    # Clean answer
    if '<start_of_turn>model' in raw_answer:
        parts = raw_answer.split('<start_of_turn>model')
        if len(parts) > 1:
            answer = parts[-1].split('<end_of_turn>')[0].strip()
        else:
            answer = raw_answer
    elif 'model\n' in raw_answer:
        answer = raw_answer.split('model\n')[-1].strip()
    else:
        answer = raw_answer.split(test_question)[-1].strip()
    
    print(f"✓ Answer: {answer}")
    
    # Test attention extraction
    attention, token_indices, method = extractor.extract_token_conditioned_attention_robust(
        medgemma_model, medgemma_processor, outputs,
        ["pneumonia", "consolidation", "opacity"], test_image, prompt
    )
    
    print(f"  Attention shape: {attention.shape}, method: {method}")
    
    # Check if we actually got cross-attention
    if method == 'cross_attention' or method == 'cross_attention_forward':
        print("✓ Cross-attention extraction successful!")
    else:
        print(f"⚠ Fell back to {method} method")
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            print(f"  outputs.attentions exists but extraction failed")
    
except Exception as e:
    print(f"✗ MedGemma error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test Complete ===")
print("\nSummary:")
print("- LLaVA-Rad: Check if answer is coherent and attention was extracted")
print("- MedGemma: Check if answer is clean (no 'model' prefix) and attention method")
print("- If both work, you can run the full pipeline with confidence!")
print("\nNext step: !python run_medical_vlm_analysis_colab.py --n_studies 5")