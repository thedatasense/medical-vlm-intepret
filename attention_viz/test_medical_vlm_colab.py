#!/usr/bin/env python3
"""
Test script for Medical VLM Analysis - Debug version

This simplified script helps debug issues before running the full pipeline.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

print("=== Medical VLM Test Script ===")

# 1. Setup paths
print("\n1. Setting up paths...")
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

# 3. Test image path
print("\n3. Testing image path...")
test_image_path = "/content/drive/MyDrive/Robust_Medical_LLM_Dataset/MIMIC_JPG/hundred_vqa/10000032.jpg"
if os.path.exists(test_image_path):
    print(f"✓ Test image found: {test_image_path}")
    test_image = Image.open(test_image_path).convert('RGB')
    print(f"  Image size: {test_image.size}")
else:
    print(f"✗ Test image not found: {test_image_path}")
    # Try to find any image
    image_dir = "/content/drive/MyDrive/Robust_Medical_LLM_Dataset/MIMIC_JPG/hundred_vqa"
    if os.path.exists(image_dir):
        images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')][:5]
        print(f"  Found {len(images)} images in directory")
        if images:
            test_image_path = os.path.join(image_dir, images[0])
            test_image = Image.open(test_image_path).convert('RGB')
            print(f"  Using: {images[0]}")

# 4. Test LLaVA-Rad
print("\n4. Testing LLaVA-Rad...")
try:
    llava_config = AttentionConfig(
        use_medical_colormap=True,
        multi_head_mode='mean',  # Start with mean instead of entropy_weighted
        percentile_clip=(5, 95)
    )
    llava_vis = EnhancedLLaVARadVisualizer(config=llava_config)
    print("  Loading model...")
    llava_vis.load_model(load_in_8bit=True)
    print("✓ LLaVA-Rad loaded successfully")
    
    # Test inference
    print("  Testing inference...")
    test_question = "Is there evidence of pneumonia?"
    result = llava_vis.generate_with_attention(
        test_image_path,
        test_question,
        max_new_tokens=50,
        use_cache=False
    )
    print(f"✓ Answer: {result['answer']}")
    print(f"  Attention method: {result.get('attention_method', 'unknown')}")
    
    # Check attention shape
    att = result.get('visual_attention')
    if att is not None:
        if isinstance(att, list):
            print(f"  Attention: list of {len(att)} heads, first shape: {att[0].shape}")
        else:
            print(f"  Attention shape: {att.shape}")
    
except Exception as e:
    print(f"✗ LLaVA-Rad error: {e}")
    import traceback
    traceback.print_exc()

# 5. Test MedGemma
print("\n5. Testing MedGemma...")
try:
    print("  Loading model...")
    medgemma_model, medgemma_processor = load_model_enhanced(
        model_id="google/medgemma-4b-it",
        load_in_8bit=True
    )
    print("✓ MedGemma loaded successfully")
    
    # Create extractor
    extractor = EnhancedAttentionExtractor(
        AttentionExtractionConfig(
            attention_head_reduction='mean',  # Start with mean
            fallback_chain=['cross_attention', 'gradcam', 'uniform']
        )
    )
    
    # Test inference
    print("  Testing inference...")
    
    # Prefer chat template to ensure image token presence
    prompts_to_try = []
    if hasattr(medgemma_processor, 'apply_chat_template'):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": test_question},
            ],
        }]
        prompt_chat = medgemma_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts_to_try.append(("chat", prompt_chat))
    # Fallback prompts
    prompts_to_try.extend([
        ("inline1", f"<image>{test_question}"),
        ("inline2", f"<image>\n{test_question}"),
    ])

    for i, (label, prompt) in enumerate(prompts_to_try, start=1):
        print(f"  Trying prompt format {i} ({label}): {repr(prompt[:60])}")
        try:
            inputs = medgemma_processor(text=prompt, images=test_image, return_tensors="pt")
            
            # Check input shapes
            print(f"    Input shapes: {[(k, v.shape) for k, v in inputs.items()]}")
            
            device = next(medgemma_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = medgemma_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    output_attentions=True,
                    return_dict_in_generate=True
                )
            
            # Decode answer
            answer = medgemma_processor.tokenizer.decode(
                outputs.sequences[0], skip_special_tokens=True
            )
            answer = answer.split("Assistant:")[-1].strip()
            print(f"✓ Answer: {answer}")
            
            # Test attention extraction
            attention, token_indices, method = extractor.extract_token_conditioned_attention_robust(
                medgemma_model, medgemma_processor, outputs,
                ["pneumonia"], test_image, prompt
            )
            print(f"  Attention shape: {attention.shape}, method: {method}")
            
            break  # Success, stop trying
            
        except Exception as e:
            print(f"    Error: {e}")
            if i == len(prompts_to_try) - 1:
                import traceback
                traceback.print_exc()
    
except Exception as e:
    print(f"✗ MedGemma error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test Complete ===")
print("\nIf both models loaded and produced answers, you can run the full pipeline.")
print("If there were errors, please check:")
print("1. GPU memory (use nvidia-smi)")
print("2. Package versions (transformers >= 4.36.0)")
print("3. Model access permissions")
print("4. Data paths")
