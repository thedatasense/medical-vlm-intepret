#!/usr/bin/env python3
"""
Simple test script for Colab to verify model loading
Run this first to ensure models load correctly
"""

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test imports
print("\n1. Testing imports...")
try:
    from llava_rad_enhanced import EnhancedLLaVARadVisualizer, AttentionConfig
    print("✓ LLaVA imports OK")
except Exception as e:
    print(f"✗ LLaVA import error: {e}")

try:
    from medgemma_enhanced import load_medgemma, build_inputs, generate_answer
    print("✓ MedGemma imports OK")
except Exception as e:
    print(f"✗ MedGemma import error: {e}")

# Test LLaVA loading with HF models
print("\n2. Testing LLaVA loading...")
try:
    # Use a known working HuggingFace LLaVA model
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    
    # Try without quantization first on A100
    print("Loading llava-hf/llava-1.5-7b-hf...")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        dtype=torch.float16,  # Use dtype, not torch_dtype
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()
    print("✓ LLaVA loaded successfully!")
    
    # Clean up
    del model
    del processor
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"✗ LLaVA loading error: {e}")
    import traceback
    traceback.print_exc()

# Test with wrapper class
print("\n3. Testing LLaVA wrapper...")
try:
    config = AttentionConfig(use_medical_colormap=True)
    llava_vis = EnhancedLLaVARadVisualizer(config=config)
    
    # Try without 8-bit first on A100
    llava_vis.load_model(load_in_8bit=False)
    print("✓ LLaVA wrapper loaded successfully!")
    
except Exception as e:
    print(f"✗ LLaVA wrapper error: {e}")
    import traceback
    traceback.print_exc()

# Test MedGemma
print("\n4. Testing MedGemma loading...")
try:
    medgemma_model, medgemma_processor = load_medgemma(
        dtype=torch.float16,
        device_map="auto"
    )
    print("✓ MedGemma loaded successfully!")
    
except Exception as e:
    print(f"✗ MedGemma loading error: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ Setup complete! You can now run the full analysis.")