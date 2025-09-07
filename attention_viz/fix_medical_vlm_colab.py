#!/usr/bin/env python3
"""
Fixed version for Medical VLM Analysis - handles various model architectures

This script automatically detects and adapts to different medical VLM architectures.
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, Optional

# Setup paths
sys.path.insert(0, '/content/LLaVA')
sys.path.insert(0, '/content/medical-vlm-intepret/attention_viz')

def detect_model_architecture(model_id: str) -> str:
    """Detect the architecture type from model ID"""
    model_id_lower = model_id.lower()
    
    if 'paligemma' in model_id_lower:
        return 'paligemma'
    elif 'llava' in model_id_lower:
        return 'llava'
    elif 'flamingo' in model_id_lower:
        return 'flamingo'
    elif 'biomedclip' in model_id_lower:
        return 'clip'
    else:
        return 'unknown'

def load_medical_vlm(model_id: str = "google/paligemma-3b-mix-224", load_in_8bit: bool = True):
    """Load medical VLM with automatic architecture detection"""
    
    from transformers import AutoProcessor, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig
    
    print(f"Loading model: {model_id}")
    architecture = detect_model_architecture(model_id)
    print(f"Detected architecture: {architecture}")
    
    # Configure quantization
    bnb_config = None
    if load_in_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
    
    # Load processor
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except:
        # Fallback for models without AutoProcessor
        from transformers import AutoTokenizer, AutoImageProcessor
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        image_processor = AutoImageProcessor.from_pretrained(model_id)
        processor = {'tokenizer': tokenizer, 'image_processor': image_processor}
    
    # Load model based on architecture
    try:
        # Try AutoModelForCausalLM first (most common for VLMs)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True
        )
    except:
        try:
            # Fallback to AutoModel
            model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                quantization_config=bnb_config,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
    
    model.eval()
    return model, processor, architecture

def generate_with_medical_vlm(
    model, 
    processor, 
    architecture: str,
    image: Image.Image,
    question: str,
    max_new_tokens: int = 50
) -> Tuple[str, Optional[Any]]:
    """Generate answer with automatic prompt formatting"""
    
    device = next(model.parameters()).device
    
    # Format prompt based on architecture
    if architecture == 'paligemma':
        # PaliGemma uses specific prompt format
        if hasattr(processor, 'apply_chat_template'):
            messages = [{
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = question  # PaliGemma might not need image tokens
    else:
        # Default format
        prompt = f"<image>{question}"
    
    # Prepare inputs
    try:
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        if isinstance(inputs, dict):
            inputs = {k: v.to(device) for k, v in inputs.items()}
    except:
        # Handle processors with different APIs
        inputs = {
            'input_ids': processor.tokenizer(prompt, return_tensors="pt").input_ids.to(device),
            'pixel_values': processor.image_processor(image, return_tensors="pt").pixel_values.to(device)
        }
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True
        )
    
    # Decode answer
    if hasattr(outputs, 'sequences'):
        if hasattr(processor, 'decode'):
            answer = processor.decode(outputs.sequences[0], skip_special_tokens=True)
        elif hasattr(processor, 'tokenizer'):
            answer = processor.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        else:
            answer = processor['tokenizer'].decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Clean answer
        answer = answer.split("Assistant:")[-1].strip()
        if question in answer:
            answer = answer.split(question)[-1].strip()
    else:
        answer = "Failed to generate"
    
    return answer, outputs

def test_medical_vlm():
    """Test the medical VLM with proper error handling"""
    
    # Test parameters
    model_ids_to_try = [
        "google/paligemma-3b-mix-224",
        "google/paligemma-3b-mix-448",
        "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    ]
    
    test_image_path = "/content/drive/MyDrive/Robust_Medical_LLM_Dataset/MIMIC_JPG/hundred_vqa/10000032.jpg"
    test_question = "Is there evidence of pneumonia?"
    
    # Try to load image
    if os.path.exists(test_image_path):
        test_image = Image.open(test_image_path).convert('RGB')
        print(f"Test image loaded: {test_image.size}")
    else:
        print(f"Test image not found: {test_image_path}")
        return
    
    # Try each model
    for model_id in model_ids_to_try:
        print(f"\n{'='*50}")
        print(f"Testing: {model_id}")
        print('='*50)
        
        try:
            # Load model
            model, processor, architecture = load_medical_vlm(model_id, load_in_8bit=True)
            print(f"✓ Model loaded successfully")
            
            # Test generation
            answer, outputs = generate_with_medical_vlm(
                model, processor, architecture, 
                test_image, test_question
            )
            
            print(f"✓ Answer: {answer}")
            
            # Check if we got attentions
            if hasattr(outputs, 'attentions') and outputs.attentions:
                print(f"✓ Attentions available: {len(outputs.attentions)} tokens")
            else:
                print("✗ No attentions in output")
            
            # Success - save working config
            config = {
                'model_id': model_id,
                'architecture': architecture,
                'working': True,
                'answer': answer
            }
            
            with open('working_model_config.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"\n✓ SUCCESS! Using {model_id}")
            return model, processor, architecture
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            continue
    
    print("\n✗ All models failed to load")
    return None, None, None

if __name__ == "__main__":
    print("Testing Medical VLM Models...")
    model, processor, architecture = test_medical_vlm()
    
    if model is not None:
        print("\n" + "="*50)
        print("RECOMMENDED NEXT STEPS:")
        print("="*50)
        print("1. Update your code to use the working model ID from working_model_config.json")
        print("2. Use the detected architecture type for proper prompt formatting")
        print("3. Run the full pipeline with the working configuration")
    else:
        print("\nPlease check:")
        print("1. GPU memory availability")
        print("2. Model access permissions")
        print("3. Internet connectivity for model downloads")