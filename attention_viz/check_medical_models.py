#!/usr/bin/env python3
"""
Check available medical vision-language models on Hugging Face
"""

# Common medical VLM models to try:
medical_vlm_models = [
    # Google medical models
    "google/paligemma-3b-mix-224",
    "google/paligemma-3b-mix-448", 
    "google/paligemma-3b-ft-vqav2-224",
    "google/paligemma-3b-ft-vqav2-448",
    
    # Alternative medical VLMs
    "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "microsoft/BiomedVLP-CXR-BERT-specialized",
    "StanfordAIMI/RadBERT",
    
    # Med-Flamingo variants
    "med-flamingo/med-flamingo",
    
    # LLaVA medical variants
    "microsoft/llava-med-v1.5-mistral-7b",
    "llava-hf/llava-v1.6-mistral-7b-hf",
]

print("Potential medical VLM models to try:")
print("="*50)
for model in medical_vlm_models:
    print(f"- {model}")

print("\nFor Google Colab, try running:")
print("from transformers import AutoModel, AutoProcessor")
print("model_id = 'google/paligemma-3b-mix-224'  # or another from the list")
print("processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)")
print("model = AutoModel.from_pretrained(model_id, trust_remote_code=True)")