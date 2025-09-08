"""
Colab-Ready Notebook Cells for Medical VLM Analysis
Copy these cells into your Google Colab notebook
"""

# ========================================
# Cell 1: Install Dependencies
# ========================================
"""
# Install modern transformers and dependencies
%pip install -U "transformers>=4.56.1" accelerate bitsandbytes
%pip install opencv-python scipy matplotlib pillow einops
%pip install torch torchvision --upgrade
"""

# ========================================
# Cell 2: Clone Repository and Setup
# ========================================
"""
import os
import sys

# Clone repository
if not os.path.exists('/content/medical-vlm-intepret'):
    !git clone https://github.com/thedatasense/medical-vlm-intepret.git
    
# Add to path
sys.path.insert(0, '/content/medical-vlm-intepret/attention_viz')

# Change directory
%cd /content/medical-vlm-intepret/attention_viz

print("✓ Repository setup complete")
"""

# ========================================
# Cell 3: Mount Google Drive
# ========================================
"""
from google.colab import drive
drive.mount('/content/drive')

# Verify data paths
data_root = '/content/drive/MyDrive/Robust_Medical_LLM_Dataset'
image_dir = f'{data_root}/MIMIC_JPG/hundred_vqa'
csv_path = f'{data_root}/attention_viz/medical-cxr-vqa-questions_sample.csv'

import os
print(f"Data root exists: {os.path.exists(data_root)}")
print(f"Image dir exists: {os.path.exists(image_dir)}")
print(f"CSV exists: {os.path.exists(csv_path)}")

if os.path.exists(image_dir):
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    print(f"Found {len(images)} images")
"""

# ========================================
# Cell 4: Import Modules
# ========================================
"""
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Import our modules
from llava_rad_enhanced import EnhancedLLaVARadVisualizer, AttentionConfig
from medgemma_enhanced import load_medgemma, build_inputs, generate_answer, extract_token_to_image_attention
from compare_attention_colab import compare_models_on_input

print("✓ Modules imported successfully")
"""

# ========================================
# Cell 5: Load Models (Memory Optimized)
# ========================================
"""
# Load LLaVA-Rad with 8-bit quantization
print("Loading LLaVA-Rad...")
llava_config = AttentionConfig(
    use_medical_colormap=True,
    multi_head_mode='mean'
)
llava_vis = EnhancedLLaVARadVisualizer(config=llava_config)
llava_vis.load_model(load_in_8bit=True)  # 8-bit with CPU offload
print("✓ LLaVA-Rad loaded")

# Load MedGemma
print("\\nLoading MedGemma...")
medgemma_model, medgemma_processor = load_medgemma(
    dtype=torch.float16,
    device_map="auto"
)
print("✓ MedGemma loaded")

# Clear cache
torch.cuda.empty_cache()
"""

# ========================================
# Cell 6: Test Single Image
# ========================================
"""
from PIL import Image
import matplotlib.pyplot as plt

# Load a test image
test_image_path = None
if os.path.exists(image_dir):
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')][:5]
    if images:
        test_image_path = os.path.join(image_dir, images[0])
        
if test_image_path is None:
    print("No medical images found. Please check your data paths.")
else:
    # Load and display image
    image = Image.open(test_image_path).convert('RGB')
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(f"Test Image: {os.path.basename(test_image_path)}")
    plt.axis('off')
    plt.show()
    
    # Test question
    question = "Is there evidence of pneumonia?"
    
    # Test LLaVA-Rad
    print(f"\\nQuestion: {question}")
    print("\\nTesting LLaVA-Rad...")
    llava_result = llava_vis.generate_with_attention(
        test_image_path,
        question,
        max_new_tokens=50
    )
    print(f"Answer: {llava_result['answer']}")
    print(f"Attention method: {llava_result.get('attention_method', 'N/A')}")
    
    # Visualize LLaVA attention
    if llava_result.get('visual_attention') is not None:
        llava_viz = llava_vis.visualize_attention(
            image,
            llava_result['visual_attention'],
            title="LLaVA-Rad Attention"
        )
        plt.figure(figsize=(10, 10))
        plt.imshow(llava_viz)
        plt.axis('off')
        plt.show()
"""

# ========================================
# Cell 7: Test MedGemma
# ========================================
"""
# Test MedGemma
print("\\nTesting MedGemma...")

# Build inputs correctly
inputs = build_inputs(
    medgemma_processor,
    image,
    question,
    do_pan_and_scan=False,
    device=medgemma_model.device
)

# Generate answer
answer = generate_answer(medgemma_model, medgemma_processor, inputs)
print(f"Answer: {answer}")

# Extract attention
try:
    attention_map, metadata = extract_token_to_image_attention(
        medgemma_model,
        medgemma_processor,
        image,
        question
    )
    print(f"Attention shape: {attention_map.shape}")
    print(f"Image tokens: {metadata['num_image_tokens']}")
    
    # Visualize
    from medgemma_enhanced import visualize_attention_on_image
    medgemma_viz = visualize_attention_on_image(
        image,
        attention_map,
        title="MedGemma Attention"
    )
    plt.figure(figsize=(10, 10))
    plt.imshow(medgemma_viz)
    plt.axis('off')
    plt.show()
except Exception as e:
    print(f"Attention extraction error: {e}")
"""

# ========================================
# Cell 8: Compare Models
# ========================================
"""
# Compare both models on the same input
print("\\nComparing models...")
results = compare_models_on_input(
    image_path=test_image_path,
    prompt=question,
    save_outputs=True,
    output_dir="/content/comparison_outputs"
)

# Display results
print(f"\\nLLaVA-Rad: {results.get('llava', {}).get('answer', 'N/A')}")
print(f"MedGemma: {results.get('medgemma', {}).get('answer', 'N/A')}")

if os.path.exists("/content/comparison_outputs"):
    print(f"\\nOutputs saved to /content/comparison_outputs")
    !ls /content/comparison_outputs
"""

# ========================================
# Cell 9: Run Full Analysis
# ========================================
"""
# Run robustness analysis on multiple samples
!python run_medical_vlm_analysis.py --n_studies 10 --output_dir /content/results

# Check results
if os.path.exists("/content/results"):
    import pandas as pd
    results_df = pd.read_csv("/content/results/detailed_results.csv")
    print(results_df.head())
    
    # Plot accuracy comparison
    import matplotlib.pyplot as plt
    
    if 'llava_correct' in results_df.columns and 'medgemma_correct' in results_df.columns:
        accuracies = {
            'LLaVA-Rad': results_df['llava_correct'].mean(),
            'MedGemma': results_df['medgemma_correct'].mean()
        }
        
        plt.figure(figsize=(8, 6))
        plt.bar(accuracies.keys(), accuracies.values())
        plt.ylabel('Accuracy')
        plt.title('Model Comparison on Medical VQA')
        plt.ylim(0, 1)
        for i, (model, acc) in enumerate(accuracies.items()):
            plt.text(i, acc + 0.02, f'{acc:.2%}', ha='center')
        plt.show()
"""

# ========================================
# Cell 10: Memory Management
# ========================================
"""
# Clear GPU memory if needed
import gc
torch.cuda.empty_cache()
gc.collect()

# Check memory usage
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
"""

# ========================================
# Additional Helper Functions
# ========================================
"""
# Function to process multiple images
def process_batch(image_paths, questions, models_dict):
    results = []
    for img_path, question in zip(image_paths, questions):
        print(f"Processing {os.path.basename(img_path)}...")
        result = compare_models_on_input(
            image_path=img_path,
            prompt=question,
            **models_dict
        )
        results.append(result)
        # Clear cache periodically
        if len(results) % 5 == 0:
            torch.cuda.empty_cache()
    return results

# Function to analyze consistency
def analyze_consistency(results_list):
    llava_answers = [r.get('llava', {}).get('answer', '') for r in results_list]
    medgemma_answers = [r.get('medgemma', {}).get('answer', '') for r in results_list]
    
    # Count unique answers
    llava_unique = len(set(llava_answers))
    medgemma_unique = len(set(medgemma_answers))
    
    print(f"LLaVA-Rad unique answers: {llava_unique}")
    print(f"MedGemma unique answers: {medgemma_unique}")
    
    return {
        'llava_consistency': 1.0 / llava_unique if llava_unique > 0 else 0,
        'medgemma_consistency': 1.0 / medgemma_unique if medgemma_unique > 0 else 0
    }
"""