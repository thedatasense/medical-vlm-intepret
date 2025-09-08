"""
Complete setup for LLaVA-Rad in Google Colab
Copy and run these cells in order
"""

# ========================================
# Cell 1: Clone and Setup LLaVA-Rad
# ========================================
"""
# Clone official LLaVA-Rad repository
!git clone https://github.com/microsoft/LLaVA-Rad.git
%cd LLaVA-Rad

# Install LLaVA-Rad
!pip install -e .

# Install additional dependencies
!pip install -U "transformers>=4.56.1" accelerate bitsandbytes
!pip install opencv-python scipy matplotlib einops

# Go back to main directory
%cd /content
"""

# ========================================
# Cell 2: Clone Medical VLM Interpret
# ========================================
"""
# Clone medical VLM analysis repository
!git clone https://github.com/thedatasense/medical-vlm-intepret.git

# Add both to Python path
import sys
sys.path.insert(0, '/content/LLaVA-Rad')
sys.path.insert(0, '/content/medical-vlm-intepret/attention_viz')

print("✓ Repositories cloned and paths set")
"""

# ========================================
# Cell 3: Verify Imports
# ========================================
"""
# Test LLaVA-Rad imports
try:
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    print("✓ LLaVA-Rad imports successful")
except ImportError as e:
    print(f"✗ LLaVA-Rad import error: {e}")

# Test our modules
try:
    from llava_rad_enhanced import EnhancedLLaVARadVisualizer, AttentionConfig
    from medgemma_enhanced import load_medgemma
    print("✓ Enhanced modules imported")
except ImportError as e:
    print(f"✗ Module import error: {e}")
"""

# ========================================
# Cell 4: Mount Drive and Check Data
# ========================================
"""
from google.colab import drive
drive.mount('/content/drive')

import os

# Define paths
data_root = '/content/drive/MyDrive/Robust_Medical_LLM_Dataset'
image_dir = f'{data_root}/MIMIC_JPG/hundred_vqa'

if os.path.exists(image_dir):
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    print(f"✓ Found {len(images)} medical images")
    test_image_path = os.path.join(image_dir, images[0])
else:
    print("✗ Medical images not found")
    test_image_path = None
"""

# ========================================
# Cell 5: Load LLaVA-Rad Model
# ========================================
"""
from llava_rad_enhanced import EnhancedLLaVARadVisualizer, AttentionConfig

# Configure attention visualization
config = AttentionConfig(
    use_medical_colormap=True,
    multi_head_mode='mean',
    alpha=0.5
)

# Create visualizer
llava_vis = EnhancedLLaVARadVisualizer(config=config)

# Load model - try different paths
print("Loading LLaVA-Rad model...")
try:
    # Try the medical-specific model first
    llava_vis.load_model(
        model_path="microsoft/llava-med-v1.5-mistral-7b",
        model_base=None,
        load_in_8bit=False  # A100 has enough memory
    )
    print("✓ Loaded medical LLaVA-Rad model")
except Exception as e:
    print(f"Medical model failed: {e}")
    # Try standard LLaVA-Rad
    try:
        llava_vis.load_model(
            model_path="liuhaotian/llava-v1.5-7b",
            model_base=None,
            load_in_8bit=False
        )
        print("✓ Loaded standard LLaVA-Rad model")
    except Exception as e2:
        print(f"Standard model also failed: {e2}")
"""

# ========================================
# Cell 6: Test LLaVA-Rad
# ========================================
"""
from PIL import Image
import matplotlib.pyplot as plt

if test_image_path and os.path.exists(test_image_path):
    # Load and display test image
    image = Image.open(test_image_path).convert('RGB')
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title("Test Medical Image")
    plt.axis('off')
    plt.show()
    
    # Test generation
    question = "Is there evidence of pneumonia in this chest X-ray?"
    print(f"\\nQuestion: {question}")
    
    # Generate with attention
    result = llava_vis.generate_with_attention(
        test_image_path,
        question,
        max_new_tokens=100,
        conv_mode="llava_v1"  # or "llava_med" if using medical model
    )
    
    print(f"\\nAnswer: {result['answer']}")
    print(f"Attention method: {result.get('attention_method', 'N/A')}")
    
    # Visualize attention if available
    if result.get('visual_attention') is not None:
        viz = llava_vis.visualize_attention(
            image,
            result['visual_attention'],
            title="LLaVA-Rad Attention Map"
        )
        plt.figure(figsize=(10, 10))
        plt.imshow(viz)
        plt.axis('off')
        plt.show()
else:
    print("No test image available")
"""

# ========================================
# Cell 7: Load MedGemma
# ========================================
"""
from medgemma_enhanced import load_medgemma, build_inputs, generate_answer

print("Loading MedGemma...")
medgemma_model, medgemma_processor = load_medgemma(
    dtype=torch.float16,
    device_map="auto"
)
print("✓ MedGemma loaded")
"""

# ========================================
# Cell 8: Compare Models
# ========================================
"""
# Run comparison if both models loaded
if test_image_path and 'llava_vis' in locals() and 'medgemma_model' in locals():
    print("Comparing models on:", question)
    
    # LLaVA-Rad result (already computed above)
    print(f"\\nLLaVA-Rad: {result['answer']}")
    
    # MedGemma result
    inputs = build_inputs(
        medgemma_processor,
        image,
        question,
        do_pan_and_scan=False
    )
    
    medgemma_answer = generate_answer(
        medgemma_model,
        medgemma_processor,
        inputs
    )
    
    print(f"\\nMedGemma: {medgemma_answer}")
    
    # Side-by-side visualization
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Chest X-Ray")
    axes[0].axis('off')
    
    # LLaVA-Rad attention
    if result.get('visual_attention') is not None:
        axes[1].imshow(image)
        im = axes[1].imshow(
            cv2.resize(result['visual_attention'], (image.width, image.height)),
            cmap='hot',
            alpha=0.5
        )
        axes[1].set_title("LLaVA-Rad Attention")
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
"""

# ========================================
# Cell 9: Run Full Analysis
# ========================================
"""
# Run robustness analysis
%cd /content/medical-vlm-intepret/attention_viz
!python run_medical_vlm_analysis.py --n_studies 10 --output_dir /content/results
"""