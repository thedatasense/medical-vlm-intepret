#!/usr/bin/env python3
"""
Helper module for importing enhanced attention visualization modules in Google Colab
Updated to match the simplified codebase
"""

import sys
import os

# Add the attention_viz directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import enhanced modules - only what actually exists
try:
    from llava_rad_enhanced import (
        EnhancedLLaVARadVisualizer,
        AttentionConfig,
        AttentionMetrics,
        load_llava_rad,
        extract_query_to_image_attention
    )
    print("✓ LLaVA-Rad enhanced modules imported successfully")
except ImportError as e:
    print(f"❌ Error importing LLaVA-Rad enhanced modules: {e}")

try:
    from medgemma_enhanced import (
        EnhancedAttentionExtractor,
        AttentionExtractionConfig,
        load_medgemma,
        build_inputs,
        generate_answer,
        extract_token_to_image_attention,
        visualize_attention_on_image
    )
    print("✓ MedGemma enhanced modules imported successfully")
except ImportError as e:
    print(f"❌ Error importing MedGemma enhanced modules: {e}")

# Import comparison module
try:
    from compare_attention_colab import compare_models_on_input
    print("✓ Comparison module imported successfully")
except ImportError as e:
    print(f"❌ Error importing comparison module: {e}")

# Convenience function for Colab setup
def setup_colab_environment():
    """Setup the Colab environment with necessary paths and imports"""
    # Check if in Colab
    try:
        import google.colab
        IN_COLAB = True
        print("✓ Running in Google Colab")
    except ImportError:
        IN_COLAB = False
        print("✓ Running in local environment")
    
    # Mount Drive if in Colab
    if IN_COLAB:
        try:
            from google.colab import drive
            if not os.path.exists('/content/drive'):
                drive.mount('/content/drive')
                print("✓ Google Drive mounted")
            else:
                print("✓ Google Drive already mounted")
        except Exception as e:
            print(f"❌ Error mounting Google Drive: {e}")
    
    # Set default paths
    if IN_COLAB:
        base_path = "/content/drive/MyDrive/Robust_Medical_LLM_Dataset"
    else:
        base_path = os.path.expanduser("~/Robust_Medical_LLM_Dataset")
    
    paths = {
        "base_csv": os.path.join(base_path, "attention_viz/medical-cxr-vqa-questions_sample.csv"),
        "var_csv": os.path.join(base_path, "attention_viz/medical-cxr-vqa-questions_sample_hardpositives.csv"),
        "image_root": os.path.join(base_path, "MIMIC_JPG/hundred_vqa"),
        "output_dir": "/content/outputs" if IN_COLAB else "outputs"
    }
    
    return IN_COLAB, paths

# Export key classes and functions that actually exist
__all__ = [
    # LLaVA-Rad
    'EnhancedLLaVARadVisualizer',
    'AttentionConfig',
    'AttentionMetrics',
    'load_llava_rad',
    'extract_query_to_image_attention',
    # MedGemma
    'EnhancedAttentionExtractor',
    'AttentionExtractionConfig',
    'load_medgemma',
    'build_inputs',
    'generate_answer',
    'extract_token_to_image_attention',
    'visualize_attention_on_image',
    # Comparison
    'compare_models_on_input',
    # Helper
    'setup_colab_environment'
]

if __name__ == "__main__":
    print("\nEnhanced Attention Visualization Modules")
    print("=" * 50)
    print("Available modules imported successfully!")
    print("\nUsage:")
    print("  from colab_imports import *")
    print("  IN_COLAB, paths = setup_colab_environment()")