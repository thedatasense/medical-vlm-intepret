#!/usr/bin/env python3
"""
Helper module for importing enhanced attention visualization modules in Google Colab
"""

import sys
import os

# Add the attention_viz directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import enhanced modules
try:
    from llava_rad_enhanced import (
        EnhancedLLaVARadVisualizer,
        AttentionConfig,
        AttentionMetrics,
        AttentionDifferenceAnalyzer,
        AttentionCache
    )
    print("✓ LLaVA-Rad enhanced modules imported successfully")
except ImportError as e:
    print(f"❌ Error importing LLaVA-Rad enhanced modules: {e}")

try:
    from medgemma_enhanced import (
        EnhancedAttentionExtractor,
        AttentionExtractionConfig,
        AttentionVisualizationEnhanced,
        RobustAttentionAnalyzer,
        create_tight_body_mask
    )
    print("✓ MedGemma enhanced modules imported successfully")
except ImportError as e:
    print(f"❌ Error importing MedGemma enhanced modules: {e}")

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
        "output_dir": "outputs_enhanced"
    }
    
    return IN_COLAB, paths

# Export key classes and functions
__all__ = [
    # LLaVA-Rad enhanced
    'EnhancedLLaVARadVisualizer',
    'AttentionConfig',
    'AttentionMetrics',
    'AttentionDifferenceAnalyzer',
    'AttentionCache',
    # MedGemma enhanced
    'EnhancedAttentionExtractor',
    'AttentionExtractionConfig',
    'AttentionVisualizationEnhanced',
    'RobustAttentionAnalyzer',
    'create_tight_body_mask',
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