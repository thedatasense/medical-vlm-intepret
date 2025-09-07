# Medical VLM Attention Analysis

A streamlined codebase for comparing attention mechanisms between LLaVA-Rad and MedGemma-4b-it on medical imaging tasks.

## Essential Files

### Core Implementation
- `llava_rad_enhanced.py` - Enhanced LLaVA-Rad visualizer with attention extraction
- `medgemma_enhanced.py` - MedGemma model with attention extraction capabilities
- `compare_attention_colab.py` - Side-by-side model comparison pipeline
- `run_medical_vlm_analysis.py` - Complete analysis pipeline for robustness studies

### Helpers
- `colab_imports.py` - Import helper for Google Colab environment
- `setup_colab_environment.py` - Environment setup with proper error handling

### Data
- `medical-cxr-vqa-questions_sample.csv` - Sample medical VQA questions
- `medical-cxr-vqa-questions_sample_hardpositives.csv` - Hard positive test cases

### Documentation
- `CLAUDE.md` - Guidance for Claude Code assistant
- `README.md` - This file

## Quick Start (Google Colab)

```python
# 1. Clone repository
!git clone https://github.com/thedatasense/medical-vlm-intepret.git
%cd medical-vlm-intepret/attention_viz

# 2. Install dependencies
!pip install torch transformers opencv-python scipy matplotlib pillow bitsandbytes

# 3. Run comparison
from compare_attention_colab import compare_models_on_input

results = compare_models_on_input(
    image_path="path/to/chest_xray.jpg",
    prompt="Is there evidence of pneumonia?",
    save_outputs=True
)

# 4. Setup environment first (handles LLaVA installation properly)
!python setup_colab_environment.py

# 5. Run full analysis
!python run_medical_vlm_analysis.py --n_studies 100
```

## Archive

Non-essential files (test scripts, old implementations, documentation) have been moved to the `archive/` directory for reference.