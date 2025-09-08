# Medical VLM Attention Analysis

A streamlined codebase for comparing attention mechanisms between LLaVA-Rad and MedGemma-4b-it on medical imaging tasks.

**Updated for modern Transformers (4.56.1+) with Gemma3 support and HuggingFace-first approach.**

## Essential Files

### Core Implementation
- `llava_rad_enhanced.py` - Enhanced LLaVA-Rad visualizer with HF-first loading
- `medgemma_enhanced.py` - MedGemma with Gemma3 API support and robust attention extraction
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

# 2. Install dependencies (uses pinned versions from requirements.txt)
!pip install -r requirements.txt

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

## Key Updates

1. **Modern Dependencies**: Pinned to Transformers 4.56.1+ for Gemma3 support
2. **HuggingFace First**: Default to HF model loading (microsoft/llava-rad, google/medgemma-4b-it)
3. **Robust Attention**: Handles variable vision token counts with SigLIP encoder support
4. **Clean Architecture**: Removed redundant files and synthetic data generation

## Models

- **LLaVA-Rad**: `microsoft/llava-rad` (falls back to `llava-hf/llava-1.5-7b-hf` if needed)
- **MedGemma**: `google/medgemma-4b-it` (uses Gemma3 APIs with bfloat16 by default)

## Archive

Non-essential files (test scripts, old implementations, documentation) have been moved to the `archive/` directory for reference.