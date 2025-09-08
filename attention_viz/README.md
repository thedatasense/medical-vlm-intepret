# Medical VLM Attention Analysis

Professional implementation for comparing attention mechanisms between LLaVA-Rad (Microsoft's medical vision-language model) and MedGemma (Google's medical language model) on chest X-ray interpretation tasks.

## Overview

This repository provides a robust framework for analyzing and comparing attention patterns in medical vision-language models, specifically designed for:
- Medical image interpretation with focus on chest X-rays
- Attention mechanism extraction and visualization
- Statistical robustness analysis across paraphrase variants
- Production-ready code for research reproducibility

## Project Structure

```
attention_viz/
├── llava_rad_medical_only.py    # LLaVA-Rad medical model (no HF fallback)
├── medgemma_enhanced.py          # MedGemma with attention extraction
├── run_medical_vlm_analysis.py   # Main analysis pipeline
├── requirements.txt              # Python dependencies
├── notebooks/
│   └── colab_complete_setup.ipynb # Google Colab setup notebook
└── archive/                      # Reference implementations
```

## Installation

### Google Colab (Recommended)

Use the provided notebook: `notebooks/colab_complete_setup.ipynb`

### Manual Setup

```bash
# Clone repository
git clone https://github.com/thedatasense/medical-vlm-intepret.git
cd medical-vlm-intepret/attention_viz

# Install dependencies
pip install -r requirements.txt

# Clone and install LLaVA-Rad
git clone https://github.com/microsoft/LLaVA-Rad.git
cd LLaVA-Rad && pip install -e .
```

## Usage

### Quick Start

```python
from llava_rad_medical_only import LLaVARadMedical, MedicalAttentionConfig
from medgemma_enhanced import load_medgemma

# Configure and load models
config = MedicalAttentionConfig(
    colormap='hot',
    attention_head_mode='mean'
)

llava_medical = LLaVARadMedical(config=config)
llava_medical.load_model()

medgemma_model, medgemma_processor = load_medgemma()
```

### Run Analysis

```bash
python run_medical_vlm_analysis.py --n_studies 100 --output_dir results
```

## Key Features

- **Medical-Specific**: Uses Microsoft's LLaVA-Rad medical model, not generic LLaVA
- **Query-to-Visual Attention**: Correct attention extraction pattern
- **Statistical Robustness**: Cluster bootstrap for proper confidence intervals
- **Production Quality**: No synthetic data, proper error handling

## Models

- **LLaVA-Rad**: `microsoft/llava-med-v1.5-mistral-7b` (medical-specific)
- **MedGemma**: `google/medgemma-4b-it` (medical-focused)

## Data Format

CSV files with columns:
- `study_id`: Unique study identifier
- `image_path`: Path to chest X-ray
- `finding`: Medical condition
- `variant_id`: Paraphrase variant (0-5)
- `question`: Medical question
- `answer_gt`: Ground truth (yes/no)