# Medical VLM Interpretation - Essential Files

This repository contains code for comparing attention mechanisms between LLaVA-Rad (Microsoft's medical VLM) and MedGemma (Google's medical VLM).

## Essential Files

### Core Implementation
- `llava_rad_medical_only.py` - LLaVA-Rad medical model handler (no HF fallback)
- `medgemma_enhanced.py` - MedGemma model with attention extraction
- `run_medical_vlm_analysis.py` - Main analysis script

### Colab Setup
- `colab_llava_rad_medical.py` - Complete Colab notebook cells for setup and testing
- `fix_llava_conflict.py` - Utility to fix transformers conflict

### Documentation
- `CLAUDE.md` - Project documentation and fixes
- `requirements.txt` - Python dependencies

### Data Samples
- `medical-cxr-vqa-questions_sample.csv` - Sample medical questions
- `medical-cxr-vqa-questions_sample_hardpositives.csv` - Hard positive samples

## Quick Start (Google Colab)

1. Run the cells from `colab_llava_rad_medical.py` in order
2. This will:
   - Fix the transformers conflict
   - Clone Microsoft's LLaVA-Rad repository
   - Load both medical models
   - Run comparison analysis

## Key Features

- Query-to-visual attention extraction (not visual-to-visual)
- Multi-head attention aggregation with entropy weighting
- JS divergence for consistency analysis
- Cluster bootstrap for statistical analysis
- Professional medical imaging focus (no synthetic data)

## Models

- **LLaVA-Rad**: Microsoft's medical VLM (`microsoft/llava-med-v1.5-mistral-7b`)
- **MedGemma**: Google's medical VLM (`google/medgemma-4b-it`)

Both models are specifically trained for medical image interpretation.