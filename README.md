Medical Vision–Language Model Interpretation and Robustness Analysis

This repository contains a practical, research‑grade toolkit to interrogate and compare visual attention and answer robustness of state‑of‑the‑art medical vision–language models (VLMs) on chest X‑ray question answering. It focuses on two representative models with different training recipes and interfaces:

- LLaVA‑Rad (via the LLaVA library or HF fallback)
- MedGemma 3‑4B (instruction‑tuned)

The code provides consistent attention extraction across both models, quantitative attention quality metrics, and a reproducible robustness study over paraphrased question variants. The implementation is optimized for Colab usage but is modular enough to run locally with the appropriate dependencies and GPU.

## Contributions

- Unified attention extraction for heterogeneous VLM stacks:
  - Query‑to‑visual token attention for LLaVA‑style architectures with robust visual token localization.
  - Token‑conditioned cross‑attention extraction for MedGemma, with fallbacks to Grad‑CAM and uniform attention.
- Head aggregation with entropy‑weighted fusion and per‑layer aggregation over the final quarter of transformer layers.
- Attention quality metrics tailored for medical imaging:
  - Focus (entropy‑based), Sparsity (Gini), ROI support, and inter‑prompt JS divergence.
- End‑to‑end robustness pipeline over paraphrase variants with accuracy, flip‑rate, consistency, latency, and attention divergence.
- Turn‑key Colab workflows for environment setup, data mounting, model loading, visualization, and report generation.

## Repository Structure

- `attention_viz/`
  - `llava_rad_enhanced.py`: Enhanced LLaVA‑Rad visualizer and attention extractor.
  - `medgemma_enhanced.py`: Enhanced MedGemma attention extractor with fallbacks.
  - `compare_attention_colab.py`: Single‑case, side‑by‑side comparison pipeline (answers + overlays).
  - `run_medical_vlm_analysis_colab.py`: Full robustness study pipeline (batch evaluation + report/plots).
  - `colab_imports.py`: Convenience imports and Colab path setup helpers.
  - `medgemma_launch_mimic_fixed.py`: Compatibility alias for legacy notebook imports.
  - `medical-cxr-vqa-questions_sample*.csv`: Example question files with paraphrase variants.
  - `archive/`: Prior versions and experimental artifacts.
- `pyproject.toml`: Minimal project metadata (runtime deps are installed at runtime in Colab).

## Methods

### Attention extraction (LLaVA‑Rad)
- Visual token localization: image anchor token detection with fallbacks; expected grid size `(image_size / patch_size)^2` (default 336/14 → 24×24).
- Query‑to‑visual attention: extract attention from the final generated/query token to visual token indices, not visual‑to‑visual.
- Layer aggregation: aggregate over the last quarter of layers (empirically more task‑specific) and optionally over heads with:
  - mean, max, or entropy‑weighted head fusion (heads with lower entropy receive higher weight).
- Outputs normalized to a probability map over the visual grid; percentiles used for visualization clipping.

Reference implementation: `EnhancedLLaVARadVisualizer.extract_attention_robust` and `extract_visual_attention_multihead` in `attention_viz/llava_rad_enhanced.py`.

### Attention extraction (MedGemma)
- Primary: token‑conditioned cross‑attention. Identify target text tokens (robust matching across tokenizations) and aggregate attention from those tokens to visual tokens across the last quarter of layers.
- Fallbacks: Grad‑CAM over the vision backbone (with hooks on late encoder blocks) when cross‑attention is unavailable; final fallback is a uniform prior.
- Aggregations: per‑head mean/max/entropy‑weighted; per‑token mean/max/weighted; per‑layer mean over last quarter.

Reference implementation: `EnhancedAttentionExtractor.extract_token_conditioned_attention_robust` in `attention_viz/medgemma_enhanced.py`.

### Attention quality metrics
- Focus (entropy‑based): let `A` be the normalized attention over `N` patches. `H(A) = -∑ A log A`, `H_max = log N`, focus = `1 − H(A)/H_max` (higher is more concentrated).
- Sparsity (Gini): Gini coefficient over flattened, normalized attention, quantifying inequality (higher indicates sparse, peaky attentions).
- Consistency: mean pairwise `1 − JS(A_i, A_j)` across prompts; also compute JS distribution between models per sample.

Reference implementation: `AttentionMetrics` in `attention_viz/llava_rad_enhanced.py` and usage throughout.

### Robustness study design
- Data: chest X‑ray JPEGs (e.g., MIMIC‑CXR JPG subset) with per‑study paraphrase variants of the same clinical query (e.g., pneumonia). Example CSVs show the required structure (`study_id`, `image_path`, `question`, `answer`, `question_variant`, etc.).
- Protocol per study:
  - For each variant, run both models on the same image.
  - Record answers, correctness (string‑prefix match to GT for yes/no), latency, attention maps, and attention metrics.
  - Compute per‑study flip rate (fraction of variants that deviate from the base answer), full‑consistency rate, and inter‑model JS divergence.
- Outputs: JSONL of per‑sample results, aggregate stats JSON, plots (accuracy bars, focus/sparsity histograms, JS divergence histogram), and a Markdown report.

Reference implementation: `run_medical_vlm_analysis_colab.py` (see `InferenceResult`, `run_robustness_study`, `analyze_results`, `visualize_results`, `generate_report`).

## Quick Start (Colab)

The pipelines are designed for Colab GPUs where model weights can be installed on demand.

1) Upload `attention_viz/run_medical_vlm_analysis_colab.py` to Colab and run:

    !python run_medical_vlm_analysis_colab.py --n_studies 25 --output_dir robustness_output

What it does:
- Clones this repo into `/content/medical-vlm-intepret` if missing.
- Installs runtime packages: `torch`, `torchvision`, `transformers>=4.36.0`, `opencv-python`, `scipy`, `matplotlib`, `pillow`, `bitsandbytes`, `accelerate`, `gradio`.
- Installs LLaVA library and adds it to the Python path.
- Mounts Google Drive and verifies the following paths:
  - `data_root`: `.../Robust_Medical_LLM_Dataset`
  - `image_root`: `.../MIMIC_JPG/hundred_vqa`
  - `csv_path`: `.../attention_viz/medical-cxr-vqa-questions_sample.csv`
  - `csv_variants_path`: `.../attention_viz/medical-cxr-vqa-questions_sample_hardpositives.csv`
- Loads LLaVA‑Rad (8‑bit by default) and MedGemma (8‑bit by default).
- Runs the robustness study and writes outputs into `robustness_output/`:
  - `robustness_results.jsonl`, `analysis_results.json`, `visualizations/*.png`, `robustness_report.md`.

2) For a single‑case qualitative comparison in Colab/Jupyter, use:

```python
from attention_viz.compare_attention_colab import setup_colab_environment, compare_models_on_input
IN_COLAB, paths = setup_colab_environment()

image_path = f"{paths['image_root']}/10000032.jpg"
prompt = "Is there right lower lobe consolidation suggestive of pneumonia?"

out = compare_models_on_input(
    image_path=image_path,
    prompt=prompt,
    save_outputs=True,
    output_dir=paths['output_dir']
)
```

This produces a side‑by‑side figure (input / LLaVA‑Rad overlay / MedGemma overlay) and optional multi‑head grids for LLaVA.

## Local Setup (advanced)

- Python ≥ 3.9 recommended (code targets PyTorch + Transformers; `pyproject.toml` is minimal). For local runs replicate the Colab installs:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.36" opencv-python scipy matplotlib pillow bitsandbytes accelerate gradio
# Optional: LLaVA library (for native LLaVA‑Rad loading)
git clone https://github.com/haotian-liu/LLaVA.git
pip install -e LLaVA
```

- GPU: ≥ 15 GB free VRAM recommended. Both pipelines support 8‑bit quantization via `bitsandbytes`; set `load_in_8bit=True` (default) to fit consumer GPUs.
- Models: the LLaVA‑Rad HF card is not a standard HF model; the loader falls back to `llava-hf/llava-1.5-7b-hf` if LLaVA library loading fails. MedGemma model id defaults to `google/med-gemma-3-4b-it` but can be swapped.

## Data Expectations

The CSVs are examples; you must provide the actual images on disk and ensure the CSV paths point to them. Minimal required columns:
- `study_id`: groups variants of the same case.
- `image_path`: relative path from `image_root` or absolute path.
- `question`: the natural‑language query.
- `answer`: short string answer for correctness (e.g., yes/no, present/absent).
- Optional: `question_variant` or `variant_id` for the paraphrase method.

## Reproducing the Robustness Study

- Entrypoint: `attention_viz/run_medical_vlm_analysis_colab.py`
- Key arguments:
  - `--n_studies`: number of unique `study_id`s to process.
  - `--output_dir`: output directory for results and figures.
  - `--skip_setup`: skip environment bootstrap when you manage packages manually.
- Outputs:
  - JSONL of per‑sample `InferenceResult` including answers, correctness, latency, attention metrics, and JS divergence.
  - Aggregate statistics and histograms saved under `visualizations/`.
  - A Markdown executive report aggregating findings and sample disagreements.

## Interpreting Metrics

- Accuracy: fraction of yes/no answers whose prefix matches ground truth; conservative by design for noisy LLM decoding.
- Focus vs Sparsity: high focus and high sparsity indicate compact, peaked attention; pathological extremes (single‑patch spikes) are discouraged via percentile clipping in visualization.
- JS divergence: distributional dissimilarity between two attention maps; used for within‑model consistency across variants and between‑model comparisons on the same item.
- Flip rate: fraction of paraphrases whose answer differs from the base prompt’s answer for the same image; lower is better.
- Full consistency: fraction of studies with identical answers across all paraphrases; higher is better.


## Troubleshooting

- LLaVA‑Rad loading fails via HF: the script automatically falls back to the native LLaVA library or to `llava-hf/llava-1.5-7b-hf` when appropriate.
- CUDA OOM: enable 8‑bit loading (`load_in_8bit=True`), reduce `max_new_tokens`, or switch to smaller backbones.
- No attentions in generation outputs: the extractors transparently fall back to single‑forward attentions, Grad‑CAM, or uniform priors; see logs for the method used.
- Missing images/paths: verify Drive mount in Colab and the `image_root`/CSV paths printed during setup.

## Citation

If you use this code or ideas in your research, please cite this repository:

```
@software{medical_vlm_interpret_2025,
  title        = {Medical VLM Interpretation and Robustness Analysis},
  author       = {Binesh Kumar},
  year         = {2025},
  url          = {https://github.com/thedatasense/medical-vlm-intepret},
}
```

You may also wish to cite upstream model works (LLaVA, MedGemma) and any datasets you evaluate on (e.g., MIMIC‑CXR).

## License

This repository includes code intended for research and educational purposes. Check the licenses of upstream models and datasets before redistribution or deployment.
