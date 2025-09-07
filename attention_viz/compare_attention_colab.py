#!/usr/bin/env python3
"""
Colab-friendly comparison pipeline for MedGemma3-4B and LLaVA-Rad

- Reuses enhanced utilities from this repo (model loading, attention extraction, visualization)
- Loads both models (assume available via HF/LLaVA libraries)
- Preprocesses input chest X-ray
- Runs inference with a hard positive diagnostic prompt
- Extracts attention maps for both models
- Visualizes inline and optionally saves figures

Usage (in Colab / Jupyter):

  # 1) Optional: environment prep (paths + imports)
  from compare_attention_colab import *
  IN_COLAB, paths = setup_colab_environment()

  # 2) Update image_path and prompt as needed
  image_path = f"{paths['image_root']}/10000032.jpg"
  prompt = "Is there right lower lobe consolidation suggestive of pneumonia?"

  # 3) Run comparison
  results = compare_models_on_input(image_path=image_path,
                                   prompt=prompt,
                                   save_outputs=True,
                                   output_dir=paths['output_dir'])

  # 4) Access results dict for answers, attention maps, and paths
  results.keys()

Notes:
- For LLaVA-Rad, EnhancedLLaVARadVisualizer handles the special loading logic.
- For MedGemma3 4B, set the model id below to your available checkpoint.
- If network access or packages are missing, install in Colab per AGENTS.md.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import csv
import random

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Local enhanced utilities - direct imports
try:
    from llava_rad_enhanced import EnhancedLLaVARadVisualizer, AttentionConfig, AttentionMetrics
    from medgemma_enhanced import EnhancedAttentionExtractor, AttentionExtractionConfig
    # Optional colab helper
    try:
        from colab_imports import setup_colab_environment
    except ImportError:
        def setup_colab_environment():
            """Fallback for non-colab environments"""
            import os
            return os.getcwd().endswith('content'), {
                'image_root': '/content/drive/MyDrive/Robust_Medical_LLM_Dataset/MIMIC_JPG/hundred_vqa',
                'output_dir': 'outputs_enhanced'
            }
except ImportError:
    # If direct imports fail, try with sys.path manipulation
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from llava_rad_enhanced import EnhancedLLaVARadVisualizer, AttentionConfig, AttentionMetrics
    from medgemma_enhanced import EnhancedAttentionExtractor, AttentionExtractionConfig
    try:
        from colab_imports import setup_colab_environment
    except ImportError:
        def setup_colab_environment():
            """Fallback for non-colab environments"""
            return False, {
                'image_root': 'images',
                'output_dir': 'outputs_enhanced'
            }

try:
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
except Exception:
    # Delay import failures to runtime to keep module importable in non-Colab contexts
    torch = None
    AutoProcessor = None
    AutoModelForCausalLM = None
    BitsAndBytesConfig = None


# ----------------------------
# Helpers
# ----------------------------

def _ensure_pil_image(image: str | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def _overlay_attention_on_image(image: Image.Image, attn: np.ndarray, alpha: float = 0.5,
                                cmap: str = "jet") -> Image.Image:
    # Convert to RGB array
    img = np.array(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    H, W = img.shape[:2]

    # Normalize and resize attention
    attn = attn.astype(np.float32)
    attn = attn / (attn.max() + 1e-8)
    attn_resized = cv2.resize(attn, (W, H), interpolation=cv2.INTER_CUBIC)

    # Colorize
    heat = (plt.get_cmap(cmap)(attn_resized)[..., :3] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img, 1.0, heat, alpha, 0)
    return Image.fromarray(overlay)


def _plot_side_by_side(image: Image.Image,
                       attn_llava: np.ndarray | List[np.ndarray],
                       attn_med: np.ndarray,
                       answers: Dict[str, str],
                       save_path: Optional[str] = None) -> None:
    # Handle multi-head from LLaVA if provided
    if isinstance(attn_llava, list) and len(attn_llava) > 0:
        # Aggregate heads by mean for top-level comparison
        attn_llava_agg = np.mean(np.stack(attn_llava, axis=0), axis=0)
    else:
        attn_llava_agg = attn_llava

    ov_llava = _overlay_attention_on_image(image, attn_llava_agg, alpha=0.45, cmap="jet")
    ov_med = _overlay_attention_on_image(image, attn_med, alpha=0.45, cmap="jet")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(ov_llava)
    axes[1].set_title(f"LLaVA-Rad\n{answers.get('llava', '')[:64]}")
    axes[1].axis("off")

    axes[2].imshow(ov_med)
    axes[2].set_title(f"MedGemma3-4B\n{answers.get('medgemma', '')[:64]}")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ----------------------------
# Model loading
# ----------------------------

@dataclass
class MedGemmaHandles:
    model: Any
    processor: Any
    device: str


def load_medgemma_model(model_id: str,
                        load_in_8bit: bool = True,
                        load_in_4bit: bool = False) -> MedGemmaHandles:
    if torch is None:
        raise ImportError("PyTorch/transformers not available. Install in Colab first.")

    bnb = None
    if load_in_8bit or load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb,
        trust_remote_code=True,
    )
    device = next(model.parameters()).device
    model.eval()
    return MedGemmaHandles(model=model, processor=processor, device=str(device))


# ----------------------------
# Pipeline
# ----------------------------

def compare_models_on_input(
    image_path: str,
    prompt: str,
    medgemma_model_id: str = "google/medgemma-4b-it",  # Medical Gemma model
    target_words: Optional[List[str]] = None,
    llava_load_in_8bit: bool = True,
    medgemma_load_in_8bit: bool = True,
    save_outputs: bool = False,
    output_dir: str = "outputs_enhanced",
    multi_head_mode: str = "entropy_weighted",
    # CSV sampling options
    from_csv: bool = False,
    csv_path: Optional[str] = None,
    image_root: Optional[str] = None,
    csv_row_index: Optional[int] = None,
    csv_strategy_filter: Optional[str] = None,
    # Multi-head grid visualization
    show_multi_head_grid: bool = True,
    multi_head_grid_filename: str = "llava_multi_head_grid.png",
) -> Dict[str, Any]:
    """
    Runs the end-to-end comparison on a single image and prompt.

    Returns a dict with answers, attention maps, metrics, and optional file paths.
    """
    # 1) Inputs (optionally sample from CSV)
    if from_csv and csv_path and os.path.exists(csv_path):
        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if csv_strategy_filter:
                rows = [r for r in rows if r.get("question_variant", "").startswith(csv_strategy_filter)]
            if not rows:
                raise RuntimeError("CSV is empty or filter removed all rows")
            idx = csv_row_index if (csv_row_index is not None and 0 <= csv_row_index < len(rows)) else random.randrange(len(rows))
            row = rows[idx]
            # Columns expected: image_path, question
            img_rel = row.get("image_path") or row.get("image") or row.get("image_file")
            q = row.get("question") or row.get("baseline_question")
            if not img_rel or not q:
                raise RuntimeError("CSV missing required columns: image_path/question")
            if image_root:
                image_path = os.path.join(image_root, img_rel)
            else:
                image_path = img_rel
            prompt = q
            print(f"Sampled CSV row {idx}: image={os.path.basename(image_path)}, prompt='{prompt[:64]}'")
        except Exception as e:
            print(f"CSV sampling failed, falling back to provided inputs: {e}")

    image = _ensure_pil_image(image_path)

    # 2) Load LLaVA-Rad via enhanced visualizer
    llava_config = AttentionConfig(use_medical_colormap=True, multi_head_mode=multi_head_mode)
    llava_vis = EnhancedLLaVARadVisualizer(config=llava_config)
    llava_vis.load_model(load_in_8bit=llava_load_in_8bit)

    # 3) Load MedGemma3-4B
    med_handles = load_medgemma_model(medgemma_model_id, load_in_8bit=medgemma_load_in_8bit)

    # 4) Prepare hard-positive prompt and target words for attention
    #    If no target words given, derive simple keywords from prompt
    if not target_words:
        # Heuristic: keep alphanumeric words longer than 3 chars (medical terms)
        toks = [t.strip(",.?:;!()[]{}\"\'") for t in prompt.split()]
        target_words = [t for t in toks if len(t) > 3]
        if not target_words:
            target_words = ["pneumonia", "consolidation", "opacity"]

    # 5) LLaVA-Rad inference + attention
    llava_out = llava_vis.generate_with_attention(image, prompt, max_new_tokens=128)
    attn_llava = llava_out.get("visual_attention")
    ans_llava = llava_out.get("answer", "")

    # 6) MedGemma inference
    extractor_cfg = AttentionExtractionConfig(
        attention_head_reduction="entropy_weighted",
        fallback_chain=["cross_attention", "gradcam", "uniform"],
        debug_mode=False,
    )
    extractor = EnhancedAttentionExtractor(extractor_cfg)

    # Build prompt with chat template (ensures image token is present)
    if hasattr(med_handles.processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        prompt_text = med_handles.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = med_handles.processor(
            text=prompt_text, images=image, return_tensors="pt"
        )
    else:
        inputs = med_handles.processor(
            text=f"<image>{prompt}", images=image, return_tensors="pt"
        )
    inputs = {k: v.to(med_handles.device) for k, v in inputs.items()}

    with torch.no_grad():
        med_gen = med_handles.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True,
        )

    # Decode MedGemma answer
    try:
        ans_med = med_handles.processor.tokenizer.decode(
            med_gen.sequences[0], skip_special_tokens=True
        )
        # Handle Gemma-3 format
        if '<start_of_turn>model' in ans_med:
            parts = ans_med.split('<start_of_turn>model')
            if len(parts) > 1:
                ans_med = parts[-1].split('<end_of_turn>')[0]
        elif '<start_of_turn>assistant' in ans_med:
            parts = ans_med.split('<start_of_turn>assistant')
            if len(parts) > 1:
                ans_med = parts[-1].split('<end_of_turn>')[0]
        else:
            # Fallback
            ans_med = ans_med.split("Assistant:")[-1]
            ans_med = ans_med.split("model\n")[-1]
        
        ans_med = ans_med.strip()
    except Exception:
        ans_med = ""

    # 7) MedGemma attention extraction (robust with fallbacks)
    attn_med, token_idxs, method_used = extractor.extract_token_conditioned_attention_robust(
        med_handles.model, med_handles.processor, med_gen, target_words, image, prompt
    )

    # 8) Metrics
    # Normalize to non-negative and sum-1 for metric stability
    def _safe_norm(a: np.ndarray) -> np.ndarray:
        a = np.maximum(a, 0)
        s = a.sum() + 1e-9
        return a / s

    attn_llava_norm = (
        [_safe_norm(h) for h in attn_llava] if isinstance(attn_llava, list) else _safe_norm(attn_llava)
    )
    attn_med_norm = _safe_norm(attn_med)

    # Compute focus/sparsity for the main comparison map(s)
    if isinstance(attn_llava_norm, list):
        # Aggregate for metrics
        llava_agg = np.mean(np.stack(attn_llava_norm, axis=0), axis=0)
    else:
        llava_agg = attn_llava_norm

    metrics = {
        "llava": {
            **AttentionMetrics.calculate_focus_score(llava_agg),
            "sparsity": AttentionMetrics.calculate_sparsity(llava_agg),
        },
        "medgemma": {
            **AttentionMetrics.calculate_focus_score(attn_med_norm),
            "sparsity": AttentionMetrics.calculate_sparsity(attn_med_norm),
        },
    }

    # 9) Visualization inline and optional saving
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "comparison_llava_medgemma.png") if save_outputs else None
    _plot_side_by_side(image, attn_llava_norm, attn_med_norm, {"llava": ans_llava, "medgemma": ans_med}, save_path=fig_path)

    # Optionally save raw overlays
    if save_outputs:
        ov_llava_path = os.path.join(output_dir, "overlay_llava.png")
        ov_med_path = os.path.join(output_dir, "overlay_medgemma.png")

        if isinstance(attn_llava_norm, list):
            ov_llava_img = _overlay_attention_on_image(image, np.mean(np.stack(attn_llava_norm, axis=0), axis=0))
        else:
            ov_llava_img = _overlay_attention_on_image(image, attn_llava_norm)
        ov_med_img = _overlay_attention_on_image(image, attn_med_norm)

        ov_llava_img.save(ov_llava_path)
        ov_med_img.save(ov_med_path)

    # Optional multi-head grid visualization for LLaVA
    multi_head_grid_path = None
    if show_multi_head_grid and isinstance(attn_llava, list) and len(attn_llava) > 1:
        try:
            # Reuse visualizer's helper for consistent styling
            multi_head_grid_path = os.path.join(output_dir, multi_head_grid_filename) if save_outputs else None
            llava_vis.create_multi_head_visualization(attn_llava, image, save_path=multi_head_grid_path)
            plt.show()
        except Exception as e:
            print(f"Multi-head grid visualization failed: {e}")

    return {
        "answers": {"llava": ans_llava, "medgemma": ans_med},
        "attention_maps": {"llava": attn_llava_norm, "medgemma": attn_med_norm},
        "medgemma": {"token_indices": token_idxs, "method": method_used},
        "metrics": metrics,
        "paths": {"figure": fig_path, "llava_multi_head_grid": multi_head_grid_path} if save_outputs else {},
    }


if __name__ == "__main__":
    # Minimal smoke run placeholder; in practice, use from a notebook.
    IN_COLAB, paths = setup_colab_environment()
    sample_image = os.path.join(paths["image_root"], "10000032.jpg")
    sample_prompt = "Is there right lower lobe consolidation suggestive of pneumonia?"
    print("Running example comparison...")
    try:
        out = compare_models_on_input(
            image_path=sample_image,
            prompt=sample_prompt,
            save_outputs=True,
            output_dir=paths["output_dir"],
        )
        print({k: list(v.keys()) if isinstance(v, dict) else type(v).__name__ for k, v in out.items()})
    except Exception as e:
        print(f"Example run failed (expected if models not available locally): {e}")
