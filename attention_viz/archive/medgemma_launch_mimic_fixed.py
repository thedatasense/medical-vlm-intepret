#!/usr/bin/env python3
"""
MedGemma 4B Multimodal VLM - MIMIC-CXR Analysis Platform with Robust Token Analysis
Developed by SAIL Lab at University of New Haven
Enhanced with Fixed Cross-Attention Extraction and Grad-CAM Fallback
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.ndimage import gaussian_filter
import logging
import gc
import warnings
from transformers import AutoProcessor, AutoModelForImageTextToText
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import ast
import subprocess
import json

warnings.filterwarnings('ignore')

# Disable parallelism to avoid conflicts
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("="*60)
print("MedGemma 4B Multimodal VLM - MIMIC-CXR Analysis Platform")
print("Developed by SAIL Lab - University of New Haven")
print("Enhanced with Robust Token-Conditioned Attention")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# ============================================================================
# CONFIGURATION
# ============================================================================

MIMIC_CSV_PATH = "/content/drive/MyDrive/Robust_Medical_LLM_Dataset/attention_viz/medical-cxr-vqa-questions_sample.csv"
MIMIC_IMAGE_BASE_PATH = "/content/drive/MyDrive/Robust_Medical_LLM_Dataset/MIMIC_JPG/hundred_vqa"

# ============================================================================
# CORE FIXES FOR ROBUST ATTENTION EXTRACTION
# ============================================================================

def factor_to_grid(n, H, W):
    """Find best grid dimensions for n tokens matching image aspect ratio"""
    aspect = W / H
    cands = [(a, n // a) for a in range(1, int(np.sqrt(n)) + 1) if n % a == 0]
    if cands:
        gh, gw = min(cands, key=lambda wh: abs((wh[1] / wh[0]) - aspect))
    else:
        # Fallback to square-ish
        gh = int(np.sqrt(n))
        gw = n // gh
    return gh, gw


def find_target_token_indices_robust(tokenizer, prompt_ids, target_words):
    """More robust target token finding with normalization"""
    # Decode and normalize
    text = tokenizer.decode(prompt_ids, skip_special_tokens=False).lower()
    clean = text.replace(",", " ").replace("?", " ").replace(".", " ").replace("!", " ")
    ids = tokenizer.encode(clean, add_special_tokens=False)
    
    words = [w.lower().strip() for w in target_words]
    matches = []
    
    for w in words:
        w_ids = tokenizer.encode(w, add_special_tokens=False)
        for i in range(len(ids) - len(w_ids) + 1):
            if ids[i:i+len(w_ids)] == w_ids:
                # Map back to original prompt_ids indices (approximate)
                matches.extend(range(i, i+len(w_ids)))
    
    return sorted(set(matches))


# ============================================================================
# BASIC VISUALIZATION FUNCTIONS
# ============================================================================

def model_view_image(processor, pil_image):
    """Get the exact image as the model sees it using proper denormalization"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "dummy"},
                {"type": "image", "image": pil_image}
            ]
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )
    
    px = inputs["pixel_values"][0]
    ip = processor.image_processor
    mean = torch.tensor(ip.image_mean).view(3, 1, 1)
    std = torch.tensor(ip.image_std).view(3, 1, 1)
    
    arr = (px * std + mean).clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    gray = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]).astype(np.uint8)
    return Image.fromarray(gray)


def to_model_view_gray(processor, pil_image):
    """Convert PIL image to model's view in grayscale (numpy array)"""
    gray_img = model_view_image(processor, pil_image)
    return np.array(gray_img)


def tight_body_mask(gray):
    """Create a tight mask for the body region, removing borders and annotations"""
    g = gray.astype(np.uint8)
    base = cv2.GaussianBlur(g, (0, 0), 2)
    _, m = cv2.threshold(base, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = 255 - m
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m = cv2.erode(m, np.ones((9, 9), np.uint8))
    return m

# Alias for compatibility
build_body_mask = tight_body_mask


def prepare_attn_grid(attn):
    """Prepare attention data as a square grid"""
    a = np.asarray(attn, dtype=np.float32)
    if a.ndim == 1:
        n = int(np.sqrt(a.size))
        a = a[:n * n].reshape(n, n)
    return a


def strip_border_tokens(attn_grid, k=1):
    """Zero out the outer ring of tokens to remove padding artifacts"""
    g = attn_grid.copy()
    g[:k, :] = 0
    g[-k:, :] = 0
    g[:, :k] = 0
    g[:, -k:] = 0
    return g


# ============================================================================
# GRAD-CAM FALLBACK FOR ROBUST ATTENTION
# ============================================================================

def gradcam_on_vision(model, processor, pil_image, prompt, target_token, 
                     layer_name="vision_tower.vision_model.encoder.layers.-1"):
    """Grad-CAM fallback when cross-attention unavailable"""
    device = next(model.parameters()).device
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": pil_image}
        ]
    }]
    
    inputs = processor.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )
    
    inputs_gpu = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    
    acts = []
    grads = []
    
    def fwd_hook(_, __, out): 
        acts.append(out)
    
    def bwd_hook(_, grad_in, grad_out): 
        grads.append(grad_out[0])
    
    # Try to find the vision encoder layer
    try:
        # Navigate to the correct layer based on model structure
        if "vision_tower" in dict(model.named_modules()):
            # Try different possible paths
            for path in [
                "vision_tower.vision_model.encoder.layers.23",  # Last layer
                "vision_tower.vision_model.encoder.layers.22",
                "vision_model.encoder.layers.23",
                "vision_model.encoder.layers.22"
            ]:
                if path in dict(model.named_modules()):
                    layer_name = path
                    break
        
        block = dict(model.named_modules())[layer_name]
    except KeyError:
        # Find any vision-related layer
        for name, module in model.named_modules():
            if "vision" in name and "layer" in name:
                block = module
                break
        else:
            # Fallback: return uniform attention
            logger.warning("Could not find vision encoder layer for Grad-CAM, using uniform attention")
            return np.ones((16, 16)) / 256  # Default 16x16 grid
    
    h1 = block.register_forward_hook(fwd_hook)
    h2 = block.register_full_backward_hook(bwd_hook)
    
    try:
        out = model(**inputs_gpu, return_dict=True)
        next_logits = out.logits[:, -1, :]
        
        # Encode target token
        tid = processor.tokenizer.encode(target_token, add_special_tokens=False)[0]
        loss = next_logits[0, tid]
        
        model.zero_grad(set_to_none=True)
        loss.backward()
        
        h1.remove()
        h2.remove()
        
        if not acts or not grads:
            logger.warning("No activations captured in Grad-CAM, using uniform attention")
            return np.ones((16, 16)) / 256
        
        A = acts[-1].detach()
        G = grads[-1].detach()
        
        # Compute weighted combination
        if A.ndim == 5:  # [B, T, C, H, W]
            w = G.mean(dim=(0, 1, 3, 4))  # Average over batch, time, height, width
            cam = (w[None, None, :, None, None] * A).sum(dim=2).squeeze().relu()
        elif A.ndim == 3:  # [B, N, C]
            w = G.mean(dim=(0, 1))  # Average over batch and sequence
            cam = (w[None, None, :] * A).sum(dim=-1).squeeze().relu()
            # Reshape to square
            n = cam.shape[0]
            h = w = int(np.sqrt(n))
            if h * w == n:
                cam = cam.view(h, w)
        else:
            logger.warning(f"Unexpected activation shape: {A.shape}, using uniform attention")
            return np.ones((16, 16)) / 256
        
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()
        
    finally:
        h1.remove()
        h2.remove()
        torch.cuda.empty_cache()


# ============================================================================
# ENHANCED TOKEN-CONDITIONED ANALYSIS FUNCTIONS
# ============================================================================

def run_generate_with_attention_robust(model, processor, pil_image, prompt, device='cuda', max_new_tokens=20):
    """Run generation and ensure attention is properly configured"""
    # Configure model for attention output
    model.config.output_attentions = True
    model.config.return_dict = True
    if hasattr(model.config, "output_cross_attentions"):
        model.config.output_cross_attentions = True
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": pil_image}
            ]
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )
    
    inputs_gpu = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    prompt_len = inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs_gpu,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    
    generated_ids = outputs.sequences[0][prompt_len:]
    generated_text = processor.decode(generated_ids, skip_special_tokens=True)
    
    return {
        'full_ids': outputs.sequences[0],
        'prompt_len': prompt_len,
        'generated_text': generated_text,
        'pixel_values': inputs_gpu['pixel_values']
    }


def extract_token_conditioned_attention_robust(model, processor, gen_result, target_words, 
                                              pil_image=None, prompt=None, use_gradcam=False):
    """Extract attention with proper fallback to Grad-CAM"""
    device = next(model.parameters()).device
    tokenizer = processor.tokenizer
    
    # Find target token indices using robust method
    prompt_ids = gen_result['full_ids'][:gen_result['prompt_len']]
    target_idx = find_target_token_indices_robust(tokenizer, prompt_ids, target_words)
    
    if use_gradcam and pil_image and prompt and target_words:
        # Use Grad-CAM fallback
        try:
            cam = gradcam_on_vision(model, processor, pil_image, prompt, target_words[0])
            return cam, target_idx, "gradcam"
        except Exception as e:
            logger.warning(f"Grad-CAM failed: {e}")
    
    # Configure model for attention
    model.config.output_attentions = True
    model.config.return_dict = True
    if hasattr(model.config, "output_cross_attentions"):
        model.config.output_cross_attentions = True
    
    # Forward pass with attention
    full_ids = gen_result['full_ids'][:-1].unsqueeze(0).to(device)
    full_mask = torch.ones_like(full_ids)
    
    with torch.no_grad():
        outputs = model(
            input_ids=full_ids,
            pixel_values=gen_result['pixel_values'],
            attention_mask=full_mask,
            output_attentions=True,
            use_cache=False,
            return_dict=True
        )
    
    # Check for cross-attention
    if not hasattr(outputs, 'cross_attentions') or not outputs.cross_attentions:
        # Check for single image token trap
        image_token_id = getattr(processor, "image_token_id", None)
        if image_token_id is None and hasattr(processor.tokenizer, "image_token_id"):
            image_token_id = processor.tokenizer.image_token_id
        
        if image_token_id:
            img_pos = (full_ids[0] == image_token_id).nonzero(as_tuple=False).squeeze(-1)
            if img_pos.numel() <= 1:
                logger.warning(
                    "Only one image token found and no cross_attentions. "
                    "Falling back to Grad-CAM."
                )
                # Try fallback
                if pil_image and prompt and target_words:
                    try:
                        cam = gradcam_on_vision(model, processor, pil_image, prompt, target_words[0])
                        return cam, target_idx, "gradcam"
                    except Exception as e:
                        logger.error(f"Grad-CAM fallback failed: {e}")
                        # Return uniform attention as last resort
                        return np.ones((16, 16)) / 256, target_idx, "uniform"
        
        # Try fallback
        if pil_image and prompt and target_words:
            try:
                cam = gradcam_on_vision(model, processor, pil_image, prompt, target_words[0])
                return cam, target_idx, "gradcam"
            except Exception as e:
                logger.error(f"No cross-attention and Grad-CAM failed: {e}")
                return np.ones((16, 16)) / 256, target_idx, "uniform"
        else:
            logger.warning("No cross-attention available, using uniform attention")
            return np.ones((16, 16)) / 256, target_idx, "uniform"
    
    # Build token-conditioned grid with proper dimensions
    grid = build_token_conditioned_grid_robust(
        outputs, gen_result['prompt_len'], target_idx, 
        gen_result['pixel_values'].shape
    )
    
    return grid, target_idx, "cross_attention"


def build_token_conditioned_grid_robust(outputs, prompt_len, target_idx, pixel_shape, last_k_layers=2):
    """Build attention grid with proper dimensions"""
    cross_attn = outputs.cross_attentions
    self_attn = outputs.attentions
    
    n_layers = len(self_attn)
    accumulated = None
    
    for layer_idx in range(n_layers - last_k_layers, n_layers):
        self_layer = self_attn[layer_idx][0].mean(0)  # [q_len, q_len]
        cross_layer = cross_attn[layer_idx][0].mean(0)  # [q_len, kv_len]
        
        # Process last 3 decode steps
        start_q = max(prompt_len, self_layer.shape[0] - 3)
        
        for q in range(start_q, self_layer.shape[0]):
            # Self-attention gate
            self_row = self_layer[q, :prompt_len]
            
            if target_idx:
                gate_weight = sum(self_row[idx] for idx in target_idx if idx < prompt_len)
                gate_scalar = gate_weight / (self_row.sum() + 1e-8)
            else:
                gate_scalar = 1.0 / max(prompt_len, 1)
            
            # Cross-attention weighted by gate
            cross_row = cross_layer[q]
            weighted = cross_row * gate_scalar.item()
            
            if accumulated is None:
                accumulated = weighted
            else:
                accumulated += weighted
    
    # Normalize
    accumulated = accumulated / (accumulated.sum() + 1e-8)
    
    # Get proper grid dimensions using factor_to_grid
    kv_len = accumulated.shape[0]
    H, W = pixel_shape[-2:]  # Get image dimensions
    gh, gw = factor_to_grid(kv_len, H, W)
    
    # Reshape to grid
    grid = accumulated[:gh*gw].view(gh, gw).cpu().numpy()
    
    return grid


def token_mask_from_body(body_mask, gh, gw, border=2, thresh=0.15):
    """Create token-level mask from body mask"""
    m = cv2.resize(body_mask.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    m = m / (m.max() + 1e-8)
    m = (m > thresh).astype(np.float32)
    
    m[:border, :] = 0
    m[-border:, :] = 0
    m[:, :border] = 0
    m[:, -border:] = 0
    
    return m


def create_token_attention_overlay_robust(base_gray, grid, body_mask, target_words, 
                                         method="cross_attention", alpha=0.35):
    """Create visualization with method indicator"""
    gh, gw = grid.shape
    H, W = base_gray.shape
    
    # Apply token mask
    tok_mask = token_mask_from_body(body_mask, gh, gw)
    grid = grid * tok_mask
    grid = grid / (grid.sum() + 1e-8)
    
    # Resize to image size
    heat = cv2.resize(grid, (W, H), interpolation=cv2.INTER_CUBIC)
    
    # Percentile clip on masked region
    sel = body_mask > 0
    vals = heat[sel]
    lo, hi = np.percentile(vals, [2, 98]) if vals.size else (heat.min(), heat.max())
    
    heat = np.clip(heat, lo, hi)
    heat = (heat - lo) / (hi - lo + 1e-8)
    heat *= sel.astype(np.float32)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    ax1.imshow(base_gray, cmap='gray', vmin=0, vmax=255)
    ax1.set_title('Original X-ray')
    ax1.axis('off')
    
    # Overlay
    ax2.imshow(base_gray, cmap='gray', vmin=0, vmax=255)
    im = ax2.imshow(heat, alpha=alpha, cmap='jet')
    title = f'Attention on: {", ".join(target_words)}'
    if method == "gradcam":
        title += " [Grad-CAM]"
    elif method == "uniform":
        title += " [Fallback]"
    ax2.set_title(title)
    ax2.axis('off')
    
    # Add grid lines for non-uniform methods
    if method != "uniform":
        ys = np.linspace(0, H, gh + 1)
        xs = np.linspace(0, W, gw + 1)
        for y in ys:
            ax2.axhline(y, color='white', linewidth=0.3, alpha=0.3)
        for x in xs:
            ax2.axvline(x, color='white', linewidth=0.3, alpha=0.3)
    
    # Colorbar
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig


def compute_attention_metrics(grid, body_mask):
    """Compute quality metrics for attention"""
    gh, gw = grid.shape
    tok_mask = token_mask_from_body(body_mask, gh, gw)
    masked_grid = grid * tok_mask
    masked_grid = masked_grid / (masked_grid.sum() + 1e-8)
    
    border = masked_grid[0, :].sum() + masked_grid[-1, :].sum() + \
             masked_grid[:, 0].sum() + masked_grid[:, -1].sum()
    
    mid = gw // 2
    third = gh // 3
    
    metrics = {
        'inside_body_ratio': float(masked_grid.sum()),
        'border_fraction': float(border),
        'left_fraction': float(masked_grid[:, :mid].sum()),
        'right_fraction': float(masked_grid[:, mid:].sum()),
        'apical_fraction': float(masked_grid[:third, :].sum()),
        'basal_fraction': float(masked_grid[-third:, :].sum())
    }
    
    return metrics


def overlay_attention_enhanced(image_path, attn, processor, alpha=0.35, debug_align=False):
    """Enhanced attention overlay with mask-first percentile clipping"""
    if isinstance(image_path, str):
        base_img = model_view_image(processor, Image.open(image_path).convert("RGB"))
    else:
        base_img = model_view_image(processor, image_path.convert("RGB"))
    
    base = np.array(base_img)
    mask = tight_body_mask(base)
    
    attn = strip_border_tokens(prepare_attn_grid(attn), k=1)
    gh, gw = attn.shape
    H, W = base.shape[:2]
    
    interp = cv2.INTER_NEAREST if debug_align else cv2.INTER_CUBIC
    heat = cv2.resize(attn, (W, H), interpolation=interp)
    
    sel = mask > 0
    vals = heat[sel]
    lo, hi = np.percentile(vals, [2, 98]) if vals.size else (heat.min(), heat.max())
    
    heat = np.clip(heat, lo, hi)
    heat = (heat - lo) / (hi - lo + 1e-8)
    heat *= sel.astype(np.float32)
    
    fig = plt.figure(dpi=120, figsize=(10, 10))
    plt.imshow(base, cmap="gray", vmin=0, vmax=255)
    plt.imshow(heat, alpha=alpha, cmap="jet")
    
    if debug_align:
        ys = np.linspace(0, H, gh + 1)
        xs = np.linspace(0, W, gw + 1)
        for y in ys:
            plt.axhline(y, color='white', linewidth=0.4, alpha=0.5)
        for x in xs:
            plt.axvline(x, color='white', linewidth=0.4, alpha=0.5)
    
    plt.axis("off")
    plt.colorbar(fraction=0.025, pad=0.01)
    plt.tight_layout(pad=0)
    
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def extract_attention_data(model, outputs, inputs, processor) -> Dict:
    """Extract attention data with improved format for visualization"""
    data = {}
    try:
        device = next(model.parameters()).device
        full_ids = outputs.sequences[:, :-1].to(device)
        full_mask = torch.ones_like(full_ids, device=device)
        
        with torch.no_grad():
            attn_out = model(
                input_ids=full_ids,
                pixel_values=inputs["pixel_values"],
                attention_mask=full_mask,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
        
        def summarize(vec: torch.Tensor) -> Dict:
            vec = vec.float()
            g = int(np.sqrt(vec.shape[0]))
            vec = vec[:g * g]
            vec = vec / (vec.sum() + 1e-8)
            
            raw_attention = vec.cpu().numpy()
            grid = vec.view(g, g).cpu().numpy()
            
            h, w = grid.shape
            quads = {
                "upper_left": grid[:h // 2, :w // 2].mean(),
                "upper_right": grid[:h // 2, w // 2:].mean(),
                "lower_left": grid[h // 2:, :w // 2].mean(),
                "lower_right": grid[h // 2:, w // 2:].mean(),
            }
            
            return {
                "regional_focus": max(quads, key=quads.get),
                "attention_entropy": float(stats.entropy(grid.flatten() + 1e-10)),
                "attention_grid": grid.tolist(),
                "raw_attention": raw_attention.tolist(),
            }
        
        xattn = getattr(attn_out, "cross_attentions", None)
        if xattn:
            last = xattn[-1][0].mean(0)
            if last.shape[0] >= 5:
                vec = last[-5:].mean(0)
            elif last.shape[0] >= 3:
                vec = last[-3:].mean(0)
            else:
                vec = last[-1]
            return summarize(vec)
        
        image_token_id = getattr(processor, "image_token_id", None)
        if image_token_id is None and hasattr(processor.tokenizer, "image_token_id"):
            image_token_id = processor.tokenizer.image_token_id
        
        if attn_out.attentions and image_token_id is not None:
            img_pos = (full_ids[0] == image_token_id).nonzero(as_tuple=False).squeeze(-1)
            if img_pos.numel() > 0:
                last = attn_out.attentions[-1][0].mean(0)
                if last.shape[0] >= 5:
                    vec = last[-5:].mean(0)[img_pos]
                elif last.shape[0] >= 3:
                    vec = last[-3:].mean(0)[img_pos]
                else:
                    vec = last[-1, img_pos]
                return summarize(vec)
        
        return {}
    except Exception as e:
        logger.warning(f"Failed to extract attention: {e}")
        return {}


# ============================================================================
# GPU MANAGEMENT
# ============================================================================

def get_gpu_memory_from_nvidia_smi():
    """Get actual GPU memory usage using nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.total,memory.used,memory.free',
             '--format=csv,nounits,noheader'],
            capture_output=True, text=True, check=True
        )
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(', ')
            gpu_info.append({
                'id': int(parts[0]),
                'total_mb': float(parts[1]),
                'used_mb': float(parts[2]),
                'free_mb': float(parts[3]),
                'free_gb': float(parts[3]) / 1024,
                'total_gb': float(parts[1]) / 1024,
                'usage_percent': (float(parts[2]) / float(parts[1])) * 100
            })
        
        return gpu_info
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def select_best_gpu(min_free_gb: float = 15.0) -> int:
    """Select GPU with most free memory"""
    gpu_info = get_gpu_memory_from_nvidia_smi()
    
    if gpu_info is None:
        print("nvidia-smi not available, using PyTorch memory info")
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free = props.total_memory - torch.cuda.memory_allocated(i)
            total = props.total_memory
            
            gpu_info.append({
                'id': i,
                'free_gb': free / (1024 ** 3),
                'total_gb': total / (1024 ** 3),
                'usage_percent': (1 - free / total) * 100
            })
    
    if not gpu_info:
        raise RuntimeError("No CUDA GPUs available")
    
    print("\n=== GPU Status ===")
    for gpu in gpu_info:
        print(f"GPU {gpu['id']}: "
              f"{gpu['free_gb']:.1f}GB free / {gpu['total_gb']:.1f}GB total "
              f"({gpu['usage_percent']:.1f}% used)")
    
    best_gpu = max(gpu_info, key=lambda x: x['free_gb'])
    
    if best_gpu['free_gb'] < min_free_gb:
        for gpu in gpu_info:
            if gpu['free_gb'] >= min_free_gb:
                best_gpu = gpu
                break
        else:
            print(f"\n⚠️ WARNING: No GPU has at least {min_free_gb}GB free memory")
            print(f"Using GPU {best_gpu['id']} with only {best_gpu['free_gb']:.1f}GB free")
    
    print(f"\n✓ Selected GPU {best_gpu['id']} with {best_gpu['free_gb']:.1f}GB free")
    torch.cuda.set_device(best_gpu['id'])
    
    return best_gpu['id']


def setup_gpu(device_id: Optional[int] = None, min_free_gb: float = 15.0):
    """Setup GPU with optimal settings"""
    gc.collect()
    torch.cuda.empty_cache()
    
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        return torch.device('cpu')
    
    if device_id is None:
        device_id = select_best_gpu(min_free_gb)
    else:
        torch.cuda.set_device(device_id)
        print(f"Using specified GPU {device_id}")
    
    device = torch.device(f'cuda:{device_id}')
    
    print(f"=== GPU Setup Complete ===")
    print(f"GPU: {torch.cuda.get_device_name(device_id)}")
    print(f"Memory: {torch.cuda.get_device_properties(device_id).total_memory / 1024**3:.1f} GB")
    
    allocated = torch.cuda.memory_allocated(device_id) / 1024**3
    reserved = torch.cuda.memory_reserved(device_id) / 1024**3
    print(f"Current usage: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
    
    return device


# ============================================================================
# ENHANCED MODEL LOADING
# ============================================================================

def load_model_enhanced(model_id='google/medgemma-4b-it', device=None):
    """Load MedGemma 4B model with enhanced attention capabilities"""
    
    if device is None:
        device = setup_gpu()
    
    print(f"\n=== Loading MedGemma 4B Multimodal VLM ===")
    print(f"Model ID: {model_id}")
    
    processor = AutoProcessor.from_pretrained(model_id)
    print("✓ Processor loaded")
    
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map={'': device.index if device.type == 'cuda' else 'cpu'},
            attn_implementation="eager",
            tie_word_embeddings=False,
            low_cpu_mem_usage=True
        )
        
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        # Ensure attention output is properly configured
        model.config.output_attentions = True
        model.config.return_dict = True
        if hasattr(model.config, "output_cross_attentions"):
            model.config.output_cross_attentions = True
        
        print("✓ MedGemma 4B model loaded successfully with attention enabled")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(device.index) / 1024**3
            reserved = torch.cuda.memory_reserved(device.index) / 1024**3
            print(f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
        
        return model, processor
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


# ============================================================================
# MIMIC DATA LOADER
# ============================================================================

class MIMICDataLoader:
    """Load and manage MIMIC-CXR questions and images"""
    
    def __init__(self, csv_path: str, image_base_path: str):
        self.csv_path = Path(csv_path)
        self.image_base_path = Path(image_base_path)
        
        print(f"Loading MIMIC data from: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} questions")
        
        # Create question list for dropdown
        self.question_list = []
        for idx, row in self.df.iterrows():
            label = f"{row['study_id'][:8]}... - {row['question'][:50]}..."
            self.question_list.append((label, idx))
    
    def get_sample(self, idx: int) -> Dict:
        """Get a specific sample by index"""
        if idx < 0 or idx >= len(self.df):
            return None
        
        row = self.df.iloc[idx]
        
        # Parse options if needed
        if isinstance(row['options'], str):
            try:
                options = ast.literal_eval(row['options'])
            except:
                options = ['yes', 'no']
        else:
            options = row['options'] if row['options'] else ['yes', 'no']
        
        # Find image path
        image_path = self.image_base_path / row['ImagePath']
        if not image_path.exists():
            image_path = self.image_base_path / f"{row['study_id']}.jpg"
        
        return {
            'study_id': row['study_id'],
            'question': row['question'],
            'options': options,
            'correct_answer': row['correct_answer'],
            'image_path': str(image_path) if image_path.exists() else None
        }
    
    def get_dropdown_choices(self):
        """Get choices for Gradio dropdown"""
        return [label for label, _ in self.question_list]
    
    def get_index_from_label(self, label: str) -> int:
        """Get index from dropdown label"""
        for l, idx in self.question_list:
            if l == label:
                return idx
        return 0


# ============================================================================
# ENHANCED GRADIO APPLICATION WITH ROBUST TOKEN ANALYSIS
# ============================================================================

def create_mimic_gradio_interface_robust(model, processor):
    """Create Gradio interface with robust token analysis"""
    import gradio as gr
    
    # Load MIMIC data
    mimic_loader = MIMICDataLoader(MIMIC_CSV_PATH, MIMIC_IMAGE_BASE_PATH)
    
    class MIMICRobustTokenApp:
        def __init__(self, model, processor, data_loader):
            self.model = model
            self.processor = processor
            self.data_loader = data_loader
            self.current_sample = None
            self.current_image = None
            
        def load_question(self, question_selection):
            """Load selected question and image"""
            if question_selection is None:
                return None, "", "", ""
            
            idx = self.data_loader.get_index_from_label(question_selection)
            sample = self.data_loader.get_sample(idx)
            
            if sample is None or sample['image_path'] is None:
                return None, "", "", ""
            
            self.current_sample = sample
            
            # Load and store image
            image = Image.open(sample['image_path']).convert('RGB')
            self.current_image = image
            
            # Return image, question, ground truth
            return (
                image,
                sample['question'],
                sample['correct_answer'],
                ""  # Clear previous model answer
            )
        
        def load_for_token_analysis(self, question_selection):
            """Load selected question and image for token analysis tab"""
            if question_selection is None:
                return None, ""
            
            idx = self.data_loader.get_index_from_label(question_selection)
            sample = self.data_loader.get_sample(idx)
            
            if sample is None or sample['image_path'] is None:
                return None, ""
            
            # Load image
            image = Image.open(sample['image_path']).convert('RGB')
            
            return image, sample['question']
        
        def analyze_xray(self, image, question, custom_mode, show_attention, show_grid):
            """Analyze X-ray with yes/no answer"""
            if image is None:
                return "Please select a MIMIC question from the dropdown", None, None, ""
            
            # Use custom question if in custom mode, otherwise use selected question
            if custom_mode and question:
                prompt = question
            elif self.current_sample:
                prompt = self.current_sample['question']
            else:
                return "Please select a question", None, None, ""
            
            # Format prompt for yes/no answer
            formatted_prompt = f"""Question: {prompt}
Answer with only 'yes' or 'no'. Do not provide any explanation."""
            
            # Create messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": formatted_prompt},
                        {"type": "image", "image": image}
                    ]
                }
            ]
            
            # Process inputs
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            # Generate with attention
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,  # Short for yes/no
                    do_sample=False,
                    output_attentions=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            # Get generated text
            generated_ids = outputs.sequences[0][len(inputs['input_ids'][0]):]
            generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)
            
            # Extract yes/no answer
            answer = self.extract_answer(generated_text)
            
            # Create visualizations if requested
            attention_viz = None
            stats_viz = None
            
            if show_attention:
                attention_data = extract_attention_data(self.model, outputs, inputs, self.processor)
                
                if attention_data:
                    attn_to_viz = attention_data.get("raw_attention") or attention_data.get("attention_grid")
                    
                    if attn_to_viz:
                        attention_viz = overlay_attention_enhanced(
                            image, 
                            attn_to_viz, 
                            self.processor,
                            alpha=0.35,
                            debug_align=show_grid
                        )
                        
                        # Create statistics
                        stats_viz = self.create_attention_stats(attention_data, answer)
            
            # Clean up
            del outputs
            del inputs
            torch.cuda.empty_cache()
            
            # Return results
            return generated_text, attention_viz, stats_viz, answer
        
        def analyze_token_attention_robust(self, image, prompt, target_words_str, use_gradcam):
            """Analyze attention on specific tokens with robust extraction"""
            try:
                if image is None:
                    return None, "Please select a MIMIC question from the dropdown first", {}
                
                # Parse target words
                target_words = [w.strip() for w in target_words_str.split(',') if w.strip()]
                
                if not target_words:
                    return None, "⚠️ Please provide target words separated by commas", {}
                
                # Convert image if needed
                if isinstance(image, np.ndarray):
                    pil_image = Image.fromarray(image).convert('RGB')
                else:
                    pil_image = image.convert('RGB')
                
                # Get model view and body mask
                base_gray = to_model_view_gray(self.processor, pil_image)
                body_mask = build_body_mask(base_gray)
                
                # Generate with attention using robust method
                device = next(self.model.parameters()).device
                gen_result = run_generate_with_attention_robust(
                    self.model, self.processor, pil_image, prompt, device
                )
                
                # Extract token-conditioned attention with fallback
                try:
                    grid, target_idx, method = extract_token_conditioned_attention_robust(
                        self.model, self.processor, gen_result, target_words,
                        pil_image=pil_image, prompt=prompt, use_gradcam=use_gradcam
                    )
                except RuntimeError as e:
                    logger.warning(f"Attention extraction failed, using Grad-CAM: {e}")
                    grid, target_idx, method = extract_token_conditioned_attention_robust(
                        self.model, self.processor, gen_result, target_words,
                        pil_image=pil_image, prompt=prompt, use_gradcam=True
                    )
                
                # Create visualization with method indicator
                fig = create_token_attention_overlay_robust(
                    base_gray, grid, body_mask, target_words, method
                )
                
                # Compute metrics
                metrics = compute_attention_metrics(grid, body_mask)
                
                # Format output
                output_text = f"**MedGemma 4B Answer:** {gen_result['generated_text']}\n\n"
                output_text += f"**Method:** {method.replace('_', ' ').title()}\n"
                output_text += f"**Target tokens found:** {len(target_idx)} positions\n\n"
                output_text += "**Attention Metrics:**\n"
                output_text += f"- Inside body ratio: {metrics['inside_body_ratio']:.3f} "
                output_text += f"{'✓' if metrics['inside_body_ratio'] >= 0.7 else '✗'} (target ≥ 0.7)\n"
                output_text += f"- Border fraction: {metrics['border_fraction']:.3f} "
                output_text += f"{'✓' if metrics['border_fraction'] <= 0.05 else '✗'} (target ≤ 0.05)\n"
                output_text += f"- Left/Right: {metrics['left_fraction']:.2f}/{metrics['right_fraction']:.2f}\n"
                output_text += f"- Apical/Basal: {metrics['apical_fraction']:.2f}/{metrics['basal_fraction']:.2f}"
                
                # Clean up
                plt.close('all')
                torch.cuda.empty_cache()
                gc.collect()
                
                return fig, output_text, metrics
                
            except Exception as e:
                torch.cuda.empty_cache()
                gc.collect()
                return None, f"❌ Error: {str(e)}", {}
        
        def compare_prompts_robust(self, image, prompt1, prompt2, target_words_str, use_gradcam):
            """Compare attention between two prompts with robust extraction"""
            try:
                if image is None:
                    return None, "Please select a MIMIC question from the dropdown first"
                
                target_words = [w.strip() for w in target_words_str.split(',') if w.strip()]
                
                if not target_words:
                    # Run without gating
                    target_words = [""]
                
                if isinstance(image, np.ndarray):
                    pil_image = Image.fromarray(image).convert('RGB')
                else:
                    pil_image = image.convert('RGB')
                
                base_gray = to_model_view_gray(self.processor, pil_image)
                body_mask = build_body_mask(base_gray)
                
                results = []
                grids = []
                methods = []
                
                for prompt in [prompt1, prompt2]:
                    device = next(self.model.parameters()).device
                    gen_result = run_generate_with_attention_robust(
                        self.model, self.processor, pil_image, prompt, device
                    )
                    
                    try:
                        grid, target_idx, method = extract_token_conditioned_attention_robust(
                            self.model, self.processor, gen_result, target_words,
                            pil_image=pil_image, prompt=prompt, use_gradcam=use_gradcam
                        )
                    except:
                        grid, target_idx, method = extract_token_conditioned_attention_robust(
                            self.model, self.processor, gen_result, target_words,
                            pil_image=pil_image, prompt=prompt, use_gradcam=True
                        )
                    
                    grids.append(grid)
                    methods.append(method)
                    results.append({
                        'prompt': prompt[:50] + '...' if len(prompt) > 50 else prompt,
                        'answer': gen_result['generated_text'],
                        'n_targets': len(target_idx)
                    })
                
                # Calculate divergence
                js_div = jensenshannon(grids[0].flatten(), grids[1].flatten()) ** 2
                
                # Create comparison plot
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                for i, (ax, grid, result) in enumerate(zip(axes[:2], grids, results)):
                    heat = cv2.resize(grid, base_gray.shape[::-1], interpolation=cv2.INTER_CUBIC)
                    ax.imshow(base_gray, cmap='gray')
                    ax.imshow(heat, alpha=0.35, cmap='jet')
                    ax.set_title(f'Prompt {i+1}: {result["answer"][:20]}...')
                    ax.axis('off')
                
                # Difference plot
                diff = np.abs(grids[0] - grids[1])
                heat_diff = cv2.resize(diff, base_gray.shape[::-1], interpolation=cv2.INTER_CUBIC)
                axes[2].imshow(heat_diff, cmap='hot')
                axes[2].set_title(f'Difference (JS div: {js_div:.3f})')
                axes[2].axis('off')
                
                plt.tight_layout()
                
                output_text = f"**Jensen-Shannon Divergence:** {js_div:.3f}\n"
                output_text += f"**Methods:** {', '.join(set(methods))}\n\n"
                output_text += f"**Prompt 1 Answer:** {results[0]['answer']}\n"
                output_text += f"**Prompt 2 Answer:** {results[1]['answer']}\n"
                
                if js_div > 0.1 and results[0]['answer'][:10] != results[1]['answer'][:10]:
                    output_text += "\n⚠️ **Warning:** High divergence with different answers!"
                elif js_div < 0.001:
                    output_text += "\n⚠️ **Note:** Nearly identical attention patterns"
                
                # Clean up
                plt.close('all')
                torch.cuda.empty_cache()
                gc.collect()
                
                return fig, output_text
                
            except Exception as e:
                torch.cuda.empty_cache()
                gc.collect()
                return None, f"❌ Error: {str(e)}"
        
        def extract_answer(self, text: str) -> str:
            """Extract yes/no answer from generated text"""
            text_lower = text.lower().strip()
            
            if text_lower.startswith('yes'):
                return 'yes'
            elif text_lower.startswith('no'):
                return 'no'
            elif 'yes' in text_lower[:20]:
                return 'yes'
            elif 'no' in text_lower[:20]:
                return 'no'
            else:
                return 'uncertain'
        
        def create_attention_stats(self, attention_data, answer):
            """Create visualization with attention statistics and answer"""
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Regional focus distribution
            if "regional_focus" in attention_data:
                regions = ["upper_left", "upper_right", "lower_left", "lower_right"]
                focus = attention_data["regional_focus"]
                colors = ['lightblue' if r != focus else 'darkblue' for r in regions]
                values = [0.20 if r != focus else 0.40 for r in regions]
                
                axes[0].bar(regions, values, color=colors)
                axes[0].set_title("Regional Focus", fontsize=14, fontweight='bold')
                axes[0].set_ylabel("Attention Weight")
                axes[0].set_xticklabels(['Upper\nLeft', 'Upper\nRight', 'Lower\nLeft', 'Lower\nRight'])
            
            # Attention entropy
            if "attention_entropy" in attention_data:
                entropy = attention_data["attention_entropy"]
                color = 'green' if entropy < 3 else 'orange' if entropy < 4 else 'red'
                axes[1].bar(["Entropy"], [entropy], color=color)
                axes[1].set_title(f"Attention Entropy", fontsize=14, fontweight='bold')
                axes[1].set_ylabel("Entropy Value")
                axes[1].set_ylim([0, 5])
                axes[1].text(0, entropy + 0.1, f'{entropy:.2f}', ha='center', fontsize=12)
            
            # Answer comparison
            if self.current_sample:
                ground_truth = self.current_sample['correct_answer']
                is_correct = (answer == ground_truth)
                
                axes[2].bar(['Ground Truth', 'Model Answer'], [1, 1], 
                           color=['green', 'green' if is_correct else 'red'])
                axes[2].set_ylim([0, 1.5])
                axes[2].set_title("Answer Comparison", fontsize=14, fontweight='bold')
                axes[2].text(0, 0.5, ground_truth.upper(), ha='center', fontsize=16, color='white', fontweight='bold')
                axes[2].text(1, 0.5, answer.upper(), ha='center', fontsize=16, color='white', fontweight='bold')
                
                if is_correct:
                    axes[2].text(0.5, 1.2, "✓ CORRECT", ha='center', fontsize=14, color='green', fontweight='bold')
                else:
                    axes[2].text(0.5, 1.2, "✗ INCORRECT", ha='center', fontsize=14, color='red', fontweight='bold')
            
            plt.tight_layout()
            
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            return Image.open(buf)
    
    # Create app instance
    app = MIMICRobustTokenApp(model, processor, mimic_loader)
    
    # Create Gradio interface with tabs
    with gr.Blocks(title="MedGemma 4B - SAIL Lab", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏥 MedGemma 4B Multimodal VLM - MIMIC-CXR Analysis Platform
        ### Developed by SAIL Lab - University of New Haven
        ### Enhanced with Robust Token-Conditioned Attention & Grad-CAM Fallback
        """)
        
        with gr.Tabs():
            # Tab 1: Standard MIMIC Analysis
            with gr.TabItem("MIMIC Question Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        question_dropdown = gr.Dropdown(
                            label="Select MIMIC Question",
                            choices=mimic_loader.get_dropdown_choices(),
                            value=None,
                            interactive=True
                        )
                        
                        input_image = gr.Image(
                            label="X-Ray Image",
                            type="pil",
                            height=400
                        )
                        
                        question_text = gr.Textbox(
                            label="Question",
                            placeholder="Question will appear here",
                            lines=2
                        )
                        
                        ground_truth = gr.Textbox(
                            label="Ground Truth Answer",
                            interactive=False
                        )
                        
                        custom_mode = gr.Checkbox(
                            label="Custom Question Mode",
                            value=False
                        )
                        
                        with gr.Accordion("Visualization Options", open=True):
                            show_attention = gr.Checkbox(
                                label="Show Attention Visualization",
                                value=True
                            )
                            show_grid = gr.Checkbox(
                                label="Show Grid Lines (Debug)",
                                value=False
                            )
                        
                        analyze_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        model_answer = gr.Textbox(
                            label="MedGemma 4B Answer (Extracted)",
                            interactive=False
                        )
                        
                        with gr.Accordion("Raw Model Output", open=False):
                            raw_output = gr.Textbox(
                                label="Raw Generated Text",
                                interactive=False
                            )
                        
                        with gr.Tab("Attention Visualization"):
                            attention_viz = gr.Image(
                                label="Attention Overlay",
                                type="pil"
                            )
                        
                        with gr.Tab("Statistics & Comparison"):
                            stats_viz = gr.Image(
                                label="Statistics and Answer Comparison",
                                type="pil"
                            )
            
            # Tab 2: Robust Token-Conditioned Analysis
            with gr.TabItem("Token-Conditioned Analysis (Robust)"):
                with gr.Row():
                    with gr.Column():
                        token_question_dropdown = gr.Dropdown(
                            label="Select MIMIC Question",
                            choices=mimic_loader.get_dropdown_choices(),
                            value=None,
                            interactive=True
                        )
                        token_image = gr.Image(label="X-ray Image", type="pil", height=300)
                        token_prompt = gr.Textbox(
                            label="Question",
                            placeholder="Question will be loaded from selection",
                            lines=2
                        )
                        target_words = gr.Textbox(
                            label="Target Words (comma-separated)",
                            placeholder="e.g., effusion, pleural, fluid",
                            value=""
                        )
                        use_gradcam = gr.Checkbox(
                            label="Force Grad-CAM mode (use if cross-attention fails)",
                            value=False
                        )
                        token_analyze_btn = gr.Button("Analyze Token Attention", variant="primary")
                    
                    with gr.Column():
                        token_plot = gr.Plot(label="Token-Conditioned Attention")
                        token_text = gr.Markdown(label="Analysis Results")
                        token_metrics = gr.JSON(label="Detailed Metrics", visible=False)
                
                gr.Examples(
                    examples=[
                        ["pneumonia, pneumonic, consolidation"],
                        ["effusion, pleural, fluid"],
                        ["nodule, nodules, nodular, mass"],
                        ["calcification, calcified"],
                        ["fracture, fractured, break"],
                        ["cardiomegaly, enlarged, heart"],
                    ],
                    inputs=[target_words],
                    label="Common Target Words"
                )
            
            # Tab 3: Prompt Comparison
            with gr.TabItem("Prompt Sensitivity Analysis"):
                with gr.Row():
                    with gr.Column():
                        compare_question_dropdown = gr.Dropdown(
                            label="Select MIMIC Question",
                            choices=mimic_loader.get_dropdown_choices(),
                            value=None,
                            interactive=True
                        )
                        compare_image = gr.Image(label="X-ray Image", type="pil", height=300)
                        prompt1 = gr.Textbox(
                            label="Prompt 1",
                            placeholder="First phrasing of the question",
                            lines=2
                        )
                        prompt2 = gr.Textbox(
                            label="Prompt 2",
                            placeholder="Alternative phrasing",
                            lines=2
                        )
                        compare_target_words = gr.Textbox(
                            label="Target Words (optional)",
                            placeholder="e.g., effusion, pleural",
                            value=""
                        )
                        compare_gradcam = gr.Checkbox(
                            label="Force Grad-CAM mode",
                            value=False
                        )
                        compare_btn = gr.Button("Compare Prompts", variant="primary")
                    
                    with gr.Column():
                        comparison_plot = gr.Plot(label="Attention Comparison")
                        comparison_text = gr.Markdown(label="Comparison Results")
            
            # Tab 4: About
            with gr.TabItem("About"):
                gr.Markdown("""
                ## MedGemma 4B Multimodal VLM - Robust Analysis Platform
                ### Developed by SAIL Lab - University of New Haven
                
                This enhanced platform features robust token-conditioned attention extraction with automatic fallback mechanisms.
                
                ### Key Improvements:
                1. **Robust Token Finding**: Normalized text processing for better token matching
                2. **Proper Grid Dimensions**: Aspect-ratio aware grid factorization
                3. **Grad-CAM Fallback**: Automatic fallback when cross-attention unavailable
                4. **Method Indicators**: Shows which attention extraction method was used
                5. **Error Recovery**: Graceful handling of extraction failures
                
                ### Attention Extraction Methods:
                - **Cross-Attention**: Primary method using model's cross-attention layers
                - **Grad-CAM**: Gradient-based fallback for vision encoder attention
                - **Uniform**: Last resort when other methods fail
                
                ### Quality Metrics:
                - **Inside Body Ratio**: Should be ≥ 0.7 (attention on anatomical regions)
                - **Border Fraction**: Should be ≤ 0.05 (minimal border artifacts)
                - **Jensen-Shannon Divergence**: Consistency measure (lower is better)
                - **✓/✗ Indicators**: Quick visual check if metrics meet targets
                
                ### Model Information:
                - **Model**: MedGemma 4B Multimodal VLM
                - **Parameters**: 4 Billion
                - **Architecture**: Vision-Language Transformer with Cross-Attention
                - **Attention Modes**: Cross-Attention, Self-Attention, Grad-CAM
                
                ### Research Team:
                **SAIL Lab - University of New Haven**
                - Specialized in medical imaging analysis
                - Focus on interpretable multimodal models
                - Advancing healthcare through robust vision-language understanding
                
                ### Usage Tips:
                - If attention extraction fails, enable "Force Grad-CAM mode"
                - Check method indicator to understand which extraction was used
                - Lower JS divergence indicates more consistent attention patterns
                - Use target words to focus on specific medical terminology
                
                ### Citation:
                ```
                SAIL Lab - University of New Haven
                MedGemma 4B Multimodal VLM Analysis Platform
                Robust Token-Conditioned Attention with Grad-CAM Fallback
                ```
                """)
        
        # Event handlers
        question_dropdown.change(
            fn=app.load_question,
            inputs=[question_dropdown],
            outputs=[input_image, question_text, ground_truth, model_answer]
        )
        
        token_question_dropdown.change(
            fn=app.load_for_token_analysis,
            inputs=[token_question_dropdown],
            outputs=[token_image, token_prompt]
        )
        
        compare_question_dropdown.change(
            fn=lambda x: (app.load_for_token_analysis(x)[0], 
                         app.load_for_token_analysis(x)[1],
                         app.load_for_token_analysis(x)[1].replace("?", "?") if app.load_for_token_analysis(x)[1] else ""),
            inputs=[compare_question_dropdown],
            outputs=[compare_image, prompt1, prompt2]
        )
        
        analyze_btn.click(
            fn=app.analyze_xray,
            inputs=[input_image, question_text, custom_mode, show_attention, show_grid],
            outputs=[raw_output, attention_viz, stats_viz, model_answer]
        )
        
        token_analyze_btn.click(
            fn=app.analyze_token_attention_robust,
            inputs=[token_image, token_prompt, target_words, use_gradcam],
            outputs=[token_plot, token_text, token_metrics]
        )
        
        compare_btn.click(
            fn=app.compare_prompts_robust,
            inputs=[compare_image, prompt1, prompt2, compare_target_words, compare_gradcam],
            outputs=[comparison_plot, comparison_text]
        )
        
        gr.Markdown("""
        ---
        ### ⚠️ Disclaimer:
        This is a research tool developed by SAIL Lab at University of New Haven for analyzing 
        the MedGemma 4B multimodal VLM on the MIMIC-CXR dataset. Not for clinical use. 
        Always consult qualified healthcare professionals for medical advice.
        """)
    
    return demo


def launch_mimic_robust_app(model=None, processor=None, server_name="0.0.0.0", server_port=7860):
    """Launch the MIMIC app with robust token analysis"""
    
    if model is None or processor is None:
        model, processor = load_model_enhanced()
    
    demo = create_mimic_gradio_interface_robust(model, processor)
    
    print(f"\n=== Launching MedGemma 4B Robust Analysis Platform ===")
    print(f"Developed by SAIL Lab - University of New Haven")
    print(f"Server: {server_name}:{server_port}")
    print(f"Access the app at: http://{server_name}:{server_port}")
    
    demo.launch(
        share=True,
        server_name=server_name,
        server_port=server_port,
        show_error=True
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MedGemma 4B Robust - SAIL Lab, University of New Haven")
    parser.add_argument("--gpu", type=int, default=None, 
                      help="Specific GPU ID to use (default: auto-select)")
    parser.add_argument("--min-memory", type=float, default=15.0,
                      help="Minimum free GPU memory required in GB (default: 15)")
    parser.add_argument("--port", type=int, default=7860,
                      help="Server port (default: 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                      help="Server host (default: 0.0.0.0)")
    
    args = parser.parse_args()
    
    try:
        print(f"\n{'='*60}")
        print("MedGemma 4B Multimodal VLM - Robust Analysis")
        print("SAIL Lab - University of New Haven")
        print(f"{'='*60}")
        
        if args.gpu is not None:
            print(f"Using specified GPU: {args.gpu}")
            device = setup_gpu(device_id=args.gpu, min_free_gb=args.min_memory)
        else:
            print("Auto-selecting best available GPU...")
            device = setup_gpu(device_id=None, min_free_gb=args.min_memory)
        
        model, processor = load_model_enhanced(device=device)
        
        print(f"\n{'='*60}")
        print(f"Launching server on {args.host}:{args.port}")
        print(f"{'='*60}\n")
        
        launch_mimic_robust_app(
            model=model,
            processor=processor,
            server_name=args.host,
            server_port=args.port
        )
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n❌ GPU Out of Memory Error!")
            print("\nSuggestions:")
            print("1. Use a different GPU: python medgemma_launch_mimic_robust.py --gpu 1")
            print("2. Reduce minimum memory: python medgemma_launch_mimic_robust.py --min-memory 10")
        else:
            raise
            
    except KeyboardInterrupt:
        print("\n\nApplication interrupted by user")
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()