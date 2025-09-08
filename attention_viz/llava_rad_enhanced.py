#!/usr/bin/env python3
"""
Enhanced LLaVA-Rad Attention Visualizer with HuggingFace Support
Updated to prioritize HF checkpoints and modern Transformers APIs
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, Dict, List, Tuple, Any, Union
import logging
import warnings
from pathlib import Path
import json
from transformers import (
    AutoProcessor, 
    LlavaForConditionalGeneration,
    BitsAndBytesConfig
)
import gc
from datetime import datetime
from scipy.spatial.distance import jensenshannon
import cv2
from dataclasses import dataclass
from functools import lru_cache
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Default HuggingFace model IDs
LLAVA_RAD_ID = "microsoft/llava-rad"
LLAVA_HF_FALLBACK = "llava-hf/llava-1.5-7b-hf"


@dataclass
class AttentionConfig:
    """Centralized configuration for attention visualization"""
    colormap: str = 'jet'
    alpha: float = 0.5
    percentile_clip: Tuple[int, int] = (2, 98)
    border_strip: int = 2
    default_layers: str = 'last_quarter'  # or specific indices
    patch_size: int = 14
    image_size: int = 336
    body_ratio_threshold: float = 0.7
    border_fraction_threshold: float = 0.05
    use_medical_colormap: bool = False
    multi_head_mode: str = 'mean'  # 'mean', 'max', or 'individual'


def load_llava_rad_hf(
    model_id: str = LLAVA_RAD_ID,
    dtype: torch.dtype = torch.float16,
    device_map: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
) -> Tuple[Any, Any]:
    """
    Load LLaVA-Rad using HuggingFace Transformers
    
    Args:
        model_id: Model identifier (default: microsoft/llava-rad)
        dtype: Model dtype (default: float16)
        device_map: Device mapping strategy
        load_in_8bit: Use 8-bit quantization
        load_in_4bit: Use 4-bit quantization
        
    Returns:
        Tuple of (model, processor)
    """
    # Configure quantization
    quantization_config = None
    if load_in_8bit or load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        dtype = torch.float16  # Override for quantized models
    
    try:
        # First try the requested model
        logger.info(f"Loading {model_id} from HuggingFace...")
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            quantization_config=quantization_config,
            trust_remote_code=True,
            attn_implementation="eager"  # Enable attention outputs
        )
        processor = AutoProcessor.from_pretrained(model_id)
        logger.info(f"✓ Successfully loaded {model_id}")
        
    except Exception as e:
        # Fallback to a known working HF model
        logger.warning(f"Failed to load {model_id}: {e}")
        logger.info(f"Falling back to {LLAVA_HF_FALLBACK}")
        
        model = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_HF_FALLBACK,
            torch_dtype=dtype,
            device_map=device_map,
            quantization_config=quantization_config,
            attn_implementation="eager"
        )
        processor = AutoProcessor.from_pretrained(LLAVA_HF_FALLBACK)
        logger.info(f"✓ Successfully loaded fallback model")
    
    model.eval()
    return model, processor


def extract_query_to_image_attention_llava(
    model: Any,
    processor: Any,
    pil_image: Image.Image,
    text: str,
    layer_indices: Optional[List[int]] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Extract query-to-image attention for LLaVA models
    
    Args:
        model: The loaded LLaVA model
        processor: The processor for the model
        pil_image: Input PIL image
        text: Input text query
        layer_indices: Which layers to extract (None = last quarter)
        
    Returns:
        Tuple of (attention_map, metadata_dict)
    """
    # Prepare inputs
    prompt = f"USER: <image>\n{text} ASSISTANT:"
    inputs = processor(text=prompt, images=pil_image, return_tensors="pt")
    
    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    
    # Forward pass with attention
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, use_cache=False)
    
    # Determine which layers to use
    num_layers = len(outputs.attentions)
    if layer_indices is None:
        # Use last quarter of layers
        last_quarter_start = max(1, 3 * num_layers // 4)
        layer_indices = list(range(last_quarter_start, num_layers))
    
    # Find image token positions
    # LLaVA typically places image tokens after the <image> token
    image_token_id = processor.tokenizer.encode("<image>", add_special_tokens=False)[0]
    image_positions = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0]
    
    if len(image_positions) == 0:
        # Estimate based on typical LLaVA patterns
        # Usually 576 tokens for 336x336 image (24x24 patches)
        num_patches = (336 // 14) ** 2  # Default LLaVA uses 14x14 patches
        # Image tokens typically start after prompt tokens
        prompt_len = len(processor.tokenizer.encode(prompt.split("<image>")[0]))
        image_start = prompt_len + 1
        image_end = image_start + num_patches
        image_positions = torch.arange(image_start, image_end)
    
    # Extract attention from query (last token) to image tokens
    query_position = inputs["input_ids"].shape[1] - 1
    
    # Aggregate attention across selected layers
    attention_maps = []
    for layer_idx in layer_indices:
        # Get attention for this layer: (batch, heads, seq, seq)
        layer_attention = outputs.attentions[layer_idx][0]  # Remove batch dim
        
        # Extract query-to-image attention
        query_to_image = layer_attention[:, query_position, image_positions]  # (heads, image_tokens)
        
        # Average over heads
        avg_attention = query_to_image.mean(dim=0)  # (image_tokens,)
        attention_maps.append(avg_attention)
    
    # Average over layers
    final_attention = torch.stack(attention_maps).mean(dim=0)
    
    # Reshape to 2D grid
    grid_size = int(np.sqrt(len(final_attention)))
    if grid_size * grid_size == len(final_attention):
        attention_2d = final_attention.reshape(grid_size, grid_size)
    else:
        # Handle non-square attention
        attention_2d = _reshape_attention_robust(final_attention, len(final_attention))
    
    metadata = {
        "num_patches": len(image_positions),
        "grid_size": grid_size,
        "query_position": query_position,
        "layers_used": layer_indices,
        "model_id": getattr(model, 'name_or_path', 'unknown')
    }
    
    return attention_2d.cpu(), metadata


def _reshape_attention_robust(attention_1d: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Reshape 1D attention to 2D with common aspect ratios"""
    attention_1d = attention_1d.cpu().numpy()
    
    # Try common aspect ratios
    aspect_ratios = [(1, 1), (4, 3), (3, 4), (16, 9), (9, 16)]
    
    for w_ratio, h_ratio in aspect_ratios:
        scale = np.sqrt(num_tokens / (w_ratio * h_ratio))
        w = int(w_ratio * scale)
        h = int(h_ratio * scale)
        
        if w * h == num_tokens:
            return torch.from_numpy(attention_1d.reshape(h, w))
    
    # Fallback: pad to nearest square
    grid_size = int(np.ceil(np.sqrt(num_tokens)))
    padded = np.zeros(grid_size * grid_size)
    padded[:num_tokens] = attention_1d
    return torch.from_numpy(padded.reshape(grid_size, grid_size))


class AttentionMetrics:
    """Enhanced attention quality metrics"""
    
    @staticmethod
    def calculate_focus_score(attention_map: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate how focused the attention is using entropy-based measure"""
        # Ensure attention is normalized
        attention_map = attention_map / (attention_map.sum() + 1e-10)
        
        # Calculate normalized entropy
        entropy = -np.sum(attention_map * np.log(attention_map + 1e-10))
        max_entropy = np.log(attention_map.size)
        focus_score = 1 - (entropy / max_entropy)
        
        result = {'focus': float(focus_score)}
        
        # ROI focus if mask provided
        if roi_mask is not None:
            roi_attention = attention_map[roi_mask > 0].sum()
            total_attention = attention_map.sum()
            roi_focus = roi_attention / (total_attention + 1e-10)
            result['roi_focus'] = float(roi_focus)
            
            # Calculate in/out ratio
            out_attention = attention_map[roi_mask == 0].sum()
            in_out_ratio = roi_attention / (out_attention + 1e-10)
            result['in_out_ratio'] = float(in_out_ratio)
        
        return result
    
    @staticmethod
    def calculate_consistency(attention_maps: List[np.ndarray]) -> float:
        """Calculate consistency across multiple attention maps using JS divergence"""
        if len(attention_maps) < 2:
            return 1.0
            
        # Normalize all maps
        normalized_maps = []
        for att_map in attention_maps:
            flat = att_map.flatten()
            normalized = flat / (flat.sum() + 1e-10)
            normalized_maps.append(normalized)
        
        # Calculate pairwise JS divergence
        js_divs = []
        for i in range(len(normalized_maps)):
            for j in range(i + 1, len(normalized_maps)):
                js_div = jensenshannon(normalized_maps[i], normalized_maps[j])
                js_divs.append(js_div)
        
        # Consistency is inverse of mean JS divergence
        mean_js = np.mean(js_divs) if js_divs else 0
        consistency = 1 - mean_js
        
        return float(consistency)


class EnhancedLLaVARadVisualizer:
    """Enhanced LLaVA-Rad visualizer with HuggingFace-first approach"""
    
    def __init__(self, config: Optional[AttentionConfig] = None):
        self.config = config or AttentionConfig()
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.attention_cache = {}
        self.using_hf = True  # Default to HF approach
        
        # Check memory for auto-device selection
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory < 10:  # Less than 10GB
                logger.warning(f"Low GPU memory ({gpu_memory:.1f}GB). Consider using 8-bit quantization.")
    
    def load_model(self, model_id: str = LLAVA_RAD_ID, load_in_8bit: bool = False, load_in_4bit: bool = False):
        """Load LLaVA-Rad model with HF priority"""
        
        # First try HuggingFace approach
        try:
            self.model, self.processor = load_llava_rad_hf(
                model_id=model_id,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit
            )
            self.using_hf = True
            logger.info("✓ Model loaded using HuggingFace Transformers")
            
        except Exception as e:
            logger.error(f"Failed to load model via HuggingFace: {e}")
            
            # Try LLaVA library as fallback if available
            try:
                from llava.model.builder import load_pretrained_model
                from llava.utils import disable_torch_init
                
                disable_torch_init()
                
                tokenizer, model, image_processor, context_len = load_pretrained_model(
                    model_path=model_id,
                    model_base="lmsys/vicuna-7b-v1.5",
                    model_name="llavarad",
                    device_map=self.device if self.device != "cpu" else "auto",
                    load_8bit=load_in_8bit,
                    load_4bit=load_in_4bit
                )
                
                self.model = model
                self.processor = image_processor
                self.tokenizer = tokenizer
                self.using_hf = False
                
                logger.info("✓ Model loaded using LLaVA library")
                
            except Exception as e2:
                raise RuntimeError(f"Failed to load model via both HF and LLaVA library: HF error: {e}, LLaVA error: {e2}")
    
    def generate_with_attention(
        self,
        image_path: Union[str, Image.Image],
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.1,
        do_sample: bool = False,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Generate response with attention extraction"""
        
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        # Check cache
        cache_key = hashlib.md5(f"{str(image_path)}_{prompt}".encode()).hexdigest()
        if use_cache and cache_key in self.attention_cache:
            logger.info("Using cached attention")
            return self.attention_cache[cache_key]
        
        try:
            if self.using_hf:
                # HuggingFace approach
                attention_map, metadata = extract_query_to_image_attention_llava(
                    self.model, self.processor, image, prompt
                )
                
                # Generate answer
                inputs = self.processor(
                    text=f"USER: <image>\n{prompt} ASSISTANT:",
                    images=image,
                    return_tensors="pt"
                ).to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=do_sample
                    )
                
                answer = self.processor.decode(outputs[0], skip_special_tokens=True)
                answer = answer.split("ASSISTANT:")[-1].strip()
                
            else:
                # LLaVA library approach (implement if needed)
                raise NotImplementedError("LLaVA library approach not implemented in this version")
            
            # Calculate metrics
            metrics = AttentionMetrics.calculate_focus_score(attention_map.numpy())
            
            result = {
                'answer': answer,
                'visual_attention': attention_map.numpy(),
                'metrics': metrics,
                'attention_method': metadata.get('method', 'query_to_image'),
                'metadata': metadata
            }
            
            # Cache result
            if use_cache:
                self.attention_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return {
                'answer': f"Error: {str(e)}",
                'visual_attention': None,
                'attention_method': 'error'
            }
    
    def visualize_attention(
        self,
        image: Union[str, Image.Image],
        attention_map: np.ndarray,
        title: str = "Attention Visualization",
        save_path: Optional[str] = None
    ) -> Image.Image:
        """Visualize attention map overlaid on image"""
        
        # Load image if path
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Resize attention to match image
        attention_resized = cv2.resize(
            attention_map,
            (image.width, image.height),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Apply medical colormap if configured
        if self.config.use_medical_colormap:
            cmap = 'hot'
        else:
            cmap = self.config.colormap
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image)
        im = ax.imshow(
            attention_resized,
            cmap=cmap,
            alpha=self.config.alpha,
            vmin=np.percentile(attention_resized, self.config.percentile_clip[0]),
            vmax=np.percentile(attention_resized, self.config.percentile_clip[1])
        )
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        # Convert to PIL Image
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        plt.close(fig)
        
        img_array = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
        return Image.fromarray(img_array)