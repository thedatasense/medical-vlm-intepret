#!/usr/bin/env python3
"""
Enhanced LLaVA-Rad Attention Visualizer for Google Colab
Optimized for memory efficiency with 8-bit quantization and CPU offload
"""

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple, Any, Union
import logging
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import cv2
from dataclasses import dataclass
import hashlib
from scipy.spatial.distance import jensenshannon

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model ID
LLAVA_RAD_ID = "microsoft/llava-rad"


@dataclass
class AttentionConfig:
    """Configuration for attention visualization"""
    colormap: str = 'jet'
    alpha: float = 0.5
    percentile_clip: Tuple[int, int] = (2, 98)
    use_medical_colormap: bool = True
    multi_head_mode: str = 'mean'


def load_llava_rad(dtype=torch.float16, eight_bit=True, device_map="auto"):
    """
    Load LLaVA-Rad with 8-bit quantization and CPU offload for Colab
    
    Args:
        dtype: Model dtype (default: float16)
        eight_bit: Use 8-bit quantization (default: True)
        device_map: Device mapping (default: "auto")
        
    Returns:
        Tuple of (model, processor)
    """
    # Configure 8-bit quantization with CPU offload
    quant_config = BitsAndBytesConfig(
        load_in_8bit=eight_bit,
        llm_int8_enable_fp32_cpu_offload=True  # Keeps overflowed modules on CPU
    )
    
    logger.info(f"Loading {LLAVA_RAD_ID} with 8-bit quantization and CPU offload...")
    
    # Load model
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_RAD_ID,
        torch_dtype=dtype,
        device_map=device_map,
        quantization_config=quant_config,
        trust_remote_code=True
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(LLAVA_RAD_ID)
    
    model.eval()
    logger.info("✓ LLaVA-Rad loaded successfully")
    
    return model, processor


def extract_query_to_image_attention(
    model: Any,
    processor: Any,
    image: Image.Image,
    text: str,
    layer_indices: Optional[List[int]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extract query-to-image attention from LLaVA model
    
    Returns:
        Tuple of (attention_map, metadata)
    """
    # Format prompt
    prompt = f"USER: <image>\n{text} ASSISTANT:"
    
    # Process inputs
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    
    # Forward pass with attention
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, use_cache=False)
    
    # Determine layers to use
    num_layers = len(outputs.attentions)
    if layer_indices is None:
        # Use last quarter of layers
        layer_indices = list(range(3 * num_layers // 4, num_layers))
    
    # Find image token positions
    # For LLaVA, image tokens are typically after the image placeholder
    input_ids = inputs["input_ids"][0].cpu()
    
    # Estimate image token range (LLaVA uses 576 tokens for 336x336 image)
    num_patches = 576  # Standard for LLaVA
    text_tokens_before = len(processor.tokenizer.encode(prompt.split("<image>")[0]))
    image_start = text_tokens_before + 1
    image_end = image_start + num_patches
    
    # Query position is last token
    query_pos = inputs["input_ids"].shape[1] - 1
    
    # Extract attention
    attention_maps = []
    for layer_idx in layer_indices:
        layer_attn = outputs.attentions[layer_idx][0]  # Remove batch dim
        
        # Get attention from query to image tokens
        query_to_img = layer_attn[:, query_pos, image_start:image_end]  # (heads, img_tokens)
        
        # Average over heads
        avg_attn = query_to_img.mean(dim=0)
        attention_maps.append(avg_attn)
    
    # Average over layers
    final_attention = torch.stack(attention_maps).mean(dim=0).cpu().numpy()
    
    # Reshape to 2D
    grid_size = int(np.sqrt(num_patches))
    attention_2d = final_attention.reshape(grid_size, grid_size)
    
    metadata = {
        "num_patches": num_patches,
        "query_position": query_pos,
        "layers_used": layer_indices,
        "grid_size": grid_size
    }
    
    return attention_2d, metadata


class AttentionMetrics:
    """Calculate attention quality metrics"""
    
    @staticmethod
    def calculate_focus_score(attention_map: np.ndarray) -> Dict[str, float]:
        """Calculate focus score using entropy"""
        # Normalize
        attention_map = attention_map / (attention_map.sum() + 1e-10)
        
        # Calculate entropy
        entropy = -np.sum(attention_map * np.log(attention_map + 1e-10))
        max_entropy = np.log(attention_map.size)
        focus_score = 1 - (entropy / max_entropy)
        
        return {'focus': float(focus_score)}
    
    @staticmethod
    def calculate_consistency(attention_maps: List[np.ndarray]) -> float:
        """Calculate consistency across multiple maps"""
        if len(attention_maps) < 2:
            return 1.0
        
        # Calculate pairwise JS divergence
        js_divs = []
        for i in range(len(attention_maps)):
            for j in range(i + 1, len(attention_maps)):
                # Flatten and normalize
                map1 = attention_maps[i].flatten()
                map1 = map1 / (map1.sum() + 1e-10)
                map2 = attention_maps[j].flatten()  
                map2 = map2 / (map2.sum() + 1e-10)
                
                js_div = jensenshannon(map1, map2)
                js_divs.append(js_div)
        
        # Consistency is inverse of mean JS divergence
        mean_js = np.mean(js_divs) if js_divs else 0
        return float(1 - mean_js)


class EnhancedLLaVARadVisualizer:
    """Main LLaVA-Rad visualizer class optimized for Colab"""
    
    def __init__(self, config: Optional[AttentionConfig] = None):
        self.config = config or AttentionConfig()
        self.model = None
        self.processor = None
        self.cache = {}
        
    def load_model(self, load_in_8bit: bool = True):
        """Load model with 8-bit quantization by default"""
        self.model, self.processor = load_llava_rad(eight_bit=load_in_8bit)
        
    def generate_with_attention(
        self,
        image_path: Union[str, Image.Image],
        prompt: str,
        max_new_tokens: int = 100,
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
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Extract attention
            attention_map, metadata = extract_query_to_image_attention(
                self.model, self.processor, image, prompt
            )
            
            # Generate answer
            full_prompt = f"USER: <image>\n{prompt} ASSISTANT:"
            inputs = self.processor(
                text=full_prompt,
                images=image,
                return_tensors="pt"
            )
            
            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            # Generate (no temperature since sampling is disabled)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Deterministic
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode answer
            answer = self.processor.decode(outputs[0], skip_special_tokens=True)
            # Extract assistant response
            if "ASSISTANT:" in answer:
                answer = answer.split("ASSISTANT:")[-1].strip()
            
            # Calculate metrics
            metrics = AttentionMetrics.calculate_focus_score(attention_map)
            
            result = {
                'answer': answer,
                'visual_attention': attention_map,
                'metrics': metrics,
                'attention_method': 'query_to_image',
                'metadata': metadata
            }
            
            # Cache
            if use_cache:
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return {
                'answer': f"Error: {str(e)}",
                'visual_attention': None,
                'error': str(e)
            }
    
    def visualize_attention(
        self,
        image: Union[str, Image.Image],
        attention_map: np.ndarray,
        title: str = "LLaVA-Rad Attention",
        save_path: Optional[str] = None
    ) -> Image.Image:
        """Visualize attention overlay on image"""
        
        # Load image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Resize attention to image size
        attention_resized = cv2.resize(
            attention_map,
            (image.width, image.height),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Apply percentile clipping
        vmin = np.percentile(attention_resized, self.config.percentile_clip[0])
        vmax = np.percentile(attention_resized, self.config.percentile_clip[1])
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image)
        
        # Overlay attention
        cmap = 'hot' if self.config.use_medical_colormap else self.config.colormap
        im = ax.imshow(
            attention_resized,
            cmap=cmap,
            alpha=self.config.alpha,
            vmin=vmin,
            vmax=vmax
        )
        
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        # Convert to PIL Image
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        ncols, nrows = fig.canvas.get_width_height()
        plt.close(fig)
        
        img_array = buf.reshape(nrows, ncols, 3)
        return Image.fromarray(img_array)