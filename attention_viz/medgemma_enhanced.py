#!/usr/bin/env python3
"""
Enhanced MedGemma Attention Extraction with Gemma3 Support
Updated to use modern Transformers APIs and Gemma3-specific features
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple, Any, Union
import logging
import cv2
from scipy.spatial.distance import jensenshannon
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)

# Default model ID for MedGemma
MODEL_ID = "google/medgemma-4b-it"


def load_model_enhanced(model_id: str = MODEL_ID,
                       load_in_8bit: bool = True,
                       load_in_4bit: bool = False,
                       dtype: torch.dtype = torch.bfloat16) -> Tuple[Any, Any]:
    """
    Enhanced model loading function for MedGemma with Gemma3 support.
    
    Args:
        model_id: Model identifier (default: google/medgemma-4b-it)
        load_in_8bit: Use 8-bit quantization
        load_in_4bit: Use 4-bit quantization
        dtype: Default dtype (bfloat16 recommended for Gemma3)
    
    Returns:
        Tuple of (model, processor)
    """
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
        # Try to import Gemma3 specific classes if available
        try:
            from transformers import Gemma3ForConditionalGeneration
            use_gemma3_class = True
        except ImportError:
            use_gemma3_class = False
            logger.info("Gemma3ForConditionalGeneration not available, using AutoModelForCausalLM")
    except ImportError:
        raise ImportError("Please install transformers>=4.56.1: pip install transformers>=4.56.1")
    
    # Configure quantization if requested
    quantization_config = None
    if load_in_8bit or load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        # Override dtype for quantized models
        dtype = torch.float16
    
    # Load processor with left padding for generation
    processor = AutoProcessor.from_pretrained(
        model_id, 
        trust_remote_code=True,
        padding_side="left"
    )
    
    # Load model with appropriate class
    if use_gemma3_class:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
            attn_implementation="eager"  # Enable attention outputs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
            attn_implementation="eager"  # Enable attention outputs
        )
    
    model.eval()
    logger.info(f"MedGemma model loaded: {model_id} (dtype: {dtype})")
    
    return model, processor


def extract_token_to_image_attention(
    model: Any,
    processor: Any,
    pil_image: Image.Image,
    text: str,
    do_pan_and_scan: bool = False
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Extract token-to-image attention for Gemma3 models with SigLIP encoder.
    
    Args:
        model: The loaded Gemma3 model
        processor: The processor for the model
        pil_image: Input PIL image
        text: Input text query
        do_pan_and_scan: Enable pan-and-scan for variable image sizes
        
    Returns:
        Tuple of (attention_weights, metadata_dict)
    """
    # Prepare messages in Gemma3 format
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": text}
        ]
    }]
    
    # Apply chat template and tokenize
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
        do_pan_and_scan=do_pan_and_scan
    )
    
    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    
    # Forward pass with attention
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, use_cache=False)
    
    # Extract attention from last quarter of layers
    num_layers = model.config.num_hidden_layers
    last_quarter_start = max(1, 3 * num_layers // 4)
    
    # Stack attention from last quarter of layers
    attn_stack = torch.stack([
        outputs.attentions[i] for i in range(last_quarter_start, num_layers)
    ])  # Shape: (layers, batch, heads, seq, seq)
    
    # Find image token positions
    if hasattr(processor.tokenizer, 'convert_tokens_to_ids'):
        img_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    else:
        # Fallback: look for special tokens
        img_token_id = None
        for token, token_id in processor.tokenizer.get_vocab().items():
            if "<image>" in token:
                img_token_id = token_id
                break
    
    if img_token_id is not None:
        img_positions = (inputs["input_ids"][0] == img_token_id).nonzero(as_tuple=True)[0]
    else:
        # Fallback: estimate image positions based on token count
        # Gemma3 typically uses ~256-576 tokens per image depending on resolution
        text_tokens = processor.tokenizer(text, return_tensors="pt")["input_ids"].shape[1]
        total_tokens = inputs["input_ids"].shape[1]
        estimated_img_tokens = total_tokens - text_tokens - 10  # Account for special tokens
        img_positions = torch.arange(5, 5 + estimated_img_tokens)  # Skip initial tokens
    
    # Extract attention from last text token to image positions
    query_position = inputs["input_ids"].shape[1] - 1  # Last token position
    
    # Get attention weights: (layers, heads, img_positions)
    attn_to_img = attn_stack[..., query_position, img_positions]
    
    # Average over heads and layers
    attn_weights = attn_to_img.mean(dim=(0, 1))  # Shape: (num_image_tokens,)
    
    metadata = {
        "num_image_tokens": len(img_positions),
        "query_position": query_position,
        "num_layers_used": len(range(last_quarter_start, num_layers)),
        "do_pan_and_scan": do_pan_and_scan
    }
    
    return attn_weights.cpu(), metadata


@dataclass
class AttentionExtractionConfig:
    """Configuration for attention extraction"""
    use_gradcam: bool = False
    gradcam_layer: str = 'language_model.model.layers[-1]'
    token_gating: bool = True
    multi_token_aggregation: str = 'mean'  # 'mean', 'max', 'weighted'
    attention_head_reduction: str = 'mean'  # 'mean', 'max', 'entropy_weighted'
    use_cross_attention: bool = True
    fallback_chain: List[str] = None
    cache_enabled: bool = True
    debug_mode: bool = False
    do_pan_and_scan: bool = False  # Gemma3 specific
    
    def __post_init__(self):
        if self.fallback_chain is None:
            self.fallback_chain = ['gemma3_attention', 'cross_attention', 'gradcam', 'uniform']


class EnhancedAttentionExtractor:
    """Enhanced attention extraction with Gemma3 support"""
    
    def __init__(self, config: Optional[AttentionExtractionConfig] = None):
        self.config = config or AttentionExtractionConfig()
        self.cache = {} if self.config.cache_enabled else None
        
    def extract_attention_gemma3(
        self,
        model: Any,
        processor: Any,
        pil_image: Image.Image,
        text: str
    ) -> Tuple[np.ndarray, str]:
        """
        Extract attention using Gemma3-specific method
        """
        try:
            attn_weights, metadata = extract_token_to_image_attention(
                model, processor, pil_image, text,
                do_pan_and_scan=self.config.do_pan_and_scan
            )
            
            # Reshape to 2D if needed
            num_tokens = metadata["num_image_tokens"]
            if num_tokens > 0:
                # Assume square image grid
                grid_size = int(np.sqrt(num_tokens))
                if grid_size * grid_size == num_tokens:
                    attention_map = attn_weights.numpy().reshape(grid_size, grid_size)
                else:
                    # Non-square, try common aspect ratios
                    attention_map = self._reshape_attention(attn_weights.numpy(), num_tokens)
                
                return attention_map, 'gemma3_attention'
            
        except Exception as e:
            logger.warning(f"Gemma3 attention extraction failed: {e}")
        
        return None, None
        
    def _reshape_attention(self, attention_1d: np.ndarray, num_tokens: int) -> np.ndarray:
        """Reshape 1D attention to 2D with common aspect ratios"""
        # Try common aspect ratios
        aspect_ratios = [(1, 1), (4, 3), (3, 4), (16, 9), (9, 16)]
        
        for w_ratio, h_ratio in aspect_ratios:
            # Find scaling factor
            scale = np.sqrt(num_tokens / (w_ratio * h_ratio))
            w = int(w_ratio * scale)
            h = int(h_ratio * scale)
            
            if w * h == num_tokens:
                return attention_1d.reshape(h, w)
        
        # Fallback: pad to nearest square
        grid_size = int(np.ceil(np.sqrt(num_tokens)))
        padded = np.zeros(grid_size * grid_size)
        padded[:num_tokens] = attention_1d
        return padded.reshape(grid_size, grid_size)
    
    def extract_token_conditioned_attention_robust(
        self, 
        model: Any,
        processor: Any,
        gen_result: Dict,
        target_words: List[str],
        pil_image: Optional[Image.Image] = None,
        prompt: Optional[str] = None
    ) -> Tuple[np.ndarray, List[int], str]:
        """
        Enhanced token-conditioned attention extraction with fallback chain
        
        Returns:
            Tuple of (attention_map, token_indices, method_used)
        """
        # Try Gemma3-specific extraction first if we have image and prompt
        if pil_image is not None and prompt is not None:
            attention_map, method = self.extract_attention_gemma3(
                model, processor, pil_image, prompt
            )
            if attention_map is not None:
                # Return dummy token indices for compatibility
                return attention_map, [0], method
        
        # Fallback to other methods
        for method in self.config.fallback_chain:
            if method == 'gemma3_attention':
                continue  # Already tried
            
            if method == 'cross_attention':
                result = self._extract_cross_attention(model, processor, gen_result, target_words)
                if result[0] is not None:
                    return result
                    
            elif method == 'gradcam' and self.config.use_gradcam:
                result = self._extract_gradcam(model, processor, pil_image, target_words)
                if result[0] is not None:
                    return result
                    
            elif method == 'uniform':
                # Last resort: uniform attention
                return np.ones((32, 32)) / (32 * 32), [], 'uniform'
        
        return np.ones((32, 32)) / (32 * 32), [], 'fallback'
    
    def _extract_cross_attention(
        self,
        model: Any,
        processor: Any,
        gen_result: Dict,
        target_words: List[str]
    ) -> Tuple[Optional[np.ndarray], List[int], str]:
        """Extract cross-attention from generation results"""
        if not hasattr(gen_result, 'attentions') or gen_result.attentions is None:
            return None, [], 'none'
            
        try:
            # Implementation for cross-attention extraction
            # This is a placeholder - implement based on model architecture
            return None, [], 'none'
        except Exception as e:
            logger.warning(f"Cross-attention extraction failed: {e}")
            return None, [], 'none'
    
    def _extract_gradcam(
        self,
        model: Any,
        processor: Any,
        pil_image: Optional[Image.Image],
        target_words: List[str]
    ) -> Tuple[Optional[np.ndarray], List[int], str]:
        """Extract attention using GradCAM"""
        # Placeholder for GradCAM implementation
        return None, [], 'none'


# Backward compatibility functions
def load_medgemma(dtype=torch.bfloat16, device_map="auto"):
    """Legacy function name for compatibility"""
    return load_model_enhanced(MODEL_ID, dtype=dtype)


# Visualization utilities
def visualize_attention_on_image(
    image: Image.Image,
    attention_map: np.ndarray,
    title: str = "Attention Visualization",
    cmap: str = 'hot',
    alpha: float = 0.5
) -> Image.Image:
    """
    Visualize attention map overlaid on image
    """
    # Resize attention to match image size
    attention_resized = cv2.resize(
        attention_map,
        (image.width, image.height),
        interpolation=cv2.INTER_LINEAR
    )
    
    # Normalize attention
    attention_norm = (attention_resized - attention_resized.min()) / (
        attention_resized.max() - attention_resized.min() + 1e-8
    )
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    im = ax.imshow(attention_norm, cmap=cmap, alpha=alpha)
    ax.set_title(title)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Convert to PIL Image
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    plt.close(fig)
    
    img_array = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
    return Image.fromarray(img_array)