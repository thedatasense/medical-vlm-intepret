#!/usr/bin/env python3
"""
Enhanced MedGemma Attention Extraction for Google Colab
Fixed image placeholder count and optimized for Gemma3
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple, Any
import logging
from transformers import AutoProcessor, AutoModelForCausalLM
import cv2
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model ID
MEDGEMMA_ID = "google/medgemma-4b-it"


@dataclass
class AttentionExtractionConfig:
    """Configuration for attention extraction"""
    do_pan_and_scan: bool = False
    attention_head_reduction: str = 'mean'
    use_last_quarter_layers: bool = True


def load_medgemma(dtype=torch.float16, device_map="auto"):
    """
    Load MedGemma model optimized for Colab
    
    Args:
        dtype: Model dtype (default: float16)
        device_map: Device mapping (default: "auto")
        
    Returns:
        Tuple of (model, processor)
    """
    logger.info(f"Loading {MEDGEMMA_ID}...")
    
    # Load model - try Gemma3 specific class if available
    try:
        from transformers import Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(
            MEDGEMMA_ID, 
            dtype=dtype,  # Use dtype instead of torch_dtype
            device_map=device_map,
            trust_remote_code=True
        )
    except ImportError:
        # Fallback to AutoModel
        model = AutoModelForCausalLM.from_pretrained(
            MEDGEMMA_ID, 
            dtype=dtype,  # Use dtype instead of torch_dtype 
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation="eager"  # Enable attention outputs
        )
    
    # Load processor with left padding for generation
    processor = AutoProcessor.from_pretrained(
        MEDGEMMA_ID, 
        padding_side="left",
        trust_remote_code=True
    )
    
    model.eval()
    logger.info("✓ MedGemma loaded successfully")
    
    return model, processor


def build_inputs(processor, pil_image, question, do_pan_and_scan=False, device=None):
    """
    Build inputs with correct image placeholder count
    
    Args:
        processor: The MedGemma processor
        pil_image: PIL Image
        question: Text question
        do_pan_and_scan: Enable pan and scan (default: False)
        device: Target device
        
    Returns:
        Dict of model inputs
    """
    # Build message with exactly one image placeholder
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},  # Exactly one image placeholder
            {"type": "text", "text": question}
        ]
    }]
    
    # Apply chat template
    prompt = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        do_pan_and_scan=do_pan_and_scan  # Gemma3 supports this
    )
    
    # Process with single image
    encoded = processor(
        text=prompt, 
        images=[pil_image],  # Pass as list with one image
        do_pan_and_scan=do_pan_and_scan, 
        return_tensors="pt"
    )
    
    # Move to device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return {k: v.to(device) for k, v in encoded.items()}


def generate_answer(model, processor, inputs, max_new_tokens=64):
    """
    Generate answer with deterministic decoding
    
    Args:
        model: The model
        processor: The processor
        inputs: Model inputs
        max_new_tokens: Max tokens to generate
        
    Returns:
        Generated text answer
    """
    # Generate without temperature (deterministic)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # No sampling = deterministic
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True
        )
    
    # Decode
    answer = processor.decode(outputs[0], skip_special_tokens=True)
    
    # Clean answer - extract model response
    if '<start_of_turn>model' in answer:
        answer = answer.split('<start_of_turn>model')[-1]
        if '<end_of_turn>' in answer:
            answer = answer.split('<end_of_turn>')[0]
    
    return answer.strip()


def extract_attention_once(model, inputs, config=None):
    """
    Extract attention with single forward pass
    
    Args:
        model: The model
        inputs: Model inputs
        config: Attention extraction config
        
    Returns:
        Attention tensors
    """
    if config is None:
        config = AttentionExtractionConfig()
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, use_cache=False)
    
    return outputs.attentions  # List of attention tensors per layer


def extract_token_to_image_attention(
    model, 
    processor, 
    pil_image, 
    text,
    config=None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extract token-to-image attention for MedGemma
    
    Returns:
        Tuple of (attention_map, metadata)
    """
    if config is None:
        config = AttentionExtractionConfig()
    
    # Build inputs
    inputs = build_inputs(
        processor, pil_image, text, 
        do_pan_and_scan=config.do_pan_and_scan,
        device=next(model.parameters()).device
    )
    
    # Get attention
    attentions = extract_attention_once(model, inputs, config)
    
    # Determine layers to use
    num_layers = len(attentions)
    if config.use_last_quarter_layers:
        layer_indices = list(range(3 * num_layers // 4, num_layers))
    else:
        layer_indices = list(range(num_layers))
    
    # Find image token positions
    # For MedGemma/Gemma3, need to identify where image tokens are
    input_ids = inputs["input_ids"][0].cpu()
    
    # Try to find image token ID
    image_token_id = None
    if hasattr(processor.tokenizer, 'convert_tokens_to_ids'):
        try:
            image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        except:
            pass
    
    if image_token_id is not None:
        image_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
    else:
        # Estimate based on typical patterns
        # Gemma3 uses ~256-576 tokens per image
        text_tokens = len(processor.tokenizer.encode(text))
        total_tokens = input_ids.shape[0]
        estimated_img_tokens = max(256, total_tokens - text_tokens - 20)
        # Image tokens typically after initial tokens
        image_positions = torch.arange(10, min(10 + estimated_img_tokens, total_tokens - 10))
    
    # Query position is last token
    query_pos = input_ids.shape[0] - 1
    
    # Extract attention from query to image tokens
    attention_maps = []
    for layer_idx in layer_indices:
        layer_attn = attentions[layer_idx][0]  # Remove batch dim
        
        # Get attention from query to image positions
        if len(image_positions) > 0:
            query_to_img = layer_attn[:, query_pos, image_positions]  # (heads, img_tokens)
            
            # Reduce over heads
            if config.attention_head_reduction == 'mean':
                avg_attn = query_to_img.mean(dim=0)
            elif config.attention_head_reduction == 'max':
                avg_attn = query_to_img.max(dim=0)[0]
            else:
                avg_attn = query_to_img.mean(dim=0)
                
            attention_maps.append(avg_attn)
    
    if not attention_maps:
        # Fallback to uniform attention
        num_tokens = len(image_positions) if len(image_positions) > 0 else 256
        final_attention = np.ones(num_tokens) / num_tokens
    else:
        # Average over layers
        final_attention = torch.stack(attention_maps).mean(dim=0).cpu().numpy()
    
    # Reshape to 2D
    attention_2d = reshape_attention_to_grid(final_attention)
    
    metadata = {
        "num_image_tokens": len(image_positions),
        "query_position": query_pos,
        "layers_used": layer_indices,
        "do_pan_and_scan": config.do_pan_and_scan
    }
    
    return attention_2d, metadata


def reshape_attention_to_grid(attention_1d: np.ndarray) -> np.ndarray:
    """Reshape 1D attention to 2D grid"""
    num_tokens = len(attention_1d)
    
    # Try square first
    grid_size = int(np.sqrt(num_tokens))
    if grid_size * grid_size == num_tokens:
        return attention_1d.reshape(grid_size, grid_size)
    
    # Try common aspect ratios
    aspect_ratios = [(1, 1), (4, 3), (3, 4), (16, 9), (9, 16)]
    
    for w_ratio, h_ratio in aspect_ratios:
        scale = np.sqrt(num_tokens / (w_ratio * h_ratio))
        w = int(w_ratio * scale)
        h = int(h_ratio * scale)
        
        if w * h == num_tokens:
            return attention_1d.reshape(h, w)
    
    # Fallback: pad to square
    grid_size = int(np.ceil(np.sqrt(num_tokens)))
    padded = np.zeros(grid_size * grid_size)
    padded[:num_tokens] = attention_1d
    return padded.reshape(grid_size, grid_size)


class EnhancedAttentionExtractor:
    """Main attention extractor for MedGemma"""
    
    def __init__(self, config: Optional[AttentionExtractionConfig] = None):
        self.config = config or AttentionExtractionConfig()
        
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
        Extract attention with fallback support
        
        Returns:
            Tuple of (attention_map, token_indices, method_used)
        """
        # If we have image and prompt, use direct extraction
        if pil_image is not None and prompt is not None:
            try:
                attention_map, metadata = extract_token_to_image_attention(
                    model, processor, pil_image, prompt, self.config
                )
                return attention_map, [], 'token_to_image'
            except Exception as e:
                logger.warning(f"Token-to-image extraction failed: {e}")
        
        # Fallback to uniform attention
        return np.ones((32, 32)) / (32 * 32), [], 'uniform'


def visualize_attention_on_image(
    image: Image.Image,
    attention_map: np.ndarray,
    title: str = "MedGemma Attention",
    cmap: str = 'hot',
    alpha: float = 0.5,
    save_path: Optional[str] = None
) -> Image.Image:
    """Visualize attention overlaid on image"""
    
    # Resize attention to image size
    attention_resized = cv2.resize(
        attention_map,
        (image.width, image.height),
        interpolation=cv2.INTER_LINEAR
    )
    
    # Normalize
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
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    # Convert to PIL
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    ncols, nrows = fig.canvas.get_width_height()
    plt.close(fig)
    
    img_array = buf.reshape(nrows, ncols, 3)
    return Image.fromarray(img_array)


# Backward compatibility
def load_model_enhanced(model_id=MEDGEMMA_ID, load_in_8bit=False, load_in_4bit=False, **kwargs):
    """Legacy function for compatibility"""
    dtype = torch.float16 if (load_in_8bit or load_in_4bit) else torch.bfloat16
    return load_medgemma(dtype=dtype, **kwargs)