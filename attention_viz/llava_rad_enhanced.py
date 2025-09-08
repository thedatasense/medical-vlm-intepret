#!/usr/bin/env python3
"""
Enhanced LLaVA-Rad Attention Visualizer using Official Microsoft Repository
Uses the actual LLaVA-Rad model for medical imaging
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple, Any, Union
import logging
import cv2
from dataclasses import dataclass
import hashlib
from scipy.spatial.distance import jensenshannon

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add LLaVA-Rad to path if available
if os.path.exists('/content/LLaVA-Rad'):
    sys.path.insert(0, '/content/LLaVA-Rad')


@dataclass
class AttentionConfig:
    """Configuration for attention visualization"""
    colormap: str = 'jet'
    alpha: float = 0.5
    percentile_clip: Tuple[int, int] = (2, 98)
    use_medical_colormap: bool = True
    multi_head_mode: str = 'mean'


def load_llava_rad_official(
    model_path: str = "microsoft/llava-med-v1.5-mistral-7b",
    model_base: Optional[str] = None,
    load_8bit: bool = False,
    load_4bit: bool = False,
    device: str = "cuda"
):
    """
    Load LLaVA-Rad using the official Microsoft repository
    
    Args:
        model_path: Path to model weights or HuggingFace model ID
        model_base: Base model name (optional)
        load_8bit: Use 8-bit quantization
        load_4bit: Use 4-bit quantization
        device: Device to use
        
    Returns:
        Tuple of (tokenizer, model, image_processor, context_len)
    """
    try:
        # Import from LLaVA-Rad
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
        
        # Disable torch init for faster loading
        disable_torch_init()
        
        logger.info(f"Loading LLaVA-Rad model: {model_path}")
        
        # Load the model
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=model_base,
            model_name="llava-rad",
            load_8bit=load_8bit,
            load_4bit=load_4bit,
            device=device
        )
        
        model.eval()
        logger.info("✓ LLaVA-Rad loaded successfully")
        
        return tokenizer, model, image_processor, context_len
        
    except ImportError:
        raise ImportError(
            "LLaVA-Rad not found. Please run:\n"
            "!git clone https://github.com/microsoft/LLaVA-Rad.git\n"
            "!cd LLaVA-Rad && pip install -e ."
        )


def process_llava_rad_image(image_processor, image: Image.Image, model):
    """Process image for LLaVA-Rad"""
    try:
        from llava.mm_utils import process_images
        
        # Process image
        image_tensor = process_images(
            [image], 
            image_processor, 
            model.config
        )
        
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
            
        return image_tensor
        
    except ImportError:
        # Fallback to basic processing
        return image_processor.preprocess(
            image, 
            return_tensors='pt'
        )['pixel_values'][0]


def extract_llava_rad_attention(
    model,
    tokenizer,
    image_processor,
    image: Image.Image,
    prompt: str,
    conv_mode: str = "llava_v1"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extract attention from LLaVA-Rad model
    
    Args:
        model: LLaVA-Rad model
        tokenizer: Tokenizer
        image_processor: Image processor
        image: PIL Image
        prompt: Text prompt
        conv_mode: Conversation mode
        
    Returns:
        Tuple of (attention_map, metadata)
    """
    try:
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        
        # Setup conversation
        conv = conv_templates[conv_mode].copy()
        
        # Prepare prompt with image token
        if model.config.mm_use_im_start_end:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + prompt
        
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        
        # Tokenize
        input_ids = tokenizer_image_token(
            prompt_text,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0)
        
        # Process image
        image_tensor = process_llava_rad_image(image_processor, image, model)
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Move to device
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        image_tensor = image_tensor.to(dtype=model.dtype, device=device)
        
        # Forward pass with attention
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                images=image_tensor,
                output_attentions=True,
                use_cache=False
            )
        
        # Extract attention
        if hasattr(outputs, 'attentions') and outputs.attentions:
            # Get attention from last layers
            num_layers = len(outputs.attentions)
            layer_indices = list(range(3 * num_layers // 4, num_layers))
            
            # Find image token positions
            # LLaVA-Rad typically uses 576 tokens for 336x336 image
            num_patches = 576
            image_start = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1][0].item() if IMAGE_TOKEN_INDEX in input_ids else 10
            image_end = image_start + num_patches
            
            # Query position (last token)
            query_pos = input_ids.shape[1] - 1
            
            # Extract attention maps
            attention_maps = []
            for layer_idx in layer_indices:
                layer_attn = outputs.attentions[layer_idx][0]  # Remove batch
                
                if query_pos < layer_attn.shape[1] and image_end <= layer_attn.shape[2]:
                    # Query to image attention
                    query_to_img = layer_attn[:, query_pos, image_start:image_end]
                    avg_attn = query_to_img.mean(dim=0)
                    attention_maps.append(avg_attn)
            
            if attention_maps:
                # Average and reshape
                final_attention = torch.stack(attention_maps).mean(dim=0).cpu().numpy()
                grid_size = int(np.sqrt(num_patches))
                attention_2d = final_attention.reshape(grid_size, grid_size)
                
                return attention_2d, {
                    "method": "llava_rad_attention",
                    "num_patches": num_patches,
                    "layers_used": layer_indices
                }
        
        # Fallback to uniform attention
        logger.warning("No attention extracted, using uniform")
        return np.ones((24, 24)) / (24 * 24), {"method": "uniform"}
        
    except Exception as e:
        logger.error(f"Attention extraction error: {e}")
        return np.ones((24, 24)) / (24 * 24), {"method": "uniform", "error": str(e)}


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


class EnhancedLLaVARadVisualizer:
    """LLaVA-Rad visualizer using official Microsoft implementation"""
    
    def __init__(self, config: Optional[AttentionConfig] = None):
        self.config = config or AttentionConfig()
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None
        self.cache = {}
        
    def load_model(
        self, 
        model_path: str = "microsoft/llava-med-v1.5-mistral-7b",
        model_base: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """Load LLaVA-Rad model from official repository"""
        
        # Load using official loader
        self.tokenizer, self.model, self.image_processor, self.context_len = load_llava_rad_official(
            model_path=model_path,
            model_base=model_base,
            load_8bit=load_in_8bit,
            load_4bit=load_in_4bit
        )
        
    def generate_with_attention(
        self,
        image_path: Union[str, Image.Image],
        prompt: str,
        max_new_tokens: int = 100,
        use_cache: bool = True,
        conv_mode: str = "llava_v1"
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
            attention_map, metadata = extract_llava_rad_attention(
                self.model,
                self.tokenizer,
                self.image_processor,
                image,
                prompt,
                conv_mode
            )
            
            # Generate answer
            from llava.conversation import conv_templates
            from llava.mm_utils import tokenizer_image_token
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
            
            # Setup conversation for generation
            conv = conv_templates[conv_mode].copy()
            
            if self.model.config.mm_use_im_start_end:
                prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
            else:
                prompt = DEFAULT_IMAGE_TOKEN + prompt
                
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()
            
            # Tokenize
            input_ids = tokenizer_image_token(
                prompt_text,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors='pt'
            ).unsqueeze(0)
            
            # Process image
            image_tensor = process_llava_rad_image(self.image_processor, image, self.model)
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            # Move to device
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            image_tensor = image_tensor.to(dtype=self.model.dtype, device=device)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True
                )
            
            # Decode
            outputs = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            
            # Extract answer
            if conv.sep_style == 'two':
                answer = outputs.split(conv.sep2)[-1].strip()
            else:
                answer = outputs.split(conv.sep)[-1].strip()
            
            # Calculate metrics
            metrics = AttentionMetrics.calculate_focus_score(attention_map)
            
            result = {
                'answer': answer,
                'visual_attention': attention_map,
                'metrics': metrics,
                'attention_method': metadata.get('method', 'llava_rad'),
                'metadata': metadata
            }
            
            # Cache
            if use_cache:
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
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


# Backward compatibility - if LLaVA-Rad not available, use HF models
def load_llava_rad(dtype=torch.float16, eight_bit=True, device_map="auto"):
    """Fallback loader for when LLaVA-Rad repo is not available"""
    logger.warning("Using fallback HuggingFace LLaVA loader. For best results, install LLaVA-Rad.")
    
    from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
    
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    
    if eight_bit:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
    else:
        quant_config = None
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device_map,
        quantization_config=quant_config,
        low_cpu_mem_usage=True
    )
    
    model.eval()
    return model, processor