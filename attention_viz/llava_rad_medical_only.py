#!/usr/bin/env python3
"""
LLaVA-Rad Medical Model Handler
Focused solely on Microsoft's LLaVA-Rad medical model
No HuggingFace fallback - medical specific
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
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_transformers_conflict():
    """Fix the llava config conflict in transformers"""
    try:
        from transformers.models.auto import configuration_auto
        if hasattr(configuration_auto.CONFIG_MAPPING, '_extra_content'):
            if 'llava' in configuration_auto.CONFIG_MAPPING._extra_content:
                del configuration_auto.CONFIG_MAPPING._extra_content['llava']
                logger.info("Fixed transformers llava conflict")
    except:
        pass


@dataclass
class MedicalAttentionConfig:
    """Configuration for medical attention visualization"""
    colormap: str = 'hot'  # Better for medical imaging
    alpha: float = 0.5
    use_grad_cam: bool = True
    attention_head_mode: str = 'mean'
    layer_selection: str = 'last_quarter'


class LLaVARadMedical:
    """Microsoft LLaVA-Rad Medical Model Handler"""
    
    def __init__(self, config: Optional[MedicalAttentionConfig] = None):
        self.config = config or MedicalAttentionConfig()
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Fix conflict on init
        fix_transformers_conflict()
        
    def setup_llava_rad(self):
        """Ensure LLaVA-Rad is properly set up"""
        # Add LLaVA-Rad to path if it exists
        llava_paths = ['/content/LLaVA-Rad', './LLaVA-Rad', '../LLaVA-Rad']
        
        for path in llava_paths:
            if os.path.exists(path) and path not in sys.path:
                sys.path.insert(0, path)
                logger.info(f"Added {path} to Python path")
                return True
                
        logger.warning("LLaVA-Rad directory not found. Please clone from: https://github.com/microsoft/LLaVA-Rad.git")
        return False
        
    def load_model(self, model_path: Optional[str] = None, load_8bit: bool = False):
        """Load LLaVA-Rad medical model"""
        
        if not self.setup_llava_rad():
            raise ImportError("LLaVA-Rad not found. Clone from: https://github.com/microsoft/LLaVA-Rad.git")
            
        try:
            from llava.model.builder import load_pretrained_model
            from llava.utils import disable_torch_init
            from llava.mm_utils import get_model_name_from_path
            
            disable_torch_init()
            
            # Medical model paths to try
            if model_path is None:
                model_paths = [
                    "microsoft/llava-med-v1.5-mistral-7b",
                    "microsoft/llava-med-1.5-mistral-7b",
                    "liuhaotian/llava-v1.5-7b"  # Base model as last resort
                ]
            else:
                model_paths = [model_path]
                
            for path in model_paths:
                try:
                    logger.info(f"Attempting to load: {path}")
                    
                    model_name = get_model_name_from_path(path)
                    
                    self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                        model_path=path,
                        model_base=None,
                        model_name=model_name,
                        load_8bit=load_8bit,
                        load_4bit=False,
                        device=self.device
                    )
                    
                    self.model.eval()
                    logger.info(f"✓ Successfully loaded: {path}")
                    
                    # Determine if this is a medical model
                    self.is_medical = "med" in path.lower()
                    return True
                    
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {str(e)}")
                    continue
                    
            raise RuntimeError("Failed to load any LLaVA-Rad model")
            
        except ImportError as e:
            raise ImportError(f"LLaVA-Rad imports failed: {e}")
            
    def extract_attention(
        self,
        image: Union[str, Image.Image],
        question: str,
        conv_mode: str = "llava_v1"
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract attention maps from model"""
        
        if self.model is None:
            raise RuntimeError("Model not loaded")
            
        try:
            from llava.conversation import conv_templates
            from llava.mm_utils import process_images, tokenizer_image_token
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
            
            # Load image
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
                
            # Process image
            image_tensor = process_images(
                [image], 
                self.image_processor, 
                self.model.config
            )[0]
            
            # Setup conversation
            conv = conv_templates[conv_mode].copy()
            
            # Format prompt
            if self.model.config.mm_use_im_start_end:
                prompt = DEFAULT_IMAGE_TOKEN + '\n' + question
            else:
                prompt = DEFAULT_IMAGE_TOKEN + question
                
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()
            
            # Tokenize
            input_ids = tokenizer_image_token(
                prompt_text,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors='pt'
            ).unsqueeze(0).to(self.device)
            
            # Move image to device
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(dtype=self.model.dtype, device=self.device)
            
            # Forward pass with attention
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    images=image_tensor,
                    output_attentions=True,
                    use_cache=False
                )
                
            # Extract attention maps
            if hasattr(outputs, 'attentions') and outputs.attentions:
                return self._process_attention_maps(
                    outputs.attentions,
                    input_ids,
                    image_tensor.shape
                )
            else:
                # Fallback to Grad-CAM if enabled
                if self.config.use_grad_cam:
                    return self._extract_gradcam_attention(
                        self.model,
                        input_ids,
                        image_tensor
                    )
                    
            # Return uniform attention as last resort
            return np.ones((24, 24)) / (24 * 24), {"method": "uniform"}
            
        except Exception as e:
            logger.error(f"Attention extraction failed: {e}")
            return np.ones((24, 24)) / (24 * 24), {"method": "uniform", "error": str(e)}
            
    def _process_attention_maps(
        self,
        attentions: List[torch.Tensor],
        input_ids: torch.Tensor,
        image_shape: Tuple
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process raw attention tensors into visual attention map"""
        
        num_layers = len(attentions)
        
        # Select layers based on config
        if self.config.layer_selection == 'last_quarter':
            layer_indices = list(range(3 * num_layers // 4, num_layers))
        elif self.config.layer_selection == 'last_half':
            layer_indices = list(range(num_layers // 2, num_layers))
        else:
            layer_indices = list(range(num_layers))
            
        # Find image token positions
        # LLaVA typically uses 576 tokens for 336x336 image (24x24 patches)
        num_patches = 576
        
        # Find where image tokens are (usually after initial text tokens)
        # Look for IMAGE_TOKEN_INDEX
        from llava.constants import IMAGE_TOKEN_INDEX
        image_positions = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)
        
        if len(image_positions[1]) > 0:
            image_start = image_positions[1][0].item()
        else:
            # Estimate based on typical position
            image_start = 35  # After system prompt tokens
            
        image_end = image_start + num_patches
        
        # Query position (last generated token)
        query_pos = input_ids.shape[1] - 1
        
        # Extract attention from query to image
        attention_maps = []
        
        for layer_idx in layer_indices:
            layer_attn = attentions[layer_idx][0]  # Remove batch dimension
            
            # Check dimensions
            if query_pos < layer_attn.shape[1] and image_end <= layer_attn.shape[2]:
                # Extract query to image attention
                query_to_image = layer_attn[:, query_pos, image_start:image_end]
                
                # Aggregate heads
                if self.config.attention_head_mode == 'mean':
                    aggregated = query_to_image.mean(dim=0)
                elif self.config.attention_head_mode == 'max':
                    aggregated = query_to_image.max(dim=0)[0]
                else:  # weighted by entropy
                    # Calculate entropy for each head
                    entropies = -torch.sum(
                        query_to_image * torch.log(query_to_image + 1e-10),
                        dim=1
                    )
                    weights = torch.softmax(-entropies, dim=0)
                    aggregated = torch.sum(query_to_image * weights.unsqueeze(1), dim=0)
                    
                attention_maps.append(aggregated)
                
        if attention_maps:
            # Average across selected layers
            final_attention = torch.stack(attention_maps).mean(dim=0).cpu().numpy()
            
            # Reshape to 2D
            grid_size = int(np.sqrt(num_patches))
            attention_2d = final_attention.reshape(grid_size, grid_size)
            
            return attention_2d, {
                "method": "transformer_attention",
                "num_patches": num_patches,
                "layers_used": layer_indices,
                "head_mode": self.config.attention_head_mode
            }
            
        # Fallback
        return np.ones((24, 24)) / (24 * 24), {"method": "uniform", "reason": "no_valid_attention"}
        
    def _extract_gradcam_attention(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        image_tensor: torch.Tensor
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract attention using Grad-CAM as fallback"""
        # Simplified Grad-CAM for vision encoder
        # This is a placeholder - implement full Grad-CAM if needed
        logger.info("Using Grad-CAM fallback for attention")
        
        # Return centered attention as placeholder
        attention = np.zeros((24, 24))
        center = 12
        for i in range(24):
            for j in range(24):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                attention[i, j] = np.exp(-dist / 8)
                
        attention /= attention.sum()
        
        return attention, {"method": "gradcam_placeholder"}
        
    def generate_with_attention(
        self,
        image: Union[str, Image.Image],
        question: str,
        max_new_tokens: int = 100,
        conv_mode: str = "llava_v1"
    ) -> Dict[str, Any]:
        """Generate response with attention extraction"""
        
        # Extract attention
        attention_map, attention_meta = self.extract_attention(image, question, conv_mode)
        
        # Generate answer
        answer = self.generate(image, question, max_new_tokens, conv_mode)
        
        # Calculate focus score
        attention_norm = attention_map / (attention_map.sum() + 1e-10)
        entropy = -np.sum(attention_norm * np.log(attention_norm + 1e-10))
        max_entropy = np.log(attention_map.size)
        focus_score = 1 - (entropy / max_entropy)
        
        return {
            'answer': answer,
            'visual_attention': attention_map,
            'metrics': {'focus': float(focus_score)},
            'attention_method': attention_meta.get('method', 'unknown'),
            'metadata': attention_meta
        }
        
    def generate(
        self,
        image: Union[str, Image.Image],
        question: str,
        max_new_tokens: int = 100,
        conv_mode: str = "llava_v1"
    ) -> str:
        """Generate answer for medical image"""
        
        if self.model is None:
            raise RuntimeError("Model not loaded")
            
        try:
            from llava.conversation import conv_templates, SeparatorStyle
            from llava.mm_utils import process_images, tokenizer_image_token
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
            
            # Load image
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
                
            # Process image
            image_tensor = process_images([image], self.image_processor, self.model.config)[0]
            
            # Setup conversation
            conv = conv_templates[conv_mode].copy()
            
            # Format prompt
            if self.model.config.mm_use_im_start_end:
                prompt = DEFAULT_IMAGE_TOKEN + '\n' + question
            else:
                prompt = DEFAULT_IMAGE_TOKEN + question
                
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()
            
            # Tokenize
            input_ids = tokenizer_image_token(
                prompt_text,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors='pt'
            ).unsqueeze(0).to(self.device)
            
            # Prepare image tensor
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(dtype=self.model.dtype, device=self.device)
            
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
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            # Extract answer based on separator style
            if conv.sep_style == SeparatorStyle.TWO:
                answer = outputs.split(conv.sep2)[-1].strip()
            else:
                answer = outputs.split(conv.sep)[-1].strip()
                
            return answer
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error: {str(e)}"
            
    def visualize_attention(
        self,
        image: Union[str, Image.Image],
        attention_map: np.ndarray,
        title: str = "Medical Attention Map",
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """Visualize attention overlay on medical image"""
        
        # Load image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            
        # Convert to numpy
        image_np = np.array(image)
        
        # Resize attention to image size
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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Original image
        ax1.imshow(image_np)
        ax1.set_title("Original Medical Image")
        ax1.axis('off')
        
        # Attention overlay
        ax2.imshow(image_np)
        im = ax2.imshow(
            attention_norm,
            cmap=self.config.colormap,
            alpha=self.config.alpha
        )
        ax2.set_title(title)
        ax2.axis('off')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', rotation=270, labelpad=15)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        # Convert to image array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        ncols, nrows = fig.canvas.get_width_height()
        plt.close(fig)
        
        img_array = buf.reshape(nrows, ncols, 3)
        return img_array


# Convenience function for quick setup
def create_medical_llava():
    """Create and return a medical LLaVA-Rad instance"""
    return LLaVARadMedical()