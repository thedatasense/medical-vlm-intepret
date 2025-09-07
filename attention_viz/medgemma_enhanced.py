#!/usr/bin/env python3
"""
Enhanced MedGemma Attention Extraction with Advanced Techniques
Incorporates suggestions for improved attention extraction and analysis
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


def load_model_enhanced(model_id: str = "google/paligemma-3b-mix-224",
                       load_in_8bit: bool = True,
                       load_in_4bit: bool = False) -> Tuple[Any, Any]:
    """
    Enhanced model loading function for MedGemma with quantization support.
    
    Returns:
        Tuple of (model, processor)
    """
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError:
        raise ImportError("Please install transformers and bitsandbytes: pip install transformers bitsandbytes")
    
    # Configure quantization
    bnb_config = None
    if load_in_8bit or load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    # Load model with eager attention for attention extraction
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="eager"  # Enable attention outputs
    )
    
    model.eval()
    logger.info(f"PaliGemma model loaded: {model_id}")
    
    return model, processor


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
    
    def __post_init__(self):
        if self.fallback_chain is None:
            self.fallback_chain = ['cross_attention', 'gradcam', 'uniform']


class EnhancedAttentionExtractor:
    """Enhanced attention extraction with multiple techniques"""
    
    def __init__(self, config: Optional[AttentionExtractionConfig] = None):
        self.config = config or AttentionExtractionConfig()
        self.cache = {} if self.config.cache_enabled else None
        
    def extract_token_conditioned_attention_robust(self, 
                                                  model: Any,
                                                  processor: Any,
                                                  gen_result: Dict,
                                                  target_words: List[str],
                                                  pil_image: Optional[Image.Image] = None,
                                                  prompt: Optional[str] = None) -> Tuple[np.ndarray, List[int], str]:
        """
        Enhanced token-conditioned attention extraction with fallback chain
        
        Returns:
            - attention_grid: numpy array of shape (H, W) with attention weights
            - token_indices: list of token indices that were used
            - method: string indicating which method was used
        """
        # Try cache first
        if self.cache is not None:
            cache_key = self._get_cache_key(prompt, target_words, str(model))
            if cache_key in self.cache:
                logger.info("Using cached attention result")
                return self.cache[cache_key]
        
        # Initialize fallback methods
        extraction_methods = []
        
        if 'cross_attention' in self.config.fallback_chain:
            extraction_methods.append(
                ('cross_attention', self._extract_cross_attention)
            )
        
        if 'gradcam' in self.config.fallback_chain:
            extraction_methods.append(
                ('gradcam', self._extract_gradcam_attention)
            )
        
        if 'uniform' in self.config.fallback_chain:
            extraction_methods.append(
                ('uniform', self._extract_uniform_attention)
            )
        
        # Try each method in order
        for method_name, method_func in extraction_methods:
            try:
                logger.info(f"Trying {method_name} attention extraction...")
                
                result = method_func(
                    model, processor, gen_result, target_words,
                    pil_image, prompt
                )
                
                if result is not None and result[0] is not None:
                    # Cache successful result
                    if self.cache is not None and cache_key:
                        self.cache[cache_key] = result
                    
                    logger.info(f"✓ Successfully extracted attention using {method_name}")
                    return result
                    
            except Exception as e:
                logger.warning(f"{method_name} extraction failed: {e}")
                if self.config.debug_mode:
                    import traceback
                    traceback.print_exc()
        
        # Final fallback
        logger.warning("All methods failed, using uniform attention")
        return self._extract_uniform_attention(
            model, processor, gen_result, target_words, pil_image, prompt
        )
    
    def _extract_cross_attention(self, model, processor, gen_result, target_words,
                                pil_image, prompt) -> Tuple[np.ndarray, List[int], str]:
        """Extract cross-attention between text and image tokens"""
        # Try to use single forward pass approach first
        if pil_image is not None and prompt is not None:
            try:
                # Prepare inputs for forward pass
                if hasattr(processor, 'apply_chat_template'):
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }]
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = processor(text=text, images=pil_image, return_tensors="pt")
                else:
                    inputs = processor(
                        text=f"<image>{prompt}",
                        images=pil_image,
                        return_tensors="pt"
                    )
                
                # Move to device
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Single forward pass with attention
                with torch.no_grad():
                    outputs = model(**inputs, output_attentions=True)
                
                # Extract attentions from outputs
                if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    # Use the attentions from forward pass
                    all_attentions = outputs.attentions
                    input_ids = inputs['input_ids'][0]
                    
                    # Find target token indices
                    token_indices = self._find_token_indices_robust(
                        input_ids, target_words, processor.tokenizer
                    )
                    
                    if not token_indices:
                        logger.warning("No target tokens found in forward pass")
                    else:
                        # Extract visual token positions
                        visual_token_range = self._identify_visual_tokens_medgemma(
                            input_ids, processor
                        )
                        
                        if visual_token_range is not None:
                            # Process attention layers
                            attention_maps = []
                            n_layers = len(all_attentions)
                            
                            # Use last quarter of layers by default
                            layer_range = range(3 * n_layers // 4, n_layers)
                            
                            for layer_idx in layer_range:
                                layer_attn = all_attentions[layer_idx]
                                
                                # Handle different attention formats
                                if isinstance(layer_attn, tuple):
                                    layer_attn = layer_attn[0]
                                
                                if layer_attn is None:
                                    logger.warning(f"Layer {layer_idx} attention is None")
                                    continue
                                    
                                if isinstance(layer_attn, torch.Tensor):
                                    layer_attn = layer_attn.cpu().numpy()
                                
                                # Extract attention for the last token (as query) to visual tokens
                                if hasattr(layer_attn, 'shape') and len(layer_attn.shape) == 4:  # [batch, heads, seq, seq]
                                    # Aggregate heads based on config
                                    if self.config.attention_head_reduction == 'mean':
                                        layer_attn = layer_attn.mean(axis=1)
                                    elif self.config.attention_head_reduction == 'max':
                                        layer_attn = layer_attn.max(axis=1)
                                    elif self.config.attention_head_reduction == 'entropy_weighted':
                                        # Apply entropy weighting
                                        weights = []
                                        for h in range(layer_attn.shape[1]):
                                            head_att = layer_attn[0, h]
                                            entropy = -np.sum(head_att * np.log(head_att + 1e-10))
                                            weights.append(1.0 / (1.0 + entropy))
                                        weights = np.array(weights) / np.sum(weights)
                                        
                                        weighted_attention = np.zeros_like(layer_attn[0, 0])
                                        for h, w in enumerate(weights):
                                            weighted_attention += w * layer_attn[0, h]
                                        layer_attn = weighted_attention.reshape(1, *weighted_attention.shape)
                                    
                                    layer_attn = layer_attn[0]  # Remove batch
                                
                                elif hasattr(layer_attn, 'shape') and len(layer_attn.shape) == 3:  # [batch, seq, seq]
                                    layer_attn = layer_attn[0]
                                elif hasattr(layer_attn, 'shape'):
                                    logger.warning(f"Unexpected attention shape at layer {layer_idx}: {layer_attn.shape}")
                                    continue
                                
                                # Extract attention from last token to visual tokens
                                visual_attention = layer_attn[-1, visual_token_range.start:visual_token_range.stop]
                                
                                # Ensure we have valid attention values
                                if len(visual_attention.shape) == 0 or visual_attention.size == 0:
                                    logger.warning(f"Empty attention at layer {layer_idx}")
                                    continue
                                    
                                attention_maps.append(visual_attention)
                            
                            if attention_maps:
                                # Aggregate across layers
                                aggregated = np.mean(attention_maps, axis=0)
                                
                                # Reshape to image grid
                                grid_size = int(np.sqrt(len(visual_token_range)))
                                if grid_size * grid_size != len(visual_token_range):
                                    # Handle non-square grids
                                    # Try to find reasonable dimensions
                                    n_tokens = len(visual_token_range)
                                    h = int(np.sqrt(n_tokens))
                                    w = n_tokens // h
                                    
                                    # Ensure h*w >= n_tokens
                                    while h * w < n_tokens:
                                        w += 1
                                    
                                    # Trim or pad if necessary
                                    if h * w > n_tokens:
                                        # Pad with zeros
                                        padded = np.zeros(h * w)
                                        padded[:n_tokens] = aggregated
                                        attention_grid = padded.reshape(h, w)
                                    else:
                                        attention_grid = aggregated.reshape(h, w)
                                else:
                                    attention_grid = aggregated.reshape(grid_size, grid_size)
                                
                                # Normalize
                                attention_grid = attention_grid / (attention_grid.sum() + 1e-10)
                                
                                return attention_grid, token_indices, 'cross_attention_forward'
                
            except Exception as e:
                logger.warning(f"Forward pass attention extraction failed: {e}")
        
        # Fall back to using generation result attentions
        if not hasattr(gen_result, 'attentions') or gen_result.attentions is None:
            return None
        
        # Get generated tokens
        generated_ids = gen_result.sequences[0] if hasattr(gen_result, 'sequences') else gen_result.get('sequences', [[]])[0]
        
        # Find target token indices with improved matching
        token_indices = self._find_token_indices_robust(
            generated_ids, target_words, processor.tokenizer
        )
        
        if not token_indices:
            logger.warning("No target tokens found")
            return None
        
        # Extract visual token positions
        visual_token_range = self._identify_visual_tokens_medgemma(
            generated_ids, processor
        )
        
        if visual_token_range is None:
            logger.warning("Could not identify visual tokens")
            return None
        
        # Process attention layers
        attention_maps = []
        n_layers = len(gen_result.attentions[0])  # First token's attention
        
        # Use last quarter of layers by default
        layer_range = range(3 * n_layers // 4, n_layers)
        
        for layer_idx in layer_range:
            layer_attention = self._extract_layer_cross_attention(
                gen_result.attentions,
                token_indices,
                visual_token_range,
                layer_idx
            )
            
            if layer_attention is not None:
                attention_maps.append(layer_attention)
        
        if not attention_maps:
            return None
        
        # Aggregate across layers
        aggregated = np.mean(attention_maps, axis=0)
        
        # Reshape to image grid
        grid_size = int(np.sqrt(len(visual_token_range)))
        if grid_size * grid_size != len(visual_token_range):
            # Handle non-square grids
            n_tokens = len(visual_token_range)
            h = int(np.sqrt(n_tokens))
            w = n_tokens // h
            
            # Ensure h*w >= n_tokens
            while h * w < n_tokens:
                w += 1
            
            # Trim or pad if necessary
            if h * w > n_tokens:
                # Pad with zeros
                padded = np.zeros(h * w)
                padded[:n_tokens] = aggregated
                attention_grid = padded.reshape(h, w)
            else:
                attention_grid = aggregated.reshape(h, w)
        else:
            attention_grid = aggregated.reshape(grid_size, grid_size)
        
        # Normalize
        attention_grid = attention_grid / (attention_grid.sum() + 1e-10)
        
        return attention_grid, token_indices, 'cross_attention'
    
    def _extract_gradcam_attention(self, model, processor, gen_result, target_words,
                                  pil_image, prompt) -> Tuple[np.ndarray, List[int], str]:
        """Extract attention using Grad-CAM technique"""
        if pil_image is None:
            return None
        
        # Prepare inputs (respect chat template if available)
        if hasattr(processor, 'apply_chat_template'):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=text, images=pil_image, return_tensors="pt")
        else:
            inputs = processor(
                text=f"<image>{prompt}",
                images=pil_image,
                return_tensors="pt"
            )
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Find target token indices
        target_ids = []
        tokenizer = processor.tokenizer
        for word in target_words:
            tokens = tokenizer.encode(word, add_special_tokens=False)
            target_ids.extend(tokens)
        
        if not target_ids:
            return None
        
        # Hook for gradients
        activations = []
        gradients = []
        
        def forward_hook(module, input, output):
            activations.append(output)
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        # Register hooks on visual encoder
        visual_encoder = None
        
        # Try different ways to access vision model
        if hasattr(model, 'vision_model'):
            visual_encoder = model.vision_model
        elif hasattr(model, 'vision_encoder'):
            visual_encoder = model.vision_encoder
        elif hasattr(model, 'model') and hasattr(model.model, 'vision_model'):
            visual_encoder = model.model.vision_model
        elif hasattr(model, 'get_vision_tower'):
            try:
                visual_encoder = model.get_vision_tower()
            except:
                pass
        
        if visual_encoder is None:
            logger.warning("Could not find vision encoder for Grad-CAM")
            return None
        
        # Target the last layer of visual encoder
        target_layer = None
        candidate_patterns = [
            'encoder.layers',
            'blocks',
            'layers',
            'transformer.resblocks',
            'vision_model.encoder.layers'
        ]
        
        for name, module in visual_encoder.named_modules():
            for pattern in candidate_patterns:
                if pattern in name and 'layernorm' not in name.lower() and 'ln' not in name.lower():
                    target_layer = module
        
        if target_layer is None:
            # Try to get the last substantial layer
            layers = []
            for name, module in visual_encoder.named_modules():
                if len(list(module.children())) > 0 and not any(skip in name.lower() for skip in ['head', 'norm', 'pool']):
                    layers.append((name, module))
            
            if layers:
                target_layer = layers[-1][1]
        
        if target_layer is None:
            logger.warning("Could not find suitable layer for Grad-CAM hooks")
            return None
        
        # Register hooks
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)
        
        try:
            # Forward pass
            model.zero_grad()
            outputs = model(**inputs)
            
            # Get logits for target tokens
            logits = outputs.logits[0]  # Remove batch dimension
            target_logits = logits[-1, target_ids].mean()  # Last position, target tokens
            
            # Backward pass
            target_logits.backward()
            
            # Get gradients and activations
            if activations and gradients:
                grad = gradients[0].mean(dim=0)  # Average over heads
                act = activations[0].mean(dim=0)
                
                # Grad-CAM combination
                weights = grad.mean(dim=-1, keepdim=True)  # Global average pooling
                cam = (weights * act).sum(dim=0)  # Weighted combination
                
                # Extract visual tokens
                visual_range = self._identify_visual_tokens_medgemma(
                    inputs['input_ids'][0], processor
                )
                
                if visual_range:
                    visual_cam = cam[visual_range.start:visual_range.stop]
                    
                    # Reshape to grid
                    grid_size = int(np.sqrt(len(visual_cam)))
                    if grid_size * grid_size == len(visual_cam):
                        cam_grid = visual_cam.reshape(grid_size, grid_size)
                    else:
                        # Handle non-square
                        h = min(32, len(visual_cam) // 32)
                        w = len(visual_cam) // h
                        cam_grid = visual_cam[:h*w].reshape(h, w)
                    
                    # Convert to numpy and normalize
                    cam_grid = cam_grid.detach().cpu().numpy()
                    cam_grid = np.maximum(cam_grid, 0)  # ReLU
                    cam_grid = cam_grid / (cam_grid.sum() + 1e-10)
                    
                    return cam_grid, target_ids, 'gradcam'
            
        finally:
            # Remove hooks
            forward_handle.remove()
            backward_handle.remove()
        
        return None
    
    def _extract_uniform_attention(self, model, processor, gen_result, target_words,
                                  pil_image, prompt) -> Tuple[np.ndarray, List[int], str]:
        """Create uniform attention as final fallback"""
        # Try to infer grid size
        grid_size = 32  # Default for MedGemma
        
        if hasattr(processor, 'image_seq_length'):
            # MedGemma specific
            total_patches = processor.image_seq_length
            grid_size = int(np.sqrt(total_patches))
        
        # Create uniform grid
        attention_grid = np.ones((grid_size, grid_size))
        attention_grid = attention_grid / attention_grid.sum()
        
        return attention_grid, [], 'uniform'
    
    def _find_token_indices_robust(self, token_ids: torch.Tensor, 
                                  target_words: List[str], 
                                  tokenizer: Any) -> List[int]:
        """Robustly find token indices for target words"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()
        
        found_indices = []
        
        for word in target_words:
            # Try exact match first
            word_tokens = tokenizer.encode(word, add_special_tokens=False)
            
            # Search for token sequence
            for i in range(len(token_ids) - len(word_tokens) + 1):
                if np.array_equal(token_ids[i:i+len(word_tokens)], word_tokens):
                    found_indices.extend(range(i, i + len(word_tokens)))
                    break
            
            # Try with space prefix
            if not found_indices:
                word_tokens = tokenizer.encode(f" {word}", add_special_tokens=False)
                for i in range(len(token_ids) - len(word_tokens) + 1):
                    if np.array_equal(token_ids[i:i+len(word_tokens)], word_tokens):
                        found_indices.extend(range(i, i + len(word_tokens)))
                        break
            
            # Try lowercase
            if not found_indices:
                word_tokens = tokenizer.encode(word.lower(), add_special_tokens=False)
                for i in range(len(token_ids) - len(word_tokens) + 1):
                    if np.array_equal(token_ids[i:i+len(word_tokens)], word_tokens):
                        found_indices.extend(range(i, i + len(word_tokens)))
                        break
        
        return list(set(found_indices))  # Remove duplicates
    
    def _identify_visual_tokens_medgemma(self, input_ids: torch.Tensor, 
                                        processor: Any) -> Optional[range]:
        """Identify visual token positions for MedGemma"""
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.cpu().numpy()
        
        # Look for image token
        image_token_id = processor.tokenizer.convert_tokens_to_ids('<image>')
        image_positions = np.where(input_ids == image_token_id)[0]
        
        if len(image_positions) > 0:
            # Visual tokens typically follow the image token
            start_pos = image_positions[0] + 1
            
            # Get expected number of visual tokens
            if hasattr(processor, 'image_seq_length'):
                num_visual = processor.image_seq_length
            else:
                # Estimate based on common sizes
                num_visual = 256  # 16x16 patches
            
            # Ensure we don't exceed sequence length
            end_pos = min(start_pos + num_visual, len(input_ids))
            
            return range(start_pos, end_pos)
        
        # Fallback: estimate position
        seq_len = len(input_ids)
        if hasattr(processor, 'image_seq_length'):
            num_visual = processor.image_seq_length
            # Visual tokens often in the middle
            start = max(1, (seq_len - num_visual) // 2)
            end = min(seq_len, start + num_visual)
            return range(start, end)
        
        return None
    
    def _extract_layer_cross_attention(self, all_attentions: List, 
                                      token_indices: List[int],
                                      visual_range: range,
                                      layer_idx: int) -> Optional[np.ndarray]:
        """Extract cross attention for a specific layer"""
        if layer_idx >= len(all_attentions[0]):
            return None
        
        # Collect attention from text tokens to visual tokens
        attention_to_visual = []
        
        for token_idx in token_indices:
            if token_idx < len(all_attentions):
                token_attention = all_attentions[token_idx][layer_idx]
                
                # Handle different attention formats
                if isinstance(token_attention, tuple):
                    token_attention = token_attention[0]
                
                if isinstance(token_attention, torch.Tensor):
                    token_attention = token_attention.cpu().numpy()
                
                # Extract attention to visual tokens
                if len(token_attention.shape) == 4:  # [batch, heads, seq, seq]
                    # Average over heads based on config
                    if self.config.attention_head_reduction == 'mean':
                        token_attention = token_attention.mean(axis=1)
                    elif self.config.attention_head_reduction == 'max':
                        token_attention = token_attention.max(axis=1)
                    elif self.config.attention_head_reduction == 'entropy_weighted':
                        # Weight by inverse entropy (more focused heads get more weight)
                        weights = []
                        for h in range(token_attention.shape[1]):
                            head_att = token_attention[0, h]
                            entropy = -np.sum(head_att * np.log(head_att + 1e-10))
                            weights.append(1.0 / (1.0 + entropy))
                        weights = np.array(weights) / np.sum(weights)
                        
                        weighted_attention = np.zeros_like(token_attention[0, 0])
                        for h, w in enumerate(weights):
                            weighted_attention += w * token_attention[0, h]
                        token_attention = weighted_attention.reshape(1, *weighted_attention.shape)
                    
                    token_attention = token_attention[0]  # Remove batch
                
                elif len(token_attention.shape) == 3:  # [batch, seq, seq]
                    token_attention = token_attention[0]
                
                # Get attention to visual tokens
                visual_attention = token_attention[visual_range.start:visual_range.stop]
                attention_to_visual.append(visual_attention)
        
        if not attention_to_visual:
            return None
        
        # Aggregate across tokens based on config
        if self.config.multi_token_aggregation == 'mean':
            aggregated = np.mean(attention_to_visual, axis=0)
        elif self.config.multi_token_aggregation == 'max':
            aggregated = np.max(attention_to_visual, axis=0)
        elif self.config.multi_token_aggregation == 'weighted':
            # Weight by token importance (uniform for now)
            weights = np.ones(len(attention_to_visual)) / len(attention_to_visual)
            aggregated = np.average(attention_to_visual, axis=0, weights=weights)
        else:
            aggregated = np.mean(attention_to_visual, axis=0)
        
        return aggregated
    
    def _get_cache_key(self, prompt: str, target_words: List[str], 
                      model_id: str) -> str:
        """Generate cache key for attention results"""
        content = f"{prompt}_{','.join(target_words)}_{model_id}"
        return hashlib.md5(content.encode()).hexdigest()


class AttentionVisualizationEnhanced:
    """Enhanced visualization methods for attention maps"""
    
    @staticmethod
    def create_attention_overlay(image: Image.Image,
                               attention_map: np.ndarray,
                               alpha: float = 0.5,
                               colormap: str = 'jet',
                               percentile_clip: Tuple[int, int] = (2, 98),
                               use_body_mask: bool = True) -> Image.Image:
        """Create enhanced attention overlay with medical-specific features"""
        # Convert image to numpy
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        H, W = img_array.shape[:2]
        
        # Resize attention to image size
        attention_resized = cv2.resize(
            attention_map.astype(np.float32), 
            (W, H), 
            interpolation=cv2.INTER_CUBIC
        )
        
        # Apply body mask if requested
        if use_body_mask:
            try:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                body_mask = create_tight_body_mask(gray)
                attention_resized = attention_resized * (body_mask / 255.0)
            except Exception as e:
                logger.warning(f"Body mask creation failed: {e}")
        
        # Apply percentile clipping
        valid_attention = attention_resized[attention_resized > 0]
        if len(valid_attention) > 0:
            low, high = np.percentile(valid_attention, percentile_clip)
            attention_resized = np.clip(attention_resized, low, high)
            attention_resized = (attention_resized - low) / (high - low + 1e-8)
        
        # Create visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(img_array)
        
        # Apply colormap
        if colormap == 'medical':
            colormap = 'bone'  # Better for medical images
        
        plt.imshow(attention_resized, alpha=alpha, cmap=colormap)
        plt.axis('off')
        
        # Convert to PIL Image
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        
        return Image.open(buf)
    
    @staticmethod
    def create_multi_method_comparison(image: Image.Image,
                                     attention_results: Dict[str, np.ndarray],
                                     save_path: Optional[str] = None) -> plt.Figure:
        """Compare attention maps from different extraction methods"""
        n_methods = len(attention_results)
        fig, axes = plt.subplots(1, n_methods + 1, figsize=(5 * (n_methods + 1), 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Attention maps
        for idx, (method_name, attention_map) in enumerate(attention_results.items()):
            ax = axes[idx + 1]
            
            # Create overlay
            img_array = np.array(image)
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            
            H, W = img_array.shape[:2]
            attention_resized = cv2.resize(
                attention_map.astype(np.float32), 
                (W, H), 
                interpolation=cv2.INTER_CUBIC
            )
            
            # Normalize
            attention_resized = (attention_resized - attention_resized.min()) / (
                attention_resized.max() - attention_resized.min() + 1e-8
            )
            
            ax.imshow(img_array)
            im = ax.imshow(attention_resized, alpha=0.5, cmap='jet')
            ax.set_title(f'{method_name.title()} Method')
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Method comparison saved to {save_path}")
        
        return fig


def create_tight_body_mask(gray: np.ndarray) -> np.ndarray:
    """Create a tight mask for the body region in chest X-ray"""
    # This is a simplified version - you can enhance based on your needs
    
    # Apply threshold to get rough body region
    _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    
    # Remove small components
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find largest connected component (body)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        return mask
    
    return binary


class RobustAttentionAnalyzer:
    """Analyze attention patterns for robustness"""
    
    def __init__(self, extractor: EnhancedAttentionExtractor):
        self.extractor = extractor
    
    def analyze_prompt_sensitivity(self, model, processor, image: Image.Image,
                                  prompt_variations: List[str],
                                  target_words: List[str]) -> Dict[str, Any]:
        """Analyze how attention changes across prompt variations"""
        results = []
        attention_maps = []
        
        for prompt in prompt_variations:
            # Generate with model
            inputs = processor(
                text=f"<image>{prompt}",
                images=image,
                return_tensors="pt"
            )
            
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                gen_result = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    output_attentions=True,
                    return_dict_in_generate=True
                )
            
            # Extract attention
            attention_grid, token_indices, method = self.extractor.extract_token_conditioned_attention_robust(
                model, processor, gen_result, target_words, image, prompt
            )
            
            attention_maps.append(attention_grid)
            
            # Decode answer
            answer = processor.tokenizer.decode(
                gen_result.sequences[0], 
                skip_special_tokens=True
            ).split("Assistant:")[-1].strip()
            
            results.append({
                'prompt': prompt,
                'answer': answer,
                'attention_method': method,
                'token_indices': token_indices
            })
        
        # Compute consistency metrics
        try:
            from llava_rad_enhanced import AttentionMetrics
        except ImportError:
            # If running in a different context, try absolute import
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from llava_rad_enhanced import AttentionMetrics
        consistency = AttentionMetrics.calculate_consistency(attention_maps)
        
        # Compute pairwise JS divergences
        n_prompts = len(prompt_variations)
        js_matrix = np.zeros((n_prompts, n_prompts))
        
        for i in range(n_prompts):
            for j in range(i+1, n_prompts):
                js_div = jensenshannon(
                    attention_maps[i].flatten(),
                    attention_maps[j].flatten()
                )
                js_matrix[i, j] = js_matrix[j, i] = js_div
        
        return {
            'results': results,
            'attention_maps': attention_maps,
            'consistency_score': consistency,
            'js_divergence_matrix': js_matrix,
            'mean_js_divergence': js_matrix[js_matrix > 0].mean(),
            'max_js_divergence': js_matrix.max()
        }


def demo_enhanced_extraction():
    """Demo script for enhanced attention extraction"""
    from transformers import AutoProcessor, AutoModelForCausalLM
    
    # Load model
    model_id = "google/paligemma-3b-ft-med400k-224"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Configure extractor
    config = AttentionExtractionConfig(
        use_gradcam=True,
        attention_head_reduction='entropy_weighted',
        multi_token_aggregation='weighted',
        debug_mode=True
    )
    
    extractor = EnhancedAttentionExtractor(config)
    
    # Test image
    test_image = Image.open("/path/to/chest_xray.jpg")
    prompt = "Is there evidence of pneumonia?"
    target_words = ["pneumonia", "consolidation", "opacity"]
    
    # Generate
    inputs = processor(
        text=f"<image>{prompt}",
        images=test_image,
        return_tensors="pt"
    )
    
    gen_result = model.generate(
        **inputs,
        max_new_tokens=50,
        output_attentions=True,
        return_dict_in_generate=True
    )
    
    # Extract attention
    attention_grid, token_indices, method = extractor.extract_token_conditioned_attention_robust(
        model, processor, gen_result, target_words, test_image, prompt
    )
    
    print(f"Extraction method used: {method}")
    print(f"Attention shape: {attention_grid.shape}")
    print(f"Token indices found: {token_indices}")
    
    # Visualize
    overlay = AttentionVisualizationEnhanced.create_attention_overlay(
        test_image, attention_grid, use_body_mask=True
    )
    overlay.save("enhanced_attention_overlay.png")
    
    # Test robustness
    analyzer = RobustAttentionAnalyzer(extractor)
    
    prompt_variations = [
        "Is there evidence of pneumonia?",
        "Do you see any signs of pneumonia?",
        "Can you identify pneumonia in this X-ray?",
        "Is pneumonia present?"
    ]
    
    robustness_results = analyzer.analyze_prompt_sensitivity(
        model, processor, test_image, prompt_variations, target_words
    )
    
    print(f"\nRobustness Analysis:")
    print(f"Consistency score: {robustness_results['consistency_score']:.3f}")
    print(f"Mean JS divergence: {robustness_results['mean_js_divergence']:.3f}")
    print(f"Max JS divergence: {robustness_results['max_js_divergence']:.3f}")


if __name__ == "__main__":
    demo_enhanced_extraction()
