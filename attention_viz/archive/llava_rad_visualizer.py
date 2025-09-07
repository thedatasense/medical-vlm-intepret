#!/usr/bin/env python3
"""
LLaVA-Rad Attention Visualizer and Analysis Platform
Microsoft's LLaVA-Rad model for medical imaging
Developed by SAIL Lab - University of New Haven
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, Dict, List, Tuple, Any
import logging
import warnings
from pathlib import Path
import json
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
import gc
from datetime import datetime
from scipy.spatial.distance import jensenshannon

# Matplotlib 3D surface support (importing triggers 3D toolkit registration)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class LLaVARadVisualizer:
    """
    Comprehensive attention visualization for Microsoft's LLaVA-Rad model
    Adapted from MedGemma visualizer for consistency
    """
    
    def __init__(self, device=None):
        """Initialize LLaVA-Rad visualizer"""
        self.model = None
        self.processor = None
        self.device = device if device else self.setup_gpu()
        # Default assumptions for LLaVA image/patch sizing
        self.image_size: int = 336
        self.patch_size: int = 14
        logger.info(f"LLaVA-Rad Visualizer initialized on {self.device}")
    
    def setup_gpu(self, min_free_gb: float = 15.0) -> str:
        """Setup GPU with sufficient memory"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            return "cpu"
        
        # Check all available GPUs
        best_gpu = 0
        max_free_memory = 0
        
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            
            # Get GPU memory info
            total_memory = torch.cuda.get_device_properties(i).total_memory
            allocated_memory = torch.cuda.memory_allocated(i)
            cached_memory = torch.cuda.memory_reserved(i)
            free_memory = total_memory - cached_memory
            free_gb = free_memory / (1024**3)
            
            logger.info(f"GPU {i}: {free_gb:.1f}GB free / {total_memory/(1024**3):.1f}GB total")
            
            if free_gb > max_free_memory and free_gb >= min_free_gb:
                max_free_memory = free_gb
                best_gpu = i
        
        if max_free_memory < min_free_gb:
            raise RuntimeError(f"No GPU with {min_free_gb}GB+ free memory available")
        
        device = f"cuda:{best_gpu}"
        torch.cuda.set_device(best_gpu)
        logger.info(f"Selected GPU {best_gpu} with {max_free_memory:.1f}GB free")
        
        return device
    
    def load_model(self, model_id="microsoft/llava-rad", model_base="lmsys/vicuna-7b-v1.5", 
                   load_in_8bit=False, load_in_4bit=False):
        """Load LLaVA-Rad model and processor using the LLaVA library
        
        Microsoft's LLaVA-Rad is a medical vision-language model specifically trained for radiology.
        """
        logger.info(f"Loading LLaVA-Rad model: {model_id}")
        
        try:
            # Import LLaVA components
            try:
                from llava.model.builder import load_pretrained_model
                from llava.utils import disable_torch_init
                from llava.conversation import conv_templates
                from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
                from llava.constants import IMAGE_TOKEN_INDEX
                
                # Disable torch init for faster loading
                disable_torch_init()
                
                # Load model using LLaVA's method (official llava-rad path)
                model_name = "llavarad"
                self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                    model_path=model_id,
                    model_base=model_base,
                    model_name=model_name,
                    load_8bit=load_in_8bit,
                    load_4bit=load_in_4bit,
                )
                
                # Store LLaVA specific attributes
                self.conv_mode = "v1"
                self.conv_templates = conv_templates
                self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
                self.tokenizer_image_token = tokenizer_image_token
                self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
                
                # Create processor wrapper for compatibility
                class LLaVAProcessorWrapper:
                    def __init__(self, tokenizer, image_processor, parent):
                        self.tokenizer = tokenizer
                        self.image_processor = image_processor
                        self.parent = parent
                    
                    def __call__(self, text=None, images=None, **kwargs):
                        result = {}
                        if text:
                            # Use LLaVA's tokenizer_image_token for proper handling
                            if hasattr(self.parent, 'tokenizer_image_token'):
                                input_ids = self.parent.tokenizer_image_token(
                                    text, self.tokenizer, self.parent.IMAGE_TOKEN_INDEX, return_tensors='pt'
                                )
                                result['input_ids'] = input_ids.unsqueeze(0)
                            else:
                                result.update(self.tokenizer(text, return_tensors="pt", **kwargs))
                        if images is not None:
                            # Process image using LLaVA's image processor
                            image_tensor = self.image_processor.preprocess(images, return_tensors='pt')['pixel_values']
                            if image_tensor.dtype != torch.float16:
                                image_tensor = image_tensor.half()
                            result['pixel_values'] = image_tensor
                        return result
                    
                    def decode(self, *args, **kwargs):
                        return self.tokenizer.decode(*args, **kwargs)
                    
                    def apply_chat_template(self, messages, **kwargs):
                        # Use LLaVA conversation templates
                        conv = self.parent.conv_templates[self.parent.conv_mode].copy()
                        for msg in messages:
                            if msg['role'] == 'user':
                                content_text = ""
                                has_image = False
                                for content in msg['content']:
                                    if content['type'] == 'text':
                                        content_text += content['text']
                                    elif content['type'] == 'image':
                                        has_image = True
                                if has_image and '<image>' not in content_text:
                                    content_text = '<image>\n' + content_text
                                conv.append_message(conv.roles[0], content_text)
                            elif msg['role'] == 'assistant':
                                conv.append_message(conv.roles[1], msg.get('content', None))
                        return conv.get_prompt()
                
                self.processor = LLaVAProcessorWrapper(self.tokenizer, self.image_processor, self)

                logger.info("✓ LLaVA-Rad model loaded successfully using LLaVA library")

                # Try infer image/patch sizes from processor when possible
                try:
                    size = getattr(self.image_processor, 'size', None)
                    if isinstance(size, dict) and 'shortest_edge' in size:
                        self.image_size = int(size['shortest_edge'])
                    elif isinstance(size, int):
                        self.image_size = int(size)
                    # LLaVA typically uses ViT-L/14
                    self.patch_size = 14
                except Exception:
                    pass
                
            except ImportError as e:
                logger.warning(f"LLaVA library not found: {e}")
                logger.info("Falling back to HF/alternative loading...")
                raise
                
        except Exception as e:
            logger.warning(f"Failed to load with LLaVA library: {e}")
            # Fallback to transformers approach (or alternative for llava-rad)
            if isinstance(model_id, str) and ("llava-rad" in model_id.lower() or model_id == "microsoft/llava-rad"):
                logger.info("HF loader incompatible with microsoft/llava-rad; using alternative HF LLaVA model.")
                return self.load_alternative_model()
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
            
            # Configure quantization if requested
            bnb_config = None
            if load_in_8bit or load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=load_in_8bit,
                    load_in_4bit=load_in_4bit,
                    bnb_4bit_compute_dtype=torch.float16 if load_in_4bit else None,
                    bnb_4bit_use_double_quant=load_in_4bit
                )
            
            # Try to load processor/tokenizer
            try:
                self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                logger.info("✓ Processor loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load processor directly: {e}")
                # Fallback: load tokenizer separately
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, legacy=False)
                except:
                    # Try with legacy=True if that fails
                    self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, legacy=True)
                logger.info("✓ Tokenizer loaded")
                
                # For LLaVA-Rad, we'll create a simple processor wrapper
                class ProcessorWrapper:
                    def __init__(self, tokenizer):
                        self.tokenizer = tokenizer
                        # Add special tokens if needed
                        if not hasattr(self.tokenizer, 'image_token'):
                            self.tokenizer.add_special_tokens({'additional_special_tokens': ['<image>']})
                    
                    def __call__(self, text=None, images=None, **kwargs):
                        if text and isinstance(text, str):
                            # Ensure text is properly formatted
                            if '<image>' not in text and images is not None:
                                text = '<image>\n' + text
                            return self.tokenizer(text, return_tensors="pt", **kwargs)
                        elif text:
                            return self.tokenizer(text, return_tensors="pt", **kwargs)
                        return {}
                    
                    def decode(self, *args, **kwargs):
                        return self.tokenizer.decode(*args, **kwargs)
                    
                    def apply_chat_template(self, *args, **kwargs):
                        if hasattr(self.tokenizer, 'apply_chat_template'):
                            return self.tokenizer.apply_chat_template(*args, **kwargs)
                        # Fallback for simple conversation formatting
                        if args and isinstance(args[0], list) and len(args[0]) > 0:
                            messages = args[0]
                            text = ""
                            for msg in messages:
                                if msg.get('role') == 'user' and 'content' in msg:
                                    for content in msg['content']:
                                        if content.get('type') == 'text':
                                            text += content['text'] + "\n"
                                        elif content.get('type') == 'image':
                                            text = '<image>\n' + text
                            return text.strip()
                        return str(args[0]) if args else ""
                
                self.processor = ProcessorWrapper(self.tokenizer)
            
            # Load model with automatic device placement
            model_kwargs = {
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True
            }
            
            if bnb_config:
                model_kwargs["quantization_config"] = bnb_config
            elif self.device != "cpu":
                model_kwargs["device_map"] = {"": self.device}
            
            # Add attention implementation for all models
            model_kwargs["attn_implementation"] = "eager"  # Required for attention extraction
            model_kwargs["trust_remote_code"] = True  # Required for LLaVA-Rad
            
            # Try different loading approaches for LLaVA-Rad
            try:
                # First try: Load as AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    **model_kwargs
                )
                logger.info("✓ Model loaded as AutoModelForCausalLM")
            except Exception as e1:
                logger.warning(f"AutoModelForCausalLM failed: {e1}")
                try:
                    # Second try: Load as LlavaForConditionalGeneration
                    from transformers import LlavaForConditionalGeneration
                    self.model = LlavaForConditionalGeneration.from_pretrained(
                        model_id,
                        **model_kwargs
                    )
                    logger.info("✓ Model loaded as LlavaForConditionalGeneration")
                except Exception as e2:
                    logger.warning(f"LlavaForConditionalGeneration failed: {e2}")
                    # Final try: Load with AutoModel
                    from transformers import AutoModel
                    model_kwargs.pop("attn_implementation", None)  # Remove if not supported
                    self.model = AutoModel.from_pretrained(
                        model_id,
                        **model_kwargs
                    )
                    logger.info("✓ Model loaded as AutoModel")
            
            # Enable attention output
            self.model.config.output_attentions = True
            self.model.eval()
            
            logger.info(f"✓ LLaVA medical model loaded successfully")
            logger.info(f"Model device: {next(self.model.parameters()).device}")
            logger.info(f"Model dtype: {next(self.model.parameters()).dtype}")
            
            # Report memory usage
            if self.device != "cpu":
                allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
                reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
                logger.info(f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
            
            return self.model, self.processor
            
        except Exception as e:
            logger.error(f"Failed to load LLaVA-Rad model: {e}")
            logger.error(f"Error details: {str(e)}")
            # Try alternative approach
            logger.info("Attempting alternative loading method...")
            return self.load_alternative_model()
    
    def load_alternative_model(self):
        """Load alternative vision-language model"""
        # Use llava-hf model as alternative since microsoft/llava-rad has loading issues
        model_id = "llava-hf/llava-1.5-7b-hf"
        logger.info(f"Loading alternative model: {model_id}")
        
        try:
            from transformers import LlavaForConditionalGeneration, AutoProcessor
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_id)
            
            # Load model with eager attention for attention extraction
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map={"": self.device} if self.device != "cpu" else None,
                attn_implementation="eager"  # Required for attention extraction
            )
        except Exception as e:
            logger.error(f"Failed to load alternative model: {e}")
            # Try another alternative - BiomedCLIP
            logger.info("Trying BiomedCLIP as final alternative...")
            model_id = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            from transformers import AutoModel, AutoProcessor
            
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id)
        
        self.model.config.output_attentions = True
        self.model.eval()
        
        logger.info(f"✓ Alternative model loaded: {model_id}")
        return self.model, self.processor
    
    def extract_attention_weights(self, outputs, layer_indices=None):
        """Extract attention weights from model outputs"""
        if not hasattr(outputs, 'attentions') or outputs.attentions is None:
            logger.warning("No attention weights in outputs")
            return None
        
        attentions = outputs.attentions
        
        # Default to middle and late layers for LLaVA
        if layer_indices is None:
            num_layers = len(attentions)
            # Sample evenly across layers
            layer_indices = [num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1]
            layer_indices = [i for i in layer_indices if i < num_layers]
        
        attention_maps = []
        for layer_idx in layer_indices:
            if layer_idx < len(attentions):
                # Get attention for this layer
                layer_attention = attentions[layer_idx]
                
                if isinstance(layer_attention, tuple):
                    layer_attention = layer_attention[0]
                
                # Average across heads and batch
                if layer_attention.dim() == 4:  # [batch, heads, seq, seq]
                    avg_attention = layer_attention.mean(dim=[0, 1])
                elif layer_attention.dim() == 3:  # [heads, seq, seq]
                    avg_attention = layer_attention.mean(dim=0)
                else:
                    avg_attention = layer_attention
                
                attention_maps.append(avg_attention.cpu().numpy())
        
        if not attention_maps:
            return None
        
        # Average across selected layers
        combined_attention = np.mean(attention_maps, axis=0)
        return combined_attention
    
    def identify_visual_tokens(self, input_ids: torch.Tensor, image_token_id: int) -> Optional[range]:
        """Precisely identify visual token positions based on image token anchor.

        Args:
            input_ids: Token ids tensor of shape [seq] or [1, seq].
            image_token_id: The special token id marking the <image> position.

        Returns:
            A range covering the visual token span, or None if not identifiable.
        """
        if input_ids is None:
            return None
        ids = input_ids
        if ids.dim() == 2:
            ids = ids[0]
        pos = (ids == image_token_id).nonzero(as_tuple=False).squeeze(-1)
        if pos.numel() == 1:
            start_pos = pos[0].item() + 1
            num_patches_side = self.image_size // self.patch_size
            num_visual_tokens = int(num_patches_side * num_patches_side)
            return range(start_pos, start_pos + num_visual_tokens)
        elif pos.numel() > 1:
            # Multiple image tokens (rare); return positions directly
            return range(pos[0].item(), pos[-1].item() + 1)
        return None

    def extract_visual_attention(self, attention_matrix: np.ndarray,
                                 input_ids: Optional[torch.Tensor] = None,
                                 image_token_id: Optional[int] = None,
                                 prompt_len: Optional[int] = None) -> Optional[np.ndarray]:
        """Extract visual attention from full attention matrix using image-token anchored span.

        Falls back to a centered heuristic when precise positions are unavailable.
        """
        if attention_matrix is None:
            return None

        num_patches_side = self.image_size // self.patch_size
        num_visual_tokens = num_patches_side * num_patches_side

        seq_len = attention_matrix.shape[0]

        # Try precise identification first
        visual_span = None
        if image_token_id is not None and input_ids is not None:
            visual_span = self.identify_visual_tokens(input_ids, image_token_id)

        if visual_span is not None:
            start_idx = max(0, visual_span.start)
            end_idx = min(seq_len, visual_span.stop)
            # Queries: focus on last few tokens after prompt
            q_start = 0 if prompt_len is None else max(prompt_len, seq_len - 3)
            q_rows = slice(q_start, seq_len)
            sub = attention_matrix[q_rows, start_idx:end_idx]
            # Mean over query rows -> vector over visual tokens
            vec = sub.mean(axis=0)
            # Normalize length to grid size and reshape
            if vec.shape[0] != num_visual_tokens:
                if vec.shape[0] < num_visual_tokens:
                    vec = np.pad(vec, (0, num_visual_tokens - vec.shape[0]), mode='constant')
                else:
                    vec = vec[:num_visual_tokens]
            grid = vec.reshape(num_patches_side, num_patches_side)
            return grid

        # Fallback: centered heuristic
        if seq_len > num_visual_tokens:
            start_idx = max(0, (seq_len - num_visual_tokens) // 2)
            end_idx = min(seq_len, start_idx + num_visual_tokens)
            block = attention_matrix[start_idx:end_idx, start_idx:end_idx]
            vec = block.mean(axis=1)
            if vec.shape[0] < num_visual_tokens:
                vec = np.pad(vec, (0, num_visual_tokens - vec.shape[0]), mode='constant')
            else:
                vec = vec[:num_visual_tokens]
            return vec.reshape(num_patches_side, num_patches_side)

        return None
    
    def generate_with_attention(self, image, question, max_new_tokens=100):
        """Generate answer with attention tracking"""
        if self.model is None or self.processor is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Check if we're using LLaVA library
        if hasattr(self, 'conv_templates'):
            # Use LLaVA-specific generation
            query = f"<image>\nQuestion: {question}\nAnswer with only 'yes' or 'no'."
            
            # Create conversation
            conv = self.conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], query)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            # Process image
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            if image_tensor.dtype != torch.float16:
                image_tensor = image_tensor.half()
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Tokenize
            input_ids = self.tokenizer_image_token(
                prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0).to(self.device)
            
            # Set up stopping criteria
            stopping_criteria = self.KeywordsStoppingCriteria(["</s>"], self.tokenizer, input_ids)
            
            # Generate with attention
            with torch.no_grad():
                # Enable attention output
                self.model.config.output_attentions = True
                self.model.config.return_dict_in_generate = True
                
                outputs = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    output_attentions=True,
                    return_dict_in_generate=True
                )
            
            # Decode answer
            generated_ids = outputs.sequences[0][input_ids.shape[1]:]
            answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        else:
            # Fallback to standard generation
            prompt = f"<image>\nQuestion: {question}\nAnswer with only 'yes' or 'no'."
            
            # Process inputs
            try:
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Question: {question}\nAnswer with only 'yes' or 'no'."},
                            {"type": "image"},
                        ],
                    },
                ]
                if hasattr(self.processor, 'apply_chat_template'):
                    prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt",
                    padding=True
                )
            except Exception as e:
                logger.warning(f"Using fallback input processing: {e}")
                inputs = self.processor(
                    text=prompt,
                    return_tensors="pt",
                    padding=True
                )
        
            # Move to device
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v 
                     for k, v in inputs.items()}
            
            # Generate with attention
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    output_attentions=True,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Decode answer
            generated_ids = outputs.sequences[0][len(inputs['input_ids'][0]):]
            answer = self.processor.decode(generated_ids, skip_special_tokens=True)
            # For downstream computations
            input_ids = inputs['input_ids']
        
        # Extract attention (averaged across heads/layers for a quick map)
        attention_weights = None
        if hasattr(outputs, 'attentions') and outputs.attentions:
            # LLaVA generate returns attentions per generation step
            # We'll focus on the first generation step
            first_step_attentions = outputs.attentions[0] if outputs.attentions else None
            if first_step_attentions:
                attention_weights = self.extract_attention_weights(
                    type('', (), {'attentions': first_step_attentions})()
                )
        
        # Extract visual attention
        visual_attention = None
        if attention_weights is not None:
            # Try identify visual tokens using input ids where possible
            img_token_id = None
            if hasattr(self, 'IMAGE_TOKEN_INDEX'):
                img_token_id = self.IMAGE_TOKEN_INDEX
            # For HF path, try tokenizer attributes if set
            if img_token_id is None and hasattr(self, 'processor'):
                img_token_id = getattr(self.processor, 'image_token_id', None)
                if img_token_id is None and hasattr(getattr(self.processor, 'tokenizer', None), 'image_token_id'):
                    img_token_id = self.processor.tokenizer.image_token_id

            # Determine prompt length
            pl = None
            if 'input_ids' in locals():
                pl = input_ids.shape[1] if input_ids.dim() == 2 else input_ids.shape[0]

            visual_attention = self.extract_visual_attention(
                attention_weights,
                input_ids=input_ids if 'input_ids' in locals() else None,
                image_token_id=img_token_id,
                prompt_len=pl
            )

        # Multi-head attention grids (per-head) if available
        head_visual_attentions: List[np.ndarray] = []
        try:
            if hasattr(outputs, 'attentions') and outputs.attentions:
                step0 = outputs.attentions[0]
                if isinstance(step0, (list, tuple)) and len(step0) > 0:
                    last_layer = step0[-1]  # [batch, heads, q_len, k_len] typical
                    if isinstance(last_layer, tuple):
                        last_layer = last_layer[0]
                    if last_layer.dim() == 4:
                        # Determine image token span again
                        img_token_id = None
                        if hasattr(self, 'IMAGE_TOKEN_INDEX'):
                            img_token_id = self.IMAGE_TOKEN_INDEX
                        if img_token_id is None and hasattr(self, 'processor'):
                            img_token_id = getattr(self.processor, 'image_token_id', None)
                            if img_token_id is None and hasattr(getattr(self.processor, 'tokenizer', None), 'image_token_id'):
                                img_token_id = self.processor.tokenizer.image_token_id

                        pl = input_ids.shape[1] if input_ids.dim() == 2 else input_ids.shape[0]
                        q_start = max(pl, last_layer.shape[2] - 3)
                        num_patches_side = self.image_size // self.patch_size
                        num_visual_tokens = num_patches_side * num_patches_side

                        # Identify visual span from input ids
                        vis_span = None
                        if img_token_id is not None:
                            vis_span = self.identify_visual_tokens(input_ids, img_token_id)
                        # Fallback span: center block
                        if vis_span is None:
                            k_len = last_layer.shape[3]
                            start_idx = max(0, (k_len - num_visual_tokens) // 2)
                            vis_span = range(start_idx, min(k_len, start_idx + num_visual_tokens))

                        for h in range(last_layer.shape[1]):
                            # Mean over last few query rows
                            sub = last_layer[0, h, q_start:last_layer.shape[2], vis_span.start:vis_span.stop].mean(dim=0)
                            vec = sub.detach().float().cpu().numpy()
                            if vec.shape[0] != num_visual_tokens:
                                if vec.shape[0] < num_visual_tokens:
                                    vec = np.pad(vec, (0, num_visual_tokens - vec.shape[0]), mode='constant')
                                else:
                                    vec = vec[:num_visual_tokens]
                            grid = vec.reshape(num_patches_side, num_patches_side)
                            # Normalize to [0,1]
                            grid = grid - grid.min()
                            if grid.max() > 0:
                                grid = grid / grid.max()
                            head_visual_attentions.append(grid)
        except Exception as _e:
            # Non-fatal
            pass

        # Build ROI/body mask in grid space for metrics
        roi_mask = None
        metrics: Dict[str, Any] = {}
        try:
            if visual_attention is not None:
                # Construct body mask from image -> resize to grid
                if isinstance(image, str):
                    pil_img = Image.open(image).convert('RGB')
                else:
                    pil_img = image
                gray = np.array(pil_img.convert('L'))
                base = cv2.GaussianBlur(gray, (0, 0), 2)
                _, m = cv2.threshold(base, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                m = 255 - m
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
                m = cv2.erode(m, np.ones((9, 9), np.uint8))
                gh, gw = visual_attention.shape
                roi_mask = cv2.resize(m, (gw, gh), interpolation=cv2.INTER_AREA)
                roi_mask = (roi_mask > 0).astype(np.uint8)

                # Focus score and ROI focus
                fs = AttentionMetrics.calculate_focus_score(visual_attention, roi_mask)
                if isinstance(fs, dict):
                    metrics['focus'] = fs.get('focus')
                    metrics['roi_focus'] = fs.get('roi_focus')
                else:
                    metrics['focus'] = fs

                # Border fraction
                vv = visual_attention.astype(np.float64)
                total = vv.sum() + 1e-10
                border = (vv[0, :].sum() + vv[-1, :].sum() + vv[:, 0].sum() + vv[:, -1].sum())
                metrics['border_fraction'] = float(border / total)

                # Inside body ratio
                inside = float((vv * (roi_mask > 0)).sum() / total)
                metrics['inside_body_ratio'] = inside
        except Exception:
            pass

        return {
            'answer': answer,
            'attention_weights': attention_weights,
            'visual_attention': visual_attention,
            'head_visual_attentions': head_visual_attentions if head_visual_attentions else None,
            'metrics': metrics if metrics else None,
            'outputs': outputs
        }
    
    def visualize_attention(self, image, visual_attention, question=None, answer=None, 
                          save_path=None, show_plot=True, cmap: str = 'hot'):
        """Visualize attention on image"""
        if visual_attention is None:
            logger.warning("No visual attention to visualize")
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Attention heatmap
        im = axes[1].imshow(visual_attention, cmap=cmap, interpolation='nearest')
        axes[1].set_title('Attention Heatmap')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        # Overlay
        axes[2].imshow(image)
        
        # Resize attention to match image size
        attention_resized = np.array(Image.fromarray(
            (visual_attention * 255).astype(np.uint8)
        ).resize(image.size, Image.BILINEAR))
        
        axes[2].imshow(attention_resized, alpha=0.5, cmap=cmap)
        axes[2].set_title('Attention Overlay')
        axes[2].axis('off')
        
        # Add question and answer as title
        if question and answer:
            fig.suptitle(f'Q: {question[:100]}...\nA: {answer[:100]}...', 
                        fontsize=10, y=1.05)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig

class AttentionVisualizer:
    """Enhanced attention visualization utilities (multi-head and 3D surfaces)."""

    def __init__(self):
        self.colormaps = {
            'medical': 'bone',
            'heat': 'hot',
            'cool': 'cool',
            'diverging': 'RdBu_r',
            'perceptual': 'viridis'
        }

    def visualize_single_head(self, ax, head_attention: np.ndarray, image: Image.Image, title: str = "Head"):
        # Resize attention to image size and overlay
        att = head_attention
        att = att - att.min()
        if att.max() > 0:
            att = att / att.max()
        att_img = Image.fromarray((att * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
        ax.imshow(image)
        ax.imshow(att_img, alpha=0.5, cmap='hot')
        ax.set_title(title)
        ax.axis('off')

    def create_multi_head_visualization(self, attention_heads: List[np.ndarray], image: Image.Image):
        n_heads = len(attention_heads)
        rows = 2
        cols = (n_heads + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
        axes = np.atleast_2d(axes)
        for idx, (ax, head_attn) in enumerate(zip(axes.flat, attention_heads)):
            self.visualize_single_head(ax, head_attn, image, f"Head {idx}")
        # Hide any unused axes
        for ax in axes.flat[n_heads:]:
            ax.axis('off')
        fig.tight_layout()
        return fig

    def create_3d_attention_surface(self, attention_map: np.ndarray):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(range(attention_map.shape[1]), range(attention_map.shape[0]))
        ax.plot_surface(X, Y, attention_map, cmap='viridis')
        ax.set_title('3D Attention Surface')
        return fig

class AttentionMetrics:
    @staticmethod
    def calculate_focus_score(attention_map: np.ndarray, roi_mask: Optional[np.ndarray] = None):
        """Calculate focus score and optional ROI focus."""
        p = attention_map.astype(np.float64)
        p = p / (p.sum() + 1e-10)
        entropy = -np.sum(p * np.log(p + 1e-10))
        max_entropy = np.log(p.size + 1e-10)
        focus = 1.0 - (entropy / (max_entropy + 1e-12))
        if roi_mask is not None:
            roi_attention = p[roi_mask > 0].sum()
            return {'focus': float(focus), 'roi_focus': float(roi_attention)}
        return float(focus)

    @staticmethod
    def calculate_consistency(attention_maps: List[np.ndarray]) -> float:
        """Mean pairwise consistency (1 - JS divergence) across maps."""
        if not attention_maps:
            return 0.0
        n = len(attention_maps)
        if n == 1:
            return 1.0
        # Flatten and normalize
        flats = []
        for m in attention_maps:
            v = m.astype(np.float64).flatten()
            v = v / (v.sum() + 1e-10)
            flats.append(v)
        sim_sum = 0.0
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                jsd = jensenshannon(flats[i], flats[j])
                sim = 1.0 - float(jsd)
                sim_sum += sim
                pairs += 1
        return sim_sum / max(pairs, 1)
    
    def compute_attention_metrics(self, visual_attention):
        """Compute attention quality metrics"""
        if visual_attention is None:
            return {}
        
        # Normalize attention
        attention_norm = visual_attention / (visual_attention.sum() + 1e-10)
        
        # Compute metrics
        metrics = {
            'max_attention': float(visual_attention.max()),
            'mean_attention': float(visual_attention.mean()),
            'std_attention': float(visual_attention.std()),
            'entropy': float(-np.sum(attention_norm * np.log(attention_norm + 1e-10))),
            'sparsity': float((visual_attention > visual_attention.mean()).sum() / visual_attention.size),
            'focus_score': float(visual_attention.max() / (visual_attention.mean() + 1e-10))
        }
        
        # Regional distribution (quarters)
        h, w = visual_attention.shape
        metrics['top_half'] = float(visual_attention[:h//2, :].sum() / visual_attention.sum())
        metrics['bottom_half'] = float(visual_attention[h//2:, :].sum() / visual_attention.sum())
        metrics['left_half'] = float(visual_attention[:, :w//2].sum() / visual_attention.sum())
        metrics['right_half'] = float(visual_attention[:, w//2:].sum() / visual_attention.sum())
        
        return metrics
    
    def analyze_image(self, image_path, question, save_visualizations=False, output_dir=None):
        """Complete analysis pipeline for a single image"""
        logger.info(f"Analyzing: {image_path}")
        logger.info(f"Question: {question}")
        
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        # Generate answer with attention
        results = self.generate_with_attention(image, question)
        
        # Compute metrics
        metrics = self.compute_attention_metrics(results['visual_attention'])
        
        # Prepare analysis results
        analysis = {
            'question': question,
            'answer': results['answer'],
            'attention_metrics': metrics,
            'visual_attention': results['visual_attention']
        }
        
        # Visualize if requested
        if save_visualizations and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename from question
            safe_question = question[:50].replace('/', '_').replace(' ', '_')
            save_path = output_dir / f"llava_rad_{safe_question}.png"
            
            self.visualize_attention(
                image, 
                results['visual_attention'],
                question=question,
                answer=results['answer'],
                save_path=save_path,
                show_plot=False
            )
        
        return analysis
    
    def extract_answer(self, text):
        """Extract yes/no answer from model output"""
        text = text.lower().strip()
        
        # Direct yes/no at start
        if text.startswith('yes'):
            return 'yes'
        if text.startswith('no'):
            return 'no'
        
        # Common patterns
        no_patterns = [
            'no,', 'no.', 'there is no', 'there are no',
            'does not', "doesn't", 'absent', 'not present',
            'not visible', 'cannot be seen', 'not observed',
            'negative for', 'without', 'normal'
        ]
        
        yes_patterns = [
            'yes,', 'yes.', 'there is', 'there are',
            'present', 'visible', 'observed', 'seen',
            'positive for', 'shows', 'demonstrates',
            'evident', 'apparent', 'suggests', 'indicates'
        ]
        
        # Check patterns
        for pattern in no_patterns:
            if pattern in text[:100]:  # Check first 100 chars
                return 'no'
        
        for pattern in yes_patterns:
            if pattern in text[:100]:
                return 'yes'
        
        # Fallback
        if 'yes' in text[:50]:
            return 'yes'
        if 'no' in text[:50]:
            return 'no'
        
        return 'uncertain'

def setup_llava_rad_enhanced(device=None, model_id="microsoft/llava-rad", model_base="lmsys/vicuna-7b-v1.5"):
    """Setup LLaVA-Rad with enhanced attention extraction"""
    visualizer = LLaVARadVisualizer(device=device)
    
    # Load model with 8-bit quantization for memory efficiency
    model, processor = visualizer.load_model(model_id=model_id, model_base=model_base, load_in_8bit=True)
    
    return visualizer, model, processor

def main():
    """Test LLaVA-Rad visualizer"""
    print("="*80)
    print("LLaVA-Rad Vision-Language Model - Attention Visualizer")
    print("Microsoft's Medical Vision-Language Model for Radiology")
    print("="*80)
    
    # Setup
    visualizer = LLaVARadVisualizer()
    
    # Load model - explicitly specify microsoft/llava-rad with base model
    try:
        visualizer.load_model(
            model_id="microsoft/llava-rad", 
            model_base="lmsys/vicuna-7b-v1.5",
            load_in_8bit=True
        )
    except Exception as e:
        print(f"Warning: Could not load microsoft/llava-rad: {e}")
        print("Attempting to load with alternative configuration...")
        visualizer.load_model(load_in_8bit=True)
    
    # Test with sample image
    test_image = "/content/drive/MyDrive/Robust_Medical_LLM_Dataset/MIMIC_JPG/hundred_vqa/p10_p10000032_s50414267_2373b6a3-f5121edd-63dc44ac-0c4e33e8-ae2d83d7.jpg"
    test_question = "Is there cardiomegaly?"
    
    if os.path.exists(test_image):
        analysis = visualizer.analyze_image(
            test_image,
            test_question,
            save_visualizations=True,
            output_dir="llava_rad_visualizations"
        )
        
        print(f"\nQuestion: {analysis['question']}")
        print(f"Answer: {analysis['answer']}")
        print(f"Extracted: {visualizer.extract_answer(analysis['answer'])}")
        print("\nAttention Metrics:")
        for metric, value in analysis['attention_metrics'].items():
            print(f"  {metric}: {value:.4f}")
    else:
        print(f"Test image not found: {test_image}")
    
    # Clean up
    torch.cuda.empty_cache()
    print("\n✓ LLaVA-Rad visualizer ready for use")

if __name__ == "__main__":
    main()
