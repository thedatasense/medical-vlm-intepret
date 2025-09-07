#!/usr/bin/env python3
"""
Enhanced LLaVA-Rad Attention Visualizer with Advanced Techniques
Incorporates suggestions for improved attention extraction and analysis
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
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
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
        n_maps = len(normalized_maps)
        consistency_scores = []
        
        for i in range(n_maps):
            for j in range(i+1, n_maps):
                js_div = jensenshannon(normalized_maps[i], normalized_maps[j])
                consistency_scores.append(1 - js_div)
        
        return float(np.mean(consistency_scores))
    
    @staticmethod
    def calculate_sparsity(attention_map: np.ndarray) -> float:
        """Calculate sparsity of attention (Gini coefficient)"""
        flat = attention_map.flatten()
        flat = flat / (flat.sum() + 1e-10)
        sorted_flat = np.sort(flat)
        n = len(flat)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_flat)) / (n * np.sum(sorted_flat)) - (n + 1) / n
        return float(gini)


class AttentionCache:
    """Simple attention cache for performance"""
    
    def __init__(self, max_size: int = 50):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get_key(self, image_path: str, question: str, model_name: str) -> str:
        """Generate cache key"""
        content = f"{image_path}_{question}_{model_name}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Dict]:
        """Get cached attention if available"""
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def set(self, key: str, attention_data: Dict):
        """Cache attention data"""
        if len(self.cache) >= self.max_size:
            # Remove least accessed
            if self.access_count:
                least_accessed = min(self.access_count, key=self.access_count.get)
                del self.cache[least_accessed]
                del self.access_count[least_accessed]
        
        self.cache[key] = attention_data
        self.access_count[key] = 1


class AttentionDifferenceAnalyzer:
    """Tools for comparing attention patterns"""
    
    @staticmethod
    def compute_attention_shift(attention_before: np.ndarray, 
                               attention_after: np.ndarray) -> Dict[str, Any]:
        """Compute how attention shifts between two states"""
        # Ensure same shape
        if attention_before.shape != attention_after.shape:
            raise ValueError("Attention maps must have same shape")
        
        # Normalize both
        before_norm = attention_before / (attention_before.sum() + 1e-10)
        after_norm = attention_after / (attention_after.sum() + 1e-10)
        
        # Compute difference
        diff = after_norm - before_norm
        
        # Identify regions with significant changes
        threshold = np.std(diff) * 2
        increased = diff > threshold
        decreased = diff < -threshold
        
        # Calculate shift metrics
        total_shift = float(np.abs(diff).sum())
        max_increase = float(diff.max())
        max_decrease = float(diff.min())
        
        # Find center of mass shift
        y_before, x_before = np.unravel_index(
            np.argmax(before_norm), before_norm.shape
        )
        y_after, x_after = np.unravel_index(
            np.argmax(after_norm), after_norm.shape
        )
        
        com_shift = float(np.sqrt((x_after - x_before)**2 + (y_after - y_before)**2))
        
        return {
            'difference_map': diff,
            'increased_regions': increased,
            'decreased_regions': decreased,
            'total_shift': total_shift,
            'max_increase': max_increase,
            'max_decrease': max_decrease,
            'center_of_mass_shift': com_shift,
            'js_divergence': float(jensenshannon(before_norm.flatten(), after_norm.flatten()))
        }


class EnhancedLLaVARadVisualizer:
    """Enhanced LLaVA-Rad visualizer with advanced attention techniques"""
    
    def __init__(self, device=None, config: Optional[AttentionConfig] = None):
        """Initialize enhanced visualizer"""
        self.model = None
        self.processor = None
        self.device = device if device else self.setup_gpu()
        self.config = config or AttentionConfig()
        self.cache = AttentionCache()
        self.image_size = self.config.image_size
        self.patch_size = self.config.patch_size
        logger.info(f"Enhanced LLaVA-Rad Visualizer initialized on {self.device}")
    
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
            logger.warning(f"No GPU with {min_free_gb}GB+ free memory, trying with less")
            device = f"cuda:{best_gpu}"
        else:
            device = f"cuda:{best_gpu}"
        
        torch.cuda.set_device(best_gpu)
        logger.info(f"Selected GPU {best_gpu} with {max_free_memory:.1f}GB free")
        
        return device
    
    def identify_visual_tokens(self, input_ids: torch.Tensor, 
                              image_token_id: int = -200) -> Optional[range]:
        """Precisely identify visual token positions based on image token anchor
        
        This is a key improvement from the suggestions - more precise token identification
        """
        # Convert to numpy for easier manipulation
        ids = input_ids.cpu().numpy() if isinstance(input_ids, torch.Tensor) else input_ids
        
        # Handle batch dimension
        if len(ids.shape) > 1:
            ids = ids[0]
        
        # Find image token positions
        image_positions = np.where(ids == image_token_id)[0]
        
        if len(image_positions) == 0:
            # Fallback: look for common image token IDs
            for token_id in [-200, 32000, 32001]:  # Common image tokens
                image_positions = np.where(ids == token_id)[0]
                if len(image_positions) > 0:
                    break
        
        if len(image_positions) == 0:
            logger.warning("No image token found, using heuristic")
            # Fallback to middle portion heuristic
            seq_len = len(ids)
            visual_tokens = (self.image_size // self.patch_size) ** 2
            start = max(1, (seq_len - visual_tokens) // 2)
            end = min(seq_len - 1, start + visual_tokens)
            return range(start, end)
        
        # For models with continuous visual tokens after image token
        if len(image_positions) == 1:
            start_pos = image_positions[0] + 1
            # Calculate expected visual tokens based on image size and patch size
            num_visual_tokens = (self.image_size // self.patch_size) ** 2
            
            # Verify we have enough tokens
            if start_pos + num_visual_tokens <= len(ids):
                return range(start_pos, start_pos + num_visual_tokens)
            else:
                # Adjust if not enough tokens
                return range(start_pos, len(ids))
        
        # Multiple image tokens - they might be the visual tokens themselves
        return range(image_positions[0], image_positions[-1] + 1)
    
    def extract_attention_robust(self, outputs: Any, layer_indices: Optional[List[int]] = None,
                                input_ids: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Extract attention with multiple fallback strategies"""
        try:
            # Try to get attentions from outputs
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                attentions = outputs.attentions

                # HF generate() often returns a tuple/list per generation step; select last step
                if isinstance(attentions, (list, tuple)) and len(attentions) > 0 and isinstance(attentions[0], (list, tuple)):
                    per_layer_attns = attentions[-1]
                else:
                    # Already a list/tuple of layers
                    per_layer_attns = attentions

                # Select layers
                if layer_indices is None:
                    # Default to last quarter of layers
                    n_layers = len(per_layer_attns)
                    layer_indices = list(range(max(0, 3 * n_layers // 4), n_layers))

                # Extract and collect layer attentions tensors
                attention_maps = []
                for idx in layer_indices:
                    if idx < len(per_layer_attns):
                        layer_attention = per_layer_attns[idx]
                        # Handle different attention formats (e.g., tuple wrapping)
                        if isinstance(layer_attention, tuple):
                            layer_attention = layer_attention[0]
                        attention_maps.append(layer_attention)
                
                # Identify visual tokens if input_ids provided
                visual_range = None
                if input_ids is not None:
                    visual_range = self.identify_visual_tokens(input_ids)
                
                return {
                    'attention_maps': attention_maps,
                    'visual_range': visual_range,
                    'method': 'transformer_attention',
                    'layer_indices': layer_indices
                }
            
        except Exception as e:
            logger.warning(f"Failed to extract transformer attention: {e}")
        
        # Fallback: Create uniform attention
        logger.warning("Using uniform attention fallback")
        grid_size = self.image_size // self.patch_size
        uniform_attention = np.ones((grid_size, grid_size)) / (grid_size * grid_size)
        
        return {
            'attention_maps': [uniform_attention],
            'visual_range': None,
            'method': 'uniform_fallback',
            'layer_indices': []
        }
    
    def extract_visual_attention_multihead(self, attention_data: Dict[str, Any],
                                          aggregate_mode: str = 'mean') -> np.ndarray:
        """Extract visual attention with multi-head support"""
        attention_maps = attention_data.get('attention_maps', [])
        visual_range = attention_data.get('visual_range')
        
        if not attention_maps:
            return self._create_uniform_attention()
        
        # Process each attention map
        processed_attentions = []
        
        for att_map in attention_maps:
            if isinstance(att_map, torch.Tensor):
                att_map = att_map.cpu().numpy()
            
            # Handle different shapes
            if len(att_map.shape) == 4:  # [batch, heads, seq, seq]
                batch_size, n_heads, seq_len, _ = att_map.shape
                
                # Extract per-head attention if requested
                if aggregate_mode == 'individual':
                    head_attentions = []
                    for h in range(n_heads):
                        head_att = self._process_single_attention(
                            att_map[0, h], visual_range
                        )
                        head_attentions.append(head_att)
                    return head_attentions
                
                # Otherwise aggregate heads
                if aggregate_mode == 'mean':
                    att_map = att_map.mean(axis=1)  # Average over heads
                elif aggregate_mode == 'max':
                    att_map = att_map.max(axis=1)  # Max over heads
                elif aggregate_mode == 'entropy_weighted':
                    # Implement entropy weighting - heads with lower entropy (more focused) get higher weight
                    batch_size, n_heads, seq_len, _ = att_map.shape
                    weights = np.zeros(n_heads)
                    
                    for h in range(n_heads):
                        head_att = att_map[0, h]  # Get attention for this head
                        # Normalize to valid probability distribution
                        head_att_norm = head_att / (head_att.sum() + 1e-10)
                        # Calculate entropy
                        entropy = -np.sum(head_att_norm * np.log(head_att_norm + 1e-10))
                        # Lower entropy = more focused = higher weight
                        weights[h] = 1.0 / (1.0 + entropy)
                    
                    # Normalize weights
                    weights = weights / weights.sum()
                    
                    # Apply weighted average
                    att_map = np.zeros_like(att_map[0, 0])
                    for h in range(n_heads):
                        att_map += weights[h] * att_map[0, h]
                    
                    att_map = att_map.reshape(1, seq_len, seq_len)  # Add batch dim back
                else:
                    # Default to mean if unknown mode
                    att_map = att_map.mean(axis=1)
                
                att_map = att_map[0]  # Remove batch dimension
            
            elif len(att_map.shape) == 3:  # [batch, seq, seq]
                att_map = att_map[0]
            
            # Process single attention map
            processed = self._process_single_attention(att_map, visual_range)
            processed_attentions.append(processed)
        
        # Aggregate across layers
        if processed_attentions:
            return np.mean(processed_attentions, axis=0)
        
        return self._create_uniform_attention()
    
    def _process_single_attention(self, attention_matrix: np.ndarray, 
                                 visual_range: Optional[range]) -> np.ndarray:
        """Process a single attention matrix to extract visual attention"""
        seq_len = attention_matrix.shape[0]
        
        # Use identified visual range or estimate
        if visual_range is None:
            grid_size = self.image_size // self.patch_size
            n_visual = grid_size * grid_size
            start = max(1, (seq_len - n_visual) // 2)
            end = min(seq_len - 1, start + n_visual)
            visual_range = range(start, end)
        
        # Extract attention from last token (query) to visual tokens
        # This is the critical fix - we want query-to-visual, not visual-to-visual
        vis = np.arange(visual_range.start, visual_range.stop)
        q = attention_matrix.shape[0] - 1  # last token as query
        vec = attention_matrix[q, vis]  # attention from query to visual tokens
        
        # Reshape to grid
        grid_size = int(np.sqrt(len(vis)))
        if grid_size * grid_size == len(vis):
            visual_grid = vec.reshape(grid_size, grid_size)
        else:
            # Handle non-square grids
            visual_grid = cv2.resize(
                vec.reshape(1, -1), 
                (self.image_size // self.patch_size, self.image_size // self.patch_size)
            )
        
        # Normalize
        visual_grid = visual_grid / (visual_grid.sum() + 1e-10)
        
        return visual_grid
    
    def _create_uniform_attention(self) -> np.ndarray:
        """Create uniform attention grid as fallback"""
        grid_size = self.image_size // self.patch_size
        return np.ones((grid_size, grid_size)) / (grid_size * grid_size)
    
    def create_multi_head_visualization(self, attention_heads: List[np.ndarray], 
                                       image: Image.Image, 
                                       save_path: Optional[str] = None) -> plt.Figure:
        """Visualize attention from multiple heads in a grid"""
        n_heads = len(attention_heads)
        n_cols = min(4, n_heads)
        n_rows = (n_heads + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (ax, head_attn) in enumerate(zip(axes.flat, attention_heads)):
            # Show image
            ax.imshow(image, alpha=0.7)
            
            # Overlay attention
            H, W = image.size[1], image.size[0]
            attention_resized = cv2.resize(head_attn, (W, H), interpolation=cv2.INTER_CUBIC)
            
            # Apply percentile clipping
            low, high = np.percentile(attention_resized, self.config.percentile_clip)
            attention_clipped = np.clip(attention_resized, low, high)
            attention_normalized = (attention_clipped - low) / (high - low + 1e-8)
            
            # Select colormap
            cmap = 'bone' if self.config.use_medical_colormap else self.config.colormap
            im = ax.imshow(attention_normalized, alpha=self.config.alpha, cmap=cmap)
            
            ax.set_title(f'Head {idx}', fontsize=10)
            ax.axis('off')
        
        # Turn off remaining axes
        for idx in range(len(attention_heads), n_rows * n_cols):
            axes.flat[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Multi-head visualization saved to {save_path}")
        
        return fig
    
    def create_3d_attention_surface(self, attention_map: np.ndarray, 
                                   save_path: Optional[str] = None) -> plt.Figure:
        """Create 3D surface plot of attention distribution"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create mesh grid
        X, Y = np.meshgrid(range(attention_map.shape[1]), range(attention_map.shape[0]))
        
        # Normalize attention for better visualization
        Z = attention_map / attention_map.max()
        
        # Create surface plot
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                              linewidth=0, antialiased=True)
        
        # Add contour lines at the bottom
        ax.contour(X, Y, Z, zdir='z', offset=0, cmap='viridis', alpha=0.3)
        
        # Customize the plot
        ax.set_xlabel('Width (patches)', fontsize=10)
        ax.set_ylabel('Height (patches)', fontsize=10)
        ax.set_zlabel('Attention Weight', fontsize=10)
        ax.set_title('3D Attention Surface', fontsize=12)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Set viewing angle
        ax.view_init(elev=30, azim=45)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"3D attention surface saved to {save_path}")
        
        return fig
    
    def analyze_attention_consistency(self, image: Image.Image, 
                                     questions: List[str], 
                                     max_new_tokens: int = 100) -> Dict[str, Any]:
        """Analyze attention consistency across multiple prompts"""
        attention_maps = []
        answers = []
        
        for question in questions:
            result = self.generate_with_attention(image, question, max_new_tokens)
            if result.get('visual_attention') is not None:
                att_map = result['visual_attention']
                # Guard: aggregate if a list of head maps
                if isinstance(att_map, list) and len(att_map) > 0:
                    try:
                        att_map = np.mean(np.stack(att_map, axis=0), axis=0)
                    except Exception:
                        # Best-effort fallback
                        att_map = att_map[0]
                attention_maps.append(att_map)
                answers.append(result.get('answer', ''))
        
        if not attention_maps:
            return {'error': 'No attention maps extracted'}
        
        # Calculate consistency metrics
        consistency = AttentionMetrics.calculate_consistency(attention_maps)
        
        # Calculate pairwise JS divergences
        pairwise_js = []
        for i in range(len(attention_maps)):
            for j in range(i+1, len(attention_maps)):
                js_div = jensenshannon(
                    attention_maps[i].flatten(), 
                    attention_maps[j].flatten()
                )
                pairwise_js.append(float(js_div))
        
        # Analyze attention shifts
        shift_analysis = []
        if len(attention_maps) > 1:
            for i in range(1, len(attention_maps)):
                shift = AttentionDifferenceAnalyzer.compute_attention_shift(
                    attention_maps[0], attention_maps[i]
                )
                shift_analysis.append({
                    'question': questions[i],
                    'total_shift': shift['total_shift'],
                    'js_divergence': shift['js_divergence'],
                    'com_shift': shift['center_of_mass_shift']
                })
        
        return {
            'consistency_score': consistency,
            'mean_js_divergence': float(np.mean(pairwise_js)) if pairwise_js else 0.0,
            'std_js_divergence': float(np.std(pairwise_js)) if pairwise_js else 0.0,
            'pairwise_js': pairwise_js,
            'shift_analysis': shift_analysis,
            'n_prompts': len(questions),
            'attention_maps': attention_maps,
            'answers': answers
        }
    
    def load_model(self, model_id="microsoft/llava-rad", model_base="lmsys/vicuna-7b-v1.5", 
                   load_in_8bit=False, load_in_4bit=False):
        """Load LLaVA-Rad model with enhanced configuration"""
        logger.info(f"Loading LLaVA-Rad model: {model_id}")
        
        try:
            # Try the sophisticated llava library approach first
            from llava.model.builder import load_pretrained_model
            from llava.utils import disable_torch_init
            
            disable_torch_init()
            
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path=model_id,
                model_base=model_base,
                model_name="llavarad",
                device_map=self.device if self.device != "cpu" else "auto",
                load_8bit=load_in_8bit,
                load_4bit=load_in_4bit
            )
            
            self.model = model
            self.processor = image_processor
            self.tokenizer = tokenizer
            self.using_llava_lib = True
            
            # Set model to eval mode
            self.model.eval()
            
            logger.info("✓ Model loaded using LLaVA library")
            
        except Exception as e:
            logger.warning(f"LLaVA library load failed: {e}")
            # If user asked for microsoft/llava-rad but llava isn't available, do NOT
            # try to load it via HF (it lacks standard HF files). Switch to HF LLaVA.
            hf_fallback_id = None
            if isinstance(model_id, str) and ("llava-rad" in model_id.lower()):
                hf_fallback_id = "llava-hf/llava-1.5-7b-hf"
                logger.info(f"Switching to HF fallback model: {hf_fallback_id}")
            else:
                logger.info("Falling back to transformers loading...")
            
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
            
            # Select model id for HF path
            selected_model_id = hf_fallback_id or model_id

            # Guard: avoid trying to load microsoft/llava-rad via HF
            if isinstance(selected_model_id, str) and ("llava-rad" in selected_model_id.lower()):
                selected_model_id = "llava-hf/llava-1.5-7b-hf"
                logger.info(f"HF loader incompatible with llava-rad; using {selected_model_id}")

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                selected_model_id,
                trust_remote_code=True
            )
            
            # Load model (prefer LlavaForConditionalGeneration for HF LLaVA)
            try:
                from transformers import LlavaForConditionalGeneration
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    selected_model_id,
                    device_map=self.device if self.device != "cpu" else "auto",
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                    attn_implementation="eager"
                )
            except Exception as e_llava:
                logger.warning(f"LlavaForConditionalGeneration load failed: {e_llava}. Trying AutoModelForCausalLM with trust_remote_code.")
                self.model = AutoModelForCausalLM.from_pretrained(
                    selected_model_id,
                    device_map=self.device if self.device != "cpu" else "auto",
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                    attn_implementation="eager"
                )
            
            self.model.eval()
            # In HF path, use processor.tokenizer
            self.tokenizer = getattr(self.processor, 'tokenizer', None)
            self.using_llava_lib = False
            
            logger.info("✓ Model loaded using transformers")
    
    def generate_with_attention(self, image: Union[str, Image.Image], 
                               question: str, 
                               max_new_tokens: int = 100,
                               use_cache: bool = True) -> Dict[str, Any]:
        """Generate answer with enhanced attention extraction"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Load image if path provided, and capture a stable key for caching
        image_key: Optional[str] = None
        if isinstance(image, str):
            image_key = image
            image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            # Use filename if available; otherwise, hash bytes as a stable key
            image_key = getattr(image, 'filename', None)
            if not image_key:
                try:
                    image_key = f"md5:{hashlib.md5(image.tobytes()).hexdigest()}"
                except Exception:
                    image_key = "inmemory_image"
        
        # Check cache
        cache_key = None
        if use_cache:
            cache_key = self.cache.get_key(image_key or "inmemory_image", question, "llava-rad")
            cached = self.cache.get(cache_key)
            if cached:
                logger.info("Using cached result")
                return cached
        
        # Prepare inputs
        prompt = f"<image>\n{question}"
        
        # Process inputs based on model type
        if getattr(self, 'using_llava_lib', False):
            # LLaVA library approach (llava is installed)
            from llava.conversation import conv_templates
            from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
            from llava.constants import IMAGE_TOKEN_INDEX
            
            conv = conv_templates["v1"].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            input_ids = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0)
            
            if self.device != "cpu":
                input_ids = input_ids.to(self.device)
            
            # Process image
            image_tensor = self.processor.preprocess(image, return_tensors='pt')['pixel_values']
            if self.device != "cpu":
                image_tensor = image_tensor.half().to(self.device)
            
            # Generate with attention
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    output_attentions=True,
                    return_dict_in_generate=True
                )
            input_ids_for_extract = input_ids
        else:
            # Transformers approach
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate with attention
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    output_attentions=True,
                    return_dict_in_generate=True
                )
            input_ids_for_extract = inputs.get('input_ids')
        
        # Extract answer
        if hasattr(outputs, 'sequences'):
            answer_ids = outputs.sequences[0]
            if hasattr(self, 'tokenizer') and self.tokenizer:
                answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True)
            else:
                answer = self.processor.decode(answer_ids, skip_special_tokens=True)
            
            # Clean answer
            answer = answer.split("ASSISTANT:")[-1].strip()
        else:
            answer = "Failed to generate answer"
        
        # Extract attention
        attention_data = self.extract_attention_robust(
            outputs,
            input_ids=input_ids_for_extract
        )
        
        # Process visual attention
        visual_attention = self.extract_visual_attention_multihead(
            attention_data,
            aggregate_mode=self.config.multi_head_mode
        )
        
        # Calculate metrics
        metrics = {}
        if isinstance(visual_attention, np.ndarray):
            metrics = AttentionMetrics.calculate_focus_score(visual_attention)
            metrics['sparsity'] = AttentionMetrics.calculate_sparsity(visual_attention)
        
        result = {
            'answer': answer,
            'visual_attention': visual_attention,
            'attention_method': attention_data.get('method', 'unknown'),
            'metrics': metrics,
            'raw_attention_data': attention_data
        }
        
        # Cache result
        if use_cache and cache_key:
            self.cache.set(cache_key, result)
        
        return result
    
    def visualize_attention_difference(self, image: Image.Image,
                                      question1: str, question2: str,
                                      save_path: Optional[str] = None) -> plt.Figure:
        """Visualize the difference in attention between two questions"""
        # Get attention for both questions
        result1 = self.generate_with_attention(image, question1)
        result2 = self.generate_with_attention(image, question2)
        
        att1 = result1.get('visual_attention')
        att2 = result2.get('visual_attention')
        
        if att1 is None or att2 is None:
            raise ValueError("Failed to extract attention for one or both questions")
        
        # Compute attention shift
        shift_data = AttentionDifferenceAnalyzer.compute_attention_shift(att1, att2)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original attentions
        axes[0, 0].imshow(image)
        axes[0, 0].imshow(cv2.resize(att1, (image.width, image.height)), 
                         alpha=0.5, cmap='hot')
        axes[0, 0].set_title(f'Q1: {question1[:30]}...')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(image)
        axes[0, 1].imshow(cv2.resize(att2, (image.width, image.height)), 
                         alpha=0.5, cmap='hot')
        axes[0, 1].set_title(f'Q2: {question2[:30]}...')
        axes[0, 1].axis('off')
        
        # Difference map
        diff_resized = cv2.resize(shift_data['difference_map'], (image.width, image.height))
        im = axes[0, 2].imshow(diff_resized, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
        axes[0, 2].set_title('Attention Difference (Q2 - Q1)')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2])
        
        # Increased/decreased regions
        axes[1, 0].imshow(image)
        increased_resized = cv2.resize(
            shift_data['increased_regions'].astype(float), 
            (image.width, image.height)
        )
        axes[1, 0].imshow(increased_resized, alpha=0.5, cmap='Greens')
        axes[1, 0].set_title('Increased Attention Regions')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(image)
        decreased_resized = cv2.resize(
            shift_data['decreased_regions'].astype(float), 
            (image.width, image.height)
        )
        axes[1, 1].imshow(decreased_resized, alpha=0.5, cmap='Reds')
        axes[1, 1].set_title('Decreased Attention Regions')
        axes[1, 1].axis('off')
        
        # Metrics
        metrics_text = f"""Attention Shift Metrics:
Total Shift: {shift_data['total_shift']:.3f}
JS Divergence: {shift_data['js_divergence']:.3f}
CoM Shift: {shift_data['center_of_mass_shift']:.1f} pixels
Max Increase: {shift_data['max_increase']:.3f}
Max Decrease: {shift_data['max_decrease']:.3f}

Answer 1: {result1['answer'][:50]}...
Answer 2: {result2['answer'][:50]}..."""
        
        axes[1, 2].text(0.1, 0.5, metrics_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='center')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Attention difference visualization saved to {save_path}")
        
        return fig


def main():
    """Demo script showing enhanced features"""
    visualizer = EnhancedLLaVARadVisualizer()
    
    # Example configuration
    config = AttentionConfig(
        use_medical_colormap=True,
        multi_head_mode='individual',
        percentile_clip=(5, 95)
    )
    visualizer.config = config
    
    # Load model
    visualizer.load_model(load_in_8bit=True)
    
    # Test image
    test_image = "/content/drive/MyDrive/Robust_Medical_LLM_Dataset/MIMIC_JPG/hundred_vqa/10000032.jpg"
    
    # Basic generation with attention
    result = visualizer.generate_with_attention(
        test_image,
        "Is there evidence of pneumonia?",
        max_new_tokens=100
    )
    
    print(f"Answer: {result['answer']}")
    print(f"Attention method: {result['attention_method']}")
    print(f"Metrics: {result['metrics']}")
    
    # Test consistency analysis
    questions = [
        "Is there evidence of pneumonia?",
        "Do you see any signs of pneumonia?",
        "Can you identify pneumonia in this X-ray?",
        "Is pneumonia present in this chest radiograph?"
    ]
    
    consistency_results = visualizer.analyze_attention_consistency(
        Image.open(test_image),
        questions
    )
    
    print(f"\nConsistency Analysis:")
    print(f"Consistency score: {consistency_results['consistency_score']:.3f}")
    print(f"Mean JS divergence: {consistency_results['mean_js_divergence']:.3f}")
    
    # Test multi-head visualization if available
    if isinstance(result.get('visual_attention'), list):
        print(f"\nCreating multi-head visualization...")
        fig = visualizer.create_multi_head_visualization(
            result['visual_attention'],
            Image.open(test_image),
            save_path="multi_head_attention.png"
        )
        plt.close(fig)
    
    # Test attention difference
    print(f"\nVisualizing attention difference...")
    fig = visualizer.visualize_attention_difference(
        Image.open(test_image),
        "Is there evidence of pneumonia?",
        "Describe the cardiac silhouette.",
        save_path="attention_difference.png"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
