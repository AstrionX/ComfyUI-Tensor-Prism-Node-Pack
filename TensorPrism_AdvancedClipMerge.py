import torch
import torch.nn.functional as F
import numpy as np
import math
import gc
from typing import Dict, Any, Tuple, Optional, List
import psutil

class AdvancedCLIPMerge:
    """
    Advanced CLIP merging node with multiple interpolation methods and layer-specific control.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip1": ("CLIP",),
                "clip2": ("CLIP",),
                "merge_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "merge_method": ([
                    "linear", 
                    "slerp", 
                    "cosine", 
                    "weighted_average",
                    "spectral_blend"
                ], {"default": "slerp"}),
            },
            "optional": {
                "attention_bias": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "feedforward_bias": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "embedding_bias": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "normalization_bias": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "preserve_norms": ("BOOLEAN", {"default": True}),
                "spectral_alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "memory_efficient": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("clip", "merge_info")
    FUNCTION = "merge_clips"
    CATEGORY = "Tensor_Prism/CLIP"
    
    def __init__(self):
        self.device = self._get_optimal_device()
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    def _get_optimal_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _get_layer_type(self, key: str) -> str:
        """Determine layer type from parameter key"""
        key_lower = key.lower()
        if any(term in key_lower for term in ['attn', 'attention', 'self_attn', 'cross_attn']):
            return 'attention'
        elif any(term in key_lower for term in ['ffn', 'mlp', 'feed_forward']):
            return 'feedforward'
        elif any(term in key_lower for term in ['embed', 'position', 'token']):
            return 'embedding'
        elif any(term in key_lower for term in ['norm', 'layer_norm', 'group_norm']):
            return 'normalization'
        else:
            return 'other'
    
    def _apply_layer_bias(self, ratio: float, layer_type: str, attention_bias: float,
                         feedforward_bias: float, embedding_bias: float, normalization_bias: float) -> float:
        """Apply layer-specific bias to merge ratio"""
        bias_map = {
            'attention': attention_bias,
            'feedforward': feedforward_bias,
            'embedding': embedding_bias,
            'normalization': normalization_bias,
            'other': 0.0
        }
        
        biased_ratio = ratio + bias_map.get(layer_type, 0.0)
        return max(0.0, min(1.0, biased_ratio))
    
    def _slerp(self, t1: torch.Tensor, t2: torch.Tensor, ratio: float) -> torch.Tensor:
        """Spherical Linear Interpolation"""
        t1_flat = t1.flatten()
        t2_flat = t2.flatten()
        
        # Normalize vectors
        t1_norm = F.normalize(t1_flat, dim=0)
        t2_norm = F.normalize(t2_flat, dim=0)
        
        # Calculate angle between vectors
        dot = torch.clamp(torch.dot(t1_norm, t2_norm), -1.0, 1.0)
        theta = torch.acos(torch.abs(dot))
        
        # Handle edge cases
        if theta < 1e-6:
            return self._linear_interpolation(t1, t2, ratio)
        
        # SLERP formula
        sin_theta = torch.sin(theta)
        w1 = torch.sin((1.0 - ratio) * theta) / sin_theta
        w2 = torch.sin(ratio * theta) / sin_theta
        
        result = w1 * t1_flat + w2 * t2_flat
        return result.reshape(t1.shape)
    
    def _cosine_interpolation(self, t1: torch.Tensor, t2: torch.Tensor, ratio: float) -> torch.Tensor:
        """Cosine interpolation for smoother transitions"""
        smooth_ratio = 0.5 * (1.0 - math.cos(ratio * math.pi))
        return t1 * (1.0 - smooth_ratio) + t2 * smooth_ratio
    
    def _linear_interpolation(self, t1: torch.Tensor, t2: torch.Tensor, ratio: float) -> torch.Tensor:
        """Standard linear interpolation"""
        return t1 * (1.0 - ratio) + t2 * ratio
    
    def _weighted_average(self, t1: torch.Tensor, t2: torch.Tensor, ratio: float) -> torch.Tensor:
        """Weighted average based on tensor magnitudes"""
        mag1 = torch.norm(t1)
        mag2 = torch.norm(t2)
        total_mag = mag1 + mag2 + 1e-8
        
        w1 = (mag1 / total_mag) * (1.0 - ratio)
        w2 = (mag2 / total_mag) * ratio
        norm_factor = w1 + w2
        
        return (w1 * t1 + w2 * t2) / norm_factor
    
    def _spectral_blend(self, t1: torch.Tensor, t2: torch.Tensor, ratio: float, alpha: float) -> torch.Tensor:
        """Frequency domain blending"""
        if len(t1.shape) < 2:
            return self._linear_interpolation(t1, t2, ratio)
        
        try:
            # Reshape to 2D for FFT
            original_shape = t1.shape
            t1_2d = t1.view(t1.shape[0], -1).float()
            t2_2d = t2.view(t2.shape[0], -1).float()
            
            # Apply FFT
            fft1 = torch.fft.fft2(t1_2d)
            fft2 = torch.fft.fft2(t2_2d)
            
            # Blend in frequency domain
            magnitude1 = torch.abs(fft1)
            magnitude2 = torch.abs(fft2)
            phase1 = torch.angle(fft1)
            phase2 = torch.angle(fft2)
            
            # Blend magnitudes and phases separately
            blended_mag = (1.0 - ratio) * magnitude1 + ratio * magnitude2
            blended_phase = (1.0 - alpha) * phase1 + alpha * phase2
            
            # Reconstruct complex spectrum
            blended_fft = blended_mag * torch.exp(1j * blended_phase)
            
            # Inverse FFT
            result = torch.fft.ifft2(blended_fft).real
            return result.view(original_shape).to(t1.dtype)
            
        except:
            return self._linear_interpolation(t1, t2, ratio)
    
    def _merge_tensors(self, t1: torch.Tensor, t2: torch.Tensor, ratio: float, method: str,
                      spectral_alpha: float = 0.5) -> torch.Tensor:
        """Apply the specified merge method to two tensors"""
        if method == "linear":
            return self._linear_interpolation(t1, t2, ratio)
        elif method == "slerp":
            return self._slerp(t1, t2, ratio)
        elif method == "cosine":
            return self._cosine_interpolation(t1, t2, ratio)
        elif method == "weighted_average":
            return self._weighted_average(t1, t2, ratio)
        elif method == "spectral_blend":
            return self._spectral_blend(t1, t2, ratio, spectral_alpha)
        else:
            return self._linear_interpolation(t1, t2, ratio)
    
    def merge_clips(self, clip1, clip2, merge_ratio: float, merge_method: str,
                   attention_bias: float = 0.0, feedforward_bias: float = 0.0, 
                   embedding_bias: float = 0.0, normalization_bias: float = 0.0, 
                   preserve_norms: bool = True, spectral_alpha: float = 0.5, 
                   memory_efficient: bool = True):
        
        try:
            # Memory cleanup
            if memory_efficient and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Clone the first CLIP model
            merged_clip = clip1.clone()
            
            # Get state dictionaries
            state_dict1 = clip1.cond_stage_model.state_dict() if hasattr(clip1, 'cond_stage_model') else clip1.state_dict()
            state_dict2 = clip2.cond_stage_model.state_dict() if hasattr(clip2, 'cond_stage_model') else clip2.state_dict()
            
            # Merge statistics
            merged_layers = 0
            layer_types_merged = {'attention': 0, 'feedforward': 0, 'embedding': 0, 'normalization': 0, 'other': 0}
            
            # Merge parameters
            merged_state_dict = {}
            
            for key in state_dict1.keys():
                if key in state_dict2:
                    t1 = state_dict1[key].to(self.device)
                    t2 = state_dict2[key].to(self.device)
                    
                    # Determine merge ratio for this layer
                    layer_type = self._get_layer_type(key)
                    current_ratio = merge_ratio
                    
                    # Apply layer-specific bias
                    current_ratio = self._apply_layer_bias(
                        current_ratio, layer_type, attention_bias, 
                        feedforward_bias, embedding_bias, normalization_bias
                    )
                    
                    # Store original norms if preserving
                    original_norm = torch.norm(t1) if preserve_norms else None
                    
                    try:
                        # Perform merge
                        merged_tensor = self._merge_tensors(
                            t1, t2, current_ratio, merge_method, spectral_alpha
                        )
                        
                        # Preserve norm if requested
                        if preserve_norms and original_norm is not None:
                            current_norm = torch.norm(merged_tensor)
                            if current_norm > 1e-8:
                                merged_tensor = merged_tensor * (original_norm / current_norm)
                        
                        merged_state_dict[key] = merged_tensor.cpu()
                        merged_layers += 1
                        layer_types_merged[layer_type] += 1
                        
                    except Exception as e:
                        # Fallback to linear interpolation
                        merged_tensor = self._linear_interpolation(t1, t2, current_ratio)
                        merged_state_dict[key] = merged_tensor.cpu()
                    
                    # Memory cleanup for large tensors
                    if memory_efficient:
                        del t1, t2
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
                    # Keep original parameter if not in second model
                    merged_state_dict[key] = state_dict1[key]
            
            # Load merged state dict
            if hasattr(merged_clip, 'cond_stage_model'):
                merged_clip.cond_stage_model.load_state_dict(merged_state_dict, strict=False)
            else:
                merged_clip.load_state_dict(merged_state_dict, strict=False)
            
            # Generate merge information
            merge_info = f"""=== ADVANCED CLIP MERGE RESULTS ===
Method: {merge_method}
Base Ratio: {merge_ratio:.3f}

=== LAYER STATISTICS ===
Successfully Merged: {merged_layers}
Total Parameters: {len(state_dict1)}

=== LAYER TYPE BREAKDOWN ===
Attention: {layer_types_merged['attention']}
Feedforward: {layer_types_merged['feedforward']}
Embedding: {layer_types_merged['embedding']}
Normalization: {layer_types_merged['normalization']}
Other: {layer_types_merged['other']}

=== MERGE SETTINGS ===
Attention Bias: {attention_bias:+.3f}
Feedforward Bias: {feedforward_bias:+.3f}
Embedding Bias: {embedding_bias:+.3f}
Normalization Bias: {normalization_bias:+.3f}
Preserve Norms: {preserve_norms}
"""
            
            # Final cleanup
            if memory_efficient and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            return (merged_clip, merge_info)
            
        except Exception as e:
            error_info = f"CLIP merge failed: {str(e)}"
            return (clip1, error_info)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AdvancedCLIPMerge": AdvancedCLIPMerge
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedCLIPMerge": "Advanced CLIP Merge (Tensor Prism)"
}