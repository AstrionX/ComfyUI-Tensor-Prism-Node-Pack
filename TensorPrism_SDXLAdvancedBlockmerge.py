"""
OPTION 1: REMOVE ALL UNET BLOCK LAYERS (Simplified Advanced Version)
This keeps all the advanced memory management but removes per-block control
"""

import torch
import gc
import logging
import psutil
import threading
import time
from typing import Dict, List, Tuple, Optional, Union
import traceback
from contextmanager import contextmanager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryManager:
    """Advanced memory management for GPU/CPU processing."""
    
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.device_memory_gb = self._get_device_memory()
        self.system_memory_gb = self._get_system_memory()
        self.memory_threshold = 0.85  # Use max 85% of available memory
        
    def _get_device_memory(self) -> float:
        """Get GPU memory in GB."""
        if self.cuda_available:
            try:
                return torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except:
                return 0.0
        return 0.0
    
    def _get_system_memory(self) -> float:
        """Get system RAM in GB."""
        return psutil.virtual_memory().total / (1024**3)
    
    def get_available_memory(self, device: torch.device) -> float:
        """Get currently available memory in GB."""
        if device.type == "cuda" and self.cuda_available:
            try:
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                return free_memory / (1024**3)
            except:
                return 0.0
        else:
            available_memory = psutil.virtual_memory().available / (1024**3)
            return available_memory
    
    def estimate_tensor_memory(self, tensor: torch.Tensor) -> float:
        """Estimate tensor memory usage in GB."""
        if tensor is None:
            return 0.0
        try:
            element_size = tensor.element_size()
            num_elements = tensor.numel()
            return (element_size * num_elements) / (1024**3)
        except:
            return 0.0
    
    def can_fit_in_memory(self, tensor: torch.Tensor, device: torch.device) -> bool:
        """Check if tensor can fit in device memory."""
        tensor_memory = self.estimate_tensor_memory(tensor)
        available_memory = self.get_available_memory(device) * self.memory_threshold
        return tensor_memory <= available_memory
    
    def should_use_cpu_fallback(self, device: torch.device) -> bool:
        """Determine if should fallback to CPU based on memory."""
        if device.type == "cpu":
            return False
        
        available_gpu_memory = self.get_available_memory(device)
        available_cpu_memory = self.get_available_memory(torch.device("cpu"))
        
        return available_gpu_memory < 2.0 or (available_cpu_memory > available_gpu_memory * 2)
    
    @contextmanager
    def memory_context(self, device: torch.device):
        """Context manager for memory cleanup."""
        try:
            yield
        finally:
            self.cleanup_memory(device)
    
    def cleanup_memory(self, device: torch.device = None):
        """Comprehensive memory cleanup."""
        gc.collect()
        if self.cuda_available:
            try:
                if device is None or device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except:
                pass


class SDXLAdvancedBlockMergeTensorPrism:
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        inputs = {
            "required": {
                "model_A": ("MODEL", {}),
                "model_B": ("MODEL", {}),
                "merge_method": (["Linear Interpolation", "Add Difference", "TIES-Merging (Simplified)"],),
                "default_unet_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "memory_limit_gb": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 64.0, "step": 0.5, "round": 0.1}),
                "force_cpu": ("BOOLEAN", {"default": False}),
                "batch_size": ("INT", {"default": 50, "min": 1, "max": 500, "step": 10}),
                "auto_memory_management": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "model_C": ("MODEL", {}),
                "ties_global_alpha_A": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "ties_global_alpha_B": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "rescale_output_magnitudes": ("BOOLEAN", {"default": False}),
                "a_delta_factor": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01, "round": 0.001}),
                "b_delta_factor": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01, "round": 0.001}),
                "precision_mode": (["auto", "fp16", "fp32"], {"default": "auto"}),
                "aggressive_cleanup": ("BOOLEAN", {"default": True}),
                
                # Special components
                "out_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "time_embed_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "label_emb_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                
                # Input blocks 0-11 (SDXL has 12)
                "input_block_00_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "input_block_01_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "input_block_02_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "input_block_03_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "input_block_04_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "input_block_05_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "input_block_06_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "input_block_07_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "input_block_08_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "input_block_09_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "input_block_10_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "input_block_11_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                
                # Middle blocks 0-2
                "middle_block_00_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "middle_block_01_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "middle_block_02_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                
                # Output blocks 0-11 (SDXL has 12)
                "output_block_00_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "output_block_01_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "output_block_02_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "output_block_03_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "output_block_04_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "output_block_05_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "output_block_06_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "output_block_07_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "output_block_08_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "output_block_09_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "output_block_10_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "output_block_11_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
            }
        }
        
        return inputs

    def _build_ratio_mapping_fixed(self, keys: List[str], default_ratio: float, kwargs: Dict) -> Dict[str, float]:
        
        try:
            default_ratio = max(0.0, min(1.0, default_ratio))
            
            # Build prefix mappings
            ratio_prefixes = {
                "time_embed.": max(0.0, min(1.0, kwargs.get("time_embed_ratio", default_ratio))),
                "label_emb.": max(0.0, min(1.0, kwargs.get("label_emb_ratio", default_ratio))),
                "out.": max(0.0, min(1.0, kwargs.get("out_ratio", default_ratio))),
            }
            
            
            # Input blocks (0-11)
            for i in range(12):
                input_ratio = max(0.0, min(1.0, kwargs.get(f"input_block_{i:02d}_ratio", default_ratio)))
                ratio_prefixes[f"input_blocks.{i}."] = input_ratio
            
            # Middle blocks (0-2)
            for i in range(3):
                middle_ratio = max(0.0, min(1.0, kwargs.get(f"middle_block_{i:02d}_ratio", default_ratio)))
                ratio_prefixes[f"middle_block.{i}."] = middle_ratio
            
            # Output blocks (0-11)
            for i in range(12):
                output_ratio = max(0.0, min(1.0, kwargs.get(f"output_block_{i:02d}_ratio", default_ratio)))
                ratio_prefixes[f"output_blocks.{i}."] = output_ratio
            
            # Sort prefixes by length (descending)
            sorted_prefixes = sorted(ratio_prefixes.items(), key=lambda x: len(x[0]), reverse=True)
            
            # Build mapping
            key_to_ratio = {}
            block_stats = {}
            
            for key in keys:
                ratio = default_ratio
                matched_prefix = None
                
                for prefix, prefix_ratio in sorted_prefixes:
                    if key.startswith(prefix):
                        ratio = prefix_ratio
                        matched_prefix = prefix
                        break
                
                key_to_ratio[key] = ratio
                
                # Statistics
                if matched_prefix:
                    block_stats[matched_prefix] = block_stats.get(matched_prefix, 0) + 1
                else:
                    block_stats["unmatched"] = block_stats.get("unmatched", 0) + 1
            
            # Log block statistics
            logger.info("Block mapping statistics:")
            for prefix, count in sorted(block_stats.items()):
                if count > 0:
                    ratio = ratio_prefixes.get(prefix, default_ratio)
                    logger.info(f"  {prefix}: {count} parameters (ratio: {ratio})")
            
            return key_to_ratio
            
        except Exception as e:
            logger.error(f"Error building ratio mapping: {e}")
            return {key: default_ratio for key in keys}

    # ... (Keep all other methods the same, just use _build_ratio_mapping_fixed instead)




NODE_CLASS_MAPPINGS = {
     "SDXLAdvancedBlockMergeTensorPrism": SDXLAdvancedBlockMergeTensorPrism
}
 
NODE_DISPLAY_NAME_MAPPINGS = {
     "SDXLAdvancedBlockMergeTensorPrism": "SDXL Advanced Block Merge (Tensor Prism)"
}