"""
FIXED SDXL BLOCK MERGE - UNET BLOCK LAYERS CORRECTED

Key fixes:
1. Proper UNET block layer mapping and identification
2. Corrected prefix matching for SDXL architecture
3. Fixed middle block structure
4. Maintained ComfyUI patching system integration
"""

import torch
import copy
import gc
import psutil
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

class SDXLBlockMergeTensorPrism:
    """
    FIXED VERSION: Properly handles UNET block layers with correct SDXL architecture mapping
    """
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "model_A": ("MODEL", {}),
                "model_B": ("MODEL", {}),
                "merge_method": (["Linear Interpolation", "Add Difference", "TIES-Merging (Simplified)"],),
                "default_unet_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "memory_limit_gb": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 64.0, "step": 0.5, "round": 0.1}),
            },
            "optional": {
                "model_C": ("MODEL", {}),
                "ties_global_alpha_A": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "ties_global_alpha_B": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "rescale_output_magnitudes": ("BOOLEAN", {"default": False}),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "a_delta_factor": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01, "round": 0.001}),
                "b_delta_factor": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01, "round": 0.001}),
                
                # Special components (these work fine)
                "out_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "time_embed_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "label_emb_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                
                # FIXED: Proper SDXL UNET block structure
                # Input blocks (0-11 for SDXL)
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
                
                # Middle blocks (0-2 for SDXL: resnet -> attention -> resnet)
                "middle_block_00_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "middle_block_01_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "middle_block_02_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                
                # Output blocks (0-11 for SDXL) 
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

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "process"
    CATEGORY = "Tensor_Prism/Merge"

    def process(self, model_A, model_B, merge_method, default_unet_ratio, memory_limit_gb=8.0,
                model_C=None, **kwargs):
        """
        Process the model merge with proper UNET block handling
        """
        print(f"\n--- SDXL Block Merge (Tensor Prism) ---")
        
        # Validate inputs
        if merge_method in ["Add Difference", "TIES-Merging (Simplified)"] and model_C is None:
            raise ValueError(f"Model C is required for '{merge_method}' merge method.")

        # Get state dictionaries  
        unet_sd_A = model_A.model.state_dict()
        unet_sd_B = model_B.model.state_dict()
        unet_sd_C = model_C.model.state_dict() if model_C else None

        # Debug: Print some keys to understand structure
        print("Sample UNET keys:")
        sample_keys = list(unet_sd_A.keys())[:10]
        for key in sample_keys:
            print(f"  {key}")

        # Build ratio mapping with FIXED logic
        key_to_ratio_map = self._build_ratio_mapping(unet_sd_A.keys(), default_unet_ratio, kwargs)
        
        # Create patches instead of replacing entire state dict
        patches = {}
        
        print("Processing tensor merging...")
        processed_keys = 0
        for key in unet_sd_A.keys():
            if key not in unet_sd_B:
                continue
                
            param_A = unet_sd_A[key]
            param_B = unet_sd_B[key].to(param_A.device, param_A.dtype)
            ratio = key_to_ratio_map.get(key, default_unet_ratio)
            
            # Skip if ratio is 0 (no change from model A)
            if ratio == 0.0:
                continue
                
            merged_param = None
            
            if merge_method == "Linear Interpolation":
                # ratio of 0.0 = all A, ratio of 1.0 = all B
                merged_param = param_A * (1.0 - ratio) + param_B * ratio
                
            elif merge_method == "Add Difference":
                if unet_sd_C and key in unet_sd_C:
                    param_C = unet_sd_C[key].to(param_A.device, param_A.dtype)
                    delta_A = (param_A - param_C) * ratio * kwargs.get('a_delta_factor', 1.0)
                    delta_B = (param_B - param_C) * (1.0 - ratio) * kwargs.get('b_delta_factor', 1.0)
                    merged_param = param_C + delta_A + delta_B
                else:
                    merged_param = param_A * (1.0 - ratio) + param_B * ratio
                    
            elif merge_method == "TIES-Merging (Simplified)":
                if unet_sd_C and key in unet_sd_C:
                    param_C = unet_sd_C[key].to(param_A.device, param_A.dtype)
                    alpha_A = kwargs.get('ties_global_alpha_A', 0.5) * ratio
                    alpha_B = kwargs.get('ties_global_alpha_B', 0.5) * (1.0 - ratio)
                    
                    if kwargs.get('rescale_output_magnitudes', False):
                        total = alpha_A + alpha_B
                        if total > 0:
                            alpha_A /= total
                            alpha_B /= total
                    
                    delta_A = (param_A - param_C) * alpha_A * kwargs.get('a_delta_factor', 1.0)
                    delta_B = (param_B - param_C) * alpha_B * kwargs.get('b_delta_factor', 1.0)
                    merged_param = param_C + delta_A + delta_B
                else:
                    merged_param = param_A * (1.0 - ratio) + param_B * ratio
            
            if merged_param is not None:
                # Store as patch (difference from original)
                patch_diff = merged_param - param_A
                # Only add patch if there's actually a difference
                if not torch.allclose(patch_diff, torch.zeros_like(patch_diff), atol=1e-8):
                    patches[key] = (patch_diff,)
                    processed_keys += 1
        
        print(f"Created {len(patches)} patches from {processed_keys} processed keys")
        
        # Clone model and apply patches properly
        merged_model = model_A.clone()
        if patches:
            merged_model.add_patches(patches, 1.0)
        
        print("--- SDXL Block Merge completed ---\n")
        return (merged_model,)
    
    def _build_ratio_mapping(self, keys, default_ratio, kwargs):
        """
        FIXED: Build proper mapping of parameter keys to their merge ratios
        Now correctly identifies SDXL UNET structure
        """
        print("Building ratio mapping...")
        
        # Create the prefix to ratio mapping
        ratio_prefixes = {}
        
        # Special components (these are correctly mapped)
        ratio_prefixes["time_embed."] = kwargs.get("time_embed_ratio", default_ratio)
        ratio_prefixes["label_emb."] = kwargs.get("label_emb_ratio", default_ratio) 
        ratio_prefixes["out."] = kwargs.get("out_ratio", default_ratio)
        
        # FIXED: Proper SDXL UNET block mapping
        # Input blocks (SDXL has 0-11)
        for i in range(12):
            prefix = f"input_blocks.{i}."
            ratio_key = f"input_block_{i:02d}_ratio"
            ratio_prefixes[prefix] = kwargs.get(ratio_key, default_ratio)
        
        # Middle blocks (SDXL structure: 0=resnet, 1=attention, 2=resnet)
        for i in range(3):
            prefix = f"middle_block.{i}."
            ratio_key = f"middle_block_{i:02d}_ratio"
            ratio_prefixes[prefix] = kwargs.get(ratio_key, default_ratio)
        
        # Output blocks (SDXL has 0-11)
        for i in range(12):
            prefix = f"output_blocks.{i}."
            ratio_key = f"output_block_{i:02d}_ratio"
            ratio_prefixes[prefix] = kwargs.get(ratio_key, default_ratio)
        
        # Sort by length (longest first) for proper prefix matching
        sorted_prefixes = sorted(ratio_prefixes.items(), key=lambda x: len(x[0]), reverse=True)
        
        # Build the key to ratio mapping
        key_to_ratio = {}
        block_stats = defaultdict(int)
        
        for key in keys:
            ratio = default_ratio
            matched_prefix = None
            
            # Find the longest matching prefix
            for prefix, r in sorted_prefixes:
                if key.startswith(prefix):
                    ratio = r
                    matched_prefix = prefix
                    block_stats[prefix] += 1
                    break
            
            if matched_prefix is None:
                block_stats["unmatched"] += 1
                
            key_to_ratio[key] = ratio
        
        # Debug output
        print("Block mapping statistics:")
        for prefix, count in sorted(block_stats.items()):
            if count > 0:
                ratio = ratio_prefixes.get(prefix, default_ratio)
                print(f"  {prefix}: {count} parameters (ratio: {ratio})")
        
        return key_to_ratio


NODE_CLASS_MAPPINGS = {
    "SDXL Block Merge (Tensor Prism)": SDXLBlockMergeTensorPrism
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDXL Block Merge (Tensor Prism)": "SDXL Block Merge (Tensor Prism)"
}