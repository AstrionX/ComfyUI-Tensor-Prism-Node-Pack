import torch
import copy
import gc
import psutil
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

class SDXLBlockMergeGranularTensorPrism:
    """
    A ComfyUI node for merging two (or three, using 'Add Difference' or 'TIES-Merging') SDXL models
    with highly granular control over the merging ratio for individual UNet blocks,
    time embeddings, and label embeddings.
    
    This memory-efficient version groups tensors by shape and processes them in batches
    that respect RAM limits, significantly reducing peak memory usage during merging.
    """
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "model_A": ("MODEL", {}),
                "model_B": ("MODEL", {}),
                "merge_method": (["Linear Interpolation", "Add Difference", "TIES-Merging (Simplified)"],),
                "default_unet_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "label": "Default UNet Ratio"}),
                "memory_limit_gb": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 64.0, "step": 0.5, "round": 0.1, "label": "Memory Limit (GB)"}),
            },
            "optional": {
                "model_C": ("MODEL", {}), # Required for "Add Difference" and "TIES-Merging" methods

                # Global TIES-Merging specific ratios
                "ties_global_alpha_A": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "label": "TIES: Global Alpha A"}),
                "ties_global_alpha_B": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "label": "TIES: Global Alpha B"}),
                "rescale_output_magnitudes": ("BOOLEAN", {"default": False, "label_on": "Rescale Output Magnitudes", "label_off": "No Rescale"}),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "label": "TIES: Iterations"}),

                # Delta Factors for Add Difference / TIES-Merging
                "a_delta_factor": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01, "round": 0.001, "label": "A Delta Factor (for Add/TIES)"}),
                "b_delta_factor": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01, "round": 0.001, "label": "B Delta Factor (for Add/TIES)"}),

                # Special UNet Component Ratios
                "out_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "label": "UNet Out Layer Ratio"}),
                "time_embed_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "label": "UNet Time Embed Ratio"}),
                "label_emb_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "label": "UNet Label Embed Ratio"}),
            }
        }

        # Dynamically add Input, Middle, and Output UNet Block Ratios to optional inputs
        for i in range(12): # input_blocks 0-11
            inputs["optional"][f"input_block_{i:02d}_ratio"] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "label": f"Input Block {i:02d} Ratio"})

        # Middle blocks 0-2
        inputs["optional"][f"middle_block_{0:02d}_ratio"] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "label": f"Middle Block {0:02d} (Resnet 0) Ratio"})
        inputs["optional"][f"middle_block_{1:02d}_ratio"] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "label": f"Middle Block {1:02d} (Attention/Transformer 0) Ratio"})
        inputs["optional"][f"middle_block_{2:02d}_ratio"] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "label": f"Middle Block {2:02d} (Resnet 1) Ratio"})

        for i in range(12): # output_blocks 0-11
            inputs["optional"][f"output_block_{i:02d}_ratio"] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "label": f"Output Block {i:02d} Ratio"})

        return inputs

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "process"
    CATEGORY = "Tensor_Prism/Merge"

    @staticmethod
    def get_memory_info() -> Tuple[float, float]:
        """Get current memory usage and available memory in GB"""
        memory = psutil.virtual_memory()
        used_gb = (memory.total - memory.available) / (1024**3)
        available_gb = memory.available / (1024**3)
        return used_gb, available_gb

    @staticmethod
    def estimate_tensor_memory_gb(shape: torch.Size, dtype: torch.dtype) -> float:
        """Estimate memory usage of a tensor in GB"""
        element_size = torch.tensor([], dtype=dtype).element_size()
        total_elements = 1
        for dim in shape:
            total_elements *= dim
        return (total_elements * element_size) / (1024**3)

    def group_tensors_by_shape(self, state_dict: Dict[str, torch.Tensor]) -> Dict[torch.Size, List[str]]:
        """Group tensor keys by their shape for batch processing"""
        shape_groups = defaultdict(list)
        for key, tensor in state_dict.items():
            shape_groups[tensor.shape].append(key)
        return dict(shape_groups)

    def calculate_batch_memory_usage(self, keys: List[str], state_dicts: List[Dict[str, torch.Tensor]], 
                                   merge_method: str) -> float:
        """Calculate total memory usage for processing a batch of tensors"""
        if not keys:
            return 0.0
        
        # Get shape and dtype from first tensor
        first_key = keys[0]
        first_tensor = state_dicts[0][first_key]
        shape = first_tensor.shape
        dtype = first_tensor.dtype
        
        # Calculate memory per tensor
        memory_per_tensor = self.estimate_tensor_memory_gb(shape, dtype)
        
        # Memory multiplier based on merge method
        # Linear: A + B + result = 3x
        # Add Difference/TIES: A + B + C + result + intermediates = 5x
        multiplier = 3 if merge_method == "Linear Interpolation" else 5
        
        # Total memory = number of tensors * memory per tensor * multiplier
        return len(keys) * memory_per_tensor * multiplier

    def create_processing_batches(self, shape_groups: Dict[torch.Size, List[str]], 
                                state_dicts: List[Dict[str, torch.Tensor]], 
                                memory_limit_gb: float, merge_method: str) -> List[List[str]]:
        """Create batches of tensor keys that fit within memory limit"""
        batches = []
        
        # Sort shapes by memory usage (largest first for better packing)
        sorted_shapes = sorted(shape_groups.keys(), 
                             key=lambda s: self.estimate_tensor_memory_gb(s, state_dicts[0][shape_groups[s][0]].dtype),
                             reverse=True)
        
        for shape in sorted_shapes:
            keys = shape_groups[shape]
            
            # If single tensor exceeds memory limit, process it alone
            single_tensor_memory = self.calculate_batch_memory_usage([keys[0]], state_dicts, merge_method)
            if single_tensor_memory > memory_limit_gb:
                print(f"Warning: Single tensor {keys[0]} requires {single_tensor_memory:.2f}GB, exceeding limit of {memory_limit_gb:.2f}GB")
                for key in keys:
                    batches.append([key])
                continue
            
            # Group keys into batches that fit memory limit
            current_batch = []
            for key in keys:
                test_batch = current_batch + [key]
                batch_memory = self.calculate_batch_memory_usage(test_batch, state_dicts, merge_method)
                
                if batch_memory <= memory_limit_gb:
                    current_batch.append(key)
                else:
                    # Start new batch
                    if current_batch:
                        batches.append(current_batch)
                    current_batch = [key]
            
            # Add final batch if not empty
            if current_batch:
                batches.append(current_batch)
        
        return batches

    def merge_tensor_batch(self, keys: List[str], unet_sd_A: Dict[str, torch.Tensor], 
                          unet_sd_B: Dict[str, torch.Tensor], unet_sd_C: Optional[Dict[str, torch.Tensor]],
                          key_to_ratio_map: Dict[str, float], merge_method: str,
                          ties_global_alpha_A: float, ties_global_alpha_B: float,
                          rescale_output_magnitudes: bool, iterations: int,
                          a_delta_factor: float, b_delta_factor: float) -> Dict[str, torch.Tensor]:
        """Merge a batch of tensors and return the results"""
        
        batch_results = {}
        
        # Load tensors for this batch
        batch_A = {key: unet_sd_A[key] for key in keys if key in unet_sd_A}
        batch_B = {key: unet_sd_B[key] for key in keys if key in unet_sd_B}
        batch_C = {key: unet_sd_C[key] for key in keys if unet_sd_C and key in unet_sd_C} if unet_sd_C else {}
        
        for key in keys:
            if key not in batch_A:
                print(f"Warning: Key {key} not found in batch_A, skipping")
                continue
                
            param_A = batch_A[key]
            current_modulation_ratio = key_to_ratio_map[key]

            # If parameter only exists in Model A, use it directly
            if key not in batch_B:
                print(f"Warning: UNet parameter '{key}' found in Model A but not in Model B. Inheriting from Model A only.")
                batch_results[key] = param_A.clone()
                continue

            param_B = batch_B[key]
            # Ensure param_B is on the same device and dtype as param_A
            if param_B.dtype != param_A.dtype or param_B.device != param_A.device:
                param_B = param_B.to(dtype=param_A.dtype, device=param_A.device)

            if merge_method == "Linear Interpolation":
                batch_results[key] = torch.lerp(param_B, param_A, current_modulation_ratio)

            elif merge_method == "Add Difference":
                if key not in batch_C:
                    print(f"Warning: UNet parameter '{key}' in Model A/B but not in Model C. Falling back to linear interpolation.")
                    batch_results[key] = torch.lerp(param_B, param_A, current_modulation_ratio)
                    continue

                param_C = batch_C[key]
                if param_C.dtype != param_A.dtype or param_C.device != param_A.device:
                    param_C = param_C.to(dtype=param_A.dtype, device=param_A.device)

                delta_A_factor = current_modulation_ratio * a_delta_factor
                delta_B_factor = (1.0 - current_modulation_ratio) * b_delta_factor
                
                batch_results[key] = param_C + (param_A - param_C) * delta_A_factor + (param_B - param_C) * delta_B_factor

            elif merge_method == "TIES-Merging (Simplified)":
                if key not in batch_C:
                    print(f"Warning: UNet parameter '{key}' in Model A/B but not in Model C. Falling back to linear interpolation.")
                    batch_results[key] = torch.lerp(param_B, param_A, current_modulation_ratio)
                    continue

                param_C = batch_C[key]
                if param_C.dtype != param_A.dtype or param_C.device != param_A.device:
                    param_C = param_C.to(dtype=param_A.dtype, device=param_A.device)

                alpha_k = ties_global_alpha_A * current_modulation_ratio
                beta_k = ties_global_alpha_B * (1.0 - current_modulation_ratio)

                if rescale_output_magnitudes:
                    sum_alpha_beta = alpha_k + beta_k
                    if sum_alpha_beta > 0:
                        alpha_k = alpha_k / sum_alpha_beta
                        beta_k = beta_k / sum_alpha_beta
                    else:
                        alpha_k = 0.0
                        beta_k = 0.0

                final_factor_A = alpha_k * a_delta_factor * iterations
                final_factor_B = beta_k * b_delta_factor * iterations

                batch_results[key] = param_C + (param_A - param_C) * final_factor_A + (param_B - param_C) * final_factor_B

        return batch_results

    def process(self, model_A, model_B, merge_method, default_unet_ratio, memory_limit_gb=8.0,
                model_C=None, ties_global_alpha_A=0.5, ties_global_alpha_B=0.5,
                rescale_output_magnitudes=False, iterations=1, a_delta_factor=1.0, b_delta_factor=1.0, **kwargs):
        """
        Executes memory-efficient granular model merging by processing tensors in shape-grouped batches
        """
        print(f"\n--- SDXL Memory-Efficient Granular Block Merge (Tensor Prism) ---")
        print(f"  Merge Method: {merge_method}")
        print(f"  Default UNet Ratio: {default_unet_ratio:.3f}")
        print(f"  Memory Limit: {memory_limit_gb:.1f}GB")
        
        # Get initial memory info
        used_memory, available_memory = self.get_memory_info()
        print(f"  System Memory - Used: {used_memory:.2f}GB, Available: {available_memory:.2f}GB")

        # Validate inputs
        if merge_method in ["Add Difference", "TIES-Merging (Simplified)"] and model_C is None:
            raise ValueError(f"Error: Model C is required for '{merge_method}' merge method.")

        # Get UNet state dictionaries
        def get_unet_sd(model_patcher):
            return model_patcher.model.state_dict()

        unet_sd_A = get_unet_sd(model_A)
        unet_sd_B = get_unet_sd(model_B)
        unet_sd_C = None
        if model_C is not None:
            unet_sd_C = get_unet_sd(model_C)

        # Build ratio mapping (same as original)
        unet_modulation_ratio_prefixes = {
            "time_embed.": kwargs.get("time_embed_ratio", default_unet_ratio),
            "label_emb.": kwargs.get("label_emb_ratio", default_unet_ratio),
            "out.": kwargs.get("out_ratio", default_unet_ratio),
            "middle_block.resnets.0.": kwargs.get("middle_block_00_ratio", default_unet_ratio),
            "middle_block.attentions.0.": kwargs.get("middle_block_01_ratio", default_unet_ratio),
            "middle_block.transformer_blocks.0.": kwargs.get("middle_block_01_ratio", default_unet_ratio),
            "middle_block.resnets.1.": kwargs.get("middle_block_02_ratio", default_unet_ratio),
        }

        for i in range(12):
            unet_modulation_ratio_prefixes[f"input_blocks.{i}."] = kwargs.get(f"input_block_{i:02d}_ratio", default_unet_ratio)
            unet_modulation_ratio_prefixes[f"output_blocks.{i}."] = kwargs.get(f"output_block_{i:02d}_ratio", default_unet_ratio)

        sorted_prefixes = sorted(unet_modulation_ratio_prefixes.items(), key=lambda item: len(item[0]), reverse=True)

        key_to_ratio_map = {}
        for key in unet_sd_A.keys():
            found_ratio = default_unet_ratio
            for prefix, r in sorted_prefixes:
                if key.startswith(prefix):
                    found_ratio = r
                    break
            key_to_ratio_map[key] = found_ratio

        # Group tensors by shape
        print("  Grouping tensors by shape...")
        shape_groups = self.group_tensors_by_shape(unet_sd_A)
        print(f"  Found {len(shape_groups)} unique tensor shapes")

        # Create processing batches
        print("  Creating memory-efficient processing batches...")
        state_dicts = [unet_sd_A, unet_sd_B]
        if unet_sd_C:
            state_dicts.append(unet_sd_C)
            
        batches = self.create_processing_batches(shape_groups, state_dicts, memory_limit_gb, merge_method)
        print(f"  Created {len(batches)} processing batches")

        # Process batches
        merged_unet_sd = {}
        total_params = len(unet_sd_A)
        processed_params = 0

        for i, batch_keys in enumerate(batches):
            batch_memory = self.calculate_batch_memory_usage(batch_keys, state_dicts, merge_method)
            print(f"  Processing batch {i+1}/{len(batches)} ({len(batch_keys)} tensors, ~{batch_memory:.2f}GB)")

            # Process this batch
            batch_results = self.merge_tensor_batch(
                batch_keys, unet_sd_A, unet_sd_B, unet_sd_C, key_to_ratio_map,
                merge_method, ties_global_alpha_A, ties_global_alpha_B,
                rescale_output_magnitudes, iterations, a_delta_factor, b_delta_factor
            )

            # Add results to merged state dict
            merged_unet_sd.update(batch_results)
            processed_params += len(batch_keys)

            # Force garbage collection after each batch
            del batch_results
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Progress update
            progress = (processed_params / total_params) * 100
            used_memory, _ = self.get_memory_info()
            print(f"    Progress: {progress:.1f}% ({processed_params}/{total_params}), Memory: {used_memory:.2f}GB")

        # Create merged model
        print("  Creating merged model...")
        merged_model_patcher = copy.deepcopy(model_A)
        merged_model_patcher.model.load_state_dict(merged_unet_sd, strict=False)

        # Final memory cleanup
        del merged_unet_sd
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        final_memory, _ = self.get_memory_info()
        print(f"  Final memory usage: {final_memory:.2f}GB")
        print(f"--- Memory-Efficient SDXL Granular Block Merge completed ---\n")

        return (merged_model_patcher,)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "SDXL Block Merge (Tensor Prism)": SDXLBlockMergeGranularTensorPrism
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDXL Block Merge (Tensor Prism)": "SDXL Block Merge (Tensor Prism)"
}
