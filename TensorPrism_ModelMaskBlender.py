import copy
import gc
import psutil
from typing import Dict, List, Tuple
import numpy as np

MODEL_MASK_TYPE = ("MODEL_MASK",)

class TensorPrism_ModelMaskBlender:
    """
    Memory-efficient model mask blender that processes mask values in batches
    to handle large masks without excessive RAM usage.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask_A": MODEL_MASK_TYPE,
                "mask_B": MODEL_MASK_TYPE,
                "blend_mode": (["Add", "Multiply", "Max", "Min", "Linear Blend", "Exponential Blend"], {"default": "Linear Blend"}),
                "memory_limit_gb": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 16.0, "step": 0.1, "round": 0.1, "label": "Memory Limit (GB)"}),
            },
            "optional": {
                "blend_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "label": "Blend Strength (for Linear/Exp)"}),
                "clip_output": ("BOOLEAN", {"default": True, "label_on": "Clip to [0, 1]", "label_off": "No Clipping"}),
            }
        }

    RETURN_TYPES = MODEL_MASK_TYPE
    RETURN_NAMES = ("combined_mask",)
    FUNCTION = "blend_masks"
    CATEGORY = "Tensor_Prism/Mask"

    @staticmethod
    def get_memory_info() -> Tuple[float, float]:
        """Get current memory usage and available memory in GB"""
        memory = psutil.virtual_memory()
        used_gb = (memory.total - memory.available) / (1024**3)
        available_gb = memory.available / (1024**3)
        return used_gb, available_gb

    @staticmethod
    def estimate_dict_memory_gb(dict_size: int) -> float:
        """Estimate memory usage of a dictionary with float values in GB"""
        # Rough estimate: key string + float value + overhead
        bytes_per_entry = 100  # Conservative estimate
        return (dict_size * bytes_per_entry) / (1024**3)

    def create_key_batches(self, all_keys: List[str], memory_limit_gb: float) -> List[List[str]]:
        """Create batches of keys that fit within memory limit"""
        batches = []
        current_batch = []
        
        # Estimate how many keys we can process per batch
        max_keys_per_batch = max(1000, int((memory_limit_gb * 1024**3) / 200))  # Conservative estimate
        
        for i, key in enumerate(all_keys):
            current_batch.append(key)
            
            if len(current_batch) >= max_keys_per_batch:
                batches.append(current_batch)
                current_batch = []
        
        # Add final batch if not empty
        if current_batch:
            batches.append(current_batch)
        
        return batches

    def process_mask_batch(self, batch_keys: List[str], mask_A_dict: Dict[str, float], 
                          mask_B_dict: Dict[str, float], blend_mode: str, 
                          blend_strength: float, clip_output: bool) -> Dict[str, float]:
        """Process a batch of mask keys with the specified blending operation"""
        batch_results = {}
        
        for key in batch_keys:
            val_A = mask_A_dict.get(key, 0.0)
            val_B = mask_B_dict.get(key, 0.0)

            result_val = 0.0
            if blend_mode == "Add":
                result_val = val_A + val_B
            elif blend_mode == "Multiply":
                result_val = val_A * val_B
            elif blend_mode == "Max":
                result_val = max(val_A, val_B)
            elif blend_mode == "Min":
                result_val = min(val_A, val_B)
            elif blend_mode == "Linear Blend":
                result_val = val_A * (1.0 - blend_strength) + val_B * blend_strength
            elif blend_mode == "Exponential Blend":
                exp_strength = blend_strength ** 2 
                result_val = val_A * (1.0 - exp_strength) + val_B * exp_strength
            
            if clip_output:
                result_val = max(0.0, min(1.0, result_val))
            
            batch_results[key] = result_val
        
        return batch_results

    def blend_masks(self, mask_A, mask_B, blend_mode, memory_limit_gb=2.0, 
                   blend_strength=0.5, clip_output=True):
        
        print(f"\n--- Model Mask Blender (Tensor Prism) ---")
        print(f"  Blend Mode: {blend_mode}")
        print(f"  Blend Strength: {blend_strength}")
        print(f"  Memory Limit: {memory_limit_gb:.1f}GB")
        
        # Get initial memory info
        used_memory, available_memory = self.get_memory_info()
        print(f"  System Memory - Used: {used_memory:.2f}GB, Available: {available_memory:.2f}GB")

        # Validate input masks
        if not mask_A or "mask_dict" not in mask_A:
            raise ValueError("mask_A is invalid or missing mask_dict")
        if not mask_B or "mask_dict" not in mask_B:
            raise ValueError("mask_B is invalid or missing mask_dict")

        mask_A_dict = mask_A["mask_dict"]
        mask_B_dict = mask_B["mask_dict"]
        
        all_keys = list(set(mask_A_dict.keys()).union(set(mask_B_dict.keys())))
        print(f"  Total mask keys to process: {len(all_keys)}")

        # Create processing batches
        print("  Creating memory-efficient processing batches...")
        batches = self.create_key_batches(all_keys, memory_limit_gb)
        print(f"  Created {len(batches)} processing batches")

        # Process batches
        combined_mask_dict = {}
        total_keys = len(all_keys)
        processed_keys = 0

        for i, batch_keys in enumerate(batches):
            batch_memory = self.estimate_dict_memory_gb(len(batch_keys))
            print(f"  Processing batch {i+1}/{len(batches)} ({len(batch_keys)} keys, ~{batch_memory:.3f}GB)")

            # Process this batch
            batch_results = self.process_mask_batch(
                batch_keys, mask_A_dict, mask_B_dict, blend_mode, blend_strength, clip_output
            )

            # Add results to combined mask dict
            combined_mask_dict.update(batch_results)
            processed_keys += len(batch_keys)

            # Force garbage collection after each batch
            del batch_results
            gc.collect()

            # Progress update
            progress = (processed_keys / total_keys) * 100
            used_memory, _ = self.get_memory_info()
            print(f"    Progress: {progress:.1f}% ({processed_keys}/{total_keys}), Memory: {used_memory:.2f}GB")

        # Create the combined mask object
        combined_mask = copy.deepcopy(mask_A)
        combined_mask["mask_dict"] = combined_mask_dict
        combined_mask["mask_type"] = f"blended_{mask_A.get('mask_type', 'unknown')}_{mask_B.get('mask_type', 'unknown')}"
        
        # Calculate intensity (mean of all values)
        if combined_mask_dict:
            combined_mask["intensity"] = float(np.mean(list(combined_mask_dict.values())))
        else:
            combined_mask["intensity"] = 0.0

        # Final memory cleanup
        del combined_mask_dict
        gc.collect()

        final_memory, _ = self.get_memory_info()
        print(f"  Final memory usage: {final_memory:.2f}GB")
        print(f"  Combined mask intensity: {combined_mask['intensity']:.4f}")
        print(f"--- Model Mask Blender completed ---\n")
        
        return (combined_mask,)

NODE_CLASS_MAPPINGS = {
    "TensorPrism_ModelMaskBlender": TensorPrism_ModelMaskBlender,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorPrism_ModelMaskBlender": "Mask Blender (Tensor Prism)",
}