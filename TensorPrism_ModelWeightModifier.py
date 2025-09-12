import torch
import copy
import gc
import psutil
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np

def is_unet_key(key: str) -> bool:
    key_lower = key.lower()
    return 'unet' in key_lower or 'model.diffusion_model' in key_lower

def is_vae_key(key: str) -> bool:
    key_lower = key.lower()
    return 'vae' in key_lower or 'autoencoder' in key_lower or 'first_stage_model' in key_lower

def is_text_encoder_key(key: str) -> bool:
    key_lower = key.lower()
    return 'clip' in key_lower or 'text_encoder' in key_lower or 'cond_stage' in key_lower

def _get_unet_component_type(param_name: str) -> str:
    param_lower = param_name.lower()
    if 'time_embed' in param_lower:
        return 'time_embed'
    elif 'input_blocks' in param_lower:
        return 'input_blocks'
    elif 'middle_block' in param_lower:
        return 'middle_block'
    elif 'output_blocks' in param_lower:
        return 'output_blocks'
    elif param_lower.endswith('.out.weight') or param_lower.endswith('.out.bias'):
        return 'out'
    return 'other_unet'

class TensorPrism_ModelWeightModifier:
    """
    Memory-efficient model weight modifier that processes tensors in batches
    to avoid excessive RAM usage during modifications.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "modification_target": (["All UNet", "UNet Input Blocks", "UNet Middle Block", "UNet Output Blocks", "UNet Time Embeddings", "UNet Final Output Layer", "VAE", "Text Encoders", "All Model Parameters"], {"default": "All UNet"}),
                "modification_operation": (["Multiply", "Add", "Set Value", "Clamp Magnitude", "Scale Max Abs"], {"default": "Multiply"}),
                "value": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "round": 0.001, "label": "Factor/Value"}),
                "use_mask": ("BOOLEAN", {"default": False, "label_on": "Use Mask", "label_off": "No Mask"}),
                "memory_limit_gb": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 32.0, "step": 0.5, "round": 0.1, "label": "Memory Limit (GB)"}),
            },
            "optional": {
                "mask": ("MASK",),  # Updated to use ComfyUI standard MASK type instead of custom MODEL_MASK
                "mask_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "label": "Mask Application Strength"}),
                "max_abs_reference_model": ("MODEL", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("modified_model",)
    FUNCTION = "modify_weights"
    CATEGORY = "Tensor_Prism/Transform"

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
            if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                shape_groups[tensor.shape].append(key)
        return dict(shape_groups)

    def calculate_batch_memory_usage(self, keys: List[str], state_dict: Dict[str, torch.Tensor]) -> float:
        """Calculate total memory usage for processing a batch of tensors"""
        if not keys:
            return 0.0
        
        total_memory = 0.0
        for key in keys:
            if key in state_dict:
                tensor = state_dict[key]
                # Memory for original + modified tensor (2x for safety)
                memory_per_tensor = self.estimate_tensor_memory_gb(tensor.shape, tensor.dtype)
                total_memory += memory_per_tensor * 2
        
        return total_memory

    def create_processing_batches(self, shape_groups: Dict[torch.Size, List[str]], 
                                state_dict: Dict[str, torch.Tensor], 
                                memory_limit_gb: float) -> List[List[str]]:
        """Create batches of tensor keys that fit within memory limit"""
        batches = []
        
        # Sort shapes by memory usage (largest first for better packing)
        sorted_shapes = sorted(shape_groups.keys(), 
                             key=lambda s: self.estimate_tensor_memory_gb(s, state_dict[shape_groups[s][0]].dtype),
                             reverse=True)
        
        for shape in sorted_shapes:
            keys = shape_groups[shape]
            
            # Group keys into batches that fit memory limit
            current_batch = []
            for key in keys:
                test_batch = current_batch + [key]
                batch_memory = self.calculate_batch_memory_usage(test_batch, state_dict)
                
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

    def _should_process_key_for_target(self, key: str, target_component: str) -> bool:
        """Check if a key should be processed for the given target component"""
        if target_component == "All Model Parameters":
            return True
        elif target_component == "All UNet":
            return is_unet_key(key) and not is_vae_key(key) and not is_text_encoder_key(key)
        elif target_component == "VAE":
            return is_vae_key(key)
        elif target_component == "Text Encoders":
            return is_text_encoder_key(key)
        
        unet_component_type = _get_unet_component_type(key)
        if target_component == "UNet Time Embeddings":
            return unet_component_type == 'time_embed'
        elif target_component == "UNet Input Blocks":
            return unet_component_type == 'input_blocks'
        elif target_component == "UNet Middle Block":
            return unet_component_type == 'middle_block'
        elif target_component == "UNet Output Blocks":
            return unet_component_type == 'output_blocks'
        elif target_component == "UNet Final Output Layer":
            return unet_component_type == 'out'
        
        return False

    def process_tensor_batch(self, batch_keys: List[str], state_dict: Dict[str, torch.Tensor],
                           modification_target: str, modification_operation: str, value: float,
                           mask_dict: Dict[str, float], mask_strength: float, use_mask: bool,
                           reference_max_abs: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """Process a batch of tensors with the specified modification"""
        batch_results = {}
        
        for key in batch_keys:
            if key not in state_dict:
                continue
                
            tensor = state_dict[key]
            if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                batch_results[key] = tensor
                continue

            if not self._should_process_key_for_target(key, modification_target):
                batch_results[key] = tensor
                continue
                
            current_tensor = tensor.clone()
            
            effective_mask_value = mask_dict.get(key, 0.0) if use_mask else 1.0
            op_intensity = effective_mask_value * mask_strength

            if op_intensity > 0:
                original_tensor_for_ops = current_tensor.clone()

                if modification_operation == "Multiply":
                    modified_tensor_part = original_tensor_for_ops * value
                    current_tensor = original_tensor_for_ops * (1.0 - op_intensity) + modified_tensor_part * op_intensity
                elif modification_operation == "Add":
                    modified_tensor_part = original_tensor_for_ops + value
                    current_tensor = original_tensor_for_ops * (1.0 - op_intensity) + modified_tensor_part * op_intensity
                elif modification_operation == "Set Value":
                    set_tensor = torch.full_like(original_tensor_for_ops, value, 
                                               device=original_tensor_for_ops.device, 
                                               dtype=original_tensor_for_ops.dtype)
                    current_tensor = original_tensor_for_ops * (1.0 - op_intensity) + set_tensor * op_intensity
                elif modification_operation == "Clamp Magnitude":
                    max_mag = abs(value)
                    clamped_tensor = torch.clamp(original_tensor_for_ops, -max_mag, max_mag)
                    current_tensor = original_tensor_for_ops * (1.0 - op_intensity) + clamped_tensor * op_intensity
                elif modification_operation == "Scale Max Abs":
                    if reference_max_abs is not None and reference_max_abs > 1e-8:
                        current_max_abs = torch.abs(original_tensor_for_ops).max().item()
                        if current_max_abs > 1e-8:
                            target_max_abs = reference_max_abs * value
                            scale_factor = target_max_abs / current_max_abs
                            modified_tensor_part = original_tensor_for_ops * scale_factor
                            current_tensor = original_tensor_for_ops * (1.0 - op_intensity) + modified_tensor_part * op_intensity
                        else:
                            current_tensor = original_tensor_for_ops 
                    else:
                        current_tensor = original_tensor_for_ops
                else: 
                    current_tensor = original_tensor_for_ops 
                        
            batch_results[key] = current_tensor

        return batch_results

    def convert_comfyui_mask_to_dict(self, mask_tensor, state_dict):
        """Convert ComfyUI mask tensor to mask_dict format used internally"""
        if mask_tensor is None:
            return {}
        
        # Convert tensor to numpy for processing
        if isinstance(mask_tensor, torch.Tensor):
            mask_array = mask_tensor.cpu().numpy()
        else:
            mask_array = np.array(mask_tensor)
        
        # Flatten and get average mask value
        mask_value = float(np.mean(mask_array))
        
        # Create mask_dict with uniform mask value for all parameters
        mask_dict = {}
        for key in state_dict.keys():
            if isinstance(state_dict[key], torch.Tensor) and state_dict[key].is_floating_point():
                mask_dict[key] = mask_value
        
        return mask_dict

    def modify_weights(self, model, modification_target, modification_operation, value, 
                      use_mask=False, memory_limit_gb=4.0, mask=None, mask_strength=1.0, 
                      max_abs_reference_model=None):
        
        print(f"\n--- Model Weight Modifier (Tensor Prism) ---")
        print(f"  Target: {modification_target}")
        print(f"  Operation: {modification_operation}")
        print(f"  Value: {value}")
        print(f"  Memory Limit: {memory_limit_gb:.1f}GB")
        
        # Get initial memory info
        used_memory, available_memory = self.get_memory_info()
        print(f"  System Memory - Used: {used_memory:.2f}GB, Available: {available_memory:.2f}GB")

        modified_model = copy.deepcopy(model)
        state_dict = modified_model.model.state_dict()
        
        if use_mask and mask is not None:
            mask_dict = self.convert_comfyui_mask_to_dict(mask, state_dict)
            print(f"  Using mask with average value: {np.mean(list(mask_dict.values())) if mask_dict else 0.0:.3f}")
        else:
            mask_dict = {}

        # Calculate reference max abs if needed
        reference_max_abs = None
        if modification_operation == "Scale Max Abs" and max_abs_reference_model:
            ref_state_dict = max_abs_reference_model.model.state_dict()
            all_max_abs_values = []
            for t in ref_state_dict.values():
                if isinstance(t, torch.Tensor) and t.is_floating_point():
                    all_max_abs_values.append(torch.abs(t).max().item())
            
            if all_max_abs_values:
                reference_max_abs = max(all_max_abs_values)
            else:
                print("Warning: Reference model has no floating point tensors. 'Scale Max Abs' will be skipped.")

        # Group tensors by shape
        print("  Grouping tensors by shape...")
        shape_groups = self.group_tensors_by_shape(state_dict)
        print(f"  Found {len(shape_groups)} unique tensor shapes")

        # Create processing batches
        print("  Creating memory-efficient processing batches...")
        batches = self.create_processing_batches(shape_groups, state_dict, memory_limit_gb)
        print(f"  Created {len(batches)} processing batches")

        # Process batches
        modified_state_dict = {}
        total_params = len(state_dict)
        processed_params = 0

        for i, batch_keys in enumerate(batches):
            batch_memory = self.calculate_batch_memory_usage(batch_keys, state_dict)
            print(f"  Processing batch {i+1}/{len(batches)} ({len(batch_keys)} tensors, ~{batch_memory:.2f}GB)")

            # Process this batch
            batch_results = self.process_tensor_batch(
                batch_keys, state_dict, modification_target, modification_operation, value,
                mask_dict, mask_strength, use_mask, reference_max_abs
            )

            # Add results to modified state dict
            modified_state_dict.update(batch_results)
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

        # Copy non-processed tensors
        for key, tensor in state_dict.items():
            if key not in modified_state_dict:
                modified_state_dict[key] = tensor

        # Load the modified state dict
        modified_model.model.load_state_dict(modified_state_dict, strict=False)

        # Final memory cleanup
        del modified_state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        final_memory, _ = self.get_memory_info()
        print(f"  Final memory usage: {final_memory:.2f}GB")
        print(f"--- Model Weight Modifier completed ---\n")

        return (modified_model,)

NODE_CLASS_MAPPINGS = {
    "TensorPrism_ModelWeightModifier": TensorPrism_ModelWeightModifier,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorPrism_ModelWeightModifier": "Model Weight Modifier (Tensor Prism)",
}
