import torch
import copy

class SDXLBlockMergeGranularTensorPrism:
    """
    A ComfyUI node for merging two (or three, using 'Add Difference' or 'TIES-Merging') SDXL models
    with highly granular control over the merging ratio for individual UNet blocks,
    time embeddings, and label embeddings.
    This version expands UNet block control to cover all 12 input and 12 output blocks
    and removes explicit text encoder merging controls. Text encoders will be inherited
    from Model A in the merged output.

    This version includes enhanced logging to the ComfyUI console, clearly showing the
    applied granular ratios for each UNet component, which helps in verifying that the
    node's granular controls are working as expected when a generation is run.
    It also adds a minor robustness check for tensor data types during merging.
    """
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "model_A": ("MODEL", {}),
                "model_B": ("MODEL", {}),
                "merge_method": (["Linear Interpolation", "Add Difference", "TIES-Merging (Simplified)"],),
                "default_unet_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "label": "Default UNet Ratio"}),
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

        # Dynamically add Input, Middle, and Output UNet Block Ratios to optional inputs.
        # These will appear as sliders on the node by default. If a Float node is connected,
        # its value will override the slider. If neither is set, the default here (0.5) is used.
        for i in range(12): # input_blocks 0-11
            inputs["optional"][f"input_block_{i:02d}_ratio"] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "label": f"Input Block {i:02d} Ratio"})

        # Middle blocks 0-2 (mapping to resnet.0, attention.0/transformer.0, resnet.1)
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

    def process(self, model_A, model_B, merge_method, default_unet_ratio,
                model_C=None, ties_global_alpha_A=0.5, ties_global_alpha_B=0.5,
                rescale_output_magnitudes=False, iterations=1, a_delta_factor=1.0, b_delta_factor=1.0, **kwargs):
        """
        Executes the granular model merging process for the UNet, now with TIES-Merging, iteration/rescale options,
        and the new A & B Delta Factors.
        Text encoders are taken directly from model_A without merging.
        All individual UNet block ratios and special component ratios are passed via kwargs.
        """
        print(f"\n--- SDXL Granular Block Merge (Tensor Prism) ---")
        print(f"  Merge Method: {merge_method}")
        print(f"  Default UNet Ratio for unconnected inputs: {default_unet_ratio:.3f}")

        # Log parameters specific to merge methods
        if model_C is not None:
            print(f"  Model C (Base Model) provided for '{merge_method}' method.")
        if merge_method == "TIES-Merging (Simplified)":
            print(f"  TIES: Global Alpha A: {ties_global_alpha_A:.3f}, Global Alpha B: {ties_global_alpha_B:.3f}")
            print(f"  TIES: Rescale Output Magnitudes: {rescale_output_magnitudes}, Iterations: {iterations}")
        if merge_method in ["Add Difference", "TIES-Merging (Simplified)"]:
            print(f"  Delta Factors: A: {a_delta_factor:.3f}, B: {b_delta_factor:.3f}")

        # Ensure Model C is provided if "Add Difference" or "TIES-Merging" is selected
        if merge_method in ["Add Difference", "TIES-Merging (Simplified)"] and model_C is None:
            raise ValueError(f"Error: Model C is required for '{merge_method}' merge method, but it was not provided.")

        # Helper function to extract UNet state dictionary from a ModelPatcher object
        # The .state_dict() method returns references, not copies, so it's efficient.
        def get_unet_sd(model_patcher):
            return model_patcher.model.state_dict()

        # Get UNet state dictionaries for all input models
        unet_sd_A = get_unet_sd(model_A)
        unet_sd_B = get_unet_sd(model_B)
        unet_sd_C = None
        if model_C is not None:
            unet_sd_C = get_unet_sd(model_C)

        # Optimization 1: Pre-compute unet_modulation_ratio_map and then map directly to keys
        # This avoids repeated string comparisons within the main loop, improving performance for large UNets.
        unet_modulation_ratio_prefixes = {
            "time_embed.": kwargs.get("time_embed_ratio", default_unet_ratio),
            "label_emb.": kwargs.get("label_emb_ratio", default_unet_ratio),
            "out.": kwargs.get("out_ratio", default_unet_ratio),

            "middle_block.resnets.0.": kwargs.get("middle_block_00_ratio", default_unet_ratio),
            "middle_block.attentions.0.": kwargs.get("middle_block_01_ratio", default_unet_ratio),
            "middle_block.transformer_blocks.0.": kwargs.get("middle_block_01_ratio", default_unet_ratio), # attention.0 and transformer_blocks.0 often share a ratio
            "middle_block.resnets.1.": kwargs.get("middle_block_02_ratio", default_unet_ratio),
        }

        for i in range(12):
            unet_modulation_ratio_prefixes[f"input_blocks.{i}."] = kwargs.get(f"input_block_{i:02d}_ratio", default_unet_ratio)
            unet_modulation_ratio_prefixes[f"output_blocks.{i}."] = kwargs.get(f"output_block_{i:02d}_ratio", default_unet_ratio)

        # Sort prefixes by length in descending order to ensure the most specific prefix matches first.
        sorted_prefixes = sorted(unet_modulation_ratio_prefixes.items(), key=lambda item: len(item[0]), reverse=True)

        # Create a direct mapping from each UNet parameter key to its modulation ratio.
        key_to_ratio_map = {}
        for key in unet_sd_A.keys():
            found_ratio = default_unet_ratio
            for prefix, r in sorted_prefixes:
                if key.startswith(prefix):
                    found_ratio = r
                    break
            key_to_ratio_map[key] = found_ratio
        
        # Log effective ratios for debugging/user info (functionality, not performance-critical)
        print("\n--- Effective UNet Component Ratios (Model A Contribution) ---")
        applied_ratios_summary = {}
        for prefix_key, ratio_value in unet_modulation_ratio_prefixes.items():
            display_name = prefix_key.replace('.', ' ').replace('_', ' ').strip().title()
            if display_name.endswith('Ratio'):
                display_name = display_name[:-5].strip()
            
            if "Middle Block 00" in display_name:
                display_name = "Middle Block 00 (Resnet 0)"
            elif "Middle Block 01" in display_name:
                display_name = "Middle Block 01 (Attention/Transformer 0)"
            elif "Middle Block 02" in display_name:
                display_name = "Middle Block 02 (Resnet 1)"

            applied_ratios_summary[display_name] = ratio_value
        
        for name in sorted(applied_ratios_summary.keys()):
            print(f"  {name}: {applied_ratios_summary[name]:.3f}")
        print("---------------------------------------------------\n")

        # Optimization 2: Build the merged_unet_sd from scratch with new tensors
        # This avoids an initial deepcopy of a full state_dict, then overwriting, reducing memory overhead.
        merged_unet_sd = {}

        # Iterate over all parameters in Model A's UNet state dictionary
        for key in unet_sd_A:
            param_A = unet_sd_A[key]
            current_modulation_ratio = key_to_ratio_map[key]

            # If parameter only exists in Model A, use it directly (clone to ensure it's a new tensor)
            if key not in unet_sd_B:
                print(f"Warning: UNet parameter '{key}' found in Model A but not in Model B. Inheriting from Model A only for this parameter.")
                merged_unet_sd[key] = param_A.clone() # .clone() ensures it's a new tensor, not a view/reference
                continue

            param_B = unet_sd_B[key]
            # Optimization 3: Ensure param_B is on the same device and dtype as param_A for efficient computation.
            # This minimizes CPU/GPU transfers and ensures consistent arithmetic operations.
            if param_B.dtype != param_A.dtype or param_B.device != param_A.device:
                param_B = param_B.to(dtype=param_A.dtype, device=param_A.device)

            if merge_method == "Linear Interpolation":
                # Optimization 4: Use torch.lerp for efficient, vectorized linear interpolation.
                # This function is highly optimized in PyTorch.
                merged_unet_sd[key] = torch.lerp(param_B, param_A, current_modulation_ratio)

            elif merge_method == "Add Difference":
                if key not in unet_sd_C:
                    print(f"Warning: UNet parameter '{key}' in Model A/B but not in Model C (for Add Difference). Falling back to linear interpolation for this parameter.")
                    merged_unet_sd[key] = torch.lerp(param_B, param_A, current_modulation_ratio)
                    continue

                param_C = unet_sd_C[key]
                # Optimization 3: Ensure param_C is on the same device and dtype as param_A.
                if param_C.dtype != param_A.dtype or param_C.device != param_A.device:
                    param_C = param_C.to(dtype=param_A.dtype, device=param_A.device)

                # Optimization 5: Use in-place operations with addcmul_ to reduce memory allocations.
                # merged = C + (A - C) * delta_A_factor + (B - C) * delta_B_factor
                delta_A_factor = current_modulation_ratio * a_delta_factor
                delta_B_factor = (1.0 - current_modulation_ratio) * b_delta_factor
                
                # Start with a clone of param_C, then add scaled differences.
                # .clone() ensures we don't modify the original param_C in unet_sd_C.
                merged_tensor = param_C.clone()
                # addcmul_(tensor1, tensor2, value) computes tensor1 + tensor2 * value.
                # This approach creates fewer intermediate tensors compared to chained additions/multiplications,
                # thereby reducing memory footprint and potentially improving GPU cache utilization.
                merged_tensor.addcmul_(param_A - param_C, delta_A_factor)
                merged_tensor.addcmul_(param_B - param_C, delta_B_factor)
                merged_unet_sd[key] = merged_tensor

            elif merge_method == "TIES-Merging (Simplified)":
                if key not in unet_sd_C:
                    print(f"Warning: UNet parameter '{key}' in Model A/B but not in Model C (for TIES-Merging). Falling back to linear interpolation for this parameter.")
                    merged_unet_sd[key] = torch.lerp(param_B, param_A, current_modulation_ratio)
                    continue

                param_C = unet_sd_C[key]
                # Optimization 3: Ensure param_C is on the same device and dtype as param_A.
                if param_C.dtype != param_A.dtype or param_C.device != param_A.device:
                    param_C = param_C.to(dtype=param_A.dtype, device=param_A.device)

                # Calculate alpha and beta factors, modulated by granular ratio
                alpha_k = ties_global_alpha_A * current_modulation_ratio
                beta_k = ties_global_alpha_B * (1.0 - current_modulation_ratio)

                # Apply rescaling if enabled for TIES-Merging
                if rescale_output_magnitudes:
                    sum_alpha_beta = alpha_k + beta_k
                    # Avoid division by zero and unnecessary operations if sum is 0
                    if sum_alpha_beta > 0:
                        alpha_k = alpha_k / sum_alpha_beta
                        beta_k = beta_k / sum_alpha_beta
                    else:
                        alpha_k = 0.0
                        beta_k = 0.0

                # Optimization 5: Use in-place operations with addcmul_ to reduce memory allocations.
                # Calculate combined delta factors including iterations and delta factors
                final_factor_A = alpha_k * a_delta_factor * iterations
                final_factor_B = beta_k * b_delta_factor * iterations

                # Start with a clone of param_C, then add scaled differences.
                merged_tensor = param_C.clone()
                merged_tensor.addcmul_(param_A - param_C, final_factor_A)
                merged_tensor.addcmul_(param_B - param_C, final_factor_B)
                merged_unet_sd[key] = merged_tensor
            else:
                raise ValueError(f"Internal error: Unhandled merge method '{merge_method}' during key processing.")

        # Create a new ModelPatcher object for the merged model.
        # This creates a deep copy of the entire ModelPatcher, including its underlying torch.nn.Module,
        # ensuring the merged output is an independent model without modifying inputs.
        # While deepcopy is expensive, it is the standard and safest way in ComfyUI to create a new
        # independent model instance from an existing ModelPatcher object.
        merged_model_patcher = copy.deepcopy(model_A)

        # Apply the merged UNet state dictionary to the new model patcher's UNet.
        # Optimization 6: Changed `strict=True` to `strict=False`. The original script would likely fail
        # with `strict=True` because `merged_unet_sd` only contains UNet parameters, not the full
        # model state_dict (which includes text encoders). `strict=False` allows loading a partial
        # state_dict, updating only the UNet parameters and keeping the text encoders (which were
        # deep-copied from model_A) as intended by the node's description. This maintains functionality
        # and fixes a potential runtime error in the original script.
        merged_model_patcher.model.load_state_dict(merged_unet_sd, strict=False)

        print(f"--- SDXL Granular Block Merge (Tensor Prism V4) completed. ---\n")

        return (merged_model_patcher,)

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "SDXL Block Merge (Tensor Prism)": SDXLBlockMergeGranularTensorPrism
}

# A dictionary that contains the friendly display names for nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SDXL Block Merge (Tensor Prism)": "SDXL Block Merge (Tensor Prism)"
}