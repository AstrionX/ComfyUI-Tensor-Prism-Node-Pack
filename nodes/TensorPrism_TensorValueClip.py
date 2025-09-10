import torch
import math

class TensorPrism_ClipMerge_V2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_A": ("CLIP",),
                "clip_B": ("CLIP",),
                "merge_ratio": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "help": "0.0 = CLIP A only, 1.0 = CLIP B only"
                }),
                "method": (["linear", "cosine", "slerp", "adaptive", "entropy", "dare"], {
                    "default": "linear"
                }),
            },
            "optional": {
                "text_enc_ratio": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "help": "Custom merge ratio for text encoder layers."
                }),
                "proj_ratio": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "help": "Custom merge ratio for projection layers."
                }),
                "stability_factor": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1,
                    "help": "Factor to prevent extreme values during merging"
                }),
                "preserve_magnitude": ("BOOL", {
                    "default": True,
                    "help": "Preserve original tensor magnitudes after merging"
                }),
                "safe_merge": ("BOOL", {
                    "default": True,
                    "help": "Enable additional safety checks to prevent artifacts"
                }),
            }
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("merged_clip",)
    FUNCTION = "merge_and_clip"
    CATEGORY = "Tensor Prism/Merge"

    @staticmethod
    def _safe_normalize(tensor, eps=1e-8):
        """Safely normalize a tensor without causing artifacts"""
        norm = torch.norm(tensor, p=2, dim=-1, keepdim=True)
        return tensor / torch.clamp(norm, min=eps)

    @staticmethod
    def _blend_linear(t1, t2, alpha, stability_factor=1.0):
        """Linear interpolation with stability factor"""
        result = (1 - alpha) * t1 + alpha * t2
        if stability_factor != 1.0:
            # Prevent extreme values
            t1_norm = torch.norm(t1).item()
            result_norm = torch.norm(result).item()
            if result_norm > t1_norm * stability_factor:
                result = result * (t1_norm * stability_factor) / (result_norm + 1e-8)
        return result

    @staticmethod
    def _blend_cosine(t1, t2, alpha, stability_factor=1.0):
        """Cosine interpolation with stability"""
        mu = (1 - math.cos(alpha * math.pi)) / 2
        result = (1 - mu) * t1 + mu * t2
        if stability_factor != 1.0:
            t1_norm = torch.norm(t1).item()
            result_norm = torch.norm(result).item()
            if result_norm > t1_norm * stability_factor:
                result = result * (t1_norm * stability_factor) / (result_norm + 1e-8)
        return result

    @staticmethod
    def _blend_slerp(t1, t2, alpha, stability_factor=1.0):
        """Spherical linear interpolation with improved stability"""
        original_shape = t1.shape
        t1_flat = t1.view(-1)
        t2_flat = t2.view(-1)
        
        # Preserve original magnitudes
        t1_mag = torch.norm(t1_flat)
        t2_mag = torch.norm(t2_flat)
        
        # Normalize safely
        t1_norm = TensorPrism_ClipMerge_V2._safe_normalize(t1_flat)
        t2_norm = TensorPrism_ClipMerge_V2._safe_normalize(t2_flat)
        
        # Calculate angle
        dot = torch.clamp(torch.dot(t1_norm, t2_norm), -0.99, 0.99)  # Prevent extreme angles
        theta = torch.acos(torch.abs(dot)) * alpha
        
        # Spherical interpolation
        if torch.abs(dot) > 0.9995:  # Nearly parallel vectors
            result = TensorPrism_ClipMerge_V2._blend_linear(t1, t2, alpha, stability_factor)
        else:
            relative = TensorPrism_ClipMerge_V2._safe_normalize(t2_norm - dot * t1_norm)
            result_norm = torch.cos(theta) * t1_norm + torch.sin(theta) * relative
            
            # Restore magnitude interpolation
            target_mag = (1 - alpha) * t1_mag + alpha * t2_mag
            result = result_norm * target_mag
        
        return result.view(original_shape)

    @staticmethod
    def _blend_entropy(t1, t2, alpha, stability_factor=1.0):
        """Entropy-based blending with stability"""
        # Use more stable entropy calculation
        entropy_a = torch.var(t1, unbiased=False) + 1e-8
        entropy_b = torch.var(t2, unbiased=False) + 1e-8
        
        # Prevent extreme weights
        weight_b = entropy_b / (entropy_a + entropy_b)
        weight_b = torch.clamp(weight_b, 0.1, 0.9)  # Prevent extreme ratios
        
        result = (1 - weight_b) * t1 + weight_b * t2
        return result

    @staticmethod
    def _blend_adaptive(t1, t2, alpha, stability_factor=1.0):
        """Adaptive blending based on tensor magnitudes"""
        norm_a = torch.norm(t1) + 1e-8
        norm_b = torch.norm(t2) + 1e-8
        
        # Calculate dynamic alpha with bounds
        dynamic_alpha = norm_b / (norm_a + norm_b)
        dynamic_alpha = torch.clamp(dynamic_alpha, 0.1, 0.9)
        
        return TensorPrism_ClipMerge_V2._blend_linear(t1, t2, dynamic_alpha, stability_factor)

    @staticmethod
    def _blend_dare(t1, t2, alpha, stability_factor=1.0):
        """DARE-style merging (Drop And REscale)"""
        # Calculate delta
        delta = t2 - t1
        
        # Apply dropout to delta with probability based on alpha
        if torch.rand(1).item() < alpha:
            # Scale delta by inverse probability to maintain expected value
            scaled_delta = delta / alpha
            result = t1 + scaled_delta
        else:
            result = t1
        
        return result

    @staticmethod
    def _safe_interpolate_tensors(tA, tB):
        """Safely handle tensor shape mismatches"""
        if tA.shape == tB.shape:
            return tA, tB
        
        # Handle shape mismatches more carefully
        if tA.numel() == tB.numel():
            # Same number of elements, just reshape
            tB = tB.view(tA.shape)
        elif len(tA.shape) == len(tB.shape):
            # Same dimensions, interpolate
            if len(tA.shape) >= 2:
                tB = torch.nn.functional.interpolate(
                    tB.unsqueeze(0) if tB.dim() < 4 else tB,
                    size=tA.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0) if tB.dim() < 4 else tB
            else:
                # 1D case
                tB = torch.nn.functional.interpolate(
                    tB.unsqueeze(0).unsqueeze(0),
                    size=(tA.shape[0],),
                    mode='linear'
                ).squeeze()
        else:
            # Different dimensions - use nearest neighbor matching
            print(f"Warning: Cannot merge tensors with shapes {tA.shape} and {tB.shape}")
            return tA, tA  # Use first tensor as fallback
        
        return tA, tB

    def merge_and_clip(self, clip_A, clip_B, merge_ratio, method,
                       text_enc_ratio=0.5, proj_ratio=0.5, stability_factor=1.0,
                       preserve_magnitude=True, safe_merge=True):

        # Create a copy of clip_A to modify
        merged_clip = clip_A.clone()
        
        # Get the actual model state dictionaries
        state_dict_A = clip_A.cond_stage_model.state_dict()
        state_dict_B = clip_B.cond_stage_model.state_dict()
        merged_state_dict = {}

        for key in state_dict_A.keys():
            if key in state_dict_B:
                tA, tB = state_dict_A[key], state_dict_B[key]

                # Handle tensor merging
                if isinstance(tA, torch.Tensor) and isinstance(tB, torch.Tensor):
                    # Safe tensor shape handling
                    tA, tB = self._safe_interpolate_tensors(tA, tB)
                    
                    # Skip if still incompatible
                    if tA.shape != tB.shape:
                        merged_state_dict[key] = tA
                        continue
                    
                    # Pick merge ratio for specific layer types
                    alpha = merge_ratio
                    if "text" in key.lower() or "encoder" in key.lower():
                        alpha = text_enc_ratio
                    elif "proj" in key.lower() or "projection" in key.lower():
                        alpha = proj_ratio

                    # Preserve original magnitude if requested
                    original_norm = torch.norm(tA).item() if preserve_magnitude else 1.0

                    # Choose merge method
                    try:
                        if method == "linear":
                            blended = self._blend_linear(tA, tB, alpha, stability_factor)
                        elif method == "cosine":
                            blended = self._blend_cosine(tA, tB, alpha, stability_factor)
                        elif method == "slerp":
                            blended = self._blend_slerp(tA, tB, alpha, stability_factor)
                        elif method == "entropy":
                            blended = self._blend_entropy(tA, tB, alpha, stability_factor)
                        elif method == "adaptive":
                            blended = self._blend_adaptive(tA, tB, alpha, stability_factor)
                        elif method == "dare":
                            blended = self._blend_dare(tA, tB, alpha, stability_factor)
                        else:
                            blended = tA
                    except Exception as e:
                        print(f"Warning: Merge failed for layer {key}: {e}")
                        blended = tA

                    # Safety checks
                    if safe_merge:
                        # Check for NaN/Inf
                        if torch.isnan(blended).any() or torch.isinf(blended).any():
                            print(f"Warning: NaN/Inf detected in {key}, using fallback")
                            blended = self._blend_linear(tA, tB, alpha, stability_factor)
                        
                        # Check for extreme values
                        blended_std = torch.std(blended)
                        original_std = torch.std(tA)
                        if blended_std > original_std * 3:  # Prevent extreme deviation
                            print(f"Warning: Extreme values in {key}, applying correction")
                            blended = blended * (original_std / blended_std)
                    
                    # Restore magnitude if requested
                    if preserve_magnitude and original_norm > 1e-8:
                        current_norm = torch.norm(blended).item()
                        if current_norm > 1e-8:
                            blended = blended * (original_norm / current_norm)

                    merged_state_dict[key] = blended
                else:
                    merged_state_dict[key] = tA
            else:
                merged_state_dict[key] = state_dict_A[key]

        # Load the merged state dict back into the cloned CLIP model
        try:
            merged_clip.cond_stage_model.load_state_dict(merged_state_dict)
        except Exception as e:
            print(f"Error loading merged state dict: {e}")
            return (clip_A,)  # Return original on failure

        return (merged_clip,)


NODE_CLASS_MAPPINGS = {
    "TensorPrism_ClipMerge_V2": TensorPrism_ClipMerge_V2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorPrism_ClipMerge_V2": "Clip Merge V2 (Tensor Prism)"
}