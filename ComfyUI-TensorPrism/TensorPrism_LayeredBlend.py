import torch
import math
import re
from typing import Dict, Any, Optional
import comfy.model_management

class TensorPrism_LayeredBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_A": ("MODEL",),
                "model_B": ("MODEL",),
                "text_encoder_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "text_encoder_method": (["linear", "slerp", "cosine"],),
                "unet_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "unet_method": (["linear", "slerp", "cosine"],),
                "time_embed_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "input_blocks_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "middle_block_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "output_blocks_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "out_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "vae_A": ("VAE",),
                "vae_B": ("VAE",),
                "vae_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "vae_method": (["linear", "slerp", "cosine"],),
            }
        }

    RETURN_TYPES = ("MODEL", "VAE",)
    RETURN_NAMES = ("merged_model", "merged_vae",)
    FUNCTION = "blend_models"
    CATEGORY = "Tensor Prism/Merge"

    @staticmethod
    def _blend_linear(t1: torch.Tensor, t2: torch.Tensor, alpha: float) -> torch.Tensor:
        """Linear interpolation between two tensors"""
        return (1 - alpha) * t1 + alpha * t2

    @staticmethod
    def _blend_slerp(t1: torch.Tensor, t2: torch.Tensor, alpha: float) -> torch.Tensor:
        """Spherical linear interpolation with proper error handling"""
        if t1.numel() == 0 or t2.numel() == 0:
            return t1
            
        t1_flat = t1.view(-1)
        t2_flat = t2.view(-1)
        
        # Normalize vectors
        t1_norm = torch.nn.functional.normalize(t1_flat, dim=0, eps=1e-8)
        t2_norm = torch.nn.functional.normalize(t2_flat, dim=0, eps=1e-8)
        
        # Calculate dot product and handle edge cases
        dot = torch.clamp(torch.dot(t1_norm, t2_norm), -1.0 + 1e-7, 1.0 - 1e-7)
        
        # Handle near-parallel vectors
        if abs(dot.item()) > 0.9995:
            return TensorPrism_LayeredBlend._blend_linear(t1, t2, alpha)
        
        # SLERP calculation
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        
        if sin_theta.abs() < 1e-6:
            return TensorPrism_LayeredBlend._blend_linear(t1, t2, alpha)
        
        w1 = torch.sin((1 - alpha) * theta) / sin_theta
        w2 = torch.sin(alpha * theta) / sin_theta
        
        result = w1 * t1_norm + w2 * t2_norm
        
        # Restore original magnitude
        original_norm = torch.norm(t1_flat) * (1 - alpha) + torch.norm(t2_flat) * alpha
        result = result * original_norm
        
        return result.view(t1.shape)

    @staticmethod
    def _blend_cosine(t1: torch.Tensor, t2: torch.Tensor, alpha: float) -> torch.Tensor:
        """Cosine interpolation for smoother transitions"""
        mu = (1 - math.cos(alpha * math.pi)) / 2
        return (1 - mu) * t1 + mu * t2

    def _get_component_type(self, param_name: str) -> str:
        """Identify which component a parameter belongs to"""
        param_lower = param_name.lower()
        
        # Text encoder patterns
        if any(pattern in param_lower for pattern in ['cond_stage_model', 'text_encoder', 'clip', 'transformer.text_model']):
            return 'text_encoder'
        
        # UNet component patterns
        if any(pattern in param_lower for pattern in ['model.diffusion_model', 'unet']):
            # More specific UNet components
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
            else:
                return 'unet'
        
        # VAE patterns
        if any(pattern in param_lower for pattern in ['first_stage_model', 'vae', 'decoder', 'encoder']):
            return 'vae'
        
        # Default to unet for unrecognized parameters
        return 'unet'

    def _apply_blend_method(self, t1: torch.Tensor, t2: torch.Tensor, strength: float, method: str) -> torch.Tensor:
        """Apply the specified blending method"""
        if method == "linear":
            return self._blend_linear(t1, t2, strength)
        elif method == "slerp":
            return self._blend_slerp(t1, t2, strength)
        elif method == "cosine":
            return self._blend_cosine(t1, t2, strength)
        else:
            return self._blend_linear(t1, t2, strength)

    def blend_models(self, model_A, model_B,
                     text_encoder_strength, text_encoder_method,
                     unet_strength, unet_method,
                     time_embed_strength, input_blocks_strength,
                     middle_block_strength, output_blocks_strength, out_strength,
                     vae_A: Optional[Any] = None, vae_B: Optional[Any] = None,
                     vae_strength: float = 0.5, vae_method: str = "linear"):
        """
        Blend models with component-specific controls
        """
        
        # Clone model A as the base
        merged_model = model_A.clone()
        
        # Get state dictionaries
        state_dict_A = model_A.model.state_dict()
        state_dict_B = model_B.model.state_dict()
        
        # Create patches dictionary
        patches = {}
        
        # Component strength mapping
        component_strengths = {
            'text_encoder': (text_encoder_strength, text_encoder_method),
            'time_embed': (time_embed_strength, unet_method),
            'input_blocks': (input_blocks_strength, unet_method),
            'middle_block': (middle_block_strength, unet_method),
            'output_blocks': (output_blocks_strength, unet_method),
            'out': (out_strength, unet_method),
            'unet': (unet_strength, unet_method),  # Default UNet components
            'vae': (0.0, "linear")  # VAE handled separately
        }
        
        # Process each parameter
        for key in state_dict_A.keys():
            if key in state_dict_B:
                t1 = state_dict_A[key]
                t2 = state_dict_B[key]

                if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
                    if t1.shape == t2.shape:
                        try:
                            # Determine component type
                            component_type = self._get_component_type(key)
                            
                            # Get strength and method for this component
                            strength, method = component_strengths.get(component_type, (unet_strength, unet_method))
                            
                            # Skip if strength is 0
                            if strength == 0:
                                continue
                            
                            # Apply blending
                            merged_tensor = self._apply_blend_method(t1, t2, strength, method)
                            
                            # Only add patch if there's a meaningful difference
                            diff = merged_tensor - t1
                            if torch.abs(diff).max() > 1e-8:
                                patches[key] = (diff,)
                                
                        except Exception as e:
                            print(f"Warning: Failed to blend parameter {key}: {e}")
                            continue
        
        # Apply patches to the merged model
        if patches:
            merged_model.add_patches(patches, 1.0)
        
        # Handle VAE blending if provided
        merged_vae = None
        if vae_A is not None and vae_B is not None:
            try:
                merged_vae = self._blend_vaes(vae_A, vae_B, vae_strength, vae_method)
            except Exception as e:
                print(f"Warning: Failed to blend VAEs: {e}")
                merged_vae = vae_A  # Fallback to VAE A
        
        return (merged_model, merged_vae)

    def _blend_vaes(self, vae_A, vae_B, strength: float, method: str):
        """Blend two VAE models"""
        if strength == 0:
            return vae_A
        if strength == 1:
            return vae_B
            
        # Clone VAE A as base
        merged_vae = vae_A.clone() if hasattr(vae_A, 'clone') else vae_A
        
        try:
            # Get state dicts if available
            if hasattr(vae_A, 'first_stage_model') and hasattr(vae_B, 'first_stage_model'):
                state_dict_A = vae_A.first_stage_model.state_dict()
                state_dict_B = vae_B.first_stage_model.state_dict()
                
                # Create patches for VAE
                vae_patches = {}
                
                for key in state_dict_A.keys():
                    if key in state_dict_B:
                        t1 = state_dict_A[key]
                        t2 = state_dict_B[key]
                        
                        if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
                            if t1.shape == t2.shape:
                                merged_tensor = self._apply_blend_method(t1, t2, strength, method)
                                diff = merged_tensor - t1
                                
                                if torch.abs(diff).max() > 1e-8:
                                    vae_patches[key] = (diff,)
                
                # Apply VAE patches if the VAE supports it
                if vae_patches and hasattr(merged_vae, 'add_patches'):
                    merged_vae.add_patches(vae_patches, 1.0)
                    
        except Exception as e:
            print(f"VAE blending failed, using VAE A: {e}")
            return vae_A
            
        return merged_vae


# Register node
NODE_CLASS_MAPPINGS = {
    "TensorPrism_LayeredBlend": TensorPrism_LayeredBlend
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorPrism_LayeredBlend": "Layered Blend (Tensor Prism)"
}