"""
TensorPrism Checkpoint Reroute Notes Node
========================================

A utility node for rerouting checkpoint components (MODEL, CLIP, VAE) with optional notes
for workflow documentation and organization.

Author: AstrionX
Version: 1.2.0
License: GPL-3.0
"""

class TensorPrism_CheckpointReroute_Notes:
    """
    A utility node that reroutes MODEL, CLIP, and VAE inputs to outputs with optional notes.
    Useful for workflow organization and documentation.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model input to reroute"}),
                "clip": ("CLIP", {"tooltip": "CLIP input to reroute"}),
                "vae": ("VAE", {"tooltip": "VAE input to reroute"})
            },
            "optional": {
                "notes": ("STRING", {
                    "default": "", 
                    "multiline": True,
                    "tooltip": "Optional notes for workflow documentation"
                })
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "reroute"
    CATEGORY = "Tensor_Prism/Utilities"
    
    DESCRIPTION = """
    Reroutes checkpoint components (MODEL, CLIP, VAE) with optional documentation.
    
    Features:
    • Pass-through routing for MODEL, CLIP, and VAE
    • Optional notes field for workflow documentation
    • Maintains full compatibility with all checkpoint types
    • Zero processing overhead - direct passthrough
    
    Use Cases:
    • Organizing complex workflows
    • Adding documentation to checkpoint flows
    • Creating clean connection points
    • Workflow readability improvements
    """

    def reroute(self, model, clip, vae, notes="", **kwargs):
        """
        Reroute the inputs directly to outputs with optional notes processing.
        
        Args:
            model: The model to reroute
            clip: The CLIP to reroute  
            vae: The VAE to reroute
            notes: Optional notes (stored but not processed)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            tuple: (model, clip, vae) - Direct passthrough of inputs
        """
        # Optional: Log notes if provided (for debugging/workflow tracking)
        if notes and notes.strip():
            print(f"[TensorPrism Checkpoint Reroute] Notes: {notes.strip()}")
        
        # Direct passthrough - zero processing overhead
        return model, clip, vae

    @classmethod
    def IS_CHANGED(cls, model, clip, vae, notes="", **kwargs):
        """
        Since this is a passthrough node, we don't need change detection.
        Return None to indicate no caching needed.
        """
        return None

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "TensorPrism_CheckpointReroute_Notes": TensorPrism_CheckpointReroute_Notes
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorPrism_CheckpointReroute_Notes": "Checkpoint Reroute + Notes (Tensor Prism)"
}

# Export the class for import
__all__ = ["TensorPrism_CheckpointReroute_Notes"]