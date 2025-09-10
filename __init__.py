"""
TensorPrism ComfyUI Node Pack
Advanced model merging and enhancement nodes for ComfyUI
"""

# Import all your node classes here
from .nodes.core_merge import CoreMergeNode
from .nodes.model_enhancer import ModelEnhancerNode
from .nodes.layered_blend import LayeredBlendNode
from .nodes.prism import PrismNode
from .nodes.sdxl_block_merge import SDXLBlockMergeNode
from .nodes.model_mask_generator import ModelMaskGeneratorNode
from .nodes.weighted_mask_merge import WeightedMaskMergeNode

# Define the node class mappings - this is CRITICAL for ComfyUI to recognize your nodes
NODE_CLASS_MAPPINGS = {
    "TensorPrism_CoreMerge": CoreMergeNode,
    "TensorPrism_ModelEnhancer": ModelEnhancerNode,
    "TensorPrism_LayeredBlend": LayeredBlendNode,
    "TensorPrism_Prism": PrismNode,
    "TensorPrism_SDXLBlockMerge": SDXLBlockMergeNode,
    "TensorPrism_ModelMaskGenerator": ModelMaskGeneratorNode,
    "TensorPrism_WeightedMaskMerge": WeightedMaskMergeNode,
}

# Define display names for the UI - this makes your nodes look professional
NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorPrism_CoreMerge": "Core Merge",
    "TensorPrism_ModelEnhancer": "Model Enhancer",
    "TensorPrism_LayeredBlend": "Layered Blend",
    "TensorPrism_Prism": "Prism",
    "TensorPrism_SDXLBlockMerge": "SDXL Block Merge",
    "TensorPrism_ModelMaskGenerator": "Model Mask Generator",
    "TensorPrism_WeightedMaskMerge": "Weighted Mask Merge",
}

# Web extensions (if you have any custom web UI components)
WEB_DIRECTORY = "./web"

# Optional: Add version info
__version__ = "1.0.0"

# Optional: Add author info
__author__ = "AstrionX"

# This is required by ComfyUI to identify this as a node package
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print(f"[TensorPrism] Loaded {len(NODE_CLASS_MAPPINGS)} nodes successfully!")
