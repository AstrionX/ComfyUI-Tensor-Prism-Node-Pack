"""
TensorPrism ComfyUI Node Pack
Advanced model merging and enhancement nodes for ComfyUI
"""

print("[TensorPrism] Loading node pack...")

# Import all your node classes
from .TensorPrism_MainMerge import TensorPrism_MainMerge
from .TensorPrism_MaskedTensorMerge import TensorPrism_WeightedMaskMerge
from .TensorPrism_ModelMaskGenerator import TensorPrism_ModelMaskGenerator, TensorPrism_WeightedMaskMerge as MaskMerge
from .TensorPrism_Prism import TensorPrism_FastPrism
from .TensorPrism_SDXLBlockMerge import SDXLBlockMergeGranularTensorPrism
from .TensorPrism_SDXLAdvancedBlockmerge import SDXLAdvancedBlockMergeTensorPrism
from .TensorPrism_ModelKeyFilter import TensorPrism_ModelKeyFilter
from .TensorPrism_ModelMaskBlender import TensorPrism_ModelMaskBlender
from .TensorPrism_ModelWeightModifier import TensorPrism_ModelWeightModifier

print("[TensorPrism] All node classes imported successfully!")

# Consolidate all NODE_CLASS_MAPPINGS from individual files
NODE_CLASS_MAPPINGS = {
    # Core Merging Nodes
    "TensorPrism_MainMerge": TensorPrism_MainMerge,
    "TensorPrism_Prism": TensorPrism_FastPrism,
    
    # SDXL Block Merge Nodes
    "SDXL Block Merge (Tensor Prism)": SDXLBlockMergeGranularTensorPrism,
    "SDXLAdvancedBlockMergeTensorPrism": SDXLAdvancedBlockMergeTensorPrism,
    
    # Mask System Nodes
    "TensorPrism_ModelMaskGenerator": TensorPrism_ModelMaskGenerator,
    "TensorPrism_WeightedMaskMerge": TensorPrism_WeightedMaskMerge,
    "TensorPrism_ModelKeyFilter": TensorPrism_ModelKeyFilter,
    "TensorPrism_ModelMaskBlender": TensorPrism_ModelMaskBlender,
    
    # Model Modification Nodes
    "TensorPrism_ModelWeightModifier": TensorPrism_ModelWeightModifier,
}

# Consolidate all NODE_DISPLAY_NAME_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = {
    # Core Merging Nodes
    "TensorPrism_MainMerge": "Main Merge (Tensor Prism)",
    "TensorPrism_Prism": "Prism (Tensor Prism)",
    
    # SDXL Block Merge Nodes
    "SDXL Block Merge (Tensor Prism)": "SDXL Block Merge (Tensor Prism)",
    "SDXLAdvancedBlockMergeTensorPrism": "SDXL Advanced Block Merge (Tensor Prism)",
    
    # Mask System Nodes
    "TensorPrism_ModelMaskGenerator": "Model Mask Generator (Tensor Prism)",
    "TensorPrism_WeightedMaskMerge": "Weighted Mask Merge (Tensor Prism)",
    "TensorPrism_ModelKeyFilter": "Model Key Filter (Tensor Prism)",
    "TensorPrism_ModelMaskBlender": "Mask Blender (Tensor Prism)",
    
    # Model Modification Nodes
    "TensorPrism_ModelWeightModifier": "Model Weight Modifier (Tensor Prism)",
}

# Web extensions (if you have any)
WEB_DIRECTORY = "./web"

# Package info
__version__ = "1.1.0"
__author__ = "AstrionX"

# Required by ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print(f"[TensorPrism] Successfully registered {len(NODE_CLASS_MAPPINGS)} nodes:")
for key, value in NODE_CLASS_MAPPINGS.items():
    print(f"  - {key}")

print("[TensorPrism] Node pack loaded successfully!")