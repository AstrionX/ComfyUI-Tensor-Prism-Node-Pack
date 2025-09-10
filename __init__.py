"""
TensorPrism ComfyUI Node Pack
Advanced model merging and enhancement nodes for ComfyUI
"""

print("[TensorPrism] Loading node pack...")

# Import all your node classes
from .TensorPrism_Enhancer import ModelEnhancerTensorPrism
from .TensorPrism_LayeredBlend import TensorPrism_LayeredBlend
from .TensorPrism_MainMerge import TensorPrism_MainMerge
from .TensorPrism_MaskedTensorMerge import TensorPrism_WeightedMaskMerge
from .TensorPrism_ModelMaskGenerator import TensorPrism_ModelMaskGenerator, TensorPrism_WeightedMaskMerge as MaskMerge
from .TensorPrism_Prism import TensorPrism_FastPrism
from .TensorPrism_SDXLBlockMergeGranular import SDXLBlockMergeGranularTensorPrism

print("[TensorPrism] All node classes imported successfully!")

# Consolidate all NODE_CLASS_MAPPINGS from individual files
NODE_CLASS_MAPPINGS = {
    # From TensorPrism_Enhancer.py
    "ModelEnhancerTensorPrism": ModelEnhancerTensorPrism,
    
    # From TensorPrism_LayeredBlend.py
    "TensorPrism_LayeredBlend": TensorPrism_LayeredBlend,
    
    # From TensorPrism_MainMerge.py
    "TensorPrism_MainMerge": TensorPrism_MainMerge,
    
    # From TensorPrism_MaskedTensorMerge.py
    "TensorPrism_WeightedMaskMerge": TensorPrism_WeightedMaskMerge,
    
    # From TensorPrism_ModelMaskGenerator.py
    "TensorPrism_ModelMaskGenerator": TensorPrism_ModelMaskGenerator,
    
    # From TensorPrism_Prism.py
    "TensorPrism_Prism": TensorPrism_FastPrism,
    
    # From TensorPrism_SDXLBlockMergeGranular.py
    "SDXL Block Merge (Tensor Prism)": SDXLBlockMergeGranularTensorPrism,
}

# Consolidate all NODE_DISPLAY_NAME_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = {
    # From TensorPrism_Enhancer.py
    "ModelEnhancerTensorPrism": "Model Enhancer (Tensor Prism)",
    
    # From TensorPrism_LayeredBlend.py
    "TensorPrism_LayeredBlend": "Layered Blend (Tensor Prism)",
    
    # From TensorPrism_MainMerge.py
    "TensorPrism_MainMerge": "Main Merge (Tensor Prism)",
    
    # From TensorPrism_MaskedTensorMerge.py
    "TensorPrism_WeightedMaskMerge": "Weighted Mask Merge (Tensor Prism)",
    
    # From TensorPrism_ModelMaskGenerator.py
    "TensorPrism_ModelMaskGenerator": "Model Mask Generator (Tensor Prism)",
    
    # From TensorPrism_Prism.py
    "TensorPrism_Prism": "Prism (Tensor Prism)",
    
    # From TensorPrism_SDXLBlockMergeGranular.py
    "SDXL Block Merge (Tensor Prism)": "SDXL Block Merge (Tensor Prism)",
}

# Web extensions (if you have any)
WEB_DIRECTORY = "./web"

# Package info
__version__ = "1.0.0"
__author__ = "AstrionX"

# Required by ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print(f"[TensorPrism] Successfully registered {len(NODE_CLASS_MAPPINGS)} nodes:")
for key, value in NODE_CLASS_MAPPINGS.items():
    print(f"  - {key}")

print("[TensorPrism] Node pack loaded successfully!")
