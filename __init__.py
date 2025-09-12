"""
TensorPrism ComfyUI Node Pack
Advanced model merging and enhancement nodes for ComfyUI
"""

# Import all your node classes
try:
    from .TensorPrism_LayeredBlend import TensorPrism_LayeredBlend
except ImportError as e:
    print(f"Warning: Could not import TensorPrism_LayeredBlend: {e}")
    TensorPrism_LayeredBlend = None

try:
    from .TensorPrism_MainMerge import TensorPrism_MainMerge
except ImportError as e:
    print(f"Warning: Could not import TensorPrism_MainMerge: {e}")
    TensorPrism_MainMerge = None

try:
    from .TensorPrism_MaskedTensorMerge import TensorPrism_WeightedMaskMerge
except ImportError as e:
    print(f"Warning: Could not import TensorPrism_WeightedMaskMerge: {e}")
    TensorPrism_WeightedMaskMerge = None

try:
    from .TensorPrism_ModelKeyFilter import TensorPrism_ModelKeyFilter
except ImportError as e:
    print(f"Warning: Could not import TensorPrism_ModelKeyFilter: {e}")
    TensorPrism_ModelKeyFilter = None

try:
    from .TensorPrism_ModelMaskBlender import TensorPrism_ModelMaskBlender
except ImportError as e:
    print(f"Warning: Could not import TensorPrism_ModelMaskBlender: {e}")
    TensorPrism_ModelMaskBlender = None

try:
    from .TensorPrism_ModelMaskGenerator import (
        TensorPrism_ModelMaskGenerator,
        TensorPrism_WeightedMaskMerge as MaskGenWeightedMerge
    )
except ImportError as e:
    print(f"Warning: Could not import from TensorPrism_ModelMaskGenerator: {e}")
    TensorPrism_ModelMaskGenerator = None
    MaskGenWeightedMerge = None

try:
    from .TensorPrism_ModelWeightModifier import TensorPrism_ModelWeightModifier
except ImportError as e:
    print(f"Warning: Could not import TensorPrism_ModelWeightModifier: {e}")
    TensorPrism_ModelWeightModifier = None

try:
    from .TensorPrism_Prism import TensorPrism_FastPrism
except ImportError as e:
    print(f"Warning: Could not import TensorPrism_FastPrism: {e}")
    TensorPrism_FastPrism = None

try:
    from .TensorPrism_SDXLAdvancedBlockmerge import SDXLAdvancedBlockMergeTensorPrism
except ImportError as e:
    print(f"Warning: Could not import SDXLAdvancedBlockMergeTensorPrism: {e}")
    SDXLAdvancedBlockMergeTensorPrism = None

try:
    from .TensorPrism_SDXLBlockMerge import SDXLBlockMergeGranularTensorPrism
except ImportError as e:
    print(f"Warning: Could not import SDXLBlockMergeGranularTensorPrism: {e}")
    SDXLBlockMergeGranularTensorPrism = None

# Node class mappings - this is what ComfyUI looks for
NODE_CLASS_MAPPINGS = {}

# Add nodes to mapping if they were successfully imported
if TensorPrism_LayeredBlend is not None:
    NODE_CLASS_MAPPINGS["TensorPrism_LayeredBlend"] = TensorPrism_LayeredBlend

if TensorPrism_MainMerge is not None:
    NODE_CLASS_MAPPINGS["TensorPrism_MainMerge"] = TensorPrism_MainMerge

if TensorPrism_WeightedMaskMerge is not None:
    NODE_CLASS_MAPPINGS["TensorPrism_WeightedMaskMerge"] = TensorPrism_WeightedMaskMerge

if TensorPrism_ModelKeyFilter is not None:
    NODE_CLASS_MAPPINGS["TensorPrism_ModelKeyFilter"] = TensorPrism_ModelKeyFilter

if TensorPrism_ModelMaskBlender is not None:
    NODE_CLASS_MAPPINGS["TensorPrism_ModelMaskBlender"] = TensorPrism_ModelMaskBlender

if TensorPrism_ModelMaskGenerator is not None:
    NODE_CLASS_MAPPINGS["TensorPrism_ModelMaskGenerator"] = TensorPrism_ModelMaskGenerator

# Handle the duplicate WeightedMaskMerge from ModelMaskGenerator
if MaskGenWeightedMerge is not None and TensorPrism_WeightedMaskMerge is None:
    NODE_CLASS_MAPPINGS["TensorPrism_WeightedMaskMerge"] = MaskGenWeightedMerge

if TensorPrism_ModelWeightModifier is not None:
    NODE_CLASS_MAPPINGS["TensorPrism_ModelWeightModifier"] = TensorPrism_ModelWeightModifier

if TensorPrism_FastPrism is not None:
    NODE_CLASS_MAPPINGS["TensorPrism_Prism"] = TensorPrism_FastPrism

if SDXLAdvancedBlockMergeTensorPrism is not None:
    NODE_CLASS_MAPPINGS["SDXLAdvancedBlockMergeTensorPrism"] = SDXLAdvancedBlockMergeTensorPrism

if SDXLBlockMergeGranularTensorPrism is not None:
    NODE_CLASS_MAPPINGS["TensorPrism_SDXLBlockMerge"] = SDXLBlockMergeGranularTensorPrism

# Display name mappings - these are the human-readable names shown in ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorPrism_LayeredBlend": "Layered Blend (Tensor Prism)",
    "TensorPrism_MainMerge": "Main Merge (Tensor Prism)", 
    "TensorPrism_WeightedMaskMerge": "Weighted Mask Merge (Tensor Prism)",
    "TensorPrism_ModelKeyFilter": "Model Key Filter (Tensor Prism)",
    "TensorPrism_ModelMaskBlender": "Mask Blender (Tensor Prism)",
    "TensorPrism_ModelMaskGenerator": "Model Mask Generator (Tensor Prism)",
    "TensorPrism_ModelWeightModifier": "Model Weight Modifier (Tensor Prism)",
    "TensorPrism_Prism": "Prism (Tensor Prism)",
    "SDXLAdvancedBlockMergeTensorPrism": "SDXL Advanced Block Merge (Tensor Prism)",
    "TensorPrism_SDXLBlockMerge": "SDXL Block Merge (Tensor Prism)",
    "ModelEnhancerTensorPrism": "Model Enhancer (Tensor Prism)",
}

# Export what ComfyUI needs
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Print loaded nodes for debugging
loaded_nodes = list(NODE_CLASS_MAPPINGS.keys())
if loaded_nodes:
    print(f"TensorPrism: Successfully loaded {len(loaded_nodes)} nodes:")
    for node_name in sorted(loaded_nodes):
        display_name = NODE_DISPLAY_NAME_MAPPINGS.get(node_name, node_name)
        print(f"  - {display_name}")
else:
    print("TensorPrism: Warning - No nodes were loaded successfully!")
    print("Check that all Python files are in the same directory and have no import errors.")
