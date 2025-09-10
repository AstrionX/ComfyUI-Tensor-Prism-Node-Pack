"""
TensorPrism ComfyUI Node Pack
Advanced model merging and enhancement nodes for ComfyUI
"""

from .TensorPrism_CoreMerge import NODE_CLASS_MAPPINGS as CoreMerge_Mappings, NODE_DISPLAY_NAME_MAPPINGS as CoreMerge_Display
from .TensorPrism_Enhancer import NODE_CLASS_MAPPINGS as Enhancer_Mappings, NODE_DISPLAY_NAME_MAPPINGS as Enhancer_Display
from .TensorPrism_LayeredBlend import NODE_CLASS_MAPPINGS as LayeredBlend_Mappings, NODE_DISPLAY_NAME_MAPPINGS as LayeredBlend_Display
from .TensorPrism_MaskedTensorMerge import NODE_CLASS_MAPPINGS as MaskedMerge_Mappings, NODE_DISPLAY_NAME_MAPPINGS as MaskedMerge_Display
from .TensorPrism_ModelMaskGenerator import NODE_CLASS_MAPPINGS as ModelMask_Mappings, NODE_DISPLAY_NAME_MAPPINGS as ModelMask_Display
from .TensorPrism_Prism import NODE_CLASS_MAPPINGS as Prism_Mappings, NODE_DISPLAY_NAME_MAPPINGS as Prism_Display
from .TensorPrism_SDXLBlockMergeGranular import NODE_CLASS_MAPPINGS as SDXLBlock_Mappings, NODE_DISPLAY_NAME_MAPPINGS as SDXLBlock_Display

# Combine all node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Add all mappings
for mapping in [CoreMerge_Mappings, Enhancer_Mappings, LayeredBlend_Mappings, 
                MaskedMerge_Mappings, ModelMask_Mappings, Prism_Mappings, SDXLBlock_Mappings]:
    NODE_CLASS_MAPPINGS.update(mapping)

for mapping in [CoreMerge_Display, Enhancer_Display, LayeredBlend_Display,
                MaskedMerge_Display, ModelMask_Display, Prism_Display, SDXLBlock_Display]:
    NODE_DISPLAY_NAME_MAPPINGS.update(mapping)

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Version info
WEB_DIRECTORY = "./web"
__version__ = "1.0.0"
