"""
TensorPrism ComfyUI Node Pack - Direct Import Version
"""

print("TensorPrism: Starting node pack initialization...")

# Import all dependencies directly - fail fast if missing
import torch
import math
import copy
import gc
import psutil
import re
import logging
import threading
import time
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum
import traceback
import numpy as np

print("TensorPrism: All dependencies imported successfully")

# Import all nodes directly
from .TensorPrism_MainMerge import TensorPrism_MainMerge
from .TensorPrism_LayeredBlend import TensorPrism_LayeredBlend
from .TensorPrism_Prism import TensorPrism_FastPrism
from .TensorPrism_Enhancer import ModelEnhancerTensorPrism
from .TensorPrism_MaskedTensorMerge import TensorPrism_WeightedMaskMerge
from .TensorPrism_ModelMaskGenerator import TensorPrism_ModelMaskGenerator
from .TensorPrism_ModelKeyFilter import TensorPrism_ModelKeyFilter
from .TensorPrism_ModelMaskBlender import TensorPrism_ModelMaskBlender
from .TensorPrism_ModelWeightModifier import TensorPrism_ModelWeightModifier
from .TensorPrism_SDXLBlockMerge import SDXLBlockMergeGranularTensorPrism
from .TensorPrism_SDXLAdvancedBlockmerge import SDXLAdvancedBlockMergeTensorPrism

print("TensorPrism: All node classes imported successfully")

# Direct node mappings - no conditionals
NODE_CLASS_MAPPINGS = {
    "TensorPrism_MainMerge": TensorPrism_MainMerge,
    "TensorPrism_LayeredBlend": TensorPrism_LayeredBlend,
    "TensorPrism_Prism": TensorPrism_FastPrism,
    "ModelEnhancerTensorPrism": ModelEnhancerTensorPrism,
    "TensorPrism_WeightedMaskMerge": TensorPrism_WeightedMaskMerge,
    "TensorPrism_ModelMaskGenerator": TensorPrism_ModelMaskGenerator,
    "TensorPrism_ModelKeyFilter": TensorPrism_ModelKeyFilter,
    "TensorPrism_ModelMaskBlender": TensorPrism_ModelMaskBlender,
    "TensorPrism_ModelWeightModifier": TensorPrism_ModelWeightModifier,
    "TensorPrism_SDXLBlockMerge": SDXLBlockMergeGranularTensorPrism,
    "SDXLAdvancedBlockMergeTensorPrism": SDXLAdvancedBlockMergeTensorPrism,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorPrism_MainMerge": "Main Merge (Tensor Prism)",
    "TensorPrism_LayeredBlend": "Layered Blend (Tensor Prism)",
    "TensorPrism_Prism": "Prism (Tensor Prism)",
    "ModelEnhancerTensorPrism": "Model Enhancer (Tensor Prism)",
    "TensorPrism_WeightedMaskMerge": "Weighted Mask Merge (Tensor Prism)",
    "TensorPrism_ModelMaskGenerator": "Model Mask Generator (Tensor Prism)",
    "TensorPrism_ModelKeyFilter": "Model Key Filter (Tensor Prism)",
    "TensorPrism_ModelMaskBlender": "Mask Blender (Tensor Prism)",
    "TensorPrism_ModelWeightModifier": "Model Weight Modifier (Tensor Prism)",
    "TensorPrism_SDXLBlockMerge": "SDXL Block Merge (Tensor Prism)",
    "SDXLAdvancedBlockMergeTensorPrism": "SDXL Advanced Block Merge (Tensor Prism)",
}

print(f"TensorPrism: Successfully loaded {len(NODE_CLASS_MAPPINGS)} nodes:")
for node_name in sorted(NODE_CLASS_MAPPINGS.keys()):
    print(f"  - {NODE_DISPLAY_NAME_MAPPINGS[node_name]}")

# Required exports
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
