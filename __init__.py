"""
TensorPrism ComfyUI Node Pack
Advanced model merging and enhancement nodes for ComfyUI
"""

import sys
import os
import traceback

print("[TensorPrism] Loading node pack...")

# Initialize mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Define individual node imports with error handling
def safe_import(module_name, class_names, display_names):
    """Safely import nodes with error handling"""
    try:
        module = __import__(f".{module_name}", package=__name__, fromlist=class_names.keys())
        for class_key, class_name in class_names.items():
            if hasattr(module, class_name):
                NODE_CLASS_MAPPINGS[class_key] = getattr(module, class_name)
                NODE_DISPLAY_NAME_MAPPINGS[class_key] = display_names[class_key]
                print(f"[TensorPrism] ✓ Loaded: {display_names[class_key]}")
            else:
                print(f"[TensorPrism] ⚠ Class {class_name} not found in {module_name}")
    except ImportError as e:
        print(f"[TensorPrism] ⚠ Failed to import {module_name}: {e}")
        return False
    except Exception as e:
        print(f"[TensorPrism] ⚠ Error loading {module_name}: {e}")
        return False
    return True

# Core Merging Nodes
safe_import("TensorPrism_MainMerge", 
           {"TensorPrism_MainMerge": "TensorPrism_MainMerge"},
           {"TensorPrism_MainMerge": "Main Merge (Tensor Prism)"})

safe_import("TensorPrism_Prism",
           {"TensorPrism_Prism": "TensorPrism_FastPrism"},
           {"TensorPrism_Prism": "Prism (Tensor Prism)"})

# Legacy/Original Nodes (may not exist yet)
safe_import("TensorPrism_Enhancer",
           {"ModelEnhancerTensorPrism": "ModelEnhancerTensorPrism"},
           {"ModelEnhancerTensorPrism": "Model Enhancer (Tensor Prism)"})

safe_import("TensorPrism_LayeredBlend",
           {"TensorPrism_LayeredBlend": "TensorPrism_LayeredBlend"},
           {"TensorPrism_LayeredBlend": "Layered Blend (Tensor Prism)"})

# SDXL Block Merge Nodes
safe_import("TensorPrism_SDXLBlockMerge",
           {"SDXL Block Merge (Tensor Prism)": "SDXLBlockMergeGranularTensorPrism"},
           {"SDXL Block Merge (Tensor Prism)": "SDXL Block Merge (Tensor Prism)"})

safe_import("TensorPrism_SDXLAdvancedBlockmerge",
           {"SDXLAdvancedBlockMergeTensorPrism": "SDXLAdvancedBlockMergeTensorPrism"},
           {"SDXLAdvancedBlockMergeTensorPrism": "SDXL Advanced Block Merge (Tensor Prism)"})

# Mask System Nodes
safe_import("TensorPrism_ModelMaskGenerator",
           {"TensorPrism_ModelMaskGenerator": "TensorPrism_ModelMaskGenerator",
            "TensorPrism_WeightedMaskMerge": "TensorPrism_WeightedMaskMerge"},
           {"TensorPrism_ModelMaskGenerator": "Model Mask Generator (Tensor Prism)",
            "TensorPrism_WeightedMaskMerge": "Weighted Mask Merge (Tensor Prism)"})

safe_import("TensorPrism_MaskedTensorMerge",
           {"TensorPrism_WeightedMaskMerge_Alt": "TensorPrism_WeightedMaskMerge"},
           {"TensorPrism_WeightedMaskMerge_Alt": "Weighted Mask Merge Alt (Tensor Prism)"})

safe_import("TensorPrism_ModelKeyFilter",
           {"TensorPrism_ModelKeyFilter": "TensorPrism_ModelKeyFilter"},
           {"TensorPrism_ModelKeyFilter": "Model Key Filter (Tensor Prism)"})

safe_import("TensorPrism_ModelMaskBlender",
           {"TensorPrism_ModelMaskBlender": "TensorPrism_ModelMaskBlender"},
           {"TensorPrism_ModelMaskBlender": "Mask Blender (Tensor Prism)"})

# Model Modification Nodes
safe_import("TensorPrism_ModelWeightModifier",
           {"TensorPrism_ModelWeightModifier": "TensorPrism_ModelWeightModifier"},
           {"TensorPrism_ModelWeightModifier": "Model Weight Modifier (Tensor Prism)"})

# Web extensions (if you have any)
WEB_DIRECTORY = "./web"

# Package info
__version__ = "1.1.0"
__author__ = "AstrionX"

# Required by ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print(f"[TensorPrism] Successfully registered {len(NODE_CLASS_MAPPINGS)} nodes:")
for key in sorted(NODE_CLASS_MAPPINGS.keys()):
    print(f"  - {NODE_DISPLAY_NAME_MAPPINGS[key]}")

if len(NODE_CLASS_MAPPINGS) == 0:
    print("[TensorPrism] ⚠ WARNING: No nodes were loaded! Check your file structure and imports.")
else:
    print("[TensorPrism] Node pack loaded successfully!")
