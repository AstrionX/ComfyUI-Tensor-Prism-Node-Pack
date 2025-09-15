"""
ComfyUI Tensor Prism Node Pack
===============================

Advanced model merging and enhancement nodes for ComfyUI, providing sophisticated
techniques for blending, enhancing, and manipulating Stable Diffusion models with
GPU-optimized memory management.

Author: AstrionX
Version: 1.2.0
License: GPL-3.0
Repository: https://github.com/AstrionX/ComfyUI-Tensor-Prism-Node-Pack

Features:
- Advanced model merging with multiple interpolation methods
- Spectral frequency-domain merging
- Granular SDXL block control
- Sophisticated masking system
- GPU-optimized memory management
- Cross-platform compatibility (CUDA/MPS/CPU)
"""

import os
import sys
import importlib.util
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from .TensorPrism_MainMerge import TensorPrism_MainMerge
except ImportError:
    TensorPrism_MainMerge = None

try:
    from .TensorPrism_SDXLBlockMerge import SDXLBlockMergeTensorPrism
except ImportError:
    SDXLBlockMergeTensorPrism = None

try:
    from .TensorPrism_SDXLAdvancedBlockmerge import SDXLAdvancedBlockMergeTensorPrism
except ImportError:
    SDXLAdvancedBlockMergeTensorPrism = None

try:
    from .TensorPrism_ModelMaskGenerator import TensorPrism_ModelMaskGenerator, TensorPrism_WeightedMaskMerge
except ImportError:
    TensorPrism_ModelMaskGenerator = None
    TensorPrism_WeightedMaskMerge = None

try:
    from .TensorPrism_ModelKeyFilter import TensorPrism_ModelKeyFilter
except ImportError:
    TensorPrism_ModelKeyFilter = None

try:
    from .TensorPrism_ModelMaskBlender import TensorPrism_ModelMaskBlender
except ImportError:
    TensorPrism_ModelMaskBlender = None

try:
    from .TensorPrism_ModelWeightModifier import TensorPrism_ModelWeightModifier
except ImportError:
    TensorPrism_ModelWeightModifier = None

try:
    from .TensorPrism_WeightedTensorMerge import TensorPrism_WeightedMaskMerge as TensorPrism_WeightedTensorMerge_Class
except ImportError:
    TensorPrism_WeightedTensorMerge_Class = None

try:
    from .TensorPrism_Enhancer import ModelEnhancerTensorPrism
except ImportError:
    ModelEnhancerTensorPrism = None

try:
    from .TensorPrism_LayeredBlend import TensorPrism_LayeredBlend
except ImportError:
    TensorPrism_LayeredBlend = None

try:
    from .TensorPrism_Prism import TensorPrism_FastPrism
except ImportError:
    TensorPrism_FastPrism = None

try:
    from .TensorPrism_vpredepsilonconverter import TensorPrism_EpsilonVPredConverter
except ImportError:
    TensorPrism_EpsilonVPredConverter = None

try:
    from .TensorPrism_CheckpointReroute_Notes import TensorPrism_CheckpointReroute_Notes
except ImportError:
    TensorPrism_CheckpointReroute_Notes = None

try:
    from .TensorPrism_AdvancedClipMerge import AdvancedCLIPMerge
except ImportError:
    AdvancedCLIPMerge = None

# Version info
__version__ = "1.2.0"
__author__ = "AstrionX"
__description__ = "Advanced model merging and enhancement nodes for ComfyUI"

NODE_CLASS_MAPPINGS = {}

# Core merging nodes
if TensorPrism_MainMerge:
    NODE_CLASS_MAPPINGS["TensorPrism_MainMerge"] = TensorPrism_MainMerge

# SDXL specific nodes
if SDXLBlockMergeTensorPrism:
    NODE_CLASS_MAPPINGS["SDXL Block Merge (Tensor Prism)"] = SDXLBlockMergeTensorPrism

if SDXLAdvancedBlockMergeTensorPrism:
    NODE_CLASS_MAPPINGS["SDXLAdvancedBlockMergeTensorPrism"] = SDXLAdvancedBlockMergeTensorPrism

# Masking and filtering nodes
if TensorPrism_ModelMaskGenerator:
    NODE_CLASS_MAPPINGS["TensorPrism_ModelMaskGenerator"] = TensorPrism_ModelMaskGenerator

if TensorPrism_WeightedMaskMerge:
    NODE_CLASS_MAPPINGS["TensorPrism_WeightedMaskMerge"] = TensorPrism_WeightedMaskMerge

if TensorPrism_ModelKeyFilter:
    NODE_CLASS_MAPPINGS["TensorPrism_ModelKeyFilter"] = TensorPrism_ModelKeyFilter

if TensorPrism_ModelMaskBlender:
    NODE_CLASS_MAPPINGS["TensorPrism_ModelMaskBlender"] = TensorPrism_ModelMaskBlender

# Enhancement and transformation nodes
if TensorPrism_ModelWeightModifier:
    NODE_CLASS_MAPPINGS["TensorPrism_ModelWeightModifier"] = TensorPrism_ModelWeightModifier

if TensorPrism_WeightedTensorMerge_Class:
    NODE_CLASS_MAPPINGS["TensorPrism_WeightedTensorMerge"] = TensorPrism_WeightedTensorMerge_Class

if ModelEnhancerTensorPrism:
    NODE_CLASS_MAPPINGS["ModelEnhancerTensorPrism"] = ModelEnhancerTensorPrism

if TensorPrism_LayeredBlend:
    NODE_CLASS_MAPPINGS["TensorPrism_LayeredBlend"] = TensorPrism_LayeredBlend

if TensorPrism_FastPrism:
    NODE_CLASS_MAPPINGS["TensorPrism_Prism"] = TensorPrism_FastPrism

# CLIP merging nodes
if AdvancedCLIPMerge:
    NODE_CLASS_MAPPINGS["AdvancedCLIPMerge"] = AdvancedCLIPMerge

# Epsilon/V-Pred merge nodes
if TensorPrism_EpsilonVPredConverter:
    NODE_CLASS_MAPPINGS["TensorPrism_EpsilonVPredConverter"] = TensorPrism_EpsilonVPredConverter

if TensorPrism_CheckpointReroute_Notes:
    NODE_CLASS_MAPPINGS["TensorPrism_CheckpointReroute_Notes"] = TensorPrism_CheckpointReroute_Notes

NODE_DISPLAY_NAME_MAPPINGS = {}

# Core merging nodes
if "TensorPrism_MainMerge" in NODE_CLASS_MAPPINGS:
    NODE_DISPLAY_NAME_MAPPINGS["TensorPrism_MainMerge"] = "Main Merge (Tensor Prism)"

# SDXL specific nodes
if "SDXL Block Merge (Tensor Prism)" in NODE_CLASS_MAPPINGS:
    NODE_DISPLAY_NAME_MAPPINGS["SDXL Block Merge (Tensor Prism)"] = "SDXL Block Merge (Tensor Prism)"

if "SDXLAdvancedBlockMergeTensorPrism" in NODE_CLASS_MAPPINGS:
    NODE_DISPLAY_NAME_MAPPINGS["SDXLAdvancedBlockMergeTensorPrism"] = "SDXL Advanced Block Merge (Tensor Prism)"

# Masking and filtering nodes
if "TensorPrism_ModelMaskGenerator" in NODE_CLASS_MAPPINGS:
    NODE_DISPLAY_NAME_MAPPINGS["TensorPrism_ModelMaskGenerator"] = "Model Mask Generator (Tensor Prism)"

if "TensorPrism_WeightedMaskMerge" in NODE_CLASS_MAPPINGS:
    NODE_DISPLAY_NAME_MAPPINGS["TensorPrism_WeightedMaskMerge"] = "Weighted Mask Merge (Tensor Prism)"

if "TensorPrism_ModelKeyFilter" in NODE_CLASS_MAPPINGS:
    NODE_DISPLAY_NAME_MAPPINGS["TensorPrism_ModelKeyFilter"] = "Model Key Filter (Tensor Prism)"

if "TensorPrism_ModelMaskBlender" in NODE_CLASS_MAPPINGS:
    NODE_DISPLAY_NAME_MAPPINGS["TensorPrism_ModelMaskBlender"] = "Mask Blender (Tensor Prism)"

# Enhancement and transformation nodes
if "TensorPrism_ModelWeightModifier" in NODE_CLASS_MAPPINGS:
    NODE_DISPLAY_NAME_MAPPINGS["TensorPrism_ModelWeightModifier"] = "Model Weight Modifier (Tensor Prism)"

if "TensorPrism_WeightedTensorMerge" in NODE_CLASS_MAPPINGS:
    NODE_DISPLAY_NAME_MAPPINGS["TensorPrism_WeightedTensorMerge"] = "Weighted Tensor Merge (Tensor Prism)"

if "ModelEnhancerTensorPrism" in NODE_CLASS_MAPPINGS:
    NODE_DISPLAY_NAME_MAPPINGS["ModelEnhancerTensorPrism"] = "Model Enhancer (Tensor Prism)"

if "TensorPrism_LayeredBlend" in NODE_CLASS_MAPPINGS:
    NODE_DISPLAY_NAME_MAPPINGS["TensorPrism_LayeredBlend"] = "Layered Blend (Tensor Prism)"

if "TensorPrism_Prism" in NODE_CLASS_MAPPINGS:
    NODE_DISPLAY_NAME_MAPPINGS["TensorPrism_Prism"] = "Prism (Tensor Prism)"

# CLIP merging nodes
if "AdvancedCLIPMerge" in NODE_CLASS_MAPPINGS:
    NODE_DISPLAY_NAME_MAPPINGS["AdvancedCLIPMerge"] = "Advanced CLIP Merge (Tensor Prism)"

# Epsilon/V-Pred merge nodes
if "TensorPrism_EpsilonVPredConverter" in NODE_CLASS_MAPPINGS:
    NODE_DISPLAY_NAME_MAPPINGS["TensorPrism_EpsilonVPredConverter"] = "Epsilon/V-Pred Converter (Tensor Prism)"

if "TensorPrism_CheckpointReroute_Notes" in NODE_CLASS_MAPPINGS:
    NODE_DISPLAY_NAME_MAPPINGS["TensorPrism_CheckpointReroute_Notes"] = "Checkpoint Reroute + Notes (Tensor Prism)"

NODE_CATEGORIES = {}

for node_key in NODE_CLASS_MAPPINGS.keys():
    if "MainMerge" in node_key or "LayeredBlend" in node_key:
        NODE_CATEGORIES[node_key] = "Tensor Prism/Core"
    elif "SDXL" in node_key:
        NODE_CATEGORIES[node_key] = "Tensor_Prism/Merge"
    elif "Mask" in node_key:
        NODE_CATEGORIES[node_key] = "Tensor_Prism/Mask"
    elif "Weight" in node_key or "Enhancer" in node_key:
        NODE_CATEGORIES[node_key] = "Tensor_Prism/Transform"
    elif "EpsilonVPred" in node_key:
        NODE_CATEGORIES[node_key] = "Tensor_Prism/Merge"
    elif "CLIP" in node_key:
        NODE_CATEGORIES[node_key] = "Tensor_Prism/CLIP"
    else:
        NODE_CATEGORIES[node_key] = "Tensor_Prism/Core"

def check_dependencies():
    """Check if required dependencies are available."""
    required_packages = [
        ("torch", "PyTorch >= 1.12.0"),
        ("numpy", "NumPy >= 1.21.0"), 
        ("psutil", "psutil >= 5.8.0"),
    ]
    
    missing_packages = []
    
    for package_name, description in required_packages:
        try:
            importlib.import_module(package_name)
        except ImportError:
            missing_packages.append(description)
    
    if missing_packages:
        print(f"[TensorPrism] Warning: Missing dependencies:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("[TensorPrism] Some features may not work properly.")
    
    return len(missing_packages) == 0

def get_system_info():
    """Get system information for optimization."""
    try:
        import torch
        import psutil
        
        info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            "cpu_count": psutil.cpu_count(),
            "total_ram": round(psutil.virtual_memory().total / (1024**3), 1),
        }
        
        if info["cuda_available"]:
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_memory"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
        
        return info
    except Exception as e:
        print(f"[TensorPrism] Could not get system info: {e}")
        return {}

def print_welcome_message():
    """Print welcome message with system information."""
    print("\n" + "="*60)
    print("🎭 TensorPrism Node Pack Loaded")
    print("="*60)
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print(f"Nodes: {len(NODE_CLASS_MAPPINGS)}")
    
    # System info
    sys_info = get_system_info()
    if sys_info:
        print(f"\n📊 System Information:")
        print(f"  PyTorch: {sys_info.get('torch_version', 'Unknown')}")
        print(f"  CPU Cores: {sys_info.get('cpu_count', 'Unknown')}")
        print(f"  RAM: {sys_info.get('total_ram', 'Unknown')}GB")
        
        if sys_info.get("cuda_available"):
            print(f"  CUDA: Available ({sys_info.get('cuda_device_count', 0)} device(s))")
            print(f"  GPU Memory: {sys_info.get('cuda_memory', 'Unknown')}GB")
        elif sys_info.get("mps_available"):
            print(f"  MPS: Available (Apple Silicon)")
        else:
            print(f"  GPU: CPU fallback mode")
    
    print(f"\n🚀 Available Nodes:")
    for category in set(NODE_CATEGORIES.values()):
        print(f"\n  {category}:")
        for node_class, node_category in NODE_CATEGORIES.items():
            if node_category == category:
                display_name = NODE_DISPLAY_NAME_MAPPINGS.get(node_class, node_class)
                print(f"    • {display_name}")
    
    print(f"\n💡 Memory Management:")
    if sys_info.get("cuda_memory", 0) >= 24:
        print("  Recommended: Default settings (24GB+ GPU)")
    elif sys_info.get("cuda_memory", 0) >= 12:
        print("  Recommended: Memory limit 8GB, auto precision")
    elif sys_info.get("cuda_memory", 0) >= 8:
        print("  Recommended: Memory limit 6GB, CPU fallback for large merges")
    else:
        print("  Recommended: CPU processing for best stability")
    
    print("="*60 + "\n")

# Initialize the package
def __init_package():
    """Initialize the package and perform startup checks."""
    try:
        # Check dependencies
        deps_ok = check_dependencies()
        
        # Print welcome message
        print_welcome_message()
        
        if not deps_ok:
            print("[TensorPrism] Warning: Some dependencies are missing. Please install them for full functionality.")
        
        print("[TensorPrism] Package initialized successfully!")
        
    except Exception as e:
        print(f"[TensorPrism] Error during initialization: {e}")
        print("[TensorPrism] Package may not function correctly.")

# Run initialization
__init_package()

# Export for ComfyUI
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS", 
    "__version__",
    "__author__",
    "__description__"
]