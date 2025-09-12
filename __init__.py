"""
ComfyUI Tensor Prism Node Pack
===============================

Advanced model merging and enhancement nodes for ComfyUI, providing sophisticated
techniques for blending, enhancing, and manipulating Stable Diffusion models with
GPU-optimized memory management.

Author: AstrionX
Version: 1.0.0
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

# Node class imports
from .tensor_prism_main_merge import TensorPrismMainMerge
from .tensor_prism_prism import TensorPrismPrism  
from .tensor_prism_sdxl_block_merge import TensorPrismSDXLBlockMerge
from .tensor_prism_sdxl_advanced_block_merge import TensorPrismSDXLAdvancedBlockMerge
from .tensor_prism_model_mask_generator import TensorPrismModelMaskGenerator
from .tensor_prism_weighted_mask_merge import TensorPrismWeightedMaskMerge
from .tensor_prism_model_key_filter import TensorPrismModelKeyFilter
from .tensor_prism_mask_blender import TensorPrismMaskBlender
from .tensor_prism_model_weight_modifier import TensorPrismModelWeightModifier

# Version info
__version__ = "1.0.0"
__author__ = "AstrionX"
__description__ = "Advanced model merging and enhancement nodes for ComfyUI"

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    # Core merging nodes
    "TensorPrismMainMerge": TensorPrismMainMerge,
    "TensorPrismPrism": TensorPrismPrism,
    
    # SDXL specific nodes
    "TensorPrismSDXLBlockMerge": TensorPrismSDXLBlockMerge,
    "TensorPrismSDXLAdvancedBlockMerge": TensorPrismSDXLAdvancedBlockMerge,
    
    # Masking and filtering nodes
    "TensorPrismModelMaskGenerator": TensorPrismModelMaskGenerator,
    "TensorPrismWeightedMaskMerge": TensorPrismWeightedMaskMerge,
    "TensorPrismModelKeyFilter": TensorPrismModelKeyFilter,
    "TensorPrismMaskBlender": TensorPrismMaskBlender,
    
    # Transformation nodes
    "TensorPrismModelWeightModifier": TensorPrismModelWeightModifier,
}

# Display name mappings for ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    # Core merging nodes
    "TensorPrismMainMerge": "Main Merge",
    "TensorPrismPrism": "Prism",
    
    # SDXL specific nodes  
    "TensorPrismSDXLBlockMerge": "SDXL Block Merge",
    "TensorPrismSDXLAdvancedBlockMerge": "SDXL Advanced Block Merge",
    
    # Masking and filtering nodes
    "TensorPrismModelMaskGenerator": "Model Mask Generator",
    "TensorPrismWeightedMaskMerge": "Weighted Mask Merge", 
    "TensorPrismModelKeyFilter": "Model Key Filter",
    "TensorPrismMaskBlender": "Mask Blender",
    
    # Transformation nodes
    "TensorPrismModelWeightModifier": "Model Weight Modifier",
}

# Category definitions for node organization (matching your documentation)
NODE_CATEGORIES = {
    # Core nodes
    "TensorPrismMainMerge": "Tensor Prism/Core",
    "TensorPrismPrism": "Tensor Prism/Core",
    
    # SDXL merging nodes (note: using underscores as in your docs)
    "TensorPrismSDXLBlockMerge": "Tensor_Prism/Merge", 
    "TensorPrismSDXLAdvancedBlockMerge": "Tensor_Prism/Merge",
    
    # Masking nodes
    "TensorPrismModelMaskGenerator": "Tensor Prism/Mask",
    "TensorPrismWeightedMaskMerge": "Tensor Prism/Mask",
    "TensorPrismModelKeyFilter": "Tensor_Prism/Mask",  # Note: using underscore as in your docs
    "TensorPrismMaskBlender": "Tensor_Prism/Mask",     # Note: using underscore as in your docs
    
    # Transform nodes
    "TensorPrismModelWeightModifier": "Tensor_Prism/Transform",
}

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
                display_name = NODE_DISPLAY_NAME_MAPPINGS[node_class]
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
