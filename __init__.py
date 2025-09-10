"""
DEBUG VERSION: ComfyUI Node Pack __init__.py
Use this to identify what's going wrong
"""

print("[TensorPrism] Starting to load nodes...")

# Step 1: Try basic imports first
try:
    print("[TensorPrism] Attempting imports...")
    
    # ADJUST THESE IMPORTS TO MATCH YOUR ACTUAL FILE STRUCTURE
    # Option A: If your nodes are in separate files
    from .core_merge import CoreMergeNode
    from .model_enhancer import ModelEnhancerNode
    from .layered_blend import LayeredBlendNode
    from .prism import PrismNode
    from .sdxl_block_merge import SDXLBlockMergeNode
    from .model_mask_generator import ModelMaskGeneratorNode
    from .weighted_mask_merge import WeightedMaskMergeNode
    
    print("[TensorPrism] All imports successful!")
    
except ImportError as e:
    print(f"[TensorPrism] Import error: {e}")
    print("[TensorPrism] Trying alternative import method...")
    
    try:
        # Option B: If all nodes are in one file
        from .nodes import (
            CoreMergeNode, 
            ModelEnhancerNode, 
            LayeredBlendNode, 
            PrismNode, 
            SDXLBlockMergeNode, 
            ModelMaskGeneratorNode, 
            WeightedMaskMergeNode
        )
        print("[TensorPrism] Alternative imports successful!")
        
    except ImportError as e2:
        print(f"[TensorPrism] Alternative import also failed: {e2}")
        
        # Option C: If everything is in __init__.py itself
        try:
            # Import from current file - adjust based on your actual setup
            print("[TensorPrism] Trying to load from current file...")
            # Your node classes would be defined here or imported differently
            
        except Exception as e3:
            print(f"[TensorPrism] All import methods failed: {e3}")
            print("[TensorPrism] Please check your file structure and node class definitions")

# Step 2: Define node mappings (adjust class names to match what you actually have)
try:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    
    # Only add nodes that successfully imported
    if 'CoreMergeNode' in locals():
        NODE_CLASS_MAPPINGS["TensorPrism_CoreMerge"] = CoreMergeNode
        NODE_DISPLAY_NAME_MAPPINGS["TensorPrism_CoreMerge"] = "Core Merge"
        
    if 'ModelEnhancerNode' in locals():
        NODE_CLASS_MAPPINGS["TensorPrism_ModelEnhancer"] = ModelEnhancerNode
        NODE_DISPLAY_NAME_MAPPINGS["TensorPrism_ModelEnhancer"] = "Model Enhancer"
        
    if 'LayeredBlendNode' in locals():
        NODE_CLASS_MAPPINGS["TensorPrism_LayeredBlend"] = LayeredBlendNode
        NODE_DISPLAY_NAME_MAPPINGS["TensorPrism_LayeredBlend"] = "Layered Blend"
        
    if 'PrismNode' in locals():
        NODE_CLASS_MAPPINGS["TensorPrism_Prism"] = PrismNode
        NODE_DISPLAY_NAME_MAPPINGS["TensorPrism_Prism"] = "Prism"
        
    if 'SDXLBlockMergeNode' in locals():
        NODE_CLASS_MAPPINGS["TensorPrism_SDXLBlockMerge"] = SDXLBlockMergeNode
        NODE_DISPLAY_NAME_MAPPINGS["TensorPrism_SDXLBlockMerge"] = "SDXL Block Merge"
        
    if 'ModelMaskGeneratorNode' in locals():
        NODE_CLASS_MAPPINGS["TensorPrism_ModelMaskGenerator"] = ModelMaskGeneratorNode
        NODE_DISPLAY_NAME_MAPPINGS["TensorPrism_ModelMaskGenerator"] = "Model Mask Generator"
        
    if 'WeightedMaskMergeNode' in locals():
        NODE_CLASS_MAPPINGS["TensorPrism_WeightedMaskMerge"] = WeightedMaskMergeNode
        NODE_DISPLAY_NAME_MAPPINGS["TensorPrism_WeightedMaskMerge"] = "Weighted Mask Merge"
    
    print(f"[TensorPrism] Successfully registered {len(NODE_CLASS_MAPPINGS)} nodes:")
    for key, value in NODE_CLASS_MAPPINGS.items():
        print(f"  - {key}: {value.__name__}")
        
    if len(NODE_CLASS_MAPPINGS) == 0:
        print("[TensorPrism] WARNING: No nodes were registered! Check your class names and imports.")
        
except Exception as e:
    print(f"[TensorPrism] Error creating node mappings: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Step 3: Verify node classes have required methods
print("[TensorPrism] Checking node class requirements...")
for name, node_class in NODE_CLASS_MAPPINGS.items():
    if hasattr(node_class, 'INPUT_TYPES'):
        print(f"  ✓ {name} has INPUT_TYPES")
    else:
        print(f"  ✗ {name} missing INPUT_TYPES method")
        
    if hasattr(node_class, 'RETURN_TYPES'):
        print(f"  ✓ {name} has RETURN_TYPES")
    else:
        print(f"  ✗ {name} missing RETURN_TYPES")

print("[TensorPrism] Initialization complete!")

# Required by ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
