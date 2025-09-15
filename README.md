### ComfyUI-Tensor-Prism-Node-Pack
## Developer Notes

My First ComfyUI Node Pack, vibe coded with Gemini 2.5 Flash and Claude 4. Feel free to publish the models you make and link them to me I'd like to be able to see the models, and see what they're about to see if I need to add more nodes or if the nodes are good and make really good quality checkpoint models. This is also a node pack for those familiar with merging models.

# TensorPrism ComfyUI Node Pack

Advanced model merging and enhancement nodes for ComfyUI, providing sophisticated techniques for blending, enhancing, and manipulating Stable Diffusion models with GPU-optimized memory management.

## Features

### Core Merging Nodes

- **Main Merge**: Advanced model merging with multiple interpolation methods (linear, slerp, cosine, directional, frequency, stochastic)
- **Prism**: Fast spectral merging with frequency-based blending techniques including spectral blend, frequency bands, magnitude weighting, adaptive mixing, and harmonic merging
- **SDXL Block Merge**: Granular control over individual SDXL UNet blocks with support for TIES merging
- **SDXL Advanced Block Merge**: GPU-optimized block merging with intelligent memory management for any GPU size (including 12GB and smaller cards)
- **Epsilon/V-Pred Block Merge**: Granular block-level model merging with V-Pred/Epsilon conversion and individual control for input blocks 0-8, middle blocks 0-2, and output blocks 0-8

### Conversion and Processing Nodes

- **Epsilon/V-Pred Converter**: Pure converter between V-Prediction and Epsilon prediction types with configurable conversion strength and smart layer targeting
- **V-Pred/Epsilon Converter**: Advanced prediction type conversion with critical layer identification (output blocks, middle blocks) and secondary layer processing (time embedding, input blocks)

### Advanced Mask System

- **Model Mask Generator**: Create sophisticated masks for selective model merging with layer-based, block-based, attention-only, feedforward-only, custom patterns, random sparse, and depth gradient options
- **Weighted Mask Merge**: Apply masks to control where and how models are blended with tensor-level precision
- **Model Key Filter**: Memory-efficient filtering of model parameters with batch processing for large models
- **Mask Blender**: Combine multiple masks with various blending modes (Add, Multiply, Max, Min, Linear Blend, Exponential Blend)

### CLIP Processing

- **Advanced CLIP Merge**: Sophisticated CLIP merging with multiple interpolation methods:
  - **Linear**: Standard interpolation
  - **SLERP**: Spherical Linear Interpolation for smooth blending
  - **Cosine**: Cosine-based smooth transitions
  - **Weighted Average**: Magnitude-based weighting
  - **Spectral Blend**: Frequency domain blending with separate magnitude/phase control
  - Layer-specific bias controls for attention, feedforward, embedding, and normalization layers
  - Norm preservation options for maintaining model stability

### Model Transformation

- **Model Weight Modifier**: Memory-efficient weight modification with operations like multiply, add, set value, clamp magnitude, and scale max absolute value

### Utility Nodes

- **Checkpoint Reroute + Notes**: Convenient rerouting node for MODEL, CLIP, and VAE connections with optional note-taking functionality for workflow organization and documentation

### Advanced Features

- **GPU-Optimized Memory Management**: Intelligent memory allocation and cleanup for cards with limited VRAM
- **Batch Processing**: Process large models in memory-efficient batches
- **Multiple Merging Algorithms**: Including SLERP, frequency domain blending, stochastic merging, and TIES merging
- **Spectral Analysis**: Frequency-based model analysis and merging
- **Adaptive Precision**: Automatic FP16/FP32 selection based on available memory
- **Cross-Device Compatibility**: Works with CUDA, MPS, and CPU backends
- **Prediction Type Conversion**: Automatic conversion between Epsilon and V-Prediction model types with mathematical accuracy
- **Advanced Memory Context Management**: Context managers for automatic cleanup and memory optimization
- **Workflow Organization**: Built-in documentation and organization tools for complex merging workflows

## Installation

### Method 1: ComfyUI Manager
Note: Not available yet there.
1. Open ComfyUI Manager
2. Search for "TensorPrism"
3. Click Install

### Method 2: Manual Installation (OPTIONAL)
1. Clone or download this repository
2. Place the entire folder in your `ComfyUI/custom_nodes/` directory
3. Restart ComfyUI

### Method 3: Git Clone (OPTIONAL)
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/AstrionX/ComfyUI-Tensor-Prism-Node-Pack.git
```

## Usage

### Basic Model Merging
Use the **Main Merge** node to blend two models with various interpolation methods:
- Connect two MODEL inputs
- Adjust merge ratio (0.0 = Model A only, 1.0 = Model B only)
- Choose merging method based on your needs:
  - **Linear**: Standard interpolation
  - **SLERP**: Spherical interpolation for smoother blending
  - **Cosine**: Smoother transitions with cosine curve
  - **Directional**: Vector-based interpolation
  - **Frequency**: FFT-based frequency domain blending
  - **Stochastic**: Random pattern-based merging

### Advanced Spectral Merging
The **Prism** node offers frequency-domain merging:
- **Spectral Blend**: Magnitude-based frequency separation
- **Frequency Bands**: Different weights for low/high frequency components
- **Magnitude Weighted**: Stronger tensor gets more influence
- **Adaptive Mix**: Similarity-based adaptive merging
- **Harmonic Merge**: Phase relationship-based blending

### Granular SDXL Block Control
**SDXL Block Merge** and **SDXL Advanced Block Merge** provide:
- Individual control over input blocks (0-11)
- Individual control over output blocks (0-11)
- Middle block control (3 components)
- Time embedding and label embedding control
- Final output layer control
- Memory-optimized processing for any GPU size
- Advanced memory management with context managers
- Automatic GPU/CPU fallback based on available memory

### V-Pred/Epsilon Block Merging
The **Epsilon/V-Pred Block Merge** node provides:
- Automatic conversion between V-Prediction and Epsilon prediction types
- Granular control over individual blocks:
  - Input blocks 0-8 (9 individual controls)
  - Middle blocks 0-2 (3 individual controls)
  - Output blocks 0-8 (9 individual controls)
  - Final output layer control
- Support for mixed prediction type merging (e.g., Epsilon + V-Pred → Epsilon)
- Mathematical conversion factors for accurate type conversion

### Pure Prediction Type Conversion
The **Epsilon/V-Pred Converter** provides:
- Pure conversion between prediction types without merging
- Configurable conversion strength (0.0 to 2.0)
- Smart layer targeting:
  - **Critical layers**: Output blocks, middle blocks, final output
  - **Secondary layers**: Time embedding, late input blocks
- Empirically derived conversion factors for mathematical accuracy
- Automatic model options updating for ComfyUI compatibility

### Advanced CLIP Merging
The **Advanced CLIP Merge** node offers:
- **Multiple Interpolation Methods**:
  - **Linear**: Standard weighted average
  - **SLERP**: Spherical interpolation for vector-like parameters
  - **Cosine**: Smooth cosine-based transitions
  - **Weighted Average**: Magnitude-based automatic weighting
  - **Spectral Blend**: Frequency domain blending with separate magnitude/phase control
  
- **Layer-Specific Controls**:
  - **Attention Bias**: Adjust merging for attention layers
  - **Feedforward Bias**: Control feedforward network blending
  - **Embedding Bias**: Modify text/positional embedding merge ratios
  - **Normalization Bias**: Adjust layer normalization merging
  
- **Advanced Options**:
  - **Preserve Norms**: Maintain original parameter magnitudes
  - **Memory Efficient**: Optimize for GPU memory usage
  - **Spectral Alpha**: Control phase vs magnitude blending in spectral mode

### Advanced Masking System
Create sophisticated merging patterns:

1. **Generate Masks**:
   - Layer-based: Target specific model layers
   - Block-based: Target transformer blocks
   - Component-based: Target attention or feedforward layers
   - Custom patterns: Use regex patterns
   - Random sparse: Create random merging patterns
   - Depth gradients: Gradual transitions through model depth

2. **Filter and Blend Masks**:
   - Filter model keys by component type
   - Combine masks with various blending modes
   - Memory-efficient batch processing

3. **Apply Masks**:
   - Use masks to control merge strength per parameter
   - Selective merging based on model structure
   - Tensor-level precision control

### Model Weight Modification
Transform model weights directly:
- **Multiply**: Scale weights by a factor
- **Add**: Add constant values
- **Set Value**: Replace weights with specific values
- **Clamp Magnitude**: Limit weight magnitudes
- **Scale Max Abs**: Normalize based on reference model

### Workflow Organization
Use the **Checkpoint Reroute + Notes** node to:
- Clean up complex workflows with multiple model connections
- Add documentation and notes directly in your workflow
- Maintain MODEL, CLIP, and VAE connections without modification
- Keep track of model versions and merge parameters
- Zero processing overhead - direct passthrough routing
- Organize workflow structure for better readability

## Node Reference

| Node | Category | Purpose |
|------|----------|---------|
| Main Merge | Tensor Prism/Core | Advanced merging with multiple methods |
| Prism | Tensor Prism/Core | Spectral frequency-domain merging |
| SDXL Block Merge | Tensor_Prism/Merge | Basic granular SDXL merging |
| SDXL Advanced Block Merge | Tensor_Prism/Merge | GPU-optimized SDXL merging |
| Epsilon/V-Pred Converter | Tensor_Prism/Convert | Pure prediction type conversion |
| Advanced CLIP Merge | Tensor_Prism/CLIP | Sophisticated CLIP merging with multiple methods |
| Model Mask Generator | Tensor Prism/Mask | Create structural masks |
| Weighted Mask Merge | Tensor Prism/Mask | Apply masks to merging |
| Model Key Filter | Tensor_Prism/Mask | Filter model parameters |
| Mask Blender | Tensor_Prism/Mask | Combine multiple masks |
| Model Weight Modifier | Tensor_Prism/Transform | Direct weight manipulation |
| Checkpoint Reroute + Notes | Tensor_Prism/Utilities | MODEL/CLIP/VAE rerouting with workflow documentation |

## Memory Management

The TensorPrism pack includes advanced memory management features:

- **Automatic GPU Detection**: Optimizes for your specific GPU memory
- **Adaptive Batch Sizes**: Adjusts processing based on available memory
- **Precision Selection**: Automatic FP16/FP32 based on memory constraints
- **Progressive Cleanup**: Aggressive garbage collection for low-memory systems
- **CPU Fallback**: Automatic fallback when GPU memory is insufficient
- **Memory Context Managers**: Automatic cleanup and resource management
- **Threshold-Based Processing**: Memory usage monitoring with configurable limits

### Recommended Settings by GPU:
- **24GB+ (RTX 4090, etc.)**: Use default settings, batch size 50+
- **12GB (RTX 4070 Ti, etc.)**: Set memory limit to 8GB, enable auto precision, batch size 30-50
- **8GB (RTX 4060 Ti, etc.)**: Set memory limit to 6GB, force CPU for large merges, batch size 10-30
- **6GB and below**: Use CPU processing for best stability, enable aggressive cleanup

## Requirements

- ComfyUI
- PyTorch >= 1.12.0
- NumPy >= 1.21.0
- psutil >= 5.8.0 (for memory management)

## Tips and Best Practices

1. **Start Conservative**: Begin with lower merge ratios (0.3-0.7) and adjust based on results
2. **Use SLERP for Dissimilar Models**: When merging very different models, SLERP often produces better results
3. **Leverage Spectral Methods**: Frequency domain merging can preserve details better than linear methods
4. **Use Masks for Precision**: Create masks to merge only specific model components
5. **Memory Management**: Monitor memory usage and adjust batch sizes for your hardware
6. **Experiment with Spectral Parameters**: Different frequency biases can dramatically change results
7. **Layer-Selective Merging**: Use depth gradients for smooth transitions through model layers
8. **Prediction Type Awareness**: Use the Epsilon/V-Pred nodes when working with models of different prediction types
9. **CLIP Merging Strategy**: Use spectral blend for CLIP when preserving text understanding is critical
10. **Conversion Strength**: Start with 1.0 conversion strength and adjust if results seem over/under-converted
11. **Document Your Workflows**: Use the Checkpoint Reroute + Notes node to keep track of your merging experiments
12. **Organize Complex Workflows**: Use reroute nodes with notes to create clean, documented workflow structures
13. **Layer-Specific CLIP Control**: Use attention/feedforward bias to fine-tune CLIP behavior for specific use cases
14. **Zero Overhead Documentation**: The reroute node adds no processing time while providing workflow organization

## Troubleshooting

- **Memory Issues**: Reduce batch sizes, lower memory limits, or enable CPU fallback
- **Poor Results**: Try different merging methods or adjust spectral parameters
- **Compatibility**: Ensure models are the same architecture (SDXL with SDXL, etc.)
- **Slow Performance**: Check if you're accidentally using CPU when GPU is available
- **Artifacts**: Try more conservative merge ratios or use SLERP for smoother blending
- **Prediction Type Issues**: Use the Epsilon/V-Pred nodes for automatic type conversion
- **CLIP Problems**: Use preserve_norms=True and lower merge ratios for CLIP stability
- **Conversion Artifacts**: Reduce conversion strength or use pure converter instead of block merge
- **Workflow Complexity**: Use Checkpoint Reroute + Notes nodes to organize and document complex merging chains

## Performance Tips

- **Batch Size**: Larger batches are more efficient but use more memory
- **Precision Mode**: FP16 saves memory but may affect quality on some operations
- **Memory Cleanup**: Enable aggressive cleanup for systems with limited RAM
- **Device Selection**: Let the system auto-detect optimal device unless you have specific needs
- **Workflow Organization**: Use reroute nodes to reduce visual complexity without performance impact
- **CLIP Memory**: CLIP merging is less memory-intensive than UNet merging
- **Conversion vs Merging**: Pure conversion uses less memory than block-level merge+convert

## License

https://www.gnu.org/licenses/gpl-3.0.en.html

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Changelog

### Version 1.2.0
- Added **Advanced CLIP Merge** node with multiple interpolation methods and layer-specific controls
- Added **Epsilon/V-Pred Converter** node for pure prediction type conversion
- Enhanced **SDXL Advanced Block Merge** with improved memory management and context managers
- Improved **Weighted Mask Merge** with tensor-level precision control
- Added spectral blending capabilities to CLIP merging
- Enhanced memory management with automatic GPU/CPU fallback
- Improved layer identification and targeting for prediction type conversion
- Better mathematical accuracy in conversion factors
- Added **Epsilon/V-Pred Block Merge** node with granular block control and prediction type conversion
- Added **Checkpoint Reroute + Notes (Tensor Prism)** node for workflow organization and documentation
- Enhanced block-level merging capabilities with individual layer control (input blocks 0-8, middle blocks 0-2, output blocks 0-8)
- Improved prediction type conversion with mathematical accuracy
- Better workflow documentation and organization features
- Zero-overhead utility nodes for complex workflow management
- Improved tooltip system and user experience enhancements

### Version 1.1.0
- GPU-optimized memory management
- Cross-platform compatibility (CUDA/MPS/CPU)
- Bunch of Bug Fixes
- Addition of 3 Nodes

### Version 1.0.0
- Initial release
- Core merging nodes with advanced interpolation methods
- Advanced mask system with filtering and blending
- Spectral analysis and frequency-domain merging
- Advanced mask system with filtering and blending
- Support for SDXL models with granular block control
- Model weight modification tools