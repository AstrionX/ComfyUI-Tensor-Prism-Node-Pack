### ComfyUI-Tensor-Prism-Node-Pack
## Developer Notes

My First ComfyUI Node Pack, vibe coded with Gemini 2.5 Flash and Claude 4.

# TensorPrism ComfyUI Node Pack

Advanced model merging and enhancement nodes for ComfyUI, providing sophisticated techniques for blending, enhancing, and manipulating Stable Diffusion models with GPU-optimized memory management.

## Features

### Core Merging Nodes

- **Main Merge**: Advanced model merging with multiple interpolation methods (linear, slerp, cosine, directional, frequency, stochastic)
- **Prism**: Fast spectral merging with frequency-based blending techniques including spectral blend, frequency bands, magnitude weighting, adaptive mixing, and harmonic merging
- **SDXL Block Merge**: Granular control over individual SDXL UNet blocks with support for TIES merging
- **SDXL Advanced Block Merge**: GPU-optimized block merging with intelligent memory management for any GPU size (including 12GB and smaller cards)

### Advanced Mask System

- **Model Mask Generator**: Create sophisticated masks for selective model merging with layer-based, block-based, attention-only, feedforward-only, custom patterns, random sparse, and depth gradient options
- **Weighted Mask Merge**: Apply masks to control where and how models are blended
- **Model Key Filter**: Memory-efficient filtering of model parameters with batch processing for large models
- **Mask Blender**: Combine multiple masks with various blending modes (Add, Multiply, Max, Min, Linear Blend, Exponential Blend)

### Model Transformation

- **Model Weight Modifier**: Memory-efficient weight modification with operations like multiply, add, set value, clamp magnitude, and scale max absolute value

### Advanced Features

- **GPU-Optimized Memory Management**: Intelligent memory allocation and cleanup for cards with limited VRAM
- **Batch Processing**: Process large models in memory-efficient batches
- **Multiple Merging Algorithms**: Including SLERP, frequency domain blending, stochastic merging, and TIES merging
- **Spectral Analysis**: Frequency-based model analysis and merging
- **Adaptive Precision**: Automatic FP16/FP32 selection based on available memory
- **Cross-Device Compatibility**: Works with CUDA, MPS, and CPU backends

## Installation

### Method 1: ComfyUI Manager (OPTIONAL)
Note: Not available yet there.
1. Open ComfyUI Manager
2. Search for "TensorPrism"
3. Click Install

### Method 2: Manual Installation
1. Clone or download this repository
2. Place the entire folder in your `ComfyUI/custom_nodes/` directory
3. Restart ComfyUI

### Method 3: Git Clone
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

### Model Weight Modification
Transform model weights directly:
- **Multiply**: Scale weights by a factor
- **Add**: Add constant values
- **Set Value**: Replace weights with specific values
- **Clamp Magnitude**: Limit weight magnitudes
- **Scale Max Abs**: Normalize based on reference model

## Node Reference

| Node | Category | Purpose |
|------|----------|---------|
| Main Merge | Tensor Prism/Core | Advanced merging with multiple methods |
| Prism | Tensor Prism/Core | Spectral frequency-domain merging |
| SDXL Block Merge | Tensor_Prism/Merge | Basic granular SDXL merging |
| SDXL Advanced Block Merge | Tensor_Prism/Merge | GPU-optimized SDXL merging |
| Model Mask Generator | Tensor Prism/Mask | Create structural masks |
| Weighted Mask Merge | Tensor Prism/Mask | Apply masks to merging |
| Model Key Filter | Tensor_Prism/Mask | Filter model parameters |
| Mask Blender | Tensor_Prism/Mask | Combine multiple masks |
| Model Weight Modifier | Tensor_Prism/Transform | Direct weight manipulation |

## Memory Management

The TensorPrism pack includes advanced memory management features:

- **Automatic GPU Detection**: Optimizes for your specific GPU memory
- **Adaptive Batch Sizes**: Adjusts processing based on available memory
- **Precision Selection**: Automatic FP16/FP32 based on memory constraints
- **Progressive Cleanup**: Aggressive garbage collection for low-memory systems
- **CPU Fallback**: Automatic fallback when GPU memory is insufficient

### Recommended Settings by GPU:
- **24GB+ (RTX 4090, etc.)**: Use default settings
- **12GB (RTX 4070 Ti, etc.)**: Set memory limit to 8GB, enable auto precision
- **8GB (RTX 4060 Ti, etc.)**: Set memory limit to 6GB, force CPU for large merges
- **6GB and below**: Use CPU processing for best stability

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

## Troubleshooting

- **Memory Issues**: Reduce batch sizes, lower memory limits, or enable CPU fallback
- **Poor Results**: Try different merging methods or adjust spectral parameters
- **Compatibility**: Ensure models are the same architecture (SDXL with SDXL, etc.)
- **Slow Performance**: Check if you're accidentally using CPU when GPU is available
- **Artifacts**: Try more conservative merge ratios or use SLERP for smoother blending

## Performance Tips

- **Batch Size**: Larger batches are more efficient but use more memory
- **Precision Mode**: FP16 saves memory but may affect quality on some operations
- **Memory Cleanup**: Enable aggressive cleanup for systems with limited RAM
- **Device Selection**: Let the system auto-detect optimal device unless you have specific needs

## License

https://www.gnu.org/licenses/gpl-3.0.en.html

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Changelog

### Version 1.1.0
- Initial release
- Core merging nodes with advanced interpolation methods
- Spectral analysis and frequency-domain merging
- Advanced mask system with filtering and blending
- GPU-optimized memory management
- Support for SDXL models with granular block control
- Model weight modification tools
- Cross-platform compatibility (CUDA/MPS/CPU)
