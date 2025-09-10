##### ComfyUI-Tensor-Prism-Node-Pack
## Developer Notes

My First ComfyUI Node Pack, vibe coded with Gemini 2.5 Flash and Claude 4. With `TensorPrism_SDXLBlockMergeGranular.py` if you want to run three models in it Model C is the base and you need more than 12GB of VRAM.

# TensorPrism ComfyUI Node Pack

Advanced model merging and enhancement nodes for ComfyUI, providing sophisticated techniques for blending, enhancing, and manipulating Stable Diffusion models.

## Features

### Core Nodes

- **Core Merge**: Advanced model merging with multiple interpolation methods (linear, slerp, cosine, directional, frequency, stochastic)
- **Model Enhancer**: Enhance model quality with smoothing, sharpening, and quality boost operations
- **Layered Blend**: Component-specific merging with fine control over UNet blocks, text encoders, and VAE
- **Prism**: Fast spectral merging with frequency-based blending techniques
- **SDXL Block Merge**: Granular control over individual SDXL UNet blocks with support for TIES merging

### Advanced Features

- **Model Mask Generator**: Create sophisticated masks for selective model merging
- **Weighted Mask Merge**: Apply masks to control where and how models are blended
- Multiple merging algorithms including SLERP, frequency domain blending, and stochastic merging
- Support for attention-based enhancement methods
- Adaptive quality controls with overbaking prevention

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

### Method 3:
```bash
cd ComfyUI/custom_nodes/
https://github.com/AstrionX/ComfyUI-Tensor-Prism-Node-Pack.git
```

## Usage

### Basic Model Merging
Use the **Core Merge** node to blend two models with various interpolation methods:
- Connect two MODEL inputs
- Adjust merge ratio (0.0 = Model A only, 1.0 = Model B only)
- Choose merging method based on your needs

### Advanced Layered Blending
The **Layered Blend** node provides fine control:
- Separate controls for text encoder, UNet components, and VAE
- Different merging methods for each component
- Granular control over UNet blocks

### Model Enhancement
Use the **Model Enhancer** to improve model quality:
- Apply smoothing to reduce artifacts
- Add sharpening for better detail
- Quality boost for overall improvement
- Multiple enhancement methods available

### Spectral Merging
The **Prism** node offers frequency-domain merging:
- Separate low and high frequency blending
- Spectral precision controls
- Layer-selective processing

## Node Reference

| Node | Category | Purpose |
|------|----------|---------|
| Main Merge | Tensor Prism/Core | Basic advanced merging |
| Model Enhancer | checkpoint/enhancement | Model quality improvement |
| Layered Blend | Tensor Prism/Merge | Component-specific merging |
| Prism | Tensor Prism/Core | Spectral merging |
| SDXL Block Merge | Tensor_Prism/Merge | Granular SDXL merging |
| Model Mask Generator | Tensor Prism/Mask | Create merging masks |
| Weighted Mask Merge | Tensor Prism/Mask | Apply masks to merging |

## Requirements

- ComfyUI
- PyTorch >= 1.12.0
- NumPy >= 1.21.0

## Tips and Best Practices

1. **Start Conservative**: Begin with lower merge ratios (0.3-0.7) and adjust based on results
2. **Use SLERP for Dissimilar Models**: When merging very different models, SLERP often produces better results
3. **Enhance After Merging**: Apply model enhancement after merging for best quality
4. **Experiment with Spectral Methods**: Frequency domain merging can preserve details better than linear methods
5. **Use Masks for Selective Merging**: Create masks to merge only specific model components

## Troubleshooting

- **Memory Issues**: Reduce batch sizes or use FP16 precision in the enhancer
- **Poor Results**: Try different merging methods or adjust ratios
- **Compatibility**: Ensure models are the same architecture (SDXL with SDXL, etc.)

## License

https://www.gnu.org/licenses/gpl-3.0.en.html

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Changelog

### Version 1.0.0
- Initial release
- All core nodes implemented
- Support for SDXL models
- Advanced merging algorithms
