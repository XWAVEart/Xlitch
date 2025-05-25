# XWAVE Xlitch Art Generator

A powerful web application for creating artistic glitch effects and image manipulations. This tool offers a wide range of effects and algorithms to transform your images into unique pieces of glitch art.

## Features

### 🎨 Color & Tone Effects
- **Color Filter** - Apply color overlays and filters with various blend modes
- **Color Channel Manipulation** - Swap, invert, or adjust individual RGB channels
- **RGB Channel Shift** - Shift color channels with customizable offsets and patterns
- **Hue Skrift** - Apply curved hue shifts with customizable intensity curves
- **Color Shift Expansion** - Expand color ranges with multiple pattern types
- **Histogram Glitch** - Manipulate color histograms for unique color distortions
- **Posterfy** - Reduce color depth for poster-like effects
- **Chromatic Aberration** - Simulate lens chromatic aberration effects
- **VHS Effect** - Recreate vintage VHS tape artifacts and distortions

### 🔄 Pixel Sorting Effects
- **Advanced Pixel Sorting** - Comprehensive pixel sorting with multiple algorithms:
  - **Chunk Sorting** - Sort pixels in rectangular chunks (horizontal, vertical, diagonal)
  - **Full Frame Sorting** - Sort entire rows or columns across the image
  - **Spiral Sorting** - Sort pixels in spiral patterns from chunk centers
  - **Polar Sorting** - Sort pixels based on polar coordinates (angle/radius)
  - **Wrapped Sorting** - Create staggered patterns with wrapped chunk placement
  - **Perlin Noise Sorting** - Use Perlin noise to control sorting regions
  - **Voronoi Sorting** - Sort pixels within Voronoi cell regions

### 🎭 Pixelation & Stylization
- **Pixelate** - Traditional pixelation with attribute-based color selection
- **Voronoi Pixelate** - Create Voronoi cell-based pixelation effects
- **Gaussian Blur** - Apply various blur effects with customizable kernels
- **Sharpen Effect** - Enhance image sharpness with multiple algorithms
- **Concentric Shapes** - Generate concentric patterns (circles, rectangles, polygons)
- **Contour** - Extract and stylize image contours with various modes

### 🌊 Distortion & Displacement
- **Pixel Drift** - Create flowing pixel movements with customizable bands
- **Perlin Displacement** - Displace pixels using Perlin noise patterns
- **Wave Distortion** - Apply sine wave distortions (horizontal, vertical, radial)
- **Ripple Effect** - Create water-like ripple distortions with color shifting
- **Pixel Scatter** - Scatter pixels based on various attributes and patterns
- **Offset** - Apply pixel and slice-based offset effects

### 🔀 Slice & Block Effects
- **Slice & Block Manipulation** - Comprehensive slice and block operations:
  - **Slice Shuffle** - Randomly rearrange image slices
  - **Slice Offset** - Apply offset patterns to image slices
  - **Slice Reduction** - Reduce slice resolution for compression effects
  - **Block Shuffle** - Randomly rearrange rectangular blocks

### ⚡ Glitch & Corruption
- **Bit Manipulation** - Low-level bit operations (XOR, shift, swap, invert)
- **Data Mosh Blocks** - Simulate video compression artifacts and block corruption
- **Databending** - Raw data manipulation for extreme glitch effects

### 🎵 Noise & Texture
- **Noise Effect** - Add various noise types (film grain, digital, colored, salt & pepper)
- **Perlin Noise Replacement** - Replace pixels based on Perlin noise patterns

### 🖼️ Blending & Compositing
- **Double Expose** - Blend two images with multiple blend modes
- **Masked Merge** - Merge images using various mask patterns:
  - Checkerboard (regular and random)
  - Striped patterns and gradient stripes
  - Perlin noise masks
  - Voronoi patterns
  - Concentric shapes
  - Random triangles

## Requirements

### Core Dependencies
- **Python 3.7+**
- **Flask** >= 2.0.0 (web framework)
- **Flask-WTF** >= 1.0.0 (form handling and CSRF protection)
- **Werkzeug** >= 2.0.0 (WSGI utilities)

### Image Processing & Computer Vision
- **Pillow** >= 9.0.0 (core image manipulation)
- **OpenCV** >= 4.5.0 (advanced computer vision operations)
- **scikit-image** >= 0.18.0 (specialized image processing algorithms)

### Scientific Computing
- **NumPy** >= 1.20.0 (numerical operations and array processing)
- **SciPy** >= 1.7.0 (scientific computing and spatial operations)

### Effect-Specific Libraries
- **noise** >= 1.2.2 (Perlin noise generation for distortion effects)

### Optional Dependencies
- **python-dotenv** >= 0.19.0 (environment variable management - optional but recommended)
- **gunicorn** >= 20.1.0 (production WSGI server for deployment)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/XWAVEart/Xlitch
cd glitch_art_app
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:8080`

3. Upload an image and experiment with different glitch effects

## Project Structure

The project is organized into a modular structure for better maintainability:

```
glitch_art_app/
├── app.py                  # Flask web application
├── forms.py                # Form definitions for the web interface
├── requirements.txt        # Python dependencies
├── glitch_art/             # Main package
│   ├── core/               # Core utilities
│   │   ├── image_utils.py  # Image loading, saving, and utilities
│   │   └── pixel_attributes.py # Pixel attribute calculations
│   ├── effects/            # Effect modules
│   │   ├── blend.py        # Blending and compositing effects
│   │   ├── color.py        # Color manipulation effects
│   │   ├── consolidated.py # Consolidated effect interfaces
│   │   ├── contour.py      # Contour extraction effects
│   │   ├── distortion.py   # Spatial distortion effects
│   │   ├── glitch.py       # Low-level glitch effects
│   │   ├── noise.py        # Noise-based effects
│   │   ├── patterns.py     # Pattern generation effects
│   │   ├── pixelate.py     # Pixelation effects
│   │   └── sorting.py      # Pixel sorting effects
│   └── utils/              # Additional utilities
├── static/                 # Static assets for the web interface
└── templates/              # HTML templates
```

## Effect Parameters Guide

### Pixel Sorting Parameters
- **Sort Mode**: Direction of sorting (horizontal, vertical, diagonal)
- **Chunk Size**: Dimensions of pixel chunks to sort
- **Sort By**: Attribute to sort by (brightness, hue, saturation, color sum, etc.)
- **Sort Order**: Ascending or descending order
- **Starting Corner**: For diagonal sorting - which corner to start from

### Color Effect Parameters
- **Manipulation Type**: Type of color operation (swap, invert, adjust, etc.)
- **Channel Selection**: Which color channels to affect
- **Intensity Factor**: Strength of the color manipulation
- **Blend Mode**: How colors are combined or modified

### Distortion Parameters
- **Scale/Frequency**: Size and frequency of distortion patterns
- **Amplitude**: Strength of the distortion effect
- **Distortion Type**: Type of geometric transformation
- **Noise Parameters**: Scale and characteristics of noise-based distortions

### Glitch Parameters
- **Chunk Size**: Size of data blocks to manipulate
- **Manipulation Type**: Type of bit-level operation (XOR, shift, swap, invert)
- **Skip Pattern**: Which chunks to affect (alternate, random, etc.)
- **Randomization**: Add random elements to the effect

### Mask Parameters (for Masked Merge)
- **Mask Type**: Pattern type (checkerboard, stripes, Perlin noise, etc.)
- **Pattern Dimensions**: Width and height of pattern elements
- **Threshold Values**: Control pattern generation and visibility
- **Noise Scale**: For Perlin noise-based masks

## Tips for Best Results

1. **Start with lower intensities** and gradually increase for subtle effects
2. **Combine multiple effects** by applying them sequentially
3. **Experiment with different sort attributes** for pixel sorting effects
4. **Use seeds for reproducible results** when randomization is involved
5. **Try different blend modes** for color and merge effects
6. **Adjust chunk sizes** to control the granularity of block-based effects

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Perlin noise implementation using the `noise` package
- OpenCV for advanced image processing capabilities
- SciPy for mathematical operations and algorithms
- scikit-image for specialized image processing functions
- PIL/Pillow for core image manipulation
- Flask ecosystem for the web interface

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Support

For questions, issues, or feature requests, please visit the [GitHub repository](https://github.com/XWAVEart/Xlitch).
