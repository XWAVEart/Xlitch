# XWAVE Xlitch Art Generator

A powerful web application for creating artistic glitch effects and image manipulations. This tool offers a wide range of effects and algorithms to transform your images into unique pieces of glitch art.

## Features

### Pixel Sorting Effects
- Horizontal and vertical sorting with customizable chunk sizes
- Diagonal sorting from any corner
- Full-frame sorting with multiple attributes
- Perlin noise-controlled sorting
- Voronoi-based sorting
- Spiral sorting
- Polar coordinate sorting

### Color Channel Manipulations
- Channel swapping and inversion
- Intensity adjustments
- Negative effects
- RGB channel splitting and shifting
- Histogram glitch with per-channel transformations
- Color shift expansion with multiple patterns
- Posterize effect
- Curved hue shift
- JPEG artifact simulation

### Geometric Distortions
- Perlin noise displacement
- Pixel drift with customizable bands
- Ripple effects with color shifting
- Geometric pattern generation
- Offset effects (pixel and slice-based)
- Slice shuffling and reduction
- Pixel scatter with attribute-based selection

### Advanced Effects
- Double exposure with multiple blend modes
- Data moshing and block manipulation
- Bit manipulation with various modes
- Masked image merging with patterns:
  - Checkerboard (regular and random)
  - Striped patterns and gradient stripes
  - Perlin noise masks
  - Voronoi patterns
  - Concentric rectangles

## Requirements

- Python 3.7+
- PIL (Pillow)
- NumPy
- OpenCV (cv2)
- noise
- SciPy
- Flask (for web interface)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/glitch_art_app.git
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

2. Open your web browser and navigate to `http://localhost:5000`

3. Upload an image and experiment with different glitch effects

## Project Structure

The project is organized into a modular structure for better maintainability:

```
glitch_art_app/
├── app.py                  # Flask web application
├── forms.py                # Form definitions for the web interface
├── glitch_art/             # Main package
│   ├── core/               # Core utilities
│   │   ├── image_utils.py  # Image loading, saving, and utilities
│   │   └── pixel_attributes.py # Pixel attribute calculations
│   ├── effects/            # Effect modules
│   │   ├── blend.py        # Blending and compositing effects
│   │   ├── color.py        # Color manipulation effects
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

## Effect Parameters

### Pixel Sorting
- `sort_mode`: Sorting direction (horizontal, vertical, diagonal)
- `chunk_size`: Size of pixel chunks to sort
- `sort_by`: Property to sort by (brightness, hue, saturation, etc.)
- `sort_order`: Ascending or descending order

### Channel Effects
- `manipulation_type`: Type of channel manipulation
- `choice`: Specific channel or channel pair
- `factor`: Intensity adjustment factor

### Geometric Effects
- `scale`: Scale of distortion patterns
- `intensity`: Strength of the effect
- `distortion_type`: Type of geometric distortion
- `num_points`: Number of distortion points

### Masked Merge
- `mask_type`: Type of masking pattern
- `width/height`: Dimensions of pattern elements
- `stripe_angle`: Angle for striped patterns
- `threshold`: Threshold for pattern generation
- `perlin_noise_scale`: Scale of Perlin noise patterns

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Perlin noise implementation using the `noise` package
- OpenCV for image processing capabilities
- SciPy for various mathematical operations
- PIL/Pillow for core image manipulation 
