# XWAVE Xlitch Art Generator

A powerful web application for creating artistic glitch effects and image manipulations. This tool offers a wide range of effects and algorithms to transform your images into unique pieces of glitch art.

## Features

### Pixel Sorting Effects
- Horizontal and vertical sorting
- Diagonal sorting from any corner
- Full-frame sorting
- Perlin noise-controlled sorting
- Voronoi-based sorting
- Spiral sorting
- Polar coordinate sorting

### Color Channel Manipulations
- Channel swapping
- Channel inversion
- Intensity adjustments
- Negative effects
- RGB channel splitting and shifting

### Geometric Distortions
- Perlin noise displacement
- Voronoi-based distortion
- Ripple effects
- Geometric pattern generation
- Concentric shapes

### Advanced Effects
- Double exposure with multiple blend modes
- Data moshing and block manipulation
- JPEG artifact simulation
- Pixel scatter
- Histogram glitch effects
- Masked image merging with various patterns:
  - Checkerboard
  - Random checkerboard
  - Striped patterns
  - Gradient stripes
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

## Effect Parameters

### Pixel Sorting
- `direction`: Sorting direction (horizontal, vertical, diagonal)
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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Perlin noise implementation using the `noise` package
- OpenCV for image processing capabilities
- SciPy for various mathematical operations
- PIL/Pillow for core image manipulation 