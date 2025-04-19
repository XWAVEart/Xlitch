# XWAVE Xlitch Art Generator

A powerful web application for creating artistic glitch effects and image manipulations. This tool offers a wide range of effects and algorithms to transform your images into unique pieces of glitch art.

## New and Improved Architecture

The codebase has been restructured to use a more modular and maintainable approach:

- All effects have been organized into topic-specific modules
- Improved backward compatibility with deprecated warnings
- Enhanced documentation and clear migration path
- Systematic testing for all effect functions

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

## Project Structure

The project is organized into a modular structure for better maintainability:

```
glitch_art_app/
├── app.py                  # Flask web application
├── forms.py                # Form definitions for the web interface
├── utils.py                # Legacy utility functions (deprecated)
├── MIGRATION_GUIDE.md      # Guide for migrating from utils.py to modules
├── DUPLICATION_CLEANUP_PLAN.md  # Documentation of the refactoring process
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

## Using the Library

The library can be used either through the web interface or directly in your Python code:

```python
# Import effect functions directly from their modules
from glitch_art.effects.sorting import pixel_sorting
from glitch_art.effects.color import color_channel_manipulation
from glitch_art.core.image_utils import load_image, generate_output_filename

# Load an image
image = load_image('input.jpg')

# Apply an effect
processed_image = pixel_sorting(
    image, 
    sort_mode='horizontal', 
    chunk_size='32x32', 
    sort_by='brightness',
    sort_order='ascending'
)

# Save the result
processed_image.save('output.png')
```

See the [Migration Guide](MIGRATION_GUIDE.md) for more details on using the library.

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

## Testing

To test all functions in the new module structure, run:

```bash
python test_modules.py path/to/test/image.jpg
```

This will test each function with a sample set of parameters and save the results to a `test_results` directory.

## Recent Changes

### Code Restructuring
- Migrated all effects from `utils.py` to dedicated modules under `glitch_art/effects/`
- Added proper deprecation warnings to legacy functions
- Fixed parameter inconsistencies between old and new implementations
- Improved documentation with a comprehensive migration guide

### Bug Fixes
- Fixed issues with the double expose functionality
- Corrected implementation of color_shift_expansion effect
- Fixed parameter name inconsistencies in pixel_sorting function

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Perlin noise implementation using the `noise` package
- OpenCV for image processing capabilities
- SciPy for various mathematical operations
- PIL/Pillow for core image manipulation 