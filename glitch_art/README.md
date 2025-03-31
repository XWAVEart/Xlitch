# Glitch Art Package

A Python package for applying glitch and pixel manipulation effects to images. This package was refactored from a monolithic utils.py file to provide better organization, maintainability, and extensibility.

## Package Structure

```
glitch_art/
├── __init__.py               # Package exports
├── core/
│   ├── __init__.py           # Core exports
│   ├── pixel_attributes.py   # PixelAttributes class and basic pixel operations
│   └── image_utils.py        # Common image loading/resizing/saving functions
├── effects/
│   ├── __init__.py           # Effects exports
│   ├── sorting.py            # All pixel sorting functions
│   ├── color.py              # Color manipulation (channel swapping, etc.)
│   ├── distortion.py         # Geometric distortion effects
│   ├── glitch.py             # Databending, JPEG artifacts, bit manipulation
│   ├── patterns.py           # Voronoi, spiral, concentric shapes
│   └── noise.py              # Perlin noise effects
└── utils/
    ├── __init__.py           # Utility exports
    └── helpers.py            # Shared helper functions
```

## Effects Included

- **Pixel Sorting**: Sort pixels within chunks based on various properties
- **Color Manipulation**: Swap, invert, or adjust color channels
- **Distortion Effects**: Pixel drift, Perlin noise displacement, geometric distortion
- **Glitch Effects**: Databending, JPEG artifacts, bit manipulation
- **Pattern Effects**: Voronoi cells, spiral sorting, concentric shapes
- **Noise Effects**: Various Perlin noise-based effects

## Usage

```python
from glitch_art import load_image, pixel_sorting, generate_output_filename

# Load image
image = load_image("input.jpg")

# Apply effect
processed = pixel_sorting(
    image, 
    direction='horizontal', 
    chunk_size='32x32', 
    sort_by='brightness',
    sort_order='descending'
)

# Save result
processed.save("output.jpg")
```

## Dependencies

- PIL/Pillow
- NumPy
- SciPy
- OpenCV (cv2)
- noise

## Migration Note

This package is a refactored version of the original utils.py file. The package maintains backward compatibility by importing the original functions through the top-level __init__.py file. 