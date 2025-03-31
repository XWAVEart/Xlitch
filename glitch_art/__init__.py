"""
Glitch Art Effect Library

A comprehensive collection of image processing effects for creating digital glitch art.
"""

# Import everything from effects module for easy access
from .effects import *

# Import core utilities
from .core.image_utils import load_image, resize_image_if_needed, generate_output_filename
from .core.pixel_attributes import PixelAttributes

# Version information
__version__ = '0.1.0'
__author__ = 'VibeCoding'

# Import core components
# from .core.pixel_attributes import PixelAttributes
# from .core.image_utils import load_image, resize_image_if_needed

# Import effect functions - we'll add these as we implement them
# from .effects.sorting import pixel_sorting, full_frame_sort, spiral_sort, spiral_sort_2, polar_sorting
# from .effects.color import color_channel_manipulation, split_and_shift_channels, histogram_glitch
# from .effects.distortion import pixel_drift, perlin_noise_displacement, geometric_distortion
# from .effects.glitch import databend_image, simulate_jpeg_artifacts, bit_manipulation
# from .effects.patterns import voronoi_pixel_sort, masked_merge, concentric_shapes
# from .effects.noise import perlin_noise_sorting, perlin_full_frame_sort

# For backward compatibility - these are stub imports
# that will be replaced once we implement all the modules
from utils import (
    pixel_sorting, 
    color_channel_manipulation, 
    double_expose, 
    pixel_drift, 
    bit_manipulation, 
    spiral_sort, 
    full_frame_sort, 
    spiral_sort_2, 
    polar_sorting, 
    perlin_noise_sorting, 
    perlin_full_frame_sort, 
    pixelate_by_mode, 
    concentric_shapes, 
    color_shift_expansion, 
    perlin_noise_displacement,
    data_mosh_blocks, 
    voronoi_pixel_sort, 
    split_and_shift_channels, 
    simulate_jpeg_artifacts, 
    pixel_scatter, 
    databend_image, 
    histogram_glitch, 
    Ripple, 
    masked_merge
)
