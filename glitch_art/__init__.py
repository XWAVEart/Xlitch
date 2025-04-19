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

# Import effect functions - replacing utils.py imports with module imports
# For backward compatibility
from .effects.sorting import pixel_sorting, full_frame_sort, spiral_sort_2, polar_sorting
from .effects.color import color_channel_manipulation, split_and_shift_channels, histogram_glitch, color_shift_expansion, posterize, curved_hue_shift
from .effects.distortion import pixel_drift, perlin_noise_displacement, geometric_distortion, pixel_scatter, ripple_effect, offset_effect, slice_shuffle, slice_offset, slice_reduction
from .effects.glitch import databend_image, simulate_jpeg_artifacts, bit_manipulation, data_mosh_blocks
from .effects.patterns import voronoi_pixel_sort, masked_merge, concentric_shapes
from .effects.noise import perlin_noise_sorting, perlin_full_frame_sort
from .effects.pixelate import pixelate_by_attribute
from .effects.blend import double_expose

# Note: Any functions not yet implemented in the module structure 
# should be created in the appropriate module and imported above.
# All imports from utils.py have been replaced with proper module imports.
