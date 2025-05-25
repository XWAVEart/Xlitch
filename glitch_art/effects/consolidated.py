"""
Consolidated effect functions that dispatch to individual effects based on parameters.
This module provides unified interfaces for groups of related effects.
"""

from .sorting import pixel_sorting, full_frame_sort, spiral_sort_2, polar_sorting, wrapped_sort
from .patterns import voronoi_pixel_sort
from .noise import perlin_noise_sorting, perlin_full_frame_sort
from .distortion import slice_shuffle, slice_offset, slice_reduction, block_shuffle


def advanced_pixel_sorting(image, sorting_method, **kwargs):
    """
    Unified pixel sorting function that dispatches to specific sorting methods.
    
    Args:
        image: PIL Image object
        sorting_method: Type of sorting ('chunk', 'full_frame', 'polar', 'spiral', 'voronoi', 'perlin_noise', 'perlin_full_frame', 'wrapped')
        **kwargs: Parameters specific to each sorting method
    
    Returns:
        PIL Image: Processed image
    """
    if sorting_method == 'chunk':
        # Extract chunk-specific parameters - use original PixelSortChunkForm defaults
        chunk_width = kwargs.get('chunk_width') or 48
        chunk_height = kwargs.get('chunk_height') or 48
        chunk_size = f"{chunk_width}x{chunk_height}"
        sort_mode = kwargs.get('sort_mode', 'horizontal')
        sort_by = kwargs.get('sort_by', 'brightness')
        reverse_sort = kwargs.get('reverse_sort', True)
        starting_corner = kwargs.get('starting_corner', 'top-left')
        
        return pixel_sorting(
            image=image,
            sort_mode=sort_mode,
            chunk_size=chunk_size,
            sort_by=sort_by,
            starting_corner=starting_corner,
            sort_order='descending' if reverse_sort else 'ascending'
        )
    
    elif sorting_method == 'full_frame':
        direction = kwargs.get('direction', 'vertical')
        sort_by = kwargs.get('sort_by', 'brightness')
        reverse_sort = kwargs.get('reverse_sort', True)
        
        return full_frame_sort(
            image=image,
            direction=direction,
            sort_by=sort_by,
            reverse=reverse_sort
        )
    
    elif sorting_method == 'polar':
        # Use original PolarSortForm defaults
        chunk_size = kwargs.get('chunk_size') or 64
        sort_by = kwargs.get('polar_sort_by', 'radius')
        reverse_sort = kwargs.get('reverse_sort', True)
        
        return polar_sorting(
            image=image,
            chunk_size=chunk_size,
            sort_by=sort_by,
            reverse=reverse_sort
        )
    
    elif sorting_method == 'spiral':
        # Use original SpiralSort2Form defaults
        chunk_size = kwargs.get('chunk_size') or 64
        sort_by = kwargs.get('sort_by', 'brightness')
        reverse_sort = kwargs.get('reverse_sort', True)
        
        return spiral_sort_2(
            image=image,
            chunk_size=chunk_size,
            sort_by=sort_by,
            reverse=reverse_sort
        )
    
    elif sorting_method == 'voronoi':
        # Use original VoronoiSortForm defaults
        num_cells = kwargs.get('num_cells') or 69
        size_variation = kwargs.get('size_variation') or 0.8
        sort_by = kwargs.get('sort_by', 'hue')
        sort_order = kwargs.get('sort_order', 'clockwise')
        orientation = kwargs.get('voronoi_orientation', 'spiral')
        start_position = kwargs.get('start_position', 'center')
        seed = kwargs.get('seed')
        
        return voronoi_pixel_sort(
            image=image,
            num_cells=num_cells,
            size_variation=size_variation,
            sort_by=sort_by,
            sort_order=sort_order,
            orientation=orientation,
            start_position=start_position,
            seed=seed
        )
    
    elif sorting_method == 'perlin_noise':
        # Use original PerlinNoiseSortForm defaults if not provided
        # Check for perlin-specific chunk parameters first, then fall back to general ones
        chunk_width = kwargs.get('perlin_chunk_width') or kwargs.get('chunk_width') or 120
        chunk_height = kwargs.get('perlin_chunk_height') or kwargs.get('chunk_height') or 1024
        chunk_size = f"{chunk_width}x{chunk_height}"
        noise_scale = kwargs.get('noise_scale') or 0.008
        direction = kwargs.get('direction', 'horizontal')
        reverse_sort = kwargs.get('reverse_sort', True)
        seed = kwargs.get('seed')
        
        return perlin_noise_sorting(
            image=image,
            chunk_size=chunk_size,
            noise_scale=noise_scale,
            direction=direction,
            reverse=reverse_sort,
            seed=seed
        )
    
    elif sorting_method == 'perlin_full_frame':
        # Use original PerlinFullFrameForm defaults if not provided
        noise_scale = kwargs.get('noise_scale') or 0.005
        sort_by = kwargs.get('sort_by', 'brightness')
        reverse_sort = kwargs.get('reverse_sort', True)
        pattern_width = kwargs.get('pattern_width') or 1
        seed = kwargs.get('seed')
        
        return perlin_full_frame_sort(
            image=image,
            noise_scale=noise_scale,
            sort_by=sort_by,
            reverse=reverse_sort,
            seed=seed,
            pattern_width=pattern_width
        )
    
    elif sorting_method == 'wrapped':
        # Wrapped sort parameters
        chunk_width = kwargs.get('wrapped_chunk_width') or kwargs.get('chunk_width') or 12
        chunk_height = kwargs.get('wrapped_chunk_height') or kwargs.get('chunk_height') or 123
        starting_corner = kwargs.get('wrapped_starting_corner') or kwargs.get('starting_corner', 'top-left')
        flow_direction = kwargs.get('wrapped_flow_direction', 'primary')
        sort_direction = kwargs.get('direction', 'vertical')
        sort_by = kwargs.get('sort_by', 'brightness')
        reverse_sort = kwargs.get('reverse_sort', True)
        
        return wrapped_sort(
            image=image,
            chunk_width=chunk_width,
            chunk_height=chunk_height,
            starting_corner=starting_corner,
            flow_direction=flow_direction,
            sort_direction=sort_direction,
            sort_by=sort_by,
            reverse=reverse_sort
        )
    
    else:
        raise ValueError(f"Unknown sorting method: {sorting_method}")


def slice_block_manipulation(image, manipulation_type, **kwargs):
    """
    Unified slice and block manipulation function.
    
    Args:
        image: PIL Image object
        manipulation_type: Type of manipulation ('slice_shuffle', 'slice_offset', 'slice_reduction', 'block_shuffle')
        **kwargs: Parameters specific to each manipulation type
    
    Returns:
        PIL Image: Processed image
    """
    if manipulation_type == 'slice_shuffle':
        # Use original SliceShuffleForm defaults
        count = kwargs.get('slice_count') or 16
        orientation = kwargs.get('orientation', 'rows')
        seed = kwargs.get('seed')
        
        return slice_shuffle(
            image=image,
            count=count,
            orientation=orientation,
            seed=seed
        )
    
    elif manipulation_type == 'slice_offset':
        # Use original SliceOffsetForm defaults
        count = kwargs.get('slice_count') or 16
        max_offset = kwargs.get('max_offset') or 50
        orientation = kwargs.get('orientation', 'rows')
        offset_mode = kwargs.get('offset_mode', 'random')
        frequency = kwargs.get('frequency') or 0.1
        seed = kwargs.get('seed')
        
        return slice_offset(
            image=image,
            count=count,
            max_offset=max_offset,
            orientation=orientation,
            offset_mode=offset_mode,
            sine_frequency=frequency,
            seed=seed
        )
    
    elif manipulation_type == 'slice_reduction':
        # Use original SliceReductionForm defaults
        count = kwargs.get('slice_count') or 32
        reduction_value = kwargs.get('reduction_value') or 2
        orientation = kwargs.get('orientation', 'rows')
        
        return slice_reduction(
            image=image,
            count=count,
            reduction_value=reduction_value,
            orientation=orientation
        )
    
    elif manipulation_type == 'block_shuffle':
        # Use original BlockShuffleForm defaults
        block_width = kwargs.get('block_width') or 32
        block_height = kwargs.get('block_height') or 32
        seed = kwargs.get('seed')
        
        return block_shuffle(
            image=image,
            block_width=block_width,
            block_height=block_height,
            seed=seed
        )
    
    else:
        raise ValueError(f"Unknown manipulation type: {manipulation_type}") 