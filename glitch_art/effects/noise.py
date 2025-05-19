from PIL import Image
import numpy as np
# import noise # noise is imported by the helper
import random
import math
from ..core.pixel_attributes import PixelAttributes
from ..utils.helpers import generate_noise_map # Import the helper
import logging

logger = logging.getLogger(__name__)

def perlin_noise_sorting(image, chunk_size=32, noise_scale=0.1, direction='horizontal', reverse=False, seed=None):
    """
    Apply Perlin noise-based sorting to an image by using noise values to sort pixels in chunks.

    Args:
        image (Image): PIL Image object to process.
        chunk_size (int or str): Size of chunks. Can be an integer for square chunks or a string 'widthxheight'.
        noise_scale (float): Scale of Perlin noise (higher = more detailed noise).
        direction (str): 'horizontal' or 'vertical' sorting direction.
        reverse (bool): Whether to reverse the sort order.
        seed (int, optional): Seed for the Perlin noise generator. If None, a random pattern is generated.

    Returns:
        Image: Processed image with noise-sorted pixels.
    """
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    width, height = image.size
    
    # Parse chunk size
    if isinstance(chunk_size, str) and 'x' in chunk_size:
        # Format: 'widthxheight'
        chunk_width, chunk_height = map(int, chunk_size.split('x'))
    else:
        # Square chunks
        chunk_width = chunk_height = int(chunk_size)
    
    # Handle special case for full image sorting
    if chunk_width >= width and chunk_height >= height:
        chunk_width, chunk_height = width, height
    
    # Set seed for reproducibility if provided
    # np.random.seed is not strictly needed here as noise module handles its own seeding via 'base'
    # and python's random is not used.
    # if seed is not None:
    #     np.random.seed(seed) 
        # random_state = np.random.RandomState(seed) # Not used
    # else:
        # random_state = np.random.RandomState() # Not used
    
    # Prepare a single noise map for the entire image for efficiency
    # The noise values need to be 'world-coordinated' based on original x,y and scale
    # The helper `generate_noise_map` uses x/scale, y/scale.
    # The original loops used (j + x_offset) * noise_scale and (i + y_offset) * noise_scale.
    # This is equivalent to x_img * noise_scale, y_img * noise_scale.
    # So, for the helper, the 'scale' parameter should be 1.0 / noise_scale from the args.
    
    effective_noise_scale_for_helper = 1.0 / noise_scale if noise_scale != 0 else float('inf') # Avoid div by zero
    if effective_noise_scale_for_helper == float('inf'): # Handle case of noise_scale = 0
        full_noise_map = np.zeros((height, width), dtype=np.float32) # No noise
    else:
        full_noise_map = generate_noise_map(
            shape=(height, width),
            scale=effective_noise_scale_for_helper, # Helper expects scale in pixel units per noise cycle
            octaves=1, # pnoise2 default
            base=seed if seed is not None else 0
        )

    img_array = np.array(image)
    sorted_array = np.copy(img_array) # Operate on a copy
    
    for y_start in range(0, height, chunk_height):
        for x_start in range(0, width, chunk_width):
            # Calculate actual chunk dimensions (handle edge cases)
            actual_chunk_width = min(chunk_width, width - x_start)
            actual_chunk_height = min(chunk_height, height - y_start)

            if actual_chunk_width <= 0 or actual_chunk_height <= 0:
                continue

            # Get the chunk and corresponding noise map slice
            chunk_slice = img_array[y_start : y_start + actual_chunk_height, x_start : x_start + actual_chunk_width]
            noise_slice = full_noise_map[y_start : y_start + actual_chunk_height, x_start : x_start + actual_chunk_width]
            
            sorted_chunk_slice = np.copy(chunk_slice) # Work on a copy for sorted output

            if direction == 'horizontal':
                for i in range(actual_chunk_height): # Iterate through rows of the chunk
                    row_pixels = chunk_slice[i, :] # (actual_chunk_width, 3)
                    row_noise = noise_slice[i, :]  # (actual_chunk_width,)
                    
                    # Get sorted indices based on noise values
                    sorted_indices = np.argsort(row_noise)
                    if reverse:
                        sorted_indices = sorted_indices[::-1]
                    
                    # Place sorted pixels back into the row
                    sorted_chunk_slice[i, :] = row_pixels[sorted_indices]
            else:  # vertical
                for j in range(actual_chunk_width): # Iterate through columns of the chunk
                    col_pixels = chunk_slice[:, j] # (actual_chunk_height, 3)
                    col_noise = noise_slice[:, j]  # (actual_chunk_height,)

                    # Get sorted indices based on noise values
                    sorted_indices = np.argsort(col_noise)
                    if reverse:
                        sorted_indices = sorted_indices[::-1]
                        
                    # Place sorted pixels back into the column
                    sorted_chunk_slice[:, j] = col_pixels[sorted_indices]
            
            sorted_array[y_start : y_start + actual_chunk_height, x_start : x_start + actual_chunk_width] = sorted_chunk_slice
    
    # np.random.seed(None) # Not strictly needed as it wasn't set for this scope
    return Image.fromarray(sorted_array)

def perlin_full_frame_sort(image, noise_scale=0.01, sort_by='brightness', reverse=False, seed=None, pattern_width=1):
    """
    Apply full-frame pixel sorting controlled by Perlin noise.
    
    Args:
        image (Image): PIL Image object to process.
        noise_scale (float): Scale of Perlin noise (higher = more detailed noise).
        sort_by (str): Property to sort by ('color', 'brightness', 'hue', 'red', 'green', 'blue',
                       'saturation', 'luminance', 'contrast').
        reverse (bool): Whether to reverse the sort order.
        seed (int, optional): Seed for the Perlin noise generator. If None, a random pattern is generated.
        pattern_width (int): Width of the sorted patterns (1-8 pixels). Default is 1 for single-pixel wide patterns.
    
    Returns:
        Image: Processed image with Perlin noise-controlled full-frame sorting.
    """
    # Ensure pattern_width is in valid range
    pattern_width = max(1, min(pattern_width, 8))
    
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    width, height = image.size
    img_array = np.array(image)
    # sorted_im_array = np.copy(img_array) # Will be modified in place

    sort_function_map = {
        'color': PixelAttributes.color_sum,
        'brightness': PixelAttributes.brightness,
        'hue': PixelAttributes.hue,
        'red': lambda p_tuple: p_tuple[0],
        'green': lambda p_tuple: p_tuple[1],
        'blue': lambda p_tuple: p_tuple[2],
        'saturation': PixelAttributes.saturation,
        'luminance': PixelAttributes.luminance,
        'contrast': PixelAttributes.contrast
    }
    selected_sort_func = sort_function_map.get(sort_by, PixelAttributes.brightness)
    
    # Generate Perlin noise map for the entire image.
    # The original code scaled x for noise map generation: x * noise_scale * pattern_width
    # This implies the 'scale' for the helper should be 1.0 / (noise_scale * pattern_width) for x,
    # and 1.0 / noise_scale for y.
    # The helper `generate_noise_map` takes one `scale` argument.
    # For simplicity and consistency, let's generate a standard noise map first,
    # and then if needed, can consider if pattern_width requires special noise map generation.
    # The original noise map was (noise_map_height, noise_map_width) where noise_map_width was scaled by pattern_width.
    # This means fewer unique noise values horizontally if pattern_width > 1.
    # Let's replicate this by generating a smaller noise map and then repeating/tiling it.

    noise_map_visual_width = (width + pattern_width - 1) // pattern_width
    
    effective_noise_scale_for_helper = 1.0 / noise_scale if noise_scale != 0 else float('inf')

    if effective_noise_scale_for_helper == float('inf'):
        raw_noise_map_small = np.zeros((height, noise_map_visual_width), dtype=np.float32)
    else:
        raw_noise_map_small = generate_noise_map(
            shape=(height, noise_map_visual_width),
            scale=effective_noise_scale_for_helper, # This scale applies to both x and y in helper
            octaves=1,
            base=seed if seed is not None else 0
        )

    # Expand/repeat the noise map horizontally to match image width if pattern_width > 1
    if pattern_width > 1 and noise_map_visual_width < width :
        full_noise_map = np.repeat(raw_noise_map_small, pattern_width, axis=1)[:, :width]
    else:
        full_noise_map = raw_noise_map_small[:, :width] # Ensure it's not wider than image

    # Log noise map details for debugging
    min_noise_raw, max_noise_raw = np.min(full_noise_map), np.max(full_noise_map)
    logger.debug(f"PerlinFullFrameSort: Noise map shape: {full_noise_map.shape}, Raw min: {min_noise_raw}, Raw max: {max_noise_raw}, Effective Helper Scale: {effective_noise_scale_for_helper}, Input Noise Scale: {noise_scale}")

    # Normalize noise map to [0, 1]
    min_noise, max_noise = np.min(full_noise_map), np.max(full_noise_map)
    if max_noise == min_noise: # Avoid division by zero if noise is flat
        normalized_noise_map = np.zeros_like(full_noise_map) if min_noise == 0 else np.full_like(full_noise_map, 0.5) # Or other neutral value
    else:
        normalized_noise_map = (full_noise_map - min_noise) / (max_noise - min_noise)

    # Create a new image array for the result
    sorted_array = np.array(image, dtype=np.uint8) # Start with a copy, or use np.zeros_like(img_array)
    
    # Sort each column/block based on Perlin noise values
    for x_base in range(0, width, pattern_width):
        actual_strip_width = min(pattern_width, width - x_base)
        
        # Get the noise map x index for this group.
        # normalized_noise_map has shape (height, width) after np.repeat or direct assignment.
        # So, for a given y, the noise value for the entire pattern_width strip at that y is the same
        # if pattern_width > 1 (due to np.repeat), or unique if pattern_width = 1.
        # This means we can just use x_base as the column index into normalized_noise_map
        # for the start of the strip, and all pixels in that strip row will effectively use
        # the noise value from normalized_noise_map[y, x_base] if pattern_width caused repeat,
        # or normalized_noise_map[y, x_base + c_offset] if pattern_width allowed unique values.

        # Let's stick to the old way of thinking: a block of pixels is sorted together.
        # The noise value for sorting is taken from the *start* of the block's corresponding noise map column segment.
        # Or, more accurately from old code: noise_map[y, x_base // pattern_width].

        # For the current refactored noise_map (normalized_noise_map):
        # If pattern_width > 1, normalized_noise_map has values repeated pattern_width times.
        # So normalized_noise_map[y, x_base] is the same as normalized_noise_map[y, x_base+1], ..., normalized_noise_map[y, x_base+pattern_width-1]
        # If pattern_width = 1, then normalized_noise_map[y, x_base] is unique.

        pixels_in_block = [] # This will store (pixel_tuple, original_y, noise_value, original_x_offset_in_block)
        
        for r_idx in range(height): # y-coordinate
            # The noise value for this row (r_idx) across the current pattern_width block.
            # Since normalized_noise_map is already potentially repeated,
            # noise_strip = normalized_noise_map[r_idx, x_base : x_base + actual_strip_width]
            # The old code used: noise_map[y, x_base // pattern_width].
            # With current normalized_noise_map, this is simply normalized_noise_map[r_idx, x_base]
            # because if pattern_width > 1, the values are already repeated.
            # So, for any c_idx in the block, normalized_noise_map[r_idx, x_base + c_idx] would give the correct (repeated) noise value.
            # Let's use the specific noise value at each pixel for now, as per current full_noise_map structure.
            
            for c_idx_offset in range(actual_strip_width): # x-offset within the block
                current_x = x_base + c_idx_offset
                pixel_val = tuple(img_array[r_idx, current_x]) # Get pixel from original array
                
                # Get the noise value for this specific pixel (r_idx, current_x)
                noise_val_for_pixel = normalized_noise_map[r_idx, current_x]
                
                pixels_in_block.append(
                    (pixel_val, r_idx, noise_val_for_pixel, c_idx_offset)
                )
        
        # Sort the pixels in the block: primary key noise (item[2]), secondary key attribute (sort_function(item[0]))
        pixels_in_block.sort(key=lambda item: (item[2], selected_sort_func(item[0])), reverse=reverse)
        
        # Place sorted pixels back into the sorted_array
        # The old placement: y_placement = i // actual_strip_width; x_placement = x_base + (i % actual_strip_width)
        for i, (sorted_pixel_tuple, _, _, _) in enumerate(pixels_in_block):
            y_placement = i // actual_strip_width
            x_placement_offset = i % actual_strip_width
            
            # Ensure placement is within the original image dimensions, especially for y_placement
            if y_placement < height: # x_base + x_placement_offset will be within width by loop construction
                sorted_array[y_placement, x_base + x_placement_offset] = sorted_pixel_tuple
            # If y_placement >= height, it means we sorted more pixels than can fit in the original height
            # if the block definition was too large or not clipped, but here actual_strip_width
            # and the outer loop for r_idx=0..height-1 should mean pixels_in_block has height*actual_strip_width elements.
            # The y_placement logic implies filling the block column by column if viewed as a 1D list,
            # but it's filling the (height x actual_strip_width) block in row-major order from the 1D sorted list.

    return Image.fromarray(sorted_array)


def perlin_noise_replacement(image, secondary_image, noise_scale=0.1, threshold=0.5, seed=None):
    """
    Replace pixels in the primary image with pixels from a secondary image based on Perlin noise.

    Args:
        image (Image): Primary PIL Image object to process.
        secondary_image (Image): Secondary PIL Image for replacement pixels.
        noise_scale (float): Scale of Perlin noise.
        threshold (float): Noise threshold for replacement (0 to 1).
        seed (int, optional): Seed for the Perlin noise generator. If None, a random pattern is generated.

    Returns:
        Image: Processed image with noise-based pixel replacement.
    """
    # Convert both images to RGB mode if they have alpha channels or are in different modes
    if image.mode != 'RGB':
        image = image.convert('RGB')
    if secondary_image.mode != 'RGB':
        secondary_image = secondary_image.convert('RGB')
    
    # Resize secondary image to match primary image
    secondary_image = secondary_image.resize(image.size)
    image_array = np.array(image)
    secondary_array = np.array(secondary_image)
    
    # Set seed for reproducibility if provided (only for noise base)
    # np.random.seed not strictly needed here.
    # if seed is not None:
    #     np.random.seed(seed) 
        # random_state = np.random.RandomState(seed) # Not used
    # else:
        # random_state = np.random.RandomState() # Not used
    
    # Generate Perlin noise map for the entire image
    # The helper `generate_noise_map` uses x/scale, y/scale.
    # The original loops used i * noise_scale, j * noise_scale where i is height, j is width.
    # So, for the helper, the 'scale' parameter should be 1.0 / noise_scale.
    effective_noise_scale_for_helper = 1.0 / noise_scale if noise_scale != 0 else float('inf')

    if effective_noise_scale_for_helper == float('inf'): # Handle noise_scale = 0
        noise_map = np.zeros((image.height, image.width), dtype=np.float32)
    else:
        noise_map = generate_noise_map(
            shape=(image.height, image.width),
            scale=effective_noise_scale_for_helper,
            octaves=1, # pnoise2 default
            base=seed if seed is not None else 0
        )
    
    # Normalize noise map to [0, 1]
    min_noise, max_noise = np.min(noise_map), np.max(noise_map)
    if max_noise == min_noise: # Avoid division by zero if noise is flat
        normalized_noise_map = np.zeros_like(noise_map) if threshold > 0 else np.ones_like(noise_map) # All replace or no replace
        if min_noise > threshold : # if noise is flat and above threshold, all replace
            normalized_noise_map = np.ones_like(noise_map) 
        else: # if noise is flat and below threshold, no replace
            normalized_noise_map = np.zeros_like(noise_map)

    else:
        normalized_noise_map = (noise_map - min_noise) / (max_noise - min_noise)
    
    # Replace pixels where noise exceeds threshold
    mask = normalized_noise_map > threshold
    image_array[mask] = secondary_array[mask]
    
    # np.random.seed(None) # Not needed
    
    return Image.fromarray(image_array) 