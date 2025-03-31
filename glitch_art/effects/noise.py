from PIL import Image
import numpy as np
import noise
import random
import math
from ..core.pixel_attributes import PixelAttributes

def generate_noise_map(shape, scale, octaves, base):
    """
    Generate a Perlin noise map for displacement.
    
    Args:
        shape (tuple): Height and width of the map (height, width).
        scale (float): Noise frequency (smaller values = more detailed noise).
        octaves (int): Number of noise layers for detail.
        base (int): Seed for noise pattern variation.
    
    Returns:
        np.ndarray: Noise map with values in [-1, 1].
    """
    height, width = shape
    noise_map = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            noise_map[y, x] = noise.pnoise2(x / scale, y / scale, octaves=octaves, base=base)
    return noise_map

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
    if seed is not None:
        np.random.seed(seed)
        random_state = np.random.RandomState(seed)
    else:
        random_state = np.random.RandomState()
    
    # Split image into chunks
    chunks = []
    for y in range(0, height, chunk_height):
        for x in range(0, width, chunk_width):
            # Calculate actual chunk dimensions (handle edge cases)
            actual_width = min(chunk_width, width - x)
            actual_height = min(chunk_height, height - y)
            if actual_width > 0 and actual_height > 0:
                chunks.append((image.crop((x, y, x + actual_width, y + actual_height)), (x, y)))

    sorted_image = Image.new('RGB', image.size)
    
    for chunk, (x_offset, y_offset) in chunks:
        chunk_width_actual, chunk_height_actual = chunk.size
        chunk_array = np.array(chunk)
        
        # Generate Perlin noise map for the chunk
        noise_map = np.zeros((chunk_height_actual, chunk_width_actual))
        for i in range(chunk_height_actual):
            for j in range(chunk_width_actual):
                # Add some variation based on chunk position for more interesting results
                x_noise = (j + x_offset) * noise_scale
                y_noise = (i + y_offset) * noise_scale
                
                # Use the seed if provided
                base = seed if seed is not None else 0
                noise_map[i, j] = noise.pnoise2(x_noise, y_noise, base=base)
        
        if direction == 'horizontal':
            # Sort each row by noise values
            sorted_chunk = np.zeros_like(chunk_array)
            for i in range(chunk_height_actual):
                row_pixels = [(j, chunk_array[i, j]) for j in range(chunk_width_actual)]
                row_noise = noise_map[i, :]
                
                # Sort pixels by noise values
                noise_pixel_pairs = list(zip(row_noise, row_pixels))
                noise_pixel_pairs.sort(key=lambda x: x[0], reverse=reverse)
                
                # Place sorted pixels back into the row
                for new_j, (_, (_, pixel)) in enumerate(noise_pixel_pairs):
                    sorted_chunk[i, new_j] = pixel
        else:  # vertical
            # Sort each column by noise values
            sorted_chunk = np.zeros_like(chunk_array)
            for j in range(chunk_width_actual):
                col_pixels = [(i, chunk_array[i, j]) for i in range(chunk_height_actual)]
                col_noise = noise_map[:, j]
                
                # Sort pixels by noise values
                noise_pixel_pairs = list(zip(col_noise, col_pixels))
                noise_pixel_pairs.sort(key=lambda x: x[0], reverse=reverse)
                
                # Place sorted pixels back into the column
                for new_i, (_, (_, pixel)) in enumerate(noise_pixel_pairs):
                    sorted_chunk[new_i, j] = pixel
        
        # Convert back to PIL Image and paste into the result
        sorted_chunk_img = Image.fromarray(sorted_chunk)
        sorted_image.paste(sorted_chunk_img, (x_offset, y_offset))
    
    # Reset random seed if it was set
    if seed is not None:
        np.random.seed(None)
    
    return sorted_image

def perlin_full_frame_sort(image, noise_scale=0.01, sort_by='brightness', reverse=False, seed=None):
    """
    Apply full-frame pixel sorting controlled by Perlin noise.
    
    Args:
        image (Image): PIL Image object to process.
        noise_scale (float): Scale of Perlin noise (higher = more detailed noise).
        sort_by (str): Property to sort by ('color', 'brightness', 'hue', 'red', 'green', 'blue',
                       'saturation', 'luminance', 'contrast').
        reverse (bool): Whether to reverse the sort order.
        seed (int, optional): Seed for the Perlin noise generator. If None, a random pattern is generated.
    
    Returns:
        Image: Processed image with Perlin noise-controlled full-frame sorting.
    """
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Create a new image with the same size as the input image
    width, height = image.size
    sorted_im = Image.new(image.mode, image.size)
    
    # Define the sort function based on the sort_by parameter
    sort_function = {
        'color': PixelAttributes.color_sum,
        'brightness': PixelAttributes.brightness,
        'hue': PixelAttributes.hue,
        'red': lambda p: p[0],     # Sort by red channel only
        'green': lambda p: p[1],   # Sort by green channel only
        'blue': lambda p: p[2],     # Sort by blue channel only
        'saturation': PixelAttributes.saturation,  # Sort by color saturation
        'luminance': PixelAttributes.luminance,    # Sort by luminance (value in HSV)
        'contrast': PixelAttributes.contrast       # Sort by contrast (max-min RGB)
    }.get(sort_by, PixelAttributes.brightness)  # Default to brightness if invalid choice
    
    # Set seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)
        random_state = np.random.RandomState(seed)
    else:
        random_state = np.random.RandomState()
    
    # Generate Perlin noise map for the entire image
    noise_map = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            # Use seed as base parameter if provided
            base = seed if seed is not None else 0
            noise_map[y, x] = noise.pnoise2(x * noise_scale, y * noise_scale, base=base)
    
    # Normalize noise map to [0, 1]
    noise_map = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
    
    # Sort each column based on Perlin noise values
    for x in range(width):
        # Get the pixels in the current column
        column_pixels = [(image.getpixel((x, y)), y, noise_map[y, x]) for y in range(height)]
        
        # Sort the pixels by Perlin noise value first, then by the specified criteria
        column_pixels.sort(key=lambda item: (item[2], sort_function(item[0])), reverse=reverse)
        
        # Set the pixels in the current column of the output image
        for new_y, (pixel, _, _) in enumerate(column_pixels):
            sorted_im.putpixel((x, new_y), pixel)
    
    # Reset random seed if it was set
    if seed is not None:
        np.random.seed(None)
    
    return sorted_im

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
    
    # Set seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)
        random_state = np.random.RandomState(seed)
    else:
        random_state = np.random.RandomState()
    
    # Generate Perlin noise map for the entire image
    noise_map = np.zeros((image.height, image.width))
    for i in range(image.height):
        for j in range(image.width):
            # Use seed as base parameter if provided
            base = seed if seed is not None else 0
            noise_map[i, j] = noise.pnoise2(i * noise_scale, j * noise_scale, base=base)
    
    # Normalize noise map to [0, 1]
    noise_map = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
    
    # Replace pixels where noise exceeds threshold
    mask = noise_map > threshold
    image_array[mask] = secondary_array[mask]
    
    # Reset random seed if it was set
    if seed is not None:
        np.random.seed(None)
    
    return Image.fromarray(image_array) 