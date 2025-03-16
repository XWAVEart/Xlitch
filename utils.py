from PIL import Image, ImageColor, ImageDraw
import os
import random
import numpy as np
import cv2
import colorsys

def load_image(file_path):
    """
    Load an image from the specified file path.
    
    Args:
        file_path (str): Path to the image file.
    
    Returns:
        Image or None: The loaded PIL Image object, or None if loading fails.
    """
    try:
        return Image.open(file_path)
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

def brightness(pixel):
    """
    Calculate the perceived brightness of a pixel using the luminosity formula.
    
    Args:
        pixel (tuple): RGB pixel values.
    
    Returns:
        float: Brightness value.
    """
    return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]

def hue(pixel):
    """
    Calculate the hue of a pixel.
    
    Args:
        pixel (tuple): RGB pixel values.
    
    Returns:
        int: Hue value (0-360).
    """
    return ImageColor.getcolor(f"#{pixel[0]:02x}{pixel[1]:02x}{pixel[2]:02x}", "HSV")[0]

def saturation(pixel):
    """
    Calculate the saturation of a pixel.
    
    Args:
        pixel (tuple): RGB pixel values.
    
    Returns:
        int: Saturation value (0-100).
    """
    return ImageColor.getcolor(f"#{pixel[0]:02x}{pixel[1]:02x}{pixel[2]:02x}", "HSV")[1]

def luminance(pixel):
    """
    Calculate the luminance (value in HSV) of a pixel.
    
    Args:
        pixel (tuple): RGB pixel values.
    
    Returns:
        int: Luminance value (0-100).
    """
    return ImageColor.getcolor(f"#{pixel[0]:02x}{pixel[1]:02x}{pixel[2]:02x}", "HSV")[2]

def contrast(pixel):
    """
    Calculate a contrast value based on the difference between max and min RGB values.
    
    Args:
        pixel (tuple): RGB pixel values.
    
    Returns:
        int: Contrast value.
    """
    return max(pixel[:3]) - min(pixel[:3])

def pixel_sorting(image, direction, chunk_size, sort_by, starting_corner=None):
    """
    Apply pixel sorting to the image in chunks based on a specified property.
    
    Args:
        image (Image): PIL Image object to process.
        direction (str): 'horizontal' or 'vertical' sorting direction.
        chunk_size (str): Chunk dimensions as 'widthxheight' (e.g., '32x32').
        sort_by (str): Property to sort by ('color', 'brightness', 'hue', 'red', 'green', 'blue', 
                       'saturation', 'luminance', 'contrast').
        starting_corner (str, optional): Starting corner for corner-to-corner sorting.
    
    Returns:
        Image: Processed image with sorted pixels.
    """
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    # Handle corner-to-corner sorting if starting_corner is provided
    if starting_corner:
        return pixel_sorting_corner_to_corner(
            image, 
            chunk_size, 
            sort_by, 
            starting_corner, 
            direction == 'horizontal'
        )
    
    sort_function = {
        'color': lambda p: sum(p[:3]),
        'brightness': brightness,
        'hue': hue,
        'red': lambda p: p[0],     # Sort by red channel only
        'green': lambda p: p[1],   # Sort by green channel only
        'blue': lambda p: p[2],     # Sort by blue channel only
        'saturation': saturation,  # Sort by color saturation
        'luminance': luminance,    # Sort by luminance (value in HSV)
        'contrast': contrast       # Sort by contrast (max-min RGB)
    }.get(sort_by, lambda p: sum(p[:3]))  # Default to sum of RGB if invalid

    pixels = list(image.getdata())
    width, height = image.size
    chunk_width, chunk_height = map(int, chunk_size.split('x'))
    sorted_pixels = []

    for chunk_row in range(0, height, chunk_height):
        for chunk_col in range(0, width, chunk_width):
            # Extract chunk pixels
            chunk_pixels = []
            for y in range(chunk_row, min(chunk_row + chunk_height, height)):
                for x in range(chunk_col, min(chunk_col + chunk_width, width)):
                    chunk_pixels.append(pixels[y * width + x])
            
            # Sort chunk
            if direction == 'horizontal':
                sorted_chunk = sorted(chunk_pixels, key=sort_function)
            else:  # vertical
                # Sorting columns instead of rows
                sorted_chunk = []
                chunk_width_actual = min(chunk_width, width - chunk_col)
                chunk_height_actual = min(chunk_height, height - chunk_row)
                
                # Reshape chunk pixels into a 2D array for easier column extraction
                chunk_2d = []
                for i in range(0, len(chunk_pixels), chunk_width_actual):
                    chunk_2d.append(chunk_pixels[i:i + chunk_width_actual])
                
                # Extract and sort each column
                for x in range(chunk_width_actual):
                    column = [chunk_2d[y][x] for y in range(chunk_height_actual)]
                    sorted_column = sorted(column, key=sort_function)
                    sorted_chunk.extend(sorted_column)

            # Place sorted chunk back into image
            for i, pixel in enumerate(sorted_chunk):
                if i < (min(chunk_width, width - chunk_col) * min(chunk_height, height - chunk_row)):
                    x = i % min(chunk_width, width - chunk_col) + chunk_col
                    y = i // min(chunk_width, width - chunk_col) + chunk_row
                    if x < width and y < height:
                        sorted_pixels.append((x, y, pixel))

    # Create a new image and place the sorted pixels
    result_image = Image.new(image.mode, image.size)
    
    # Fill with black first (in case there are any missing pixels)
    black_pixels = [(0, 0, 0)] * (width * height)
    result_image.putdata(black_pixels)
    
    # Place the sorted pixels
    for x, y, pixel in sorted_pixels:
        result_image.putpixel((x, y), pixel)
    
    return result_image

def pixel_sorting_corner_to_corner(image, chunk_size, sort_by, corner, horizontal):
    """
    Apply pixel sorting starting from a specified corner, either horizontally or vertically.
    
    Args:
        image (Image): PIL Image object to process.
        chunk_size (str): Chunk dimensions as 'widthxheight' (e.g., '32x32').
        sort_by (str): Property to sort by ('color', 'brightness', 'hue', 'red', 'green', 'blue',
                       'saturation', 'luminance', 'contrast').
        corner (str): Starting corner ('top-left', 'top-right', 'bottom-left', 'bottom-right').
        horizontal (bool): True for horizontal sorting, False for vertical.
    
    Returns:
        Image: Processed image with corner-to-corner sorting.
    """
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    sort_function = {
        'color': lambda p: sum(p[:3]),
        'brightness': brightness,
        'hue': hue,
        'red': lambda p: p[0],     # Sort by red channel only
        'green': lambda p: p[1],   # Sort by green channel only
        'blue': lambda p: p[2],     # Sort by blue channel only
        'saturation': saturation,  # Sort by color saturation
        'luminance': luminance,    # Sort by luminance (value in HSV)
        'contrast': contrast       # Sort by contrast (max-min RGB)
    }.get(sort_by, lambda p: sum(p[:3]))

    # Create a copy of the image to work with
    result_image = image.copy()
    width, height = image.size
    chunk_width, chunk_height = map(int, chunk_size.split('x'))
    
    # Get pixel data as a 2D array
    pixels = list(image.getdata())
    pixels_2d = []
    for y in range(height):
        row = []
        for x in range(width):
            row.append(pixels[y * width + x])
        pixels_2d.append(row)
    
    # Determine chunk processing order based on corner
    x_range = range(0, width, chunk_width)
    y_range = range(0, height, chunk_height)
    
    if corner in ['top-right', 'bottom-right']:
        x_range = range(width - chunk_width, -1, -chunk_width)
    
    if corner in ['bottom-left', 'bottom-right']:
        y_range = range(height - chunk_height, -1, -chunk_height)
    
    # Process each chunk
    for y_start in y_range:
        for x_start in x_range:
            # Calculate chunk boundaries
            y_end = min(y_start + chunk_height, height)
            x_end = min(x_start + chunk_width, width)
            
            # Make sure we're not going out of bounds
            if x_start < 0: x_start = 0
            if y_start < 0: y_start = 0
            
            # Extract all pixels in this chunk
            chunk_pixels = []
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    chunk_pixels.append(pixels_2d[y][x])
            
            # Sort the chunk pixels
            sorted_pixels = sorted(chunk_pixels, key=sort_function)
            
            # Reverse the order if needed based on corner
            if (corner in ['bottom-left', 'bottom-right'] and horizontal) or \
               (corner in ['top-right', 'bottom-right'] and not horizontal):
                sorted_pixels = sorted_pixels[::-1]
            
            # Put the sorted pixels back
            pixel_index = 0
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    if pixel_index < len(sorted_pixels):
                        result_image.putpixel((x, y), sorted_pixels[pixel_index])
                        pixel_index += 1
    
    return result_image

def color_channel_manipulation(image, manipulation_type, choice, factor=None):
    """
    Manipulate the image's color channels (swap, invert, or adjust intensity).
    
    Args:
        image (Image): PIL Image object to process.
        manipulation_type (str): 'swap', 'invert', or 'adjust'.
        choice (str): Specific channel or swap pair (e.g., 'red-green', 'red').
        factor (float, optional): Intensity adjustment factor (required for 'adjust').
    
    Returns:
        Image: Processed image with modified color channels.
    """
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        # If image has alpha channel (RGBA), convert to RGB
        image = image.convert('RGB')
    
    r, g, b = image.split()
    if manipulation_type == 'swap':
        if choice == 'red-green':
            image = Image.merge(image.mode, (g, r, b))
        elif choice == 'red-blue':
            image = Image.merge(image.mode, (b, g, r))
        elif choice == 'green-blue':
            image = Image.merge(image.mode, (r, b, g))
    elif manipulation_type == 'invert':
        if choice == 'red':
            r = r.point(lambda i: 255 - i)
        elif choice == 'green':
            g = g.point(lambda i: 255 - i)
        elif choice == 'blue':
            b = b.point(lambda i: 255 - i)
        image = Image.merge(image.mode, (r, g, b))
    elif manipulation_type == 'adjust':
        if factor is None:
            raise ValueError("Factor is required for adjust manipulation")
        if choice == 'red':
            r = r.point(lambda i: min(255, int(i * factor)))
        elif choice == 'green':
            g = g.point(lambda i: min(255, int(i * factor)))
        elif choice == 'blue':
            b = b.point(lambda i: min(255, int(i * factor)))
        image = Image.merge(image.mode, (r, g, b))
    return image

def data_moshing(image, secondary_image, blend_mode='classic', opacity=0.5):
    """
    Apply a double expose effect by blending two images with various blend modes.
    
    Args:
        image (Image): Primary PIL Image object to process.
        secondary_image (Image): Secondary image to blend with the primary image.
        blend_mode (str): Blending mode to use ('classic', 'screen', 'multiply', 'overlay', 'difference').
        opacity (float): Opacity of the secondary image (0.0 to 1.0).
    
    Returns:
        Image: Processed image with double expose effect.
    """
    # Convert both images to RGB mode if they have alpha channels or are in different modes
    if image.mode != 'RGB':
        image = image.convert('RGB')
    if secondary_image.mode != 'RGB':
        secondary_image = secondary_image.convert('RGB')
        
    secondary_image = secondary_image.resize(image.size)
    
    # Convert to numpy arrays for more advanced blending
    img1 = np.array(image).astype(float)
    img2 = np.array(secondary_image).astype(float)
    
    # Apply the selected blend mode
    if blend_mode == 'classic':
        # Simple alpha blending
        blended = img1 * (1 - opacity) + img2 * opacity
    elif blend_mode == 'screen':
        # Screen blend mode: 1 - (1 - img1) * (1 - img2)
        blended = 255 - (255 - img1) * (255 - img2) / 255
        blended = img1 * (1 - opacity) + blended * opacity
    elif blend_mode == 'multiply':
        # Multiply blend mode: img1 * img2
        blended = img1 * img2 / 255
        blended = img1 * (1 - opacity) + blended * opacity
    elif blend_mode == 'overlay':
        # Overlay blend mode
        mask = img1 > 127.5
        blended = np.zeros_like(img1)
        blended[mask] = 255 - (255 - 2 * (img1[mask] - 127.5)) * (255 - img2[mask]) / 255
        blended[~mask] = (2 * img1[~mask]) * img2[~mask] / 255
        blended = img1 * (1 - opacity) + blended * opacity
    elif blend_mode == 'difference':
        # Difference blend mode: |img1 - img2|
        blended = np.abs(img1 - img2)
        blended = img1 * (1 - opacity) + blended * opacity
    elif blend_mode == 'color_dodge':
        # Color dodge blend mode: img1 / (1 - img2)
        blended = np.zeros_like(img1)
        mask = img2 < 255
        blended[mask] = np.minimum(255, img1[mask] / (1 - img2[mask] / 255))
        blended[~mask] = 255
        blended = img1 * (1 - opacity) + blended * opacity
    else:
        # Default to classic blend if mode not recognized
        blended = img1 * (1 - opacity) + img2 * opacity
    
    # Clip values to valid range and convert back to uint8
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    # Create a new image from the blended array
    return Image.fromarray(blended)

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
    # Try to import noise packages, with fallbacks
    noise_module = None
    try:
        # Try the noise package first
        from noise import pnoise2
        noise_module = 'noise'
    except ImportError:
        try:
            # Try noise-python as a fallback
            from noise_python import snoise2
            noise_module = 'noise-python'
        except ImportError:
            raise ImportError("Either 'noise' or 'noise-python' package is required for Perlin Merge. Install with: pip install noise noise-python")
    
    # Convert both images to RGB mode if they have alpha channels or are in different modes
    if image.mode != 'RGB':
        image = image.convert('RGB')
    if secondary_image.mode != 'RGB':
        secondary_image = secondary_image.convert('RGB')
    
    # Resize secondary image to match primary image
    secondary_image = secondary_image.resize(image.size)
    image_array = np.array(image)
    secondary_array = np.array(secondary_image)
    
    # Generate Perlin noise map for the entire image
    noise_map = np.zeros((image.height, image.width))
    for i in range(image.height):
        for j in range(image.width):
            # Use the appropriate noise function based on which package is available
            if noise_module == 'noise':
                from noise import pnoise2
                # Use seed as base parameter if provided
                if seed is not None:
                    noise_map[i, j] = pnoise2(i * noise_scale, j * noise_scale, base=seed)
                else:
                    noise_map[i, j] = pnoise2(i * noise_scale, j * noise_scale)
            else:  # noise-python
                from noise_python import snoise2
                # Use seed as octaves parameter if provided (not ideal but works as a seed)
                if seed is not None:
                    noise_map[i, j] = snoise2(i * noise_scale, j * noise_scale, octaves=seed % 10 + 1)
                else:
                    noise_map[i, j] = snoise2(i * noise_scale, j * noise_scale)
    
    # Normalize noise map to [0, 1]
    noise_map = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
    
    # Replace pixels where noise exceeds threshold
    mask = noise_map > threshold
    image_array[mask] = secondary_array[mask]
    
    return Image.fromarray(image_array)

def pixel_drift(image, direction='down', num_bands=10, intensity=1.0):
    """
    Apply a pixel drift effect, shifting pixels in a specified direction with variable amounts
    to create a more glitchy effect.
    
    Args:
        image (Image): PIL Image object to process.
        direction (str): Direction of drift ('up', 'down', 'left', 'right').
        num_bands (int): Number of bands to divide the image into (more bands = more variation).
        intensity (float): Multiplier for the drift amount (higher = more extreme drift).
    
    Returns:
        Image: Processed image with drifted pixels.
    """
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    pixels = list(image.getdata())
    width, height = image.size
    drifted_pixels = []
    
    # Create a more interesting effect with variable drift amounts
    
    # Base drift amount with intensity multiplier
    base_drift = int(20 * intensity)
    
    # Create bands of different drift amounts
    band_height = height // num_bands
    band_width = width // num_bands
    
    if direction == 'down' or direction == 'up':
        # Create a more interesting vertical drift effect
        for y in range(height):
            # Calculate drift amount based on position
            # Every few rows, change the drift amount to create bands
            band_index = y // band_height
            # Alternate between different drift amounts
            if band_index % 3 == 0:
                drift_amount = base_drift
            elif band_index % 3 == 1:
                drift_amount = base_drift * 2
            else:
                drift_amount = base_drift // 2
                
            # Add some randomness to the drift
            if random.random() < 0.1:  # 10% chance of a random shift
                drift_amount = random.randint(5, int(50 * intensity))
                
            for x in range(width):
                if direction == 'down':
                    source_y = (y - drift_amount) % height
                else:  # up
                    source_y = (y + drift_amount) % height
                drifted_pixels.append(pixels[source_y * width + x])
    
    elif direction == 'left' or direction == 'right':
        # Create a more interesting horizontal drift effect
        
        for y in range(height):
            for x in range(width):
                # Calculate drift amount based on position
                band_index = x // band_width
                # Alternate between different drift amounts
                if band_index % 3 == 0:
                    drift_amount = base_drift
                elif band_index % 3 == 1:
                    drift_amount = base_drift * 2
                else:
                    drift_amount = base_drift // 2
                    
                # Add some randomness to the drift
                if random.random() < 0.1:  # 10% chance of a random shift
                    drift_amount = random.randint(5, int(50 * intensity))
                    
                if direction == 'right':
                    source_x = (x - drift_amount) % width
                else:  # left
                    source_x = (x + drift_amount) % width
                drifted_pixels.append(pixels[y * width + source_x])
    
    # Create a new image and put all the drifted pixels at once
    drifted_image = Image.new(image.mode, image.size)
    drifted_image.putdata(drifted_pixels)
    return drifted_image

def bit_manipulation(image, chunk_size=1):
    """
    Apply a bit manipulation effect by inverting bytes of the image data in chunks.
    
    Args:
        image (Image): PIL Image object to process.
        chunk_size (int): Size of chunks to process together (1=every other byte, 2=every other 2 bytes, etc.)
    
    Returns:
        Image: Processed image with bit-level glitches.
    """
    # Convert image to RGB mode if it's not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Get image data as bytes
    image_bytes = bytearray(image.tobytes())
    
    # Manipulate bytes in chunks
    chunk_size = max(1, chunk_size)  # Ensure chunk_size is at least 1
    chunk_total = chunk_size * 2  # Total size of a chunk pair (manipulated + skipped)
    
    for i in range(0, len(image_bytes), chunk_total):
        # Manipulate 'chunk_size' bytes, then skip 'chunk_size' bytes
        for j in range(chunk_size):
            if i + j < len(image_bytes):
                image_bytes[i + j] = image_bytes[i + j] ^ 0xFF  # XOR with 0xFF to invert bits
    
    # Create a new image from the manipulated bytes
    manipulated_image = Image.frombytes('RGB', image.size, bytes(image_bytes))
    return manipulated_image

def spiral_coords(size):
    """
    Generate coordinates in spiral order starting from the center of a square chunk.
    
    Args:
        size (int): Size of the square chunk.
        
    Yields:
        tuple: (x, y) coordinates in spiral order.
    """
    x, y = size // 2, size // 2  # Start at the center
    yield (x, y)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
    steps = 1
    dir_idx = 0
    while steps < size:
        for _ in range(2):  # Two sides per step increase (e.g., right then down)
            dx, dy = directions[dir_idx % 4]
            for _ in range(steps):
                x += dx
                y += dy
                if 0 <= x < size and 0 <= y < size:
                    yield (x, y)
            dir_idx += 1
        steps += 1

def spiral_sort(image, chunk_size=32, order='lightest-to-darkest'):
    """
    Apply a spiral sort effect, arranging pixels in a spiral pattern based on brightness.
    
    Args:
        image (Image): PIL Image object to process.
        chunk_size (int): Size of square chunks to process.
        order (str): 'lightest-to-darkest' or 'darkest-to-lightest'.
    
    Returns:
        Image: Processed image with spiral-sorted pixels.
    """
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert PIL image to numpy array
    np_image = np.array(image)
    
    # Get image dimensions
    height, width, _ = np_image.shape
    
    # Adjust chunk_size if needed to ensure it divides evenly into the image
    if height % chunk_size != 0 or width % chunk_size != 0:
        # Find the largest chunk size that divides evenly
        for i in range(chunk_size, 0, -1):
            if height % i == 0 and width % i == 0:
                chunk_size = i
                break
    
    # Split the image into chunks
    chunks = []
    for y in range(0, height, chunk_size):
        for x in range(0, width, chunk_size):
            if y + chunk_size <= height and x + chunk_size <= width:
                chunks.append(np_image[y:y+chunk_size, x:x+chunk_size])
    
    # Sort each chunk
    sorted_chunks = []
    for chunk in chunks:
        # Flatten the chunk and calculate luminance
        flattened_chunk = chunk.reshape(-1, chunk.shape[-1])
        luminance = np.mean(flattened_chunk, axis=-1)
        sorted_indices = np.argsort(luminance)
        
        # Reverse order if needed
        if order == 'darkest-to-lightest':
            sorted_indices = sorted_indices[::-1]
        
        # Create a new chunk with pixels arranged in a spiral
        sorted_chunk = np.zeros_like(chunk)
        spiral_order = list(spiral_coords(chunk_size))
        
        # Place pixels in spiral order
        for idx, coord in zip(sorted_indices, spiral_order):
            pixel_y, pixel_x = divmod(idx, chunk_size)
            sorted_chunk[coord[0], coord[1]] = chunk[pixel_y, pixel_x]
        
        sorted_chunks.append(sorted_chunk)
    
    # Recombine chunks into the final image
    result = np.zeros_like(np_image)
    chunk_idx = 0
    for y in range(0, height, chunk_size):
        for x in range(0, width, chunk_size):
            if y + chunk_size <= height and x + chunk_size <= width:
                result[y:y+chunk_size, x:x+chunk_size] = sorted_chunks[chunk_idx]
                chunk_idx += 1
    
    # Convert back to PIL image
    return Image.fromarray(result)

def spiral_sort_2(image, chunk_size=64, sort_by='brightness', reverse=False):
    """
    Apply spiral sorting starting from the center of each chunk, with pixels sorted by the specified property.
    
    Args:
        image (Image): PIL Image object to process.
        chunk_size (int): Size of square chunks to process.
        sort_by (str): Property to sort by ('color', 'brightness', 'hue', 'red', 'green', 'blue',
                       'saturation', 'luminance', 'contrast').
        reverse (bool): Whether to reverse the sort order.
        
    Returns:
        Image: Processed image with spiral sorting.
    """
    # Convert PIL Image to OpenCV format (BGR)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    # Convert RGB to BGR (OpenCV format)
    img_array = img_array[:, :, ::-1].copy()
    
    # Define the sort function based on the sort_by parameter
    sort_function = {
        'color': lambda p: np.sum(p),
        'brightness': lambda p: 0.299 * p[2] + 0.587 * p[1] + 0.114 * p[0],  # RGB is BGR in OpenCV
        'hue': lambda p: cv2.cvtColor(np.array([[p]]), cv2.COLOR_BGR2HSV)[0, 0, 0],
        'red': lambda p: p[2],  # Red is at index 2 in BGR
        'green': lambda p: p[1],  # Green is at index 1 in BGR
        'blue': lambda p: p[0],  # Blue is at index 0 in BGR
        'saturation': lambda p: cv2.cvtColor(np.array([[p]]), cv2.COLOR_BGR2HSV)[0, 0, 1],
        'luminance': lambda p: cv2.cvtColor(np.array([[p]]), cv2.COLOR_BGR2HSV)[0, 0, 2],
        'contrast': lambda p: np.max(p) - np.min(p)
    }.get(sort_by, lambda p: np.sum(p))
    
    # Pad the image to make dimensions multiples of chunk_size
    height, width = img_array.shape[:2]
    pad_height = (chunk_size - height % chunk_size) % chunk_size
    pad_width = (chunk_size - width % chunk_size) % chunk_size
    padded_image = np.pad(img_array, ((0, pad_height), (0, pad_width), (0, 0)), 
                         mode='constant', constant_values=0)
    
    # Split the padded image into chunks
    padded_height, padded_width = padded_image.shape[:2]
    chunks = []
    for y in range(0, padded_height, chunk_size):
        for x in range(0, padded_width, chunk_size):
            chunk = padded_image[y:y+chunk_size, x:x+chunk_size]
            chunks.append((chunk, (y, x)))
    
    # Sort each chunk in spiral order
    for chunk, (y_offset, x_offset) in chunks:
        # Get spiral order coordinates
        spiral_coords_list = list(spiral_coords(chunk_size))
        
        # Collect pixels in spiral order
        pixels = [chunk[coord] for coord in spiral_coords_list]
        
        # Calculate sort values
        sort_values = np.array([sort_function(pixel) for pixel in pixels])
        
        # Sort pixels based on the sort values
        sorted_indices = np.argsort(sort_values)
        if reverse:
            sorted_indices = sorted_indices[::-1]
        
        # Create sorted pixel list
        sorted_pixels = [pixels[idx] for idx in sorted_indices]
        
        # Place sorted pixels back into the chunk in spiral order
        for (y_coord, x_coord), pixel in zip(spiral_coords_list, sorted_pixels):
            chunk[y_coord, x_coord] = pixel
    
    # Combine chunks back into the full image
    num_cols = padded_width // chunk_size
    rows = []
    for i in range(0, len(chunks), num_cols):
        row_chunks = [chunk[0] for chunk in chunks[i:i + num_cols]]
        row = np.concatenate(row_chunks, axis=1)
        rows.append(row)
    sorted_image = np.concatenate(rows, axis=0)
    
    # Crop to original size if padded
    sorted_image = sorted_image[:height, :width]
    
    # Convert back to PIL Image (RGB)
    sorted_image_rgb = sorted_image[:, :, ::-1]  # BGR to RGB
    result_image = Image.fromarray(sorted_image_rgb)
    
    return result_image

def generate_output_filename(original_filename, effect, settings):
    """
    Generate a descriptive filename for the processed image.
    
    Args:
        original_filename (str): Original image filename.
        effect (str): Name of the applied effect.
        settings (str): String representing effect settings.
    
    Returns:
        str: New filename with effect and settings appended.
    """
    base, ext = os.path.splitext(original_filename)
    return f"{base}_{effect}_{settings}{ext}"

def full_frame_sort(image, direction='vertical', sort_by='brightness', reverse=False):
    """
    Apply full-frame pixel sorting in the specified direction.
    
    Args:
        image (Image): PIL Image object to process.
        direction (str): Direction of sorting ('vertical', 'horizontal').
        sort_by (str): Property to sort by ('color', 'brightness', 'hue', 'red', 'green', 'blue',
                       'saturation', 'luminance', 'contrast').
        reverse (bool): Whether to reverse the sort order.
    
    Returns:
        Image: Processed image with full-frame sorting.
    """
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Create a new image with the same size as the input image
    width, height = image.size
    sorted_im = Image.new(image.mode, image.size)
    
    # Define the sort function based on the sort_by parameter
    sort_function = {
        'color': lambda p: sum(p[:3]),
        'brightness': brightness,
        'hue': hue,
        'red': lambda p: p[0],     # Sort by red channel only
        'green': lambda p: p[1],   # Sort by green channel only
        'blue': lambda p: p[2],     # Sort by blue channel only
        'saturation': saturation,  # Sort by color saturation
        'luminance': luminance,    # Sort by luminance (value in HSV)
        'contrast': contrast       # Sort by contrast (max-min RGB)
    }.get(sort_by, lambda p: sum(p[:3]))
    
    if direction == 'vertical':
        # Sort each column from top to bottom
        for x in range(width):
            # Get the pixels in the current column
            column_pixels = [(image.getpixel((x, y)), y) for y in range(height)]
            
            # Sort the pixels by the specified criteria
            column_pixels.sort(key=lambda item: sort_function(item[0]), reverse=reverse)
            
            # Set the pixels in the current column of the output image
            for new_y, (pixel, _) in enumerate(column_pixels):
                sorted_im.putpixel((x, new_y), pixel)
    
    elif direction == 'horizontal':
        # Sort each row from left to right
        for y in range(height):
            # Get the pixels in the current row
            row_pixels = [(image.getpixel((x, y)), x) for x in range(width)]
            
            # Sort the pixels by the specified criteria
            row_pixels.sort(key=lambda item: sort_function(item[0]), reverse=reverse)
            
            # Set the pixels in the current row of the output image
            for new_x, (pixel, _) in enumerate(row_pixels):
                sorted_im.putpixel((new_x, y), pixel)
    
    return sorted_im

def polar_sorting(image, chunk_size, sort_by='angle', reverse=False):
    """
    Apply polar sorting to an image by sorting pixels within chunks based on polar coordinates.

    Args:
        image (Image): PIL Image object to process.
        chunk_size (int): Size of square chunks (e.g., 32 for 32x32 chunks).
        sort_by (str): 'angle' to sort by angle, 'radius' to sort by distance from center.
        reverse (bool): Whether to reverse the sort order.

    Returns:
        Image: Processed image with polar-sorted pixels.
    """
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    width, height = image.size
    # Split image into chunks
    chunks = []
    for y in range(0, height, chunk_size):
        for x in range(0, width, chunk_size):
            # Calculate actual chunk dimensions (handle edge cases)
            actual_width = min(chunk_size, width - x)
            actual_height = min(chunk_size, height - y)
            if actual_width > 0 and actual_height > 0:
                chunks.append((image.crop((x, y, x + actual_width, y + actual_height)), (x, y)))

    sorted_image = Image.new('RGB', image.size)
    
    for chunk, (x_offset, y_offset) in chunks:
        chunk_width, chunk_height = chunk.size
        chunk_array = np.array(chunk)
        
        # Calculate polar coordinates relative to chunk center
        cx, cy = chunk_width // 2, chunk_height // 2
        y_coords, x_coords = np.mgrid[:chunk_height, :chunk_width]
        angles = np.arctan2(y_coords - cy, x_coords - cx)  # Angle from center
        radii = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)  # Distance from center
        
        # Create a mapping of original positions to sorted positions
        positions = []
        for y in range(chunk_height):
            for x in range(chunk_width):
                pixel = chunk_array[y, x]
                angle = angles[y, x]
                radius = radii[y, x]
                sort_value = angle if sort_by == 'angle' else radius
                positions.append(((y, x), sort_value, pixel))
        
        # Sort by the chosen coordinate
        positions.sort(key=lambda p: p[1], reverse=reverse)
        
        # Create sorted chunk
        sorted_chunk = np.zeros_like(chunk_array)
        for i, ((orig_y, orig_x), _, pixel) in enumerate(positions):
            new_y = i // chunk_width
            new_x = i % chunk_width
            if new_y < chunk_height and new_x < chunk_width:
                sorted_chunk[new_y, new_x] = pixel
        
        # Convert back to PIL Image and paste into the result
        sorted_chunk_img = Image.fromarray(sorted_chunk)
        sorted_image.paste(sorted_chunk_img, (x_offset, y_offset))
    
    return sorted_image

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
    # Try to import noise packages, with fallbacks
    noise_module = None
    try:
        # Try the noise package first
        from noise import pnoise2
        noise_module = 'noise'
    except ImportError:
        try:
            # Try noise-python as a fallback
            from noise_python import snoise2
            noise_module = 'noise-python'
        except ImportError:
            raise ImportError("Either 'noise' or 'noise-python' package is required for Perlin noise sorting. Install with: pip install noise noise-python")
    
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
                
                # Use the appropriate noise function based on which package is available
                if noise_module == 'noise':
                    from noise import pnoise2
                    # Use seed as base parameter if provided
                    if seed is not None:
                        noise_map[i, j] = pnoise2(x_noise, y_noise, base=seed)
                    else:
                        noise_map[i, j] = pnoise2(x_noise, y_noise)
                else:  # noise-python
                    from noise_python import snoise2
                    # Use seed as octaves parameter if provided (not ideal but works as a seed)
                    if seed is not None:
                        noise_map[i, j] = snoise2(x_noise, y_noise, octaves=seed % 10 + 1)
                    else:
                        noise_map[i, j] = snoise2(x_noise, y_noise)
        
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
    # Try to import noise packages, with fallbacks
    noise_module = None
    try:
        # Try the noise package first
        from noise import pnoise2
        noise_module = 'noise'
    except ImportError:
        try:
            # Try noise-python as a fallback
            from noise_python import snoise2
            noise_module = 'noise-python'
        except ImportError:
            raise ImportError("Either 'noise' or 'noise-python' package is required for Perlin noise sorting. Install with: pip install noise noise-python")
    
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Create a new image with the same size as the input image
    width, height = image.size
    sorted_im = Image.new(image.mode, image.size)
    
    # Define the sort function based on the sort_by parameter
    sort_function = {
        'color': lambda p: sum(p[:3]),
        'brightness': brightness,
        'hue': hue,
        'red': lambda p: p[0],     # Sort by red channel only
        'green': lambda p: p[1],   # Sort by green channel only
        'blue': lambda p: p[2],     # Sort by blue channel only
        'saturation': saturation,  # Sort by color saturation
        'luminance': luminance,    # Sort by luminance (value in HSV)
        'contrast': contrast       # Sort by contrast (max-min RGB)
    }.get(sort_by, lambda p: sum(p[:3]))
    
    # Generate Perlin noise map for the entire image
    noise_map = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            # Use the appropriate noise function based on which package is available
            if noise_module == 'noise':
                from noise import pnoise2
                # Use seed as base parameter if provided
                if seed is not None:
                    noise_map[y, x] = pnoise2(x * noise_scale, y * noise_scale, base=seed)
                else:
                    noise_map[y, x] = pnoise2(x * noise_scale, y * noise_scale)
            else:  # noise-python
                from noise_python import snoise2
                # Use seed as octaves parameter if provided (not ideal but works as a seed)
                if seed is not None:
                    noise_map[y, x] = snoise2(x * noise_scale, y * noise_scale, octaves=seed % 10 + 1)
                else:
                    noise_map[y, x] = snoise2(x * noise_scale, y * noise_scale)
    
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
    
    return sorted_im

def pixelate_by_mode(image, pixel_width=8, pixel_height=8, attribute='color', num_bins=100):
    """
    Pixelate an image by creating larger blocks of pixels based on various attributes.
    
    Args:
        image (Image): PIL Image object to process.
        pixel_width (int): Width of each pixel block.
        pixel_height (int): Height of each pixel block.
        attribute (str): Attribute to use for determining pixel color ('color', 'brightness', 'hue', etc.).
        num_bins (int): Number of bins to use for quantizing attribute values.
    
    Returns:
        Image: A new pixelated image.
    """
    # Create a copy of the image
    im = image.copy()
    width, height = im.size
    
    # Create a new image with the same size
    result = Image.new('RGB', (width, height))
    
    # Define the attribute function based on the selected attribute
    if attribute == 'color':
        # For color, we'll use the most common color in the block
        def attr_func(pixel):
            return pixel
    elif attribute == 'brightness':
        def attr_func(pixel):
            return brightness(pixel)
    elif attribute == 'hue':
        def attr_func(pixel):
            return hue(pixel)
    elif attribute == 'saturation':
        def attr_func(pixel):
            return saturation(pixel)
    elif attribute == 'luminance':
        def attr_func(pixel):
            return luminance(pixel)
    
    # Process each pixel block
    for y in range(0, height, pixel_height):
        for x in range(0, width, pixel_width):
            # Define the block boundaries
            block_right = min(x + pixel_width, width)
            block_bottom = min(y + pixel_height, height)
            
            # Get all pixels in the block
            block_pixels = []
            for by in range(y, block_bottom):
                for bx in range(x, block_right):
                    pixel = im.getpixel((bx, by))
                    block_pixels.append(pixel)
            
            if attribute == 'color':
                # Find the most common color in the block
                color_counts = {}
                for pixel in block_pixels:
                    pixel_str = str(pixel)
                    if pixel_str in color_counts:
                        color_counts[pixel_str] += 1
                    else:
                        color_counts[pixel_str] = 1
                
                # Get the most common color
                most_common_color_str = max(color_counts, key=color_counts.get)
                most_common_color = eval(most_common_color_str)
                
                # Fill the block with the most common color
                for by in range(y, block_bottom):
                    for bx in range(x, block_right):
                        result.putpixel((bx, by), most_common_color)
            else:
                # Calculate the average attribute value for the block
                attr_values = [attr_func(pixel) for pixel in block_pixels]
                avg_attr = sum(attr_values) / len(attr_values)
                
                # Find a representative pixel with the closest attribute value
                closest_pixel = min(block_pixels, key=lambda p: abs(attr_func(p) - avg_attr))
                
                # Fill the block with the representative color
                for by in range(y, block_bottom):
                    for bx in range(x, block_right):
                        result.putpixel((bx, by), closest_pixel)
    
    return result

def draw_concentric_squares(image, num_points=10, num_squares=5, thickness=2):
    """
    Draw concentric squares around randomly chosen points on an image, starting from points closest to the edge.

    Args:
        image (Image): PIL Image object to process.
        num_points (int): Number of random points to select.
        num_squares (int): Number of concentric squares per point.
        thickness (int): Thickness of each square's outline and spacing between squares.

    Returns:
        Image: A new image with concentric squares drawn over the original.
    """
    # Create a copy of the image
    im = image.copy()
    draw = ImageDraw.Draw(im)
    width, height = im.size

    # Calculate the image center
    center_x = width // 2
    center_y = height // 2

    # Generate random points within image boundaries
    points = [(random.randint(0, width - 1), random.randint(0, height - 1)) for _ in range(num_points)]

    # Sort points by distance from center, farthest first (descending order)
    sorted_points = sorted(points, key=lambda p: (p[0] - center_x)**2 + (p[1] - center_y)**2, reverse=True)

    # Process each point in sorted order
    for cx, cy in sorted_points:
        # Get the color at the point's pixel
        color = im.getpixel((cx, cy))

        # Draw concentric squares centered on (cx, cy)
        for k in range(1, num_squares + 1):
            # Calculate the half-side length for the k-th square
            S = k * 2 * thickness
            left = cx - S
            top = cy - S
            right = cx + S
            bottom = cy + S
            # Draw the square as an outline with specified thickness
            draw.rectangle([left, top, right, bottom], outline=color, width=thickness)

    return im