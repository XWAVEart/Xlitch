from PIL import Image, ImageColor
import os
import random
import numpy as np

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
    Calculate the perceived brightness of a pixel using the luminance formula.
    
    Args:
        pixel (tuple): RGB tuple (red, green, blue).
    
    Returns:
        float: Brightness value.
    """
    return pixel[0] * 0.299 + pixel[1] * 0.587 + pixel[2] * 0.114

def hue(pixel):
    """
    Calculate the hue of a pixel by converting RGB to HSV.
    
    Args:
        pixel (tuple): RGB tuple (red, green, blue).
    
    Returns:
        int: Hue value (0-360).
    """
    return ImageColor.getcolor(f"#{pixel[0]:02x}{pixel[1]:02x}{pixel[2]:02x}", "HSV")[0]

def pixel_sorting(image, direction, chunk_size, sort_by, starting_corner=None):
    """
    Apply pixel sorting to the image in chunks based on a specified property.
    
    Args:
        image (Image): PIL Image object to process.
        direction (str): 'horizontal' or 'vertical' sorting direction.
        chunk_size (str): Chunk dimensions as 'widthxheight' (e.g., '32x32').
        sort_by (str): Property to sort by ('color', 'brightness', 'hue').
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
        'hue': hue
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
        sort_by (str): Property to sort by ('color', 'brightness', 'hue').
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
        'hue': hue
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
    
    if blend_mode == 'classic':
        # Classic alpha blend (weighted average)
        result = Image.blend(image, secondary_image, alpha=opacity)
    
    elif blend_mode == 'screen':
        # Screen blend mode: brightens the image, good for dark images
        # Formula: 1 - (1 - img1/255) * (1 - img2/255) * 255
        result_array = 255 - (((255 - img1) * (255 - img2)) / 255)
        # Apply opacity
        result_array = img1 * (1 - opacity) + result_array * opacity
        result = Image.fromarray(np.clip(result_array, 0, 255).astype(np.uint8))
    
    elif blend_mode == 'multiply':
        # Multiply blend mode: darkens the image, good for light images
        # Formula: (img1 * img2) / 255
        result_array = (img1 * img2) / 255
        # Apply opacity
        result_array = img1 * (1 - opacity) + result_array * opacity
        result = Image.fromarray(np.clip(result_array, 0, 255).astype(np.uint8))
    
    elif blend_mode == 'overlay':
        # Overlay blend mode: increases contrast, preserves highlights and shadows
        # Formula: if img1 < 128: 2 * img1 * img2 / 255, else: 1 - 2 * (255 - img1) * (255 - img2) / 255
        mask = img1 < 128
        result_array = np.zeros_like(img1)
        result_array[mask] = (2 * img1[mask] * img2[mask]) / 255
        result_array[~mask] = 255 - (2 * (255 - img1[~mask]) * (255 - img2[~mask])) / 255
        # Apply opacity
        result_array = img1 * (1 - opacity) + result_array * opacity
        result = Image.fromarray(np.clip(result_array, 0, 255).astype(np.uint8))
    
    elif blend_mode == 'difference':
        # Difference blend mode: subtracts the darker color from the lighter one
        # Formula: |img1 - img2|
        result_array = np.abs(img1 - img2)
        # Apply opacity
        result_array = img1 * (1 - opacity) + result_array * opacity
        result = Image.fromarray(np.clip(result_array, 0, 255).astype(np.uint8))
    
    elif blend_mode == 'color_dodge':
        # Color Dodge: brightens the base color to reflect the blend color
        # Formula: img1 / (255 - img2) * 255
        # Avoid division by zero
        img2_safe = np.where(img2 == 255, 254, img2)
        result_array = (img1 / (255 - img2_safe)) * 255
        # Apply opacity
        result_array = img1 * (1 - opacity) + result_array * opacity
        result = Image.fromarray(np.clip(result_array, 0, 255).astype(np.uint8))
    
    else:
        # Default to classic blend if an invalid mode is specified
        result = Image.blend(image, secondary_image, alpha=opacity)
    
    return result

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
    Generate coordinates for a spiral pattern starting from the center.
    
    Args:
        size (int): Size of the square grid.
    
    Yields:
        tuple: (x, y) coordinates in spiral order.
    """
    start_point = ((size - 1) // 2, (size - 1) // 2)
    yield start_point
    
    # Generate spiral coordinates
    for square in range(1, size, 2):
        for dx, dy in [(0, -1), (-1, 0), (0, 1), (1, 0)]:
            if dx:
                for _ in range(square):
                    start_point = (start_point[0] + dx, start_point[1])
                    yield start_point
            else:
                for _ in range(square):
                    start_point = (start_point[0], start_point[1] + dy)
                    yield start_point
    
    # Handle any remaining coordinates if size is even
    from itertools import cycle
    dirs = cycle(((0, 1), (1, 0), (0, -1), (-1, 0)))
    for radius in range(1, size):
        for _ in range(2):
            direction = next(dirs)
            for _ in range(radius):
                start_point = (start_point[0] + direction[0], start_point[1] + direction[1])
                if 0 <= start_point[0] < size and 0 <= start_point[1] < size:
                    yield start_point
        if radius == size - 1:
            break

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