from PIL import Image
import numpy as np
from ..core.pixel_attributes import PixelAttributes

def pixel_sorting(image, sort_mode, chunk_size, sort_by, starting_corner=None, sort_order='ascending'):
    """
    Sort pixels in chunks based on various attributes.
    
    Args:
        image (Image): Input PIL image
        sort_mode (str): Direction of sorting - 'horizontal', 'vertical', or 'diagonal'
        chunk_size (str): Size of chunks in format 'WIDTHxHEIGHT'
        sort_by (str): Attribute to sort by - 'brightness', 'hue', 'saturation', etc.
        starting_corner (str, optional): For diagonal sorting - corner to start from
        sort_order (str, optional): 'ascending' or 'descending'
    
    Returns:
        Image: Processed image with sorted pixels
    """
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Handle special case for diagonal sorting
    if sort_mode == 'diagonal':
        if not starting_corner:
            raise ValueError("starting_corner is required for diagonal sorting")
        # Determine if horizontal is True based on starting corner
        horizontal = starting_corner in ['top-left', 'bottom-left']
        # Pass the sort order parameter to the diagonal sorting function
        reverse = (sort_order == 'descending')
        return pixel_sorting_corner_to_corner(image, chunk_size, sort_by, starting_corner, horizontal, sort_order)
    
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
    }.get(sort_by, PixelAttributes.color_sum)  # Default to sum of RGB if invalid

    pixels = list(image.getdata())
    width, height = image.size
    chunk_width, chunk_height = map(int, chunk_size.split('x'))
    
    # Create result image and get pixel access object for direct pixel manipulation
    result_image = Image.new(image.mode, image.size)
    
    # Calculate how many full chunks we can fit and the remainder
    num_chunks_x = width // chunk_width
    num_chunks_y = height // chunk_height
    remainder_x = width % chunk_width
    remainder_y = height % chunk_height
    
    # Determine whether to reverse the sort based on sort_order
    reverse = (sort_order == 'descending')
    
    # Process standard chunks (full sized)
    for chunk_row in range(num_chunks_y):
        for chunk_col in range(num_chunks_x):
            # Extract chunk pixels
            chunk_pixels = []
            start_y = chunk_row * chunk_height
            start_x = chunk_col * chunk_width
            
            for y in range(start_y, start_y + chunk_height):
                for x in range(start_x, start_x + chunk_width):
                    chunk_pixels.append(pixels[y * width + x])
            
            # Sort chunk with the appropriate order
            if sort_mode == 'horizontal':
                sorted_chunk = sorted(chunk_pixels, key=sort_function, reverse=reverse)
                
                # Place sorted pixels back in horizontal order
                for i, pixel in enumerate(sorted_chunk):
                    x = start_x + (i % chunk_width)
                    y = start_y + (i // chunk_width)
                    result_image.putpixel((x, y), pixel)
            else:  # vertical
                # Reshape the chunk for column-wise sorting
                chunk_2d = []
                for i in range(0, len(chunk_pixels), chunk_width):
                    chunk_2d.append(chunk_pixels[i:i + chunk_width])
                
                # For each column in the chunk
                for x in range(chunk_width):
                    # Extract column
                    column = [chunk_2d[y][x] for y in range(chunk_height)]
                    # Sort column with the appropriate order
                    sorted_column = sorted(column, key=sort_function, reverse=reverse)
                    
                    # Place sorted column back
                    for y, pixel in enumerate(sorted_column):
                        result_image.putpixel((start_x + x, start_y + y), pixel)

    # Process right edge (if there's a remainder in width)
    if remainder_x > 0:
        for chunk_row in range(num_chunks_y):
            # Extract the right edge chunk
            start_y = chunk_row * chunk_height
            start_x = num_chunks_x * chunk_width
            
            edge_chunk = []
            for y in range(start_y, start_y + chunk_height):
                for x in range(start_x, start_x + remainder_x):
                    edge_chunk.append(pixels[y * width + x])
            
            # Sort the edge chunk with the appropriate order
            if sort_mode == 'horizontal':
                sorted_edge = sorted(edge_chunk, key=sort_function, reverse=reverse)
                
                # Place sorted pixels back
                for i, pixel in enumerate(sorted_edge):
                    x = start_x + (i % remainder_x)
                    y = start_y + (i // remainder_x)
                    result_image.putpixel((x, y), pixel)
            else:  # vertical
                # Reshape for column-wise sorting
                edge_2d = []
                for i in range(0, len(edge_chunk), remainder_x):
                    edge_2d.append(edge_chunk[i:i + remainder_x])
                
                # For each column in the edge chunk
                for x in range(remainder_x):
                    # Extract column
                    column = [row[x] for row in edge_2d if x < len(row)]
                    # Sort column with the appropriate order
                    sorted_column = sorted(column, key=sort_function, reverse=reverse)
                    
                    # Place sorted column back
                    for y, pixel in enumerate(sorted_column):
                        result_image.putpixel((start_x + x, start_y + y), pixel)
    
    # Process bottom edge (if there's a remainder in height)
    if remainder_y > 0:
        for chunk_col in range(num_chunks_x):
            # Extract the bottom edge chunk
            start_y = num_chunks_y * chunk_height
            start_x = chunk_col * chunk_width
            
            edge_chunk = []
            for y in range(start_y, start_y + remainder_y):
                for x in range(start_x, start_x + chunk_width):
                    edge_chunk.append(pixels[y * width + x])
            
            # Sort the edge chunk with the appropriate order
            if sort_mode == 'horizontal':
                sorted_edge = sorted(edge_chunk, key=sort_function, reverse=reverse)
                
                # Place sorted pixels back
                for i, pixel in enumerate(sorted_edge):
                    x = start_x + (i % chunk_width)
                    y = start_y + (i // chunk_width)
                    result_image.putpixel((x, y), pixel)
            else:  # vertical
                # Reshape for column-wise sorting
                edge_2d = []
                for i in range(0, len(edge_chunk), chunk_width):
                    edge_2d.append(edge_chunk[i:i + chunk_width])
                
                # For each column in the edge chunk
                for x in range(chunk_width):
                    # Extract column
                    column = [row[x] for row in edge_2d if x < len(row)]
                    # Sort column with the appropriate order
                    sorted_column = sorted(column, key=sort_function, reverse=reverse)
                    
                    # Place sorted column back
                    for y, pixel in enumerate(sorted_column):
                        result_image.putpixel((start_x + x, start_y + y), pixel)
    
    # Process bottom-right corner (if there's a remainder in both width and height)
    if remainder_x > 0 and remainder_y > 0:
        # Extract the corner chunk
        start_y = num_chunks_y * chunk_height
        start_x = num_chunks_x * chunk_width
        
        corner_chunk = []
        for y in range(start_y, start_y + remainder_y):
            for x in range(start_x, start_x + remainder_x):
                corner_chunk.append(pixels[y * width + x])
        
        # Sort the corner chunk with the appropriate order
        if sort_mode == 'horizontal':
            sorted_corner = sorted(corner_chunk, key=sort_function, reverse=reverse)
            
            # Place sorted pixels back
            for i, pixel in enumerate(sorted_corner):
                x = start_x + (i % remainder_x)
                y = start_y + (i // remainder_x)
                result_image.putpixel((x, y), pixel)
        else:  # vertical
            # Reshape for column-wise sorting
            corner_2d = []
            for i in range(0, len(corner_chunk), remainder_x):
                corner_2d.append(corner_chunk[i:i + remainder_x])
            
            # For each column in the corner chunk
            for x in range(remainder_x):
                # Extract column
                column = [row[x] for row in corner_2d if x < len(row)]
                # Sort column with the appropriate order
                sorted_column = sorted(column, key=sort_function, reverse=reverse)
                
                # Place sorted column back
                for y, pixel in enumerate(sorted_column):
                    result_image.putpixel((start_x + x, start_y + y), pixel)
    
    return result_image

def pixel_sorting_corner_to_corner(image, chunk_size, sort_by, corner, horizontal, sort_order='ascending'):
    """
    Apply pixel sorting starting from a specified corner, either horizontally or vertically.
    
    Args:
        image (Image): PIL Image object to process.
        chunk_size (str): Chunk dimensions as 'widthxheight' (e.g., '32x32').
        sort_by (str): Property to sort by ('color', 'brightness', 'hue', 'red', 'green', 'blue',
                       'saturation', 'luminance', 'contrast').
        corner (str): Starting corner ('top-left', 'top-right', 'bottom-left', 'bottom-right').
        horizontal (bool): True for horizontal sorting, False for vertical.
        sort_order (str): 'ascending' (low to high) or 'descending' (high to low). Default is 'ascending'.
    
    Returns:
        Image: Processed image with corner-to-corner sorting.
    """
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
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
    }.get(sort_by, PixelAttributes.color_sum)

    # Create a result image to work with
    result_image = Image.new(image.mode, image.size)
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
    
    # Initialize result_image with a copy of the original image data
    # This ensures no black regions if any pixels are missed
    for y in range(height):
        for x in range(width):
            result_image.putpixel((x, y), pixels_2d[y][x])
    
    # Calculate how many full chunks and the remainder
    full_chunks_x = width // chunk_width
    full_chunks_y = height // chunk_height
    remainder_x = width % chunk_width
    remainder_y = height % chunk_height
    
    # Create step values based on the corner
    x_step = chunk_width
    y_step = chunk_height
    
    # Determine chunk processing start points and ranges based on corner
    if corner == 'top-left':
        x_start_val = 0
        y_start_val = 0
        x_end_val = width
        y_end_val = height
        x_step = chunk_width
        y_step = chunk_height
    elif corner == 'top-right':
        x_start_val = width - chunk_width
        y_start_val = 0
        x_end_val = -chunk_width
        y_end_val = height
        x_step = -chunk_width
        y_step = chunk_height
    elif corner == 'bottom-left':
        x_start_val = 0
        y_start_val = height - chunk_height
        x_end_val = width
        y_end_val = -chunk_height
        x_step = chunk_width
        y_step = -chunk_height
    elif corner == 'bottom-right':
        x_start_val = width - chunk_width
        y_start_val = height - chunk_height
        x_end_val = -chunk_width
        y_end_val = -chunk_height
        x_step = -chunk_width
        y_step = -chunk_height
    
    # Create ranges based on the corner - ensure we include all pixels
    x_range = list(range(x_start_val, x_end_val, x_step))
    y_range = list(range(y_start_val, y_end_val, y_step))
    
    # Handle edge cases where ranges might be empty
    if not x_range and x_step > 0:
        x_range = [0]
    elif not x_range and x_step < 0:
        x_range = [width - chunk_width]
        
    if not y_range and y_step > 0:
        y_range = [0]
    elif not y_range and y_step < 0:
        y_range = [height - chunk_height]
    
    # Determine whether to reverse the sort based on sort_order
    reverse = (sort_order == 'descending')
    
    # Process each chunk
    for y_start in y_range:
        for x_start in x_range:
            # Calculate chunk boundaries
            y_end = min(y_start + abs(y_step), height) if y_step > 0 else max(y_start + y_step, 0)
            x_end = min(x_start + abs(x_step), width) if x_step > 0 else max(x_start + x_step, 0)
            
            # Make sure we're not going out of bounds
            if x_start < 0: x_start = 0
            if y_start < 0: y_start = 0
            if x_end > width: x_end = width
            if y_end > height: y_end = height
            if x_end <= 0: x_end = 1
            if y_end <= 0: y_end = 1
            
            # Ensure chunks always process from lower to higher indices
            x_start, x_end = min(x_start, x_end), max(x_start, x_end)
            y_start, y_end = min(y_start, y_end), max(y_start, y_end)
            
            # Skip empty chunks
            if x_start == x_end or y_start == y_end:
                continue
            
            # Extract all pixels in this chunk
            chunk_pixels = []
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    chunk_pixels.append(pixels_2d[y][x])
            
            # Sort the chunk pixels
            sorted_pixels = sorted(chunk_pixels, key=sort_function, reverse=reverse)
            
            # Reverse the order if needed based on corner
            if (corner in ['bottom-left', 'bottom-right'] and horizontal) or \
               (corner in ['top-right', 'bottom-right'] and not horizontal):
                sorted_pixels = sorted_pixels[::-1]
            
            # Put the sorted pixels back
            pixel_index = 0
            if horizontal:
                # Sort horizontally within chunk
                for y in range(y_start, y_end):
                    for x in range(x_start, x_end):
                        if pixel_index < len(sorted_pixels):
                            result_image.putpixel((x, y), sorted_pixels[pixel_index])
                            pixel_index += 1
            else:
                # Sort vertically within chunk
                for x in range(x_start, x_end):
                    for y in range(y_start, y_end):
                        if pixel_index < len(sorted_pixels):
                            result_image.putpixel((x, y), sorted_pixels[pixel_index])
                            pixel_index += 1
    
    return result_image

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
        'color': PixelAttributes.color_sum,
        'brightness': PixelAttributes.brightness,
        'hue': PixelAttributes.hue,
        'red': lambda p: p[0],     # Sort by red channel only
        'green': lambda p: p[1],   # Sort by green channel only
        'blue': lambda p: p[2],     # Sort by blue channel only
        'saturation': PixelAttributes.saturation,  # Sort by color saturation
        'luminance': PixelAttributes.luminance,    # Sort by luminance (value in HSV)
        'contrast': PixelAttributes.contrast       # Sort by contrast (max-min RGB)
    }.get(sort_by, PixelAttributes.brightness)
    
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
        luminance = np.apply_along_axis(PixelAttributes.brightness, 1, flattened_chunk)
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
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert PIL Image to NumPy array
    img_array = np.array(image)
    
    # Get image dimensions
    height, width, channels = img_array.shape
    
    # Calculate padding needed to make dimensions multiples of chunk_size
    pad_y = (chunk_size - (height % chunk_size)) % chunk_size
    pad_x = (chunk_size - (width % chunk_size)) % chunk_size
    
    # Pad the image if necessary using edge padding (better for visual continuity)
    if pad_y != 0 or pad_x != 0:
        img_array = np.pad(img_array, ((0, pad_y), (0, pad_x), (0, 0)), mode='edge')
        padded_height, padded_width = img_array.shape[:2]
    else:
        padded_height, padded_width = height, width
    
    # Map sort_by parameter to the appropriate function
    sort_function = {
        'color': PixelAttributes.color_sum,
        'brightness': PixelAttributes.brightness,
        'hue': PixelAttributes.hue,
        'red': lambda p: p[0],
        'green': lambda p: p[1],
        'blue': lambda p: p[2],
        'saturation': PixelAttributes.saturation,
        'luminance': PixelAttributes.luminance,
        'contrast': PixelAttributes.contrast
    }.get(sort_by, PixelAttributes.brightness)  # Default to brightness if invalid choice
    
    # Calculate number of chunks
    num_chunks_y = padded_height // chunk_size
    num_chunks_x = padded_width // chunk_size
    
    # Generate spiral coordinates once
    spiral_coords_list = list(spiral_coords(chunk_size))
    total_pixels = chunk_size * chunk_size
    
    # Sort each chunk
    sorted_chunks = []
    for y in range(num_chunks_y):
        for x in range(num_chunks_x):
            # Extract chunk
            chunk = img_array[y*chunk_size:(y+1)*chunk_size, x*chunk_size:(x+1)*chunk_size]
            
            # Flatten the chunk and calculate sort values
            flattened_chunk = chunk.reshape(-1, channels)
            sort_values = np.array([sort_function(p) for p in flattened_chunk])
            
            # Sort pixels based on the sort values
            sorted_indices = np.argsort(sort_values)
            if reverse:
                sorted_indices = sorted_indices[::-1]
            
            # Create a new chunk with pixels arranged in a spiral
            sorted_chunk = np.zeros_like(chunk)
            
            # Place sorted pixels in spiral order
            for idx, (row, col) in zip(sorted_indices[:total_pixels], spiral_coords_list):
                sorted_chunk[row, col] = flattened_chunk[idx]
            
            sorted_chunks.append(sorted_chunk)
    
    # Recombine chunks into the final image
    result = np.zeros((padded_height, padded_width, channels), dtype=img_array.dtype)
    chunk_idx = 0
    for y in range(num_chunks_y):
        for x in range(num_chunks_x):
            result[y*chunk_size:(y+1)*chunk_size, x*chunk_size:(x+1)*chunk_size] = sorted_chunks[chunk_idx]
            chunk_idx += 1
    
    # Crop to original size if padded
    if pad_y != 0 or pad_x != 0:
        result = result[:height, :width]
    
    # Convert back to PIL image
    return Image.fromarray(result)

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

def diagonal_pixel_sort(image, chunk_size, sort_by, corner, sort_order='ascending'):
    """
    Apply pixel sorting diagonally within each chunk, starting from a specified corner.
    
    Args:
        image (Image): PIL Image object to process.
        chunk_size (str): Chunk dimensions as 'widthxheight' (e.g., '32x32').
        sort_by (str): Property to sort by ('color', 'brightness', 'hue', etc.).
        corner (str): Starting corner ('top-left', 'top-right', 'bottom-left', 'bottom-right').
        sort_order (str): 'ascending' (low to high) or 'descending' (high to low). Default is 'ascending'.
    
    Returns:
        Image: Processed image with diagonally sorted pixels.
    """
    # Convert to RGB mode if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Define sort function based on sort_by parameter
    sort_function = {
        'color': PixelAttributes.color_sum,
        'brightness': PixelAttributes.brightness,
        'hue': PixelAttributes.hue,
        'red': lambda p: p[0],
        'green': lambda p: p[1],
        'blue': lambda p: p[2],
        'saturation': PixelAttributes.saturation,
        'luminance': PixelAttributes.luminance,
        'contrast': PixelAttributes.contrast
    }.get(sort_by, PixelAttributes.color_sum)
    
    # Create a new image for the result
    result_image = Image.new(image.mode, image.size)
    width, height = image.size
    chunk_width, chunk_height = map(int, chunk_size.split('x'))
    
    # Get pixel data
    pixels = list(image.getdata())
    pixels_2d = []
    for y in range(height):
        row = []
        for x in range(width):
            row.append(pixels[y * width + x])
        pixels_2d.append(row)
    
    # Calculate chunks
    num_chunks_x = width // chunk_width
    num_chunks_y = height // chunk_height
    remainder_x = width % chunk_width
    remainder_y = height % chunk_height
    
    # Determine whether to reverse the sort based on sort_order
    reverse = (sort_order == 'descending')
    
    # Process each chunk
    for chunk_y in range(num_chunks_y):
        for chunk_x in range(num_chunks_x):
            start_x = chunk_x * chunk_width
            start_y = chunk_y * chunk_height
            
            # Process diagonals
            if corner == 'top-left':
                # Top-left to bottom-right diagonals
                for diag in range(chunk_width + chunk_height - 1):
                    diagonal_pixels = []
                    
                    # Collect pixels along this diagonal
                    for i in range(diag + 1):
                        x = start_x + i
                        y = start_y + diag - i
                        
                        if (x < start_x + chunk_width and 
                            y < start_y + chunk_height and 
                            y >= start_y and x >= start_x):
                            diagonal_pixels.append(pixels_2d[y][x])
                    
                    # Sort diagonal pixels
                    sorted_pixels = sorted(diagonal_pixels, key=sort_function, reverse=reverse)
                    
                    # Place sorted pixels back on the diagonal
                    pixel_index = 0
                    for i in range(diag + 1):
                        x = start_x + i
                        y = start_y + diag - i
                        
                        if (x < start_x + chunk_width and 
                            y < start_y + chunk_height and 
                            y >= start_y and x >= start_x):
                            if pixel_index < len(sorted_pixels):
                                result_image.putpixel((x, y), sorted_pixels[pixel_index])
                                pixel_index += 1
            
            elif corner == 'top-right':
                # Top-right to bottom-left diagonals
                for diag in range(chunk_width + chunk_height - 1):
                    diagonal_pixels = []
                    
                    # Collect pixels along this diagonal
                    for i in range(diag + 1):
                        x = start_x + chunk_width - 1 - i
                        y = start_y + diag - i
                        
                        if (x >= start_x and 
                            y < start_y + chunk_height and 
                            y >= start_y and x < start_x + chunk_width):
                            diagonal_pixels.append(pixels_2d[y][x])
                    
                    # Sort diagonal pixels
                    sorted_pixels = sorted(diagonal_pixels, key=sort_function, reverse=reverse)
                    
                    # Place sorted pixels back on the diagonal
                    pixel_index = 0
                    for i in range(diag + 1):
                        x = start_x + chunk_width - 1 - i
                        y = start_y + diag - i
                        
                        if (x >= start_x and 
                            y < start_y + chunk_height and 
                            y >= start_y and x < start_x + chunk_width):
                            if pixel_index < len(sorted_pixels):
                                result_image.putpixel((x, y), sorted_pixels[pixel_index])
                                pixel_index += 1
            
            elif corner == 'bottom-left':
                # Bottom-left to top-right diagonals
                for diag in range(chunk_width + chunk_height - 1):
                    diagonal_pixels = []
                    
                    # Collect pixels along this diagonal
                    for i in range(diag + 1):
                        x = start_x + i
                        y = start_y + chunk_height - 1 - (diag - i)
                        
                        if (x < start_x + chunk_width and 
                            y >= start_y and 
                            y < start_y + chunk_height and x >= start_x):
                            diagonal_pixels.append(pixels_2d[y][x])
                    
                    # Sort diagonal pixels
                    sorted_pixels = sorted(diagonal_pixels, key=sort_function, reverse=reverse)
                    
                    # Place sorted pixels back on the diagonal
                    pixel_index = 0
                    for i in range(diag + 1):
                        x = start_x + i
                        y = start_y + chunk_height - 1 - (diag - i)
                        
                        if (x < start_x + chunk_width and 
                            y >= start_y and 
                            y < start_y + chunk_height and x >= start_x):
                            if pixel_index < len(sorted_pixels):
                                result_image.putpixel((x, y), sorted_pixels[pixel_index])
                                pixel_index += 1
            
            else:  # bottom-right
                # Bottom-right to top-left diagonals
                for diag in range(chunk_width + chunk_height - 1):
                    diagonal_pixels = []
                    
                    # Collect pixels along this diagonal
                    for i in range(diag + 1):
                        x = start_x + chunk_width - 1 - i
                        y = start_y + chunk_height - 1 - (diag - i)
                        
                        if (x >= start_x and 
                            y >= start_y and 
                            y < start_y + chunk_height and x < start_x + chunk_width):
                            diagonal_pixels.append(pixels_2d[y][x])
                    
                    # Sort diagonal pixels
                    sorted_pixels = sorted(diagonal_pixels, key=sort_function, reverse=reverse)
                    
                    # Place sorted pixels back on the diagonal
                    pixel_index = 0
                    for i in range(diag + 1):
                        x = start_x + chunk_width - 1 - i
                        y = start_y + chunk_height - 1 - (diag - i)
                        
                        if (x >= start_x and 
                            y >= start_y and 
                            y < start_y + chunk_height and x < start_x + chunk_width):
                            if pixel_index < len(sorted_pixels):
                                result_image.putpixel((x, y), sorted_pixels[pixel_index])
                                pixel_index += 1
    
    # Handle remainder chunks
    # For simplicity, we'll just copy them as-is for now
    # Handle right edge
    if remainder_x > 0:
        for y in range(height):
            for x in range(width - remainder_x, width):
                result_image.putpixel((x, y), pixels_2d[y][x])
                
    # Handle bottom edge
    if remainder_y > 0:
        for y in range(height - remainder_y, height):
            for x in range(width - remainder_x):
                result_image.putpixel((x, y), pixels_2d[y][x])
    
    return result_image 