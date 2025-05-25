from PIL import Image
import numpy as np
from ..core.pixel_attributes import PixelAttributes

# Centralized dictionary for sort functions
_SORT_FUNCTIONS = {
    'color': PixelAttributes.color_sum,
    'brightness': PixelAttributes.brightness,
    'hue': PixelAttributes.hue,
    'red': lambda p: p[0],
    'green': lambda p: p[1],
    'blue': lambda p: p[2],
    'saturation': PixelAttributes.saturation,
    'luminance': PixelAttributes.luminance,
    'contrast': PixelAttributes.contrast
}

# Helper function to process and sort a single chunk using NumPy
def _process_and_sort_chunk_np(img_array_full, start_x, start_y, chunk_width, chunk_height, 
                               sort_mode, sort_function, reverse_sort, result_array_full):
    """Extracts, sorts (using NumPy for data handling), and places back pixels for a single chunk."""
    
    # Extract chunk using NumPy slicing
    # Ensure we don't go out of bounds, though parent function should handle chunk dimensions
    current_chunk_np = img_array_full[start_y : start_y + chunk_height, start_x : start_x + chunk_width]

    if current_chunk_np.size == 0:
        return

    # Get channels for reshaping later if needed
    channels = current_chunk_np.shape[-1] if len(current_chunk_np.shape) == 3 else 1 # Handle grayscale potential
    original_shape = current_chunk_np.shape

    # Convert chunk pixels to a list of tuples for compatibility with sort_function
    # This is the part that bridges NumPy to the existing Python-based sort_key functions
    pixels_list_in_chunk = [tuple(p) for p in current_chunk_np.reshape(-1, channels)]

    if sort_mode == 'horizontal':
        sorted_pixel_tuples = sorted(pixels_list_in_chunk, key=sort_function, reverse=reverse_sort)
        
        # Convert sorted list of tuples back to a NumPy array
        # Ensure dtype matches the original chunk to avoid issues (e.g., float to int)
        sorted_pixels_np_flat = np.array(sorted_pixel_tuples, dtype=current_chunk_np.dtype)
        
        # Reshape to the chunk's original 2D pixel arrangement
        sorted_chunk_final_np = sorted_pixels_np_flat.reshape(original_shape)
        
        result_array_full[start_y : start_y + chunk_height, start_x : start_x + chunk_width] = sorted_chunk_final_np

    elif sort_mode == 'vertical':
        # For vertical sort, we sort each column of the chunk
        # We operate on a copy to build the sorted chunk
        sorted_chunk_final_np = np.copy(current_chunk_np) 

        for col_idx in range(original_shape[1]): # Iterate through columns of the chunk
            column_pixel_tuples = [tuple(p) for p in current_chunk_np[:, col_idx]] # Extract column, convert to list of tuples
            
            if not column_pixel_tuples:
                continue
                
            sorted_column_tuples = sorted(column_pixel_tuples, key=sort_function, reverse=reverse_sort)
            
            # Place sorted column back into the copy
            # Need to convert list of tuples back to NumPy array for assignment
            sorted_column_np = np.array(sorted_column_tuples, dtype=current_chunk_np.dtype)
            sorted_chunk_final_np[:, col_idx] = sorted_column_np
        
        result_array_full[start_y : start_y + chunk_height, start_x : start_x + chunk_width] = sorted_chunk_final_np

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
    if image.mode != 'RGB':
        image = image.convert('RGB')

    if sort_mode == 'diagonal':
        if not starting_corner:
            raise ValueError("starting_corner is required for diagonal sorting")
        horizontal = starting_corner in ['top-left', 'bottom-left']
        return pixel_sorting_corner_to_corner(image, chunk_size, sort_by, starting_corner, horizontal, sort_order)

    sort_function = _SORT_FUNCTIONS.get(sort_by, PixelAttributes.color_sum)
    reverse = (sort_order == 'descending')

    img_array_orig = np.array(image, dtype=np.uint8) # Explicitly use uint8 for image data
    height, width = img_array_orig.shape[:2] # Get height, width, ignore channels for now unless needed
    
    base_chunk_width, base_chunk_height = map(int, chunk_size.split('x'))

    # Create result array, initialized as a copy or zeros, then fill
    # Using a copy ensures un-processed areas (if any logic error) retain original pixels
    result_array = np.copy(img_array_orig)

    num_chunks_y = (height + base_chunk_height - 1) // base_chunk_height
    num_chunks_x = (width + base_chunk_width - 1) // base_chunk_width

    for chunk_row_idx in range(num_chunks_y):
        for chunk_col_idx in range(num_chunks_x):
            start_y = chunk_row_idx * base_chunk_height
            start_x = chunk_col_idx * base_chunk_width

            current_chunk_height = min(base_chunk_height, height - start_y)
            current_chunk_width = min(base_chunk_width, width - start_x)

            if current_chunk_width <= 0 or current_chunk_height <= 0:
                continue
            
            _process_and_sort_chunk_np(img_array_orig, start_x, start_y, 
                                       current_chunk_width, current_chunk_height, 
                                       sort_mode, sort_function, reverse, result_array)
    
    return Image.fromarray(result_array)

def pixel_sorting_corner_to_corner(image, chunk_size_str, sort_by, corner, horizontal, sort_order='ascending'):
    """
    Apply pixel sorting starting from a specified corner, either horizontally or vertically.
    
    Args:
        image (Image): PIL Image object to process.
        chunk_size_str (str): Chunk dimensions as 'widthxheight' (e.g., '32x32').
        sort_by (str): Property to sort by ('color', 'brightness', 'hue', 'red', 'green', 'blue',
                       'saturation', 'luminance', 'contrast').
        corner (str): Starting corner ('top-left', 'top-right', 'bottom-left', 'bottom-right').
        horizontal (bool): True for horizontal sorting (pixels within chunk are laid out row-wise after sort),
                         False for vertical (pixels within chunk are laid out column-wise after sort).
        sort_order (str): 'ascending' (low to high) or 'descending' (high to low). Default is 'ascending'.
    
    Returns:
        Image: Processed image with corner-to-corner sorting.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    sort_function = _SORT_FUNCTIONS.get(sort_by, PixelAttributes.color_sum)
    reverse_sort_order = (sort_order == 'descending')

    img_array_full = np.array(image, dtype=np.uint8)
    img_height, img_width, img_channels = img_array_full.shape
    result_array_full = np.copy(img_array_full)
    
    base_chunk_width, base_chunk_height = map(int, chunk_size_str.split('x'))

    # Determine iteration ranges and steps based on the corner
    # These define the starting corner of chunks
    if corner == 'top-left':
        y_coords_iter = range(0, img_height, base_chunk_height)
        x_coords_iter = range(0, img_width, base_chunk_width)
    elif corner == 'top-right':
        y_coords_iter = range(0, img_height, base_chunk_height)
        x_coords_iter = range(img_width - base_chunk_width, -1, -base_chunk_width) # Iterate right to left
    elif corner == 'bottom-left':
        y_coords_iter = range(img_height - base_chunk_height, -1, -base_chunk_height) # Iterate bottom to top
        x_coords_iter = range(0, img_width, base_chunk_width)
    elif corner == 'bottom-right':
        y_coords_iter = range(img_height - base_chunk_height, -1, -base_chunk_height) # Bottom to top
        x_coords_iter = range(img_width - base_chunk_width, -1, -base_chunk_width) # Right to left
    else:
        raise ValueError(f"Invalid corner: {corner}")

    for y_start in y_coords_iter:
        for x_start in x_coords_iter:
            # Determine actual chunk dimensions, handling edges
            current_chunk_height = min(base_chunk_height, img_height - y_start if y_start >= 0 else base_chunk_height + y_start)
            current_chunk_width = min(base_chunk_width, img_width - x_start if x_start >=0 else base_chunk_width + x_start)
            
            # Adjust for negative start indices if iterating backwards
            actual_y_start = max(0, y_start)
            actual_x_start = max(0, x_start)
            current_chunk_height = min(base_chunk_height, img_height - actual_y_start)
            current_chunk_width = min(base_chunk_width, img_width - actual_x_start)

            if current_chunk_width <= 0 or current_chunk_height <= 0:
                continue

            current_chunk_np = img_array_full[actual_y_start : actual_y_start + current_chunk_height, 
                                              actual_x_start : actual_x_start + current_chunk_width]
            
            if current_chunk_np.size == 0:
                continue

            chunk_original_shape = current_chunk_np.shape
            pixel_tuples = [tuple(p) for p in current_chunk_np.reshape(-1, img_channels)]
            
            sorted_pixel_tuples = sorted(pixel_tuples, key=sort_function, reverse=reverse_sort_order)
            
            # Conditional reversal based on corner and primary sorting direction (horizontal flag)
            if (corner in ['bottom-left', 'bottom-right'] and horizontal) or \
               (corner in ['top-right', 'bottom-right'] and not horizontal):
                # This logic might need review: it reverses the flat list of sorted pixels.
                # The original intent was related to how pixels were put back. With reshape, this needs care.
                sorted_pixel_tuples = sorted_pixel_tuples[::-1]

            sorted_pixels_np_flat = np.array(sorted_pixel_tuples, dtype=img_array_full.dtype)
            
            # Reshape and place back
            # The 'horizontal' flag determines how the 1D sorted list populates the 2D chunk space
            sorted_chunk_final_np = np.zeros(chunk_original_shape, dtype=img_array_full.dtype)
            if horizontal: # Pixels are laid out row by row (standard reshape)
                sorted_chunk_final_np = sorted_pixels_np_flat.reshape(chunk_original_shape)
            else: # Pixels are laid out column by column
                # To fill column by column, we can reshape to (width, height, channels) then transpose
                if chunk_original_shape[0] > 0 and chunk_original_shape[1] > 0: # Ensure non-empty chunk
                    temp_reshaped = sorted_pixels_np_flat.reshape(chunk_original_shape[1], chunk_original_shape[0], img_channels) # (width, height, channels)
                    sorted_chunk_final_np = temp_reshaped.transpose(1, 0, 2) # (height, width, channels)
                else:
                    sorted_chunk_final_np = sorted_pixels_np_flat.reshape(chunk_original_shape) # Fallback for empty dim

            result_array_full[actual_y_start : actual_y_start + current_chunk_height, 
                              actual_x_start : actual_x_start + current_chunk_width] = sorted_chunk_final_np
    
    return Image.fromarray(result_array_full)

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
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_array = np.array(image, dtype=np.uint8)
    height, width = img_array.shape[:2]

    result_array = np.copy(img_array) # Work on a copy for the result
    
    sort_function = _SORT_FUNCTIONS.get(sort_by, PixelAttributes.brightness) # Default for this func

    if direction == 'vertical':
        for x_col in range(width):
            # Extract column data as a NumPy array slice
            column_data_np = img_array[:, x_col] 
            # Convert NumPy 2D slice (height, channels) to list of pixel tuples for sort_function
            pixel_tuples = [tuple(p) for p in column_data_np]
            
            # Sort the list of tuples
            sorted_pixel_tuples = sorted(pixel_tuples, key=sort_function, reverse=reverse)
            
            # Convert sorted list of tuples back to a NumPy array
            sorted_column_np = np.array(sorted_pixel_tuples, dtype=img_array.dtype)
            
            # Place the sorted column back into the result array
            result_array[:, x_col] = sorted_column_np
    
    elif direction == 'horizontal':
        for y_row in range(height):
            # Extract row data as a NumPy array slice
            row_data_np = img_array[y_row, :] 
            # Convert NumPy 2D slice (width, channels) to list of pixel tuples for sort_function
            pixel_tuples = [tuple(p) for p in row_data_np]
            
            # Sort the list of tuples
            sorted_pixel_tuples = sorted(pixel_tuples, key=sort_function, reverse=reverse)
            
            # Convert sorted list of tuples back to a NumPy array
            sorted_row_np = np.array(sorted_pixel_tuples, dtype=img_array.dtype)
            
            # Place the sorted row back into the result array
            result_array[y_row, :] = sorted_row_np
    
    return Image.fromarray(result_array)

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
    
    # Get the sort function from the centralized dictionary
    sort_function = _SORT_FUNCTIONS.get(sort_by, PixelAttributes.color_sum)

    # Calculate number of chunks
    num_chunks_y = padded_height // chunk_size
    num_chunks_x = padded_width // chunk_size
    
    # Generate spiral coordinates once
    spiral_coords_list = list(spiral_coords(chunk_size))
    total_pixels = chunk_size * chunk_size
    
    # Sort each chunk
    sorted_chunks = []
    for y_idx_chunk in range(num_chunks_y):
        for x_idx_chunk in range(num_chunks_x):
            # Extract chunk
            chunk = img_array[y_idx_chunk*chunk_size:(y_idx_chunk+1)*chunk_size, x_idx_chunk*chunk_size:(x_idx_chunk+1)*chunk_size]
            
            # Flatten the chunk 
            flattened_chunk_np = chunk.reshape(-1, channels)
            
            # Convert to list of tuples for sort_function compatibility
            pixel_tuples = [tuple(p) for p in flattened_chunk_np]
            # Calculate sort values using the list of tuples
            sort_values = np.array([sort_function(t) for t in pixel_tuples])
            
            # Sort pixels based on the sort values
            sorted_indices = np.argsort(sort_values)
            if reverse:
                sorted_indices = sorted_indices[::-1]
            
            # Create a new chunk with pixels arranged in a spiral
            sorted_chunk = np.zeros_like(chunk)
            
            # Place sorted pixels in spiral order
            for idx, (row, col) in zip(sorted_indices[:total_pixels], spiral_coords_list):
                sorted_chunk[row, col] = flattened_chunk_np[idx]
            
            sorted_chunks.append(sorted_chunk)
    
    # Recombine chunks into the final image
    result = np.zeros((padded_height, padded_width, channels), dtype=img_array.dtype)
    chunk_idx = 0
    for y_idx_chunk in range(num_chunks_y):
        for x_idx_chunk in range(num_chunks_x):
            result[y_idx_chunk*chunk_size:(y_idx_chunk+1)*chunk_size, x_idx_chunk*chunk_size:(x_idx_chunk+1)*chunk_size] = sorted_chunks[chunk_idx]
            chunk_idx += 1
    
    # Crop to original size if padded
    if pad_y != 0 or pad_x != 0:
        result = result[:height, :width]
    
    # Convert back to PIL image
    return Image.fromarray(result)

def wrapped_sort(image, chunk_width, chunk_height, starting_corner='top-left', flow_direction='primary', sort_direction='vertical', sort_by='brightness', reverse=False, direction=None):
    """
    Apply wrapped pixel sorting where chunks are placed column by column, and partial chunks
    at the bottom of each column cause the next column to start with an offset, creating
    a staggered "wrapped" pattern.
    
    Args:
        image (Image): PIL Image object to process
        chunk_width (int): Width of each column of chunks
        chunk_height (int): Target height for each chunk (creates wrapping when doesn't fit evenly)
        starting_corner (str): Corner to start chunk placement from
        flow_direction (str): How chunks flow ('primary' or 'secondary' direction from corner)
        sort_direction (str): Sort direction within chunks ('vertical' or 'horizontal')
        sort_by (str): Attribute to sort by ('brightness', 'hue', 'saturation', etc.)
        reverse (bool): Whether to reverse the sort order
    
    Returns:
        Image: Processed image with wrapped chunk sorting applied
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Backward compatibility: if old 'direction' parameter is used, map it to flow_direction
    if direction is not None:
        flow_direction = 'primary' if direction == 'vertical' else 'secondary'
    
    sort_function = _SORT_FUNCTIONS.get(sort_by, PixelAttributes.brightness)
    
    img_array = np.array(image, dtype=np.uint8)
    img_height, img_width = img_array.shape[:2]
    result_array = np.copy(img_array)
    
    def generate_chunks(img_w, img_h, chunk_w, chunk_h, start_corner, flow_dir):
        """Generate chunks using the wrapped algorithm - partial chunks continue across columns."""
        chunks = []
        
        # For simplicity, start with top-left primary (can extend later)
        # The key is: when a column ends with a partial chunk, that chunk continues in the next column
        
        x = 0
        current_chunk_remaining = chunk_h  # How much height is left in the current chunk
        
        while x < img_w:
            y = 0
            col_width = min(chunk_w, img_w - x)
            
            while y < img_h:
                remaining_h = img_h - y
                
                # Use the smaller of: remaining height in image, or remaining height needed for current chunk
                curr_h = min(current_chunk_remaining, remaining_h)
                
                chunks.append((x, y, col_width, curr_h))
                y += curr_h
                current_chunk_remaining -= curr_h
                
                # If we completed a full chunk, start a new one
                if current_chunk_remaining == 0:
                    current_chunk_remaining = chunk_h
                
                # If we hit the bottom of the image, move to next column
                # The current_chunk_remaining carries over (this is the key!)
                if y >= img_h:
                    break
            
            x += chunk_w
        
        return chunks
    
    # Generate the wrapped chunks
    chunks = generate_chunks(img_width, img_height, chunk_width, chunk_height, starting_corner, flow_direction)
    
    # Process each chunk
    for chunk_x, chunk_y, chunk_w, chunk_h in chunks:
        # Collect pixels for this chunk
        chunk_pixels = []
        chunk_positions = []
        
        for dy in range(chunk_h):
            for dx in range(chunk_w):
                y = chunk_y + dy
                x = chunk_x + dx
                if 0 <= y < img_height and 0 <= x < img_width:
                    chunk_positions.append((y, x))
                    chunk_pixels.append(tuple(img_array[y, x]))
        
        # Sort the pixels in this chunk
        if chunk_pixels:
            sorted_pixels = sorted(chunk_pixels, key=sort_function, reverse=reverse)
            
            # Place sorted pixels back in the chunk area
            if sort_direction == 'vertical':
                # Sort vertically within the chunk (column by column)
                idx = 0
                for col in range(chunk_w):
                    for row in range(chunk_h):
                        if idx < len(sorted_pixels):
                            y = chunk_y + row
                            x = chunk_x + col
                            if 0 <= y < img_height and 0 <= x < img_width:
                                result_array[y, x] = sorted_pixels[idx]
                                idx += 1
            else:  # horizontal
                # Sort horizontally within the chunk (row by row)
                idx = 0
                for row in range(chunk_h):
                    for col in range(chunk_w):
                        if idx < len(sorted_pixels):
                            y = chunk_y + row
                            x = chunk_x + col
                            if 0 <= y < img_height and 0 <= x < img_width:
                                result_array[y, x] = sorted_pixels[idx]
                                idx += 1
    
    return Image.fromarray(result_array)


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
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    img_array_full = np.array(image, dtype=np.uint8)
    height, width, channels = img_array_full.shape
    result_array_full = np.copy(img_array_full)

    for y_start in range(0, height, chunk_size):
        for x_start in range(0, width, chunk_size):
            actual_height = min(chunk_size, height - y_start)
            actual_width = min(chunk_size, width - x_start)

            if actual_width <= 0 or actual_height <= 0:
                continue

            current_chunk_np = img_array_full[y_start : y_start + actual_height, x_start : x_start + actual_width]
            
            ch_height, ch_width = current_chunk_np.shape[:2]
            
            # Calculate polar coordinates relative to chunk center
            cy, cx = ch_height // 2, ch_width // 2
            y_coords, x_coords = np.mgrid[:ch_height, :ch_width]
            
            angles = np.arctan2(y_coords - cy, x_coords - cx)
            radii = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
            
            # Flatten data for sorting
            flat_pixels = current_chunk_np.reshape(-1, channels)
            flat_angles = angles.flatten()
            flat_radii = radii.flatten()
            
            sort_keys = flat_angles if sort_by == 'angle' else flat_radii
            
            # Get indices that would sort the keys
            sorted_indices = np.argsort(sort_keys)
            if reverse:
                sorted_indices = sorted_indices[::-1]
            
            # Sort the flat pixels based on these indices
            sorted_flat_pixels = flat_pixels[sorted_indices]
            
            # Reshape sorted pixels back to the chunk's original shape
            sorted_chunk_np = sorted_flat_pixels.reshape(current_chunk_np.shape)
            
            # Place the sorted chunk back into the result array
            result_array_full[y_start : y_start + actual_height, x_start : x_start + actual_width] = sorted_chunk_np
    
    return Image.fromarray(result_array_full) 