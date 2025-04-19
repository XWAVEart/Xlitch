from PIL import Image
import numpy as np
import random
from io import BytesIO

def databend_image(image, intensity=0.1, preserve_header=True, seed=None):
    """
    Apply databending to an image by simulating binary data corruption through direct pixel manipulation.
    
    Args:
        image (PIL.Image): PIL Image object to apply databending to.
        intensity (float): Intensity of the effect (0.1 to 1.0).
        preserve_header (bool): Controls whether the top portion of the image is preserved.
        seed (int or None): Random seed for reproducible results.
    
    Returns:
        PIL.Image: The glitched image.
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Make a copy of the original image to work with
    img = image.copy().convert('RGB')
    width, height = img.size
    
    # Create a pixelated version for more stable glitching
    pixels = img.load()
    
    # Determine how much of the image to protect (if preserve_header is True)
    protected_height = int(height * 0.1) if preserve_header else 0
    
    # Apply different glitch effects based on intensity
    # Scale the number of operations based on intensity - new range is 0.1 to 1.0
    # At 0.1 intensity: ~5-10 operations
    # At 1.0 intensity: ~100-150 operations
    num_operations = int(intensity * 150)
    num_operations = max(5, min(150, num_operations))  # Ensure a reasonable range
    
    # Different databending-like effects
    operations = [
        'shift_rows',
        'shift_columns',
        'swap_channels',
        'xor_block',
        'repeat_block',
        'color_shift',
        'invert_block',
        'shift_channels',
        'block_offset',
    ]
    
    # Ensure minimal dimensions for operations
    if width < 10 or height < 10:
        # For very small images, just apply simple color effects
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                if random.random() < intensity:
                    # Apply random color effect
                    effect = random.choice(['invert', 'xor', 'shift'])
                    if effect == 'invert':
                        pixels[x, y] = (255 - r, 255 - g, 255 - b)
                    elif effect == 'xor':
                        xor_val = random.randint(1, 255)
                        pixels[x, y] = (r ^ xor_val, g ^ xor_val, b ^ xor_val)
                    elif effect == 'shift':
                        shift = random.randint(-50, 50)
                        pixels[x, y] = (
                            max(0, min(255, r + shift)),
                            max(0, min(255, g + shift)),
                            max(0, min(255, b + shift))
                        )
        return img
    
    for _ in range(num_operations):
        # Select a random operation
        operation = random.choice(operations)
        
        if operation == 'shift_rows':
            # Shift a chunk of rows horizontally
            # Ensure valid y range
            max_start_y = max(protected_height, height - 10)
            if protected_height >= max_start_y:
                continue  # Skip this operation if there's not enough height
            
            start_y = random.randint(protected_height, max_start_y)
            y_size = random.randint(1, max(1, min(20, height - start_y)))
            shift_amount = random.randint(-width // 2, width // 2)
            
            for y in range(start_y, min(start_y + y_size, height)):
                row = [pixels[x, y] for x in range(width)]
                # Shift the row
                shifted_row = row[-shift_amount:] + row[:-shift_amount] if shift_amount > 0 else row[abs(shift_amount):] + row[:abs(shift_amount)]
                # Write back
                for x in range(width):
                    pixels[x, y] = shifted_row[x]
                    
        elif operation == 'shift_columns':
            # Shift a chunk of columns vertically
            # Ensure valid x range
            max_start_x = max(0, width - 10)
            if max_start_x <= 0:
                continue  # Skip this operation if there's not enough width
                
            start_x = random.randint(0, max_start_x)
            x_size = random.randint(1, max(1, min(20, width - start_x)))
            shift_amount = random.randint(-height // 3, height // 3)
            
            for x in range(start_x, min(start_x + x_size, width)):
                col = [pixels[x, y] for y in range(protected_height, height)]
                # Shift the column
                if not col:  # Skip if column is empty
                    continue
                shifted_col = col[-shift_amount:] + col[:-shift_amount] if shift_amount > 0 else col[abs(shift_amount):] + col[:abs(shift_amount)]
                # Write back
                for i, y in enumerate(range(protected_height, height)):
                    if i < len(shifted_col):
                        pixels[x, y] = shifted_col[i]
                        
        elif operation == 'swap_channels':
            # Swap color channels in a random block
            max_start_x = max(0, width - 10)
            max_start_y = max(protected_height, height - 10)
            
            if max_start_x <= 0 or max_start_y <= protected_height:
                continue  # Skip if we don't have enough space
                
            start_x = random.randint(0, max_start_x)
            start_y = random.randint(protected_height, max_start_y)
            block_width = random.randint(1, max(1, min(100, width - start_x)))
            block_height = random.randint(1, max(1, min(100, height - start_y)))
            swap_type = random.choice(['rb', 'rg', 'gb'])
            
            for y in range(start_y, min(start_y + block_height, height)):
                for x in range(start_x, min(start_x + block_width, width)):
                    r, g, b = pixels[x, y]
                    if swap_type == 'rb':
                        pixels[x, y] = (b, g, r)
                    elif swap_type == 'rg':
                        pixels[x, y] = (g, r, b)
                    elif swap_type == 'gb':
                        pixels[x, y] = (r, b, g)
                        
        elif operation == 'xor_block':
            # Apply XOR to a block
            max_start_x = max(0, width - 10)
            max_start_y = max(protected_height, height - 10)
            
            if max_start_x <= 0 or max_start_y <= protected_height:
                continue  # Skip if we don't have enough space
                
            start_x = random.randint(0, max_start_x)
            start_y = random.randint(protected_height, max_start_y)
            block_width = random.randint(1, max(1, min(100, width - start_x)))
            block_height = random.randint(1, max(1, min(100, height - start_y)))
            xor_value = random.randint(1, 255)
            
            for y in range(start_y, min(start_y + block_height, height)):
                for x in range(start_x, min(start_x + block_width, width)):
                    r, g, b = pixels[x, y]
                    pixels[x, y] = ((r ^ xor_value) & 0xFF, (g ^ xor_value) & 0xFF, (b ^ xor_value) & 0xFF)
                    
        elif operation == 'repeat_block':
            # Repeat a block somewhere else
            max_src_x = max(0, width - 20)
            max_src_y = max(protected_height, height - 20)
            
            if max_src_x <= 0 or max_src_y <= protected_height:
                continue  # Skip if we don't have enough space
                
            src_x = random.randint(0, max_src_x)
            src_y = random.randint(protected_height, max_src_y)
            block_width = random.randint(1, max(1, min(50, width - src_x)))
            block_height = random.randint(1, max(1, min(50, height - src_y)))
            
            # Ensure destination has valid space
            max_dst_x = max(0, width - block_width)
            max_dst_y = max(protected_height, height - block_height)
            
            if max_dst_x < 0 or max_dst_y < protected_height:
                continue  # Skip if destination doesn't have enough space
                
            # Destination
            dst_x = random.randint(0, max_dst_x)
            dst_y = random.randint(protected_height, max_dst_y)
            
            # Copy block
            for y_offset in range(block_height):
                for x_offset in range(block_width):
                    if (src_y + y_offset < height and src_x + x_offset < width and 
                        dst_y + y_offset < height and dst_x + x_offset < width):
                        pixels[dst_x + x_offset, dst_y + y_offset] = pixels[src_x + x_offset, src_y + y_offset]
                        
        elif operation == 'color_shift':
            # Shift color values in a region
            max_start_x = max(0, width - 10)
            max_start_y = max(protected_height, height - 10)
            
            if max_start_x <= 0 or max_start_y <= protected_height:
                continue  # Skip if we don't have enough space
                
            start_x = random.randint(0, max_start_x)
            start_y = random.randint(protected_height, max_start_y)
            block_width = random.randint(1, max(1, min(200, width - start_x)))
            block_height = random.randint(1, max(1, min(200, height - start_y)))
            
            shift_r = random.randint(-50, 50)
            shift_g = random.randint(-50, 50)
            shift_b = random.randint(-50, 50)
            
            for y in range(start_y, min(start_y + block_height, height)):
                for x in range(start_x, min(start_x + block_width, width)):
                    r, g, b = pixels[x, y]
                    pixels[x, y] = (
                        max(0, min(255, r + shift_r)),
                        max(0, min(255, g + shift_g)),
                        max(0, min(255, b + shift_b))
                    )
                    
        elif operation == 'invert_block':
            # Invert colors in a block
            max_start_x = max(0, width - 10)
            max_start_y = max(protected_height, height - 10)
            
            if max_start_x <= 0 or max_start_y <= protected_height:
                continue  # Skip if we don't have enough space
                
            start_x = random.randint(0, max_start_x)
            start_y = random.randint(protected_height, max_start_y)
            block_width = random.randint(1, max(1, min(100, width - start_x)))
            block_height = random.randint(1, max(1, min(100, height - start_y)))
            
            for y in range(start_y, min(start_y + block_height, height)):
                for x in range(start_x, min(start_x + block_width, width)):
                    r, g, b = pixels[x, y]
                    pixels[x, y] = (255 - r, 255 - g, 255 - b)
                    
        elif operation == 'shift_channels':
            # Shift RGB channels independently
            max_start_x = max(0, width - 10)
            max_start_y = max(protected_height, height - 10)
            
            if max_start_x <= 0 or max_start_y <= protected_height:
                continue  # Skip if we don't have enough space
                
            start_x = random.randint(0, max_start_x)
            start_y = random.randint(protected_height, max_start_y)
            block_width = random.randint(1, max(1, min(300, width - start_x)))
            block_height = random.randint(1, max(1, min(100, height - start_y)))
            
            # Skip if block is too small
            if block_width < 4 or block_height < 2:
                continue
                
            # Extract the block for each channel
            r_channel = []
            g_channel = []
            b_channel = []
            
            for y in range(start_y, min(start_y + block_height, height)):
                r_row = []
                g_row = []
                b_row = []
                for x in range(start_x, min(start_x + block_width, width)):
                    r, g, b = pixels[x, y]
                    r_row.append(r)
                    g_row.append(g)
                    b_row.append(b)
                r_channel.append(r_row)
                g_channel.append(g_row)
                b_channel.append(b_row)
            
            # Ensure we have data in the channels
            if not r_channel or not r_channel[0]:
                continue
                
            # Get channel lengths for safe shifting
            row_length = len(r_channel[0])
            
            # Determine safe shift amounts to avoid empty ranges
            max_shift = max(1, row_length // 4)
            
            # Shift each channel differently
            r_shift = random.randint(-max_shift, max_shift)
            g_shift = random.randint(-max_shift, max_shift)
            b_shift = random.randint(-max_shift, max_shift)
            
            # Apply shifts
            for y_idx in range(len(r_channel)):
                r_row = r_channel[y_idx]
                g_row = g_channel[y_idx]
                b_row = b_channel[y_idx]
                
                # Only shift if rows have data
                if not r_row or not g_row or not b_row:
                    continue
                    
                # Apply shifts safely
                if abs(r_shift) >= len(r_row):
                    r_shift = r_shift % len(r_row)
                if abs(g_shift) >= len(g_row):
                    g_shift = g_shift % len(g_row)
                if abs(b_shift) >= len(b_row):
                    b_shift = b_shift % len(b_row)
                
                r_row_shifted = r_row[-r_shift:] + r_row[:-r_shift] if r_shift > 0 else r_row[abs(r_shift):] + r_row[:abs(r_shift)]
                g_row_shifted = g_row[-g_shift:] + g_row[:-g_shift] if g_shift > 0 else g_row[abs(g_shift):] + g_row[:abs(g_shift)]
                b_row_shifted = b_row[-b_shift:] + b_row[:-b_shift] if b_shift > 0 else b_row[abs(b_shift):] + b_row[:abs(b_shift)]
                
                for x_idx in range(len(r_row)):
                    x = start_x + x_idx
                    y = start_y + y_idx
                    if y < height and x < width and x_idx < len(r_row_shifted) and x_idx < len(g_row_shifted) and x_idx < len(b_row_shifted):
                        pixels[x, y] = (r_row_shifted[x_idx], g_row_shifted[x_idx], b_row_shifted[x_idx])
                        
        elif operation == 'block_offset':
            # Create offset block effect
            if random.random() < 0.7:  # Horizontal offset
                # Ensure valid y range for protected height
                if height <= protected_height + 20:
                    continue  # Skip if not enough vertical space
                    
                max_y_start = max(protected_height, height - 20)
                if protected_height >= max_y_start:
                    continue  # Skip if not enough space
                    
                y_start = random.randint(protected_height, max_y_start)
                height_offset = random.randint(1, max(1, min(50, height - y_start)))
                x_offset = random.randint(-width // 3, width // 3) if width > 3 else 1
                
                for y in range(y_start, min(y_start + height_offset, height)):
                    for x in range(width):
                        src_x = (x - x_offset) % width
                        pixels[x, y] = pixels[src_x, y]
            else:  # Vertical offset
                # Ensure valid x range
                max_x_start = max(0, width - 20)
                if max_x_start <= 0:
                    continue  # Skip if not enough horizontal space
                    
                x_start = random.randint(0, max_x_start)
                width_offset = random.randint(1, max(1, min(50, width - x_start)))
                
                # Ensure vertical space considering protected height
                if height <= protected_height:
                    continue  # Skip if all vertical space is protected
                    
                y_offset = random.randint(-max(1, (height - protected_height) // 3), max(1, (height - protected_height) // 3))
                
                for x in range(x_start, min(x_start + width_offset, width)):
                    for y in range(protected_height, height):
                        # Make sure we stay within bounds
                        if height <= protected_height:
                            continue
                        src_y = protected_height + ((y - protected_height - y_offset) % max(1, (height - protected_height)))
                        pixels[x, y] = pixels[x, src_y]
    
    # Add scan lines for more glitch aesthetic
    if random.random() < 0.3 and height > protected_height + 5:
        scan_line_spacing = random.randint(4, min(20, max(4, height // 10)))
        scan_line_offset = random.randint(0, scan_line_spacing - 1) if scan_line_spacing > 1 else 0
        
        for y in range(protected_height, height):
            if scan_line_spacing > 0 and (y + scan_line_offset) % scan_line_spacing == 0:
                for x in range(width):
                    r, g, b = pixels[x, y]
                    pixels[x, y] = (min(255, int(r * 1.2)), min(255, int(g * 1.2)), min(255, int(b * 1.2)))
    
    # Add some noisy lines
    if random.random() < 0.4 and height > protected_height and width > 0:
        num_lines = random.randint(1, max(3, min(20, height // 20)))
        for _ in range(num_lines):
            if height <= protected_height:
                continue  # Skip if protected height is the whole image
                
            y = random.randint(protected_height, height - 1)
            length = random.randint(max(1, width // 5), width) if width > 5 else width
            start_x = random.randint(0, max(0, width - length))
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            for x in range(start_x, min(start_x + length, width)):
                if x < width:
                    pixels[x, y] = color
    
    # Add noise in some blocks
    if random.random() < 0.5 and width > 10 and height > protected_height + 10:
        noise_blocks = random.randint(1, max(1, min(5, (width * height) // 10000)))
        for _ in range(noise_blocks):
            max_start_x = max(0, width - 50)
            max_start_y = max(protected_height, height - 50)
            
            if max_start_x < 0 or max_start_y <= protected_height:
                continue  # Skip if not enough space
                
            start_x = random.randint(0, max_start_x)
            start_y = random.randint(protected_height, max_start_y)
            block_width = random.randint(1, max(1, min(100, width - start_x)))
            block_height = random.randint(1, max(1, min(30, height - start_y)))
            noise_intensity = random.uniform(0.2, 0.8)
            
            for y in range(start_y, min(start_y + block_height, height)):
                for x in range(start_x, min(start_x + block_width, width)):
                    if random.random() < noise_intensity:
                        r, g, b = pixels[x, y]
                        noise = random.randint(-50, 50)
                        pixels[x, y] = (
                            max(0, min(255, r + noise)),
                            max(0, min(255, g + noise)),
                            max(0, min(255, b + noise))
                        )
    
    return img

def bit_manipulation(image, chunk_size=1, offset=0, xor_value=0xFF, skip_pattern='alternate', 
                    manipulation_type='xor', bit_shift=1, randomize=False, random_seed=None):
    """
    Apply a bit manipulation effect by modifying bytes of the image data in chunks.
    
    Args:
        image (Image): PIL Image object to process.
        chunk_size (int): Size of chunks to process together.
        offset (int): Number of bytes to skip at the beginning.
        xor_value (int): Value to XOR with bytes (0-255) when manipulation_type is 'xor'.
        skip_pattern (str): Pattern for processing chunks:
            - 'alternate': Process every other chunk
            - 'every_third': Process every third chunk
            - 'every_fourth': Process every fourth chunk
            - 'random': Randomly select chunks to process
        manipulation_type (str): Type of bit manipulation:
            - 'xor': XOR with xor_value
            - 'invert': Invert all bits (same as XOR with 0xFF)
            - 'shift': Shift bits left or right
            - 'swap': Swap adjacent chunks
        bit_shift (int): Number of bits to shift when manipulation_type is 'shift'.
        randomize (bool): Add additional randomness to the effect.
        random_seed (int): Seed for random operations, for reproducible results.
    
    Returns:
        Image: Processed image with bit-level glitches.
    """
    # Convert image to RGB mode if it's not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
    
    # Get image data as bytes
    image_bytes = bytearray(image.tobytes())
    
    # Ensure parameters are valid
    chunk_size = max(1, chunk_size)  # Ensure chunk_size is at least 1
    xor_value = max(0, min(255, xor_value))  # Ensure xor_value is in valid range
    offset = max(0, min(len(image_bytes) - 1, offset))  # Ensure offset is valid
    
    # Determine chunk spacing based on skip pattern
    if skip_pattern == 'alternate':
        # Every other chunk
        chunk_spacing = 2
    elif skip_pattern == 'every_third':
        # Every third chunk
        chunk_spacing = 3
    elif skip_pattern == 'every_fourth':
        # Every fourth chunk
        chunk_spacing = 4
    elif skip_pattern == 'random':
        # Random processing will be handled specially
        chunk_spacing = 2  # Placeholder, not used for random
    else:
        # Default to every other chunk
        chunk_spacing = 2
    
    # Process chunks
    if skip_pattern == 'random':
        # Randomly select chunks to process
        for i in range(offset, len(image_bytes), chunk_size):
            if random.random() < 0.5:  # 50% chance to process each chunk
                end_idx = min(i + chunk_size, len(image_bytes))
                apply_manipulation(image_bytes, i, end_idx, manipulation_type, xor_value, bit_shift, randomize)
    elif manipulation_type == 'swap':
        # Special handling for swap - it needs two consecutive chunks
        for i in range(offset, len(image_bytes) - chunk_size * 2, chunk_size * chunk_spacing):
            # Ensure we don't go out of bounds
            if i + chunk_size * 2 <= len(image_bytes):
                # Swap chunks
                for j in range(chunk_size):
                    if i + j < len(image_bytes) and i + chunk_size + j < len(image_bytes):
                        # Swap bytes at positions i+j and i+chunk_size+j
                        image_bytes[i + j], image_bytes[i + chunk_size + j] = image_bytes[i + chunk_size + j], image_bytes[i + j]
    else:
        # Regular chunk processing
        for i in range(offset, len(image_bytes), chunk_size * chunk_spacing):
            end_idx = min(i + chunk_size, len(image_bytes))
            apply_manipulation(image_bytes, i, end_idx, manipulation_type, xor_value, bit_shift, randomize)
    
    # Create a new image from the manipulated bytes
    manipulated_image = Image.frombytes('RGB', image.size, bytes(image_bytes))
    return manipulated_image

def apply_manipulation(byte_array, start_idx, end_idx, manipulation_type, xor_value, bit_shift, randomize):
    """
    Apply a specific bit manipulation to a range of bytes in the array.
    
    Args:
        byte_array (bytearray): The byte array to manipulate.
        start_idx (int): Start index of the range to process.
        end_idx (int): End index of the range to process.
        manipulation_type (str): Type of manipulation to apply.
        xor_value (int): Value to use for XOR operation.
        bit_shift (int): Number of bits to shift.
        randomize (bool): Whether to add randomness to the effect.
    """
    for i in range(start_idx, end_idx):
        if randomize and random.random() < 0.3:  # 30% chance to skip when randomizing
            continue
            
        if manipulation_type == 'xor':
            # XOR with the specified value
            actual_xor = xor_value if not randomize else random.randint(1, 255)
            byte_array[i] = byte_array[i] ^ actual_xor
        elif manipulation_type == 'invert':
            # Invert all bits (equivalent to XOR with 0xFF)
            byte_array[i] = byte_array[i] ^ 0xFF
        elif manipulation_type == 'shift':
            # Shift bits left or right
            actual_shift = bit_shift if not randomize else random.randint(1, 7)
            # Positive shift = left, negative shift = right
            if actual_shift >= 0:
                byte_array[i] = (byte_array[i] << actual_shift) & 0xFF  # Left shift with wrap
            else:
                byte_array[i] = (byte_array[i] >> abs(actual_shift)) | ((byte_array[i] << (8 - abs(actual_shift))) & 0xFF)  # Right shift with wrap

def simulate_jpeg_artifacts(image, intensity):
    """
    Simulate JPEG compression artifacts by repeatedly compressing the image at low quality.

    Args:
        image (PIL.Image): The original image to process.
        intensity (float): A value between 0 and 1 controlling the intensity of the effect.
                           0 for minimal artifacts, 1 for extreme artifacts.

    Returns:
        PIL.Image: The image with simulated JPEG artifacts.
    """
    if not 0 <= intensity <= 1:
        raise ValueError("Intensity must be between 0 and 1.")

    # Map intensity to number of iterations and quality
    # More extreme - up to 30 iterations
    iterations = int(1 + 29 * intensity)  # 1 to 30 iterations
    # Allow quality to go down to 1
    quality = int(90 - 89 * intensity)    # 90 to 1 quality

    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        # Create a white background for RGBA images
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            # Paste the image on the background using the alpha channel as mask
            background.paste(image, mask=image.split()[3])
            current_image = background
        else:
            current_image = image.convert('RGB')
    else:
        current_image = image.copy()

    for _ in range(iterations):
        # Save the image to a BytesIO object with the specified quality
        buffer = BytesIO()
        current_image.save(buffer, format="JPEG", quality=quality)
        # Reload the image from the buffer
        buffer.seek(0)
        current_image = Image.open(buffer)
        current_image.load()  # Make sure the image data is loaded

    return current_image

def data_mosh_blocks(image, num_operations=64, max_block_size=50, block_movement='swap',
                color_swap='random', invert='never', shift='never', flip='never', seed=None):
    """
    Applies data moshing effects to an image for glitch art purposes.

    Args:
        image (Image): PIL Image object to be manipulated.
        num_operations (int): Number of manipulation operations to perform (default: 64).
        max_block_size (int): Maximum size for block width and height (default: 50).
        block_movement (str): 'swap' to swap blocks or 'in_place' to modify blocks in place (default: 'swap').
        color_swap (str): 'never', 'always', or 'random' to control color channel swapping (default: 'random').
        invert (str): 'never', 'always', or 'random' to control color inversion (default: 'never').
        shift (str): 'never', 'always', or 'random' to control channel value shifting (default: 'never').
        flip (str): 'never', 'vertical', 'horizontal', or 'random' to control block flipping (default: 'never').
        seed (int or None): Random seed for reproducibility (default: None).

    Returns:
        Image: PIL Image object with the applied glitch effects.
    """
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    # Set random seeds for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Convert PIL Image to NumPy array
    image_np = np.array(image)
    height, width, channels = image_np.shape

    # Perform the specified number of operations
    for _ in range(num_operations):
        # Determine block dimensions (non-square blocks)
        block_width = random.randint(1, min(max_block_size, width))
        block_height = random.randint(1, min(max_block_size, height))

        # Select first block position
        x1 = random.randint(0, width - block_width)
        y1 = random.randint(0, height - block_height)

        if block_movement == 'swap':
            # Select second block position, ensuring it's different from the first
            attempts = 0
            max_attempts = 10
            while attempts < max_attempts:
                x2 = random.randint(0, width - block_width)
                y2 = random.randint(0, height - block_height)
                if (x2, y2) != (x1, y1):
                    break
                attempts += 1
            else:
                # Could not find a different position, skip this operation
                continue

            # Swap the blocks
            block1 = np.copy(image_np[y1:y1 + block_height, x1:x1 + block_width, :])
            block2 = np.copy(image_np[y2:y2 + block_height, x2:x2 + block_width, :])
            image_np[y1:y1 + block_height, x1:x1 + block_width, :] = block2
            image_np[y2:y2 + block_height, x2:x2 + block_width, :] = block1

            # Define regions for transformation (both swapped blocks)
            regions = [
                (y1, y1 + block_height, x1, x1 + block_width),
                (y2, y2 + block_height, x2, x2 + block_width)
            ]
        elif block_movement == 'in_place':
            # Define region for transformation (single block)
            regions = [(y1, y1 + block_height, x1, x1 + block_width)]
        else:
            raise ValueError("block_movement must be 'swap' or 'in_place'")

        # Apply transformations to each region
        for region in regions:
            y_start, y_end, x_start, x_end = region

            # Color channel swapping
            if color_swap == 'always' or (color_swap == 'random' and random.random() < 0.5):
                swap_choice = random.choice(['red-green', 'red-blue', 'green-blue'])
                if swap_choice == 'red-green':
                    image_np[y_start:y_end, x_start:x_end, [0, 1]] = \
                        image_np[y_start:y_end, x_start:x_end, [1, 0]]
                elif swap_choice == 'red-blue':
                    image_np[y_start:y_end, x_start:x_end, [0, 2]] = \
                        image_np[y_start:y_end, x_start:x_end, [2, 0]]
                elif swap_choice == 'green-blue':
                    image_np[y_start:y_end, x_start:x_end, [1, 2]] = \
                        image_np[y_start:y_end, x_start:x_end, [2, 1]]

            # Color inversion
            if invert == 'always' or (invert == 'random' and random.random() < 0.5):
                image_np[y_start:y_end, x_start:x_end, :] = \
                    255 - image_np[y_start:y_end, x_start:x_end, :]

            # Channel value shifting
            if shift == 'always' or (shift == 'random' and random.random() < 0.5):
                offsets = np.random.randint(-50, 51, size=(3,))
                temp_block = image_np[y_start:y_end, x_start:x_end, :].astype(np.int16)
                temp_block += offsets
                image_np[y_start:y_end, x_start:x_end, :] = \
                    np.clip(temp_block, 0, 255).astype(np.uint8)

            # Block flipping
            if flip != 'never':
                if flip == 'random':
                    flip_type = random.choice(['vertical', 'horizontal'])
                else:
                    flip_type = flip
                if flip_type == 'vertical':
                    image_np[y_start:y_end, x_start:x_end, :] = \
                        np.flipud(image_np[y_start:y_end, x_start:x_end, :])
                elif flip_type == 'horizontal':
                    image_np[y_start:y_end, x_start:x_end, :] = \
                        np.fliplr(image_np[y_start:y_end, x_start:x_end, :])

    # Convert back to PIL Image and return
    moshed_image = Image.fromarray(image_np)
    return moshed_image 