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
    if seed is not None:
        random.seed(seed)
    
    if image.mode != 'RGB':
        img_rgb = image.convert('RGB')
    else:
        img_rgb = image.copy() # Work on a copy

    img_array = np.array(img_rgb)
    height, width, _ = img_array.shape
    
    protected_height = int(height * 0.1) if preserve_header else 0
    
    num_operations = int(intensity * 150)
    num_operations = max(5, min(150, num_operations))

    operations = [
        'shift_rows', 'shift_columns', 'swap_channels', 'xor_block',
        'repeat_block', 'color_shift', 'invert_block', 'shift_channels_op', # Renamed to avoid conflict
        'block_offset_op' # Renamed to avoid conflict
    ]
    
    if width < 10 or height < 10:
        # Fallback for very small images - operate on array
        for y_idx in range(height):
            for x_idx in range(width):
                if random.random() < intensity:
                    r, g, b = img_array[y_idx, x_idx]
                    effect = random.choice(['invert', 'xor', 'shift'])
                    if effect == 'invert':
                        img_array[y_idx, x_idx] = [255 - r, 255 - g, 255 - b]
                    elif effect == 'xor':
                        xor_val = random.randint(1, 255)
                        img_array[y_idx, x_idx] = [r ^ xor_val, g ^ xor_val, b ^ xor_val]
                    elif effect == 'shift':
                        s = random.randint(-50, 50)
                        img_array[y_idx, x_idx] = [
                            np.clip(r + s, 0, 255),
                            np.clip(g + s, 0, 255),
                            np.clip(b + s, 0, 255)
                        ]
        return Image.fromarray(img_array)

    for _ in range(num_operations):
        operation = random.choice(operations)
        
        # Define boundaries for operations to avoid IndexError
        # For block operations, ensure block_width/height >= 1
        min_block_dim = 1
        
        # Max start coordinates ensuring a minimal block can be formed
        # Allow operations up to the last possible start of a 1-pixel block
        max_op_start_y = max(protected_height, height - min_block_dim)
        max_op_start_x = max(0, width - min_block_dim)

        if max_op_start_y < protected_height : max_op_start_y = protected_height # ensure start_y respects protected_height
        if max_op_start_x < 0 : max_op_start_x = 0


        if operation == 'shift_rows':
            if protected_height >= height: continue # Not enough unprotected height
            start_y = random.randint(protected_height, max(protected_height, height - 1)) # Can be a single row
            y_size = random.randint(1, max(1, height - start_y)) # At least 1 row
            actual_end_y = min(start_y + y_size, height)
            
            if actual_end_y <= start_y: continue

            shift_amount = random.randint(-width // 2, width // 2)
            if shift_amount == 0: continue

            rows_to_shift = img_array[start_y:actual_end_y, :, :]
            img_array[start_y:actual_end_y, :, :] = np.roll(rows_to_shift, shift_amount, axis=1)

        elif operation == 'shift_columns':
            if width <=0 : continue
            start_x = random.randint(0, max(0, width - 1)) # Can be a single col
            x_size = random.randint(1, max(1, width - start_x))
            actual_end_x = min(start_x + x_size, width)

            if actual_end_x <= start_x : continue
            
            # Define effective height for column shift (respecting protected_header)
            eff_col_height_start = protected_height
            eff_col_height_end = height
            if eff_col_height_start >= eff_col_height_end: continue


            shift_amount = random.randint(-(eff_col_height_end - eff_col_height_start) // 3, (eff_col_height_end - eff_col_height_start) // 3)
            if shift_amount == 0: continue
            
            cols_to_shift = img_array[eff_col_height_start:eff_col_height_end, start_x:actual_end_x, :]
            img_array[eff_col_height_start:eff_col_height_end, start_x:actual_end_x, :] = np.roll(cols_to_shift, shift_amount, axis=0)

        elif operation == 'swap_channels':
            if max_op_start_y < protected_height or max_op_start_x < 0 : continue
            if height <= protected_height or width <=0 : continue

            start_y = random.randint(protected_height, max_op_start_y)
            start_x = random.randint(0, max_op_start_x)
            
            block_height = random.randint(min_block_dim, max(min_block_dim, height - start_y))
            block_width = random.randint(min_block_dim, max(min_block_dim, width - start_x))
            end_y, end_x = min(start_y + block_height, height), min(start_x + block_width, width)

            if end_y <= start_y or end_x <= start_x : continue

            block = img_array[start_y:end_y, start_x:end_x, :]
            swap_type = random.choice(['rb', 'rg', 'gb', 'brg', 'gbr', 'bgr']) # More swaps
            if swap_type == 'rb':   block[:] = block[:, :, [2, 1, 0]] # R <-> B
            elif swap_type == 'rg': block[:] = block[:, :, [1, 0, 2]] # R <-> G
            elif swap_type == 'gb': block[:] = block[:, :, [0, 2, 1]] # G <-> B
            elif swap_type == 'brg': block[:] = block[:, :, [1,2,0]] # R->G, G->B, B->R
            elif swap_type == 'gbr': block[:] = block[:, :, [2,0,1]] # R->B, B->G, G->R
            # 'bgr' is original if RGB, no change needed. For completeness:
            # elif swap_type == 'bgr': block[:] = block[:, :, [0,1,2]] 
            img_array[start_y:end_y, start_x:end_x, :] = block


        elif operation == 'xor_block':
            if max_op_start_y < protected_height or max_op_start_x < 0 : continue
            if height <= protected_height or width <=0 : continue

            start_y = random.randint(protected_height, max_op_start_y)
            start_x = random.randint(0, max_op_start_x)
            block_height = random.randint(min_block_dim, max(min_block_dim, height - start_y))
            block_width = random.randint(min_block_dim, max(min_block_dim, width - start_x))
            end_y, end_x = min(start_y + block_height, height), min(start_x + block_width, width)

            if end_y <= start_y or end_x <= start_x : continue
            
            xor_value = random.randint(1, 255)
            block = img_array[start_y:end_y, start_x:end_x, :]
            # XOR operation on uint8 will wrap around, which is fine for glitch effects
            img_array[start_y:end_y, start_x:end_x, :] = block ^ xor_value

        elif operation == 'repeat_block':
            # Define source block
            if height <= protected_height or width <=0 : continue
            max_src_start_y = max(protected_height, height - min_block_dim)
            max_src_start_x = max(0, width - min_block_dim)
            if max_src_start_y < protected_height or max_src_start_x < 0 : continue


            src_y = random.randint(protected_height, max_src_start_y)
            src_x = random.randint(0, max_src_start_x)
            block_h = random.randint(min_block_dim, max(min_block_dim, height - src_y))
            block_w = random.randint(min_block_dim, max(min_block_dim, width - src_x))
            src_end_y, src_end_x = min(src_y + block_h, height), min(src_x + block_w, width)

            if src_end_y <= src_y or src_end_x <= src_x : continue
            src_block = img_array[src_y:src_end_y, src_x:src_end_x, :]

            # Define destination (can overlap, can be anywhere not protected)
            if height <= protected_height or width <=0 : continue # Redundant but safe
            
            # Max dst_start ensures block fits
            max_dst_start_y = max(protected_height, height - block_h) 
            max_dst_start_x = max(0, width - block_w)

            if max_dst_start_y < protected_height or max_dst_start_x < 0: continue


            dst_y = random.randint(protected_height, max_dst_start_y)
            dst_x = random.randint(0, max_dst_start_x)
            dst_end_y, dst_end_x = min(dst_y + block_h, height), min(dst_x + block_w, width)
            
            # Ensure destination block has same dimensions as source after clipping
            # This can happen if block_h/block_w were large
            actual_block_h = src_end_y - src_y
            actual_block_w = src_end_x - src_x

            if dst_y + actual_block_h > height or dst_x + actual_block_w > width: continue


            img_array[dst_y : dst_y + actual_block_h, dst_x : dst_x + actual_block_w, :] = src_block[:actual_block_h, :actual_block_w, :]


        elif operation == 'color_shift':
            if max_op_start_y < protected_height or max_op_start_x < 0 : continue
            if height <= protected_height or width <=0 : continue

            start_y = random.randint(protected_height, max_op_start_y)
            start_x = random.randint(0, max_op_start_x)
            block_height = random.randint(min_block_dim, max(min_block_dim, height - start_y))
            block_width = random.randint(min_block_dim, max(min_block_dim, width - start_x))
            end_y, end_x = min(start_y + block_height, height), min(start_x + block_width, width)

            if end_y <= start_y or end_x <= start_x : continue

            # Use float for intermediate to prevent overflow before clip
            block = img_array[start_y:end_y, start_x:end_x, :].astype(np.int16)
            shifts = np.random.randint(-50, 51, size=3) # Random shift for R, G, B
            block[:, :, 0] += shifts[0]
            block[:, :, 1] += shifts[1]
            block[:, :, 2] += shifts[2]
            img_array[start_y:end_y, start_x:end_x, :] = np.clip(block, 0, 255).astype(np.uint8)

        elif operation == 'invert_block':
            if max_op_start_y < protected_height or max_op_start_x < 0 : continue
            if height <= protected_height or width <=0 : continue

            start_y = random.randint(protected_height, max_op_start_y)
            start_x = random.randint(0, max_op_start_x)
            block_height = random.randint(min_block_dim, max(min_block_dim, height - start_y))
            block_width = random.randint(min_block_dim, max(min_block_dim, width - start_x))
            end_y, end_x = min(start_y + block_height, height), min(start_x + block_width, width)
            
            if end_y <= start_y or end_x <= start_x : continue

            block = img_array[start_y:end_y, start_x:end_x, :]
            img_array[start_y:end_y, start_x:end_x, :] = 255 - block
            
        elif operation == 'shift_channels_op': # Renamed
            if max_op_start_y < protected_height or max_op_start_x < 0 : continue
            if height <= protected_height or width <=0 : continue

            start_y = random.randint(protected_height, max_op_start_y)
            start_x = random.randint(0, max_op_start_x)
            block_height = random.randint(min_block_dim, max(min_block_dim, height - start_y))
            block_width = random.randint(min_block_dim, max(min_block_dim, width - start_x))
            end_y, end_x = min(start_y + block_height, height), min(start_x + block_width, width)

            if end_y <= start_y or end_x <= start_x : continue

            block = img_array[start_y:end_y, start_x:end_x, :].copy() # Operate on a copy
            
            for i in range(3): # R, G, B
                shift_x = random.randint(-block_width // 4, block_width // 4)
                shift_y = random.randint(-block_height // 4, block_height // 4)
                # Apply roll to the specific channel of the original block slice
                # and assign it back to the same channel in img_array for that block
                rolled_channel = np.roll(block[:, :, i], shift=(shift_y, shift_x), axis=(0, 1))
                img_array[start_y:end_y, start_x:end_x, i] = rolled_channel


        elif operation == 'block_offset_op': # Renamed
            # Similar to repeat_block but shifts a block by a small random offset
            if height <= protected_height or width <=0 : continue
            max_src_start_y = max(protected_height, height - min_block_dim)
            max_src_start_x = max(0, width - min_block_dim)
            if max_src_start_y < protected_height or max_src_start_x < 0 : continue

            src_y = random.randint(protected_height, max_src_start_y)
            src_x = random.randint(0, max_src_start_x)
            block_h = random.randint(min_block_dim, max(min_block_dim, min(50, height - src_y))) # Smaller blocks
            block_w = random.randint(min_block_dim, max(min_block_dim, min(50, width - src_x)))
            src_end_y, src_end_x = min(src_y + block_h, height), min(src_x + block_w, width)

            if src_end_y <= src_y or src_end_x <= src_x : continue
            
            src_block_data = img_array[src_y:src_end_y, src_x:src_end_x, :].copy() # Copy the data

            offset_x = random.randint(-block_w // 2, block_w // 2)
            offset_y = random.randint(-block_h // 2, block_h // 2)

            dst_y_start = np.clip(src_y + offset_y, protected_height, height - block_h)
            dst_x_start = np.clip(src_x + offset_x, 0, width - block_w)
            
            # Ensure destination is valid and doesn't go out of bounds for the block size
            dst_y_end = min(dst_y_start + block_h, height)
            dst_x_end = min(dst_x_start + block_w, width)

            # Adjust block_h/w if dst is clipped smaller than src
            actual_bh = min(block_h, dst_y_end - dst_y_start)
            actual_bw = min(block_w, dst_x_end - dst_x_start)

            if actual_bh <=0 or actual_bw <=0 : continue

            img_array[dst_y_start : dst_y_start + actual_bh, dst_x_start : dst_x_start + actual_bw, :] = src_block_data[:actual_bh, :actual_bw, :]


    return Image.fromarray(img_array)

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