from PIL import Image
import numpy as np
import colorsys
from io import BytesIO
from ..core.pixel_attributes import PixelAttributes
import random
import bisect

def color_channel_manipulation(image, manipulation_type, choice, factor=None):
    """
    Manipulate the image's color channels (swap, invert, adjust intensity, or create negative).
    
    Args:
        image (Image): PIL Image object to process.
        manipulation_type (str): 'swap', 'invert', 'adjust', or 'negative'.
        choice (str): Specific channel or swap pair (e.g., 'red-green', 'red').
                     Not used for 'negative' type.
        factor (float, optional): Intensity adjustment factor (required for 'adjust').
    
    Returns:
        Image: Processed image with modified color channels.
    """
    if image.mode not in ['RGB', 'RGBA']:
        image = image.convert('RGB')
    elif image.mode == 'RGBA': # Convert RGBA to RGB by discarding alpha
        image = image.convert('RGB')

    img_array = np.array(image)
    
    if manipulation_type == 'swap':
        if choice == 'red-green':
            # R, G, B -> G, R, B
            img_array = img_array[:, :, [1, 0, 2]]
        elif choice == 'red-blue':
            # R, G, B -> B, G, R
            img_array = img_array[:, :, [2, 1, 0]] # Original logic was (b,g,r) which is [2,1,0]
        elif choice == 'green-blue':
            # R, G, B -> R, B, G
            img_array = img_array[:, :, [0, 2, 1]]
    elif manipulation_type == 'invert':
        if choice == 'red':
            img_array[:, :, 0] = 255 - img_array[:, :, 0]
        elif choice == 'green':
            img_array[:, :, 1] = 255 - img_array[:, :, 1]
        elif choice == 'blue':
            img_array[:, :, 2] = 255 - img_array[:, :, 2]
    elif manipulation_type == 'negative':
        img_array = 255 - img_array
    elif manipulation_type == 'adjust':
        if factor is None:
            raise ValueError("Factor is required for adjust manipulation")
        
        # Ensure factor is positive; negative factors would invert and are better handled by 'invert'
        # Clamping to 0 to avoid issues with large negative factors if not strictly positive.
        safe_factor = max(0, factor)

        if choice == 'red':
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * safe_factor, 0, 255).astype(np.uint8)
        elif choice == 'green':
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] * safe_factor, 0, 255).astype(np.uint8)
        elif choice == 'blue':
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * safe_factor, 0, 255).astype(np.uint8)
        # If all channels are to be adjusted, it might be more efficient to do it once
        # else: # Assuming 'all' or similar might be a choice, or if no specific channel, adjust all
        #    img_array = np.clip(img_array * safe_factor, 0, 255).astype(np.uint8)


    return Image.fromarray(img_array)

def split_and_shift_channels(image, shift_amount, direction, centered_channel, mode='shift'):
    """
    Split an RGB image into its channels, shift or mirror the channels based on mode,
    and recombine into a new image.

    Args:
        image (Image): PIL Image object in RGB mode.
        shift_amount (int): Number of pixels to shift the non-centered channels (for shift mode).
        direction (str): 'horizontal' or 'vertical' for shift mode (ignored in mirror mode).
        centered_channel (str): 'R', 'G', or 'B' to specify which channel stays centered/unchanged.
        mode (str): 'shift' to shift channels away or 'mirror' to mirror channels.

    Returns:
        Image: New PIL Image with shifted/mirrored channels.
    """
    # Convert to RGB mode if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    # Convert image to numpy array
    pixels = np.array(image)
    
    # Extract channels
    R = pixels[:, :, 0]
    G = pixels[:, :, 1]
    B = pixels[:, :, 2]
    
    # Map the centered_channel string to the appropriate channel variable
    channel_dict = {'R': R, 'G': G, 'B': B}
    centered_channel = centered_channel.upper()[0]  # Convert 'red' to 'R', etc.
    
    if mode == 'shift':
        # Define shifts: centered channel stays at 0, others shift by -X and +X
        if centered_channel == 'R':
            shifts = {'R': 0, 'G': -shift_amount, 'B': shift_amount}
        elif centered_channel == 'G':
            shifts = {'R': -shift_amount, 'G': 0, 'B': shift_amount}
        elif centered_channel == 'B':
            shifts = {'R': -shift_amount, 'G': shift_amount, 'B': 0}
        
        # Apply shifts to each channel
        R_processed = shift_channel(R, shifts['R'], direction)
        G_processed = shift_channel(G, shifts['G'], direction)
        B_processed = shift_channel(B, shifts['B'], direction)
    
    elif mode == 'mirror':
        # Always mirror horizontally and vertically regardless of direction parameter
        # Define which channels to mirror
        if centered_channel == 'R':
            # R stays the same, G mirrors horizontally, B mirrors vertically
            R_processed = R.copy()
            G_processed = mirror_channel(G, 'horizontal')
            B_processed = mirror_channel(B, 'vertical')
        elif centered_channel == 'G':
            # G stays the same, R mirrors horizontally, B mirrors vertically
            R_processed = mirror_channel(R, 'horizontal')
            G_processed = G.copy()
            B_processed = mirror_channel(B, 'vertical')
        elif centered_channel == 'B':
            # B stays the same, R mirrors horizontally, G mirrors vertically
            R_processed = mirror_channel(R, 'horizontal')
            G_processed = mirror_channel(G, 'vertical')
            B_processed = B.copy()
    else:
        raise ValueError("Mode must be 'shift' or 'mirror'.")

    # Combine processed channels
    processed_pixels = np.stack([R_processed, G_processed, B_processed], axis=2)

    # Convert back to PIL Image
    processed_image = Image.fromarray(processed_pixels)

    return processed_image

def shift_channel(channel, shift, direction):
    """
    Shift a 2D array (channel) in a specified direction.

    Args:
        channel (numpy.ndarray): The channel to shift.
        shift (int): Number of pixels to shift (can be negative).
        direction (str): 'horizontal' or 'vertical'.

    Returns:
        numpy.ndarray: The shifted channel.
    """
    height, width = channel.shape
    shifted = np.zeros_like(channel)
    
    if direction == 'horizontal':
        if shift < 0:
            shifted[:, :width+shift] = channel[:, -shift:]
        elif shift > 0:
            shifted[:, shift:] = channel[:, :width-shift]
        else:
            shifted = channel.copy()
    elif direction == 'vertical':
        if shift < 0:
            shifted[:height+shift, :] = channel[-shift:, :]
        elif shift > 0:
            shifted[shift:, :] = channel[:height-shift, :]
        else:
            shifted = channel.copy()
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'.")

    return shifted

def mirror_channel(channel, direction):
    """
    Mirror a 2D array (channel) horizontally or vertically.

    Args:
        channel (numpy.ndarray): The channel to mirror.
        direction (str): 'horizontal' or 'vertical'.

    Returns:
        numpy.ndarray: The mirrored channel.
    """
    if direction == 'horizontal':
        return np.fliplr(channel)
    elif direction == 'vertical':
        return np.flipud(channel)
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'.")

def histogram_glitch(image, r_mode='solarize', g_mode='log', b_mode='gamma', 
                     r_freq=1.0, r_phase=0.0, g_freq=1.0, g_phase=0.0, 
                     b_freq=1.0, b_phase=0.0, gamma_val=0.5):
    """
    Apply different transformations to each color channel based on its histogram.
    
    Args:
        image (PIL.Image): Input image.
        r_mode (str): Transformation for red channel ('solarize', 'log', 'gamma', 'normal').
        g_mode (str): Transformation for green channel ('solarize', 'log', 'gamma', 'normal').
        b_mode (str): Transformation for blue channel ('solarize', 'log', 'gamma', 'normal').
        r_freq (float): Frequency for red channel solarization (0.1-10.0).
        r_phase (float): Phase for red channel solarization (0.0-6.28).
        g_freq (float): Frequency for green channel solarization (0.1-10.0).
        g_phase (float): Phase for green channel solarization (0.0-6.28).
        b_freq (float): Frequency for blue channel solarization (0.1-10.0).
        b_phase (float): Phase for blue channel solarization (0.0-6.28).
        gamma_val (float): Gamma value for gamma transformation (0.1-3.0).
    
    Returns:
        PIL.Image: Processed image with transformed color channels.
    """
    # Convert to RGB mode if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Normalize the image data to 0-1 range
    img_float = img_array.astype(np.float32) / 255.0
    
    # Define helper function to select transformation
    def get_transform(mode, freq, phase, gamma):
        if mode == 'solarize':
            return lambda x: solarize(x, freq, phase)
        elif mode == 'log':
            return log_transform
        elif mode == 'gamma':
            return lambda x: gamma_transform(x, gamma)
        else:  # 'normal'
            return lambda x: x
    
    # Get transform functions for each channel
    r_transform = get_transform(r_mode, r_freq, r_phase, gamma_val)
    g_transform = get_transform(g_mode, g_freq, g_phase, gamma_val)
    b_transform = get_transform(b_mode, b_freq, b_phase, gamma_val)
    
    # Apply transforms to each channel
    img_float[:, :, 0] = r_transform(img_float[:, :, 0])
    img_float[:, :, 1] = g_transform(img_float[:, :, 1])
    img_float[:, :, 2] = b_transform(img_float[:, :, 2])
    
    # Convert back to 0-255 range, clip values, and convert back to uint8
    img_array = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
    
    # Convert back to PIL Image
    return Image.fromarray(img_array)

def solarize(x, freq=1, phase=0):
    """
    Apply a sine-based solarization transformation to a pixel value.

    Args:
        x (int or np.ndarray): Pixel value (0-1.0) or array of pixel values.
        freq (float): Frequency of the sine wave (controls inversion frequency).
        phase (float): Phase shift of the sine wave (shifts the inversion point).

    Returns:
        int or np.ndarray: Transformed pixel value(s) (0-1.0).
    """
    return 0.5 + 0.5 * np.sin(freq * np.pi * x + phase)

def log_transform(x):
    """
    Apply a logarithmic transformation to compress the dynamic range.

    Args:
        x (int or np.ndarray): Pixel value (0-1.0) or array of pixel values.

    Returns:
        int or np.ndarray: Transformed pixel value(s) (0-1.0).
    """
    return np.log(1 + x) / np.log(2)  # Normalize to approximately 0-1 range

def gamma_transform(x, gamma):
    """
    Apply a power-law (gamma) transformation to adjust brightness/contrast.

    Args:
        x (int or np.ndarray): Pixel value (0-1.0) or array of pixel values.
        gamma (float): Gamma value (e.g., <1 brightens, >1 darkens).

    Returns:
        int or np.ndarray: Transformed pixel value(s) (0-1.0).
    """
    # Handle both single values and arrays
    return np.power(x, gamma)

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
    # More extreme - up to 30 iterations (was 20)
    iterations = int(1 + 29 * intensity)  # 1 to 30 iterations
    # Allow quality to go down to 1 (was 10)
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

def color_shift_expansion(image, num_points=5, shift_amount=5, expansion_type='square', mode='xtreme', 
                        saturation_boost=0.0, value_boost=0.0, pattern_type='random', 
                        color_theme='full-spectrum', decay_factor=0.0, seed=None):
    """
    Apply a color shift expansion effect expanding colored shapes from various points.
    
    Args:
        image (Image): PIL Image object to process.
        num_points (int): Number of expansion points.
        shift_amount (int): Amount of color shifting (0-20).
        expansion_type (str): Shape of expansion ('square', 'circle', 'diamond').
        mode (str): Mode of color application ('xtreme', 'subtle', 'mono').
        saturation_boost (float): Amount to boost saturation (0.0-1.0).
        value_boost (float): Amount to boost brightness (0.0-1.0).
        pattern_type (str): Pattern of color point placement ('random', 'grid', 'edges').
        color_theme (str): Color theme to use ('full-spectrum', 'warm', 'cool', 'pastel').
        decay_factor (float): How quickly effect fades with distance (0.0-1.0).
        seed (int, optional): Seed for random number generation. Defaults to None.
    
    Returns:
        Image: Processed image with color shift expansion effect.
    """
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    width, height = image.size
    image_np = np.array(image)
    
    # Create a blank canvas for our output
    output_np = np.zeros_like(image_np)
    
    # Ensure parameters are in valid ranges
    shift_amount = max(1, min(20, shift_amount))
    num_points = max(1, min(50, num_points))
    saturation_boost = max(0.0, min(1.0, saturation_boost))
    value_boost = max(0.0, min(1.0, value_boost))
    decay_factor = max(0.0, min(1.0, decay_factor))
    
    # Generate seed points based on pattern type
    seed_points = []
    if pattern_type == 'grid':
        # Create an evenly spaced grid of points
        cols = max(2, int(np.sqrt(num_points)))
        rows = max(2, num_points // cols)
        x_step = width // cols
        y_step = height // rows
        for i in range(rows):
            for j in range(cols):
                x = j * x_step + x_step // 2
                y = i * y_step + y_step // 2
                if x < width and y < height:
                    seed_points.append((x, y))
    elif pattern_type == 'edges':
        # Points along the edges of the image
        edge_points = num_points
        # Distribute points along the edges
        for i in range(edge_points):
            if i % 4 == 0:  # Top edge
                seed_points.append((int(width * (i / edge_points)), 0))
            elif i % 4 == 1:  # Right edge
                seed_points.append((width - 1, int(height * (i / edge_points))))
            elif i % 4 == 2:  # Bottom edge
                seed_points.append((int(width * (1 - i / edge_points)), height - 1))
            elif i % 4 == 3:  # Left edge
                seed_points.append((0, int(height * (1 - i / edge_points))))
    else:  # random
        # Generate random points
        xs = np.random.randint(0, width, size=num_points)
        ys = np.random.randint(0, height, size=num_points)
        seed_points = list(zip(xs, ys))
    
    # Define the base colors for each theme
    seed_colors = []
    if color_theme == 'warm':
        # Warm colors (reds, oranges, yellows)
        for _ in range(num_points):
            h = random.uniform(0, 60) / 360  # Red to yellow
            s = random.uniform(0.6, 1.0)
            v = random.uniform(0.7, 1.0)
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            seed_colors.append((int(r * 255), int(g * 255), int(b * 255)))
    elif color_theme == 'cool':
        # Cool colors (blues, greens, purples)
        for _ in range(num_points):
            if random.random() < 0.5:
                h = random.uniform(180, 300) / 360  # Cyan to purple
            else:
                h = random.uniform(90, 180) / 360  # Yellow-green to cyan
            s = random.uniform(0.5, 1.0)
            v = random.uniform(0.6, 1.0)
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            seed_colors.append((int(r * 255), int(g * 255), int(b * 255)))
    elif color_theme == 'pastel':
        # Pastel colors (any hue but lower saturation)
        for _ in range(num_points):
            h = random.random()  # Any hue
            s = random.uniform(0.1, 0.5)  # Lower saturation
            v = random.uniform(0.8, 1.0)  # Higher value
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            seed_colors.append((int(r * 255), int(g * 255), int(b * 255)))
    else:  # 'full-spectrum'
        # Full spectrum of vibrant colors
        for i in range(num_points):
            h = i / num_points  # Evenly distributed hues
            s = random.uniform(0.7, 1.0)  # High saturation
            v = random.uniform(0.7, 1.0)  # Medium to high value
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            seed_colors.append((int(r * 255), int(g * 255), int(b * 255)))
    
    # Calculate the maximum possible distance (diagonal of the image)
    max_distance = np.sqrt(width**2 + height**2)
    
    # Create distance maps for each seed point (vectorized)
    yy, xx = np.mgrid[0:height, 0:width]
    distance_maps_list = []

    for point_idx, point_coords in enumerate(seed_points):
        x0, y0 = point_coords
        if expansion_type == 'square':
            # Chebyshev distance (L∞ norm)
            dist_map = np.maximum(np.abs(xx - x0), np.abs(yy - y0))
        elif expansion_type == 'diamond':
            # Manhattan distance (L1 norm)
            dist_map = np.abs(xx - x0) + np.abs(yy - y0)
        else:  # circle (default)
            # Euclidean distance (L2 norm)
            dist_map = np.sqrt((xx - x0)**2 + (yy - y0)**2)
        distance_maps_list.append(dist_map)
    
    # Stack distance maps into a 3D array for easier access: (num_points, height, width)
    if not distance_maps_list: # Handle case with no seed points if num_points could be 0
        # Fill output with original image if no points, though num_points is validated >= 1
        output_np = image_np.copy() 
    else:
        distance_maps_stack = np.stack(distance_maps_list, axis=0)

    # Process each pixel (still a loop, but distance calculation is now outside)
    # Further vectorization of this loop is complex due to per-pixel HSV conversions
    # and conditional logic, but the heaviest part (distance maps) is done.

    # Pre-calculate seed colors as a NumPy array for easier broadcasting later if possible
    seed_colors_np = np.array(seed_colors, dtype=np.float32) # num_points x 3

    # The main loop remains, but accesses pre-calculated distance_maps_stack
    for y in range(height):
        for x in range(width):
            original_r, original_g, original_b = image_np[y, x]
            h, s, v = colorsys.rgb_to_hsv(original_r / 255.0, original_g / 255.0, original_b / 255.0)
            
            if not distance_maps_list: # Should not happen due to num_points validation
                output_np[y, x] = image_np[y, x]
                continue

            # Get all distances for the current pixel (y,x) from all seed points
            pixel_distances = distance_maps_stack[:, y, x] # Shape: (num_points,)
            
            closest_idx = np.argmin(pixel_distances)
            min_dist = pixel_distances[closest_idx]
            
            influences = np.zeros(len(seed_points), dtype=float)
            if decay_factor > 0:
                # Higher decay_factor means faster drop-off. Normalize distance by max_distance.
                influences = np.maximum(0.0, 1.0 - (decay_factor * pixel_distances / max_distance))
            else:
                # Inverse relationship (original was 1/(1 + (d/50)^2) )
                # Avoid division by zero if distance is very small, though 1.0 + ... handles it.
                influences = 1.0 / (1.0 + (pixel_distances / 50.0)**2) # 50.0 is a sensitivity factor

            total_influence = np.sum(influences)
            
            if total_influence < 0.001 or len(seed_colors_np) == 0:
                output_np[y, x] = image_np[y, x]
                continue
            
            normalized_influences = influences / total_influence # Shape: (num_points,)
            
            # Weighted blend of seed colors (RGB)
            # normalized_influences[:, np.newaxis] gives (num_points, 1)
            # seed_colors_np is (num_points, 3)
            # Result is (num_points, 3), then sum over axis 0
            blend_rgb = np.sum(normalized_influences[:, np.newaxis] * seed_colors_np, axis=0)
            blend_r, blend_g, blend_b = blend_rgb[0], blend_rgb[1], blend_rgb[2]
            
            blend_h, blend_s, blend_v = colorsys.rgb_to_hsv(
                np.clip(blend_r / 255.0, 0, 1), 
                np.clip(blend_g / 255.0, 0, 1), 
                np.clip(blend_b / 255.0, 0, 1)
            )
            
            shift_weight = min(0.85, shift_amount / 12.0)
            
            final_h = h * (1 - shift_weight) + blend_h * shift_weight # Hue blending can be tricky, direct average here
            final_s = s * (1 - shift_weight) + (blend_s + saturation_boost) * shift_weight
            final_v = v * (1 - shift_weight) + (blend_v + value_boost) * shift_weight
            
            final_s = min(1.0, max(0.0, final_s))
            final_v = min(1.0, max(0.0, final_v))
            final_h = final_h % 1.0 # Ensure hue remains in [0,1)
            
            final_r_float, final_g_float, final_b_float = colorsys.hsv_to_rgb(final_h, final_s, final_v)
            
            output_np[y, x] = [
                int(final_r_float * 255),
                int(final_g_float * 255),
                int(final_b_float * 255)
            ]
    
    # Convert back to PIL Image
    processed_image = Image.fromarray(output_np.astype(np.uint8))
    return processed_image

def darken_color(color, amount):
    """
    Darken a color by a specified amount.
    
    Args:
        color (tuple): RGB color tuple.
        amount (float): Amount to darken (0.0-1.0).
    
    Returns:
        tuple: Darkened RGB color tuple.
    """
    r, g, b = color
    amount = max(0.0, min(1.0, amount))
    factor = 1.0 - amount
    
    return (
        int(r * factor),
        int(g * factor),
        int(b * factor)
    )

def posterize(image, levels):
    """
    Posterize an image by reducing each color channel to a specified number of levels.
    
    Args:
        image (PIL.Image): The input image.
        levels (int): The number of intensity levels per channel (must be >= 2).
    
    Returns:
        PIL.Image: The posterized image.
    """
    if not isinstance(levels, int) or levels < 2:
        raise ValueError("Levels must be an integer greater than or equal to 2")
    
    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        if image.mode == "RGBA": # Preserve transparency if applicable during conversion
            # Create a white background image
            background = Image.new("RGB", image.size, (255, 255, 255))
            # Paste the RGBA image onto the white background
            background.paste(image, mask=image.split()[3]) 
            image = background
        else:
            image = image.convert("RGB")

    img_array = np.array(image)
    
    # Calculate the scaling factor for posterization levels
    # levels-1 because if levels=2, you want 0 and 255.
    if levels == 1: # Avoid division by zero, effectively making image single color (black or white based on rounding)
        scale = 255.0 
    else:
        scale = 255.0 / (levels - 1)
    
    # Apply posterization
    # Divide by scale, round to nearest integer, then multiply by scale
    # This maps values to the discrete levels
    posterized_array = np.round(img_array / scale) * scale
    
    # Ensure values are within [0, 255] and correct data type
    posterized_array = np.clip(posterized_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(posterized_array)

def curved_hue_shift(image, C, A):
    """
    Applies a curved hue shift to an image.
    
    Args:
        image (PIL.Image): The input image.
        C (float): Curve value from 1 to 360, controlling the shift curve.
        A (float): Total shift amount in degrees.
    
    Returns:
        PIL.Image: The image with the curved hue shift applied.
    """
    # Validate inputs
    if not 1 <= C <= 360:
        raise ValueError("Curve parameter C must be between 1 and 360")
    
    original_mode = image.mode
    if image.mode not in ['RGB', 'RGBA']:
        image = image.convert('RGB') # Fallback to RGB if not directly convertible to HSV from original
    elif image.mode == 'RGBA':
        # Store alpha channel if present
        alpha = image.split()[-1] if image.mode == 'RGBA' else None
        image = image.convert('RGB') # Work with RGB for HSV conversion
    else: # Already RGB
        alpha = None

    # Convert to HSV using PIL
    hsv_image = image.convert('HSV')
    h_channel, s_channel, v_channel = hsv_image.split()

    # Convert H channel to NumPy array and normalize to 0-1 range
    h_array_pil = np.array(h_channel, dtype=np.float32)
    h_norm = h_array_pil / 255.0  # Normalized H (0.0 - 1.0)

    # Curve parameter p (normalized C to -1 to 1)
    # (C is 1-360, so C-1 maps to 0-359. (C-1)/359 * 2 - 1 would be more standard for -1 to 1 from 1-360)
    # Original formula used: p = (C - 180) / 180.0. Let's stick to this for behavioral consistency.
    p_curve = (C - 180.0) / 180.0

    # Calculate shift amount S for each pixel (A is in degrees)
    # S_shift = A * np.exp(p_curve * (h_norm - 0.5)) # A is total shift amount in degrees
    # The exponential term provides the "curve" based on original hue
    shift_factor_rad = p_curve * (h_norm - 0.5) # This term seems to be an angle or factor for exp
    # For stability and to match typical use of exp in shaping functions, ensure argument to exp is not excessively large
    # Capping the exponential factor might be useful depending on C's impact on p_curve
    # However, let's keep the direct math first to match original intent if possible.
    
    # Shift amount in degrees
    s_shift_degrees = A * np.exp(shift_factor_rad)

    # Current hue in degrees
    current_h_degrees = h_norm * 360.0

    # New hue in degrees
    new_h_degrees = (current_h_degrees + s_shift_degrees) % 360.0

    # Convert new H back to PIL's 0-255 scale for 'L' mode channel
    new_h_pil_scale = (new_h_degrees / 360.0) * 255.0
    shifted_h_array = np.clip(new_h_pil_scale, 0, 255).astype(np.uint8)
    
    shifted_h_channel_pil = Image.fromarray(shifted_h_array, mode='L')

    # Merge channels and convert back to RGB
    final_hsv_image = Image.merge('HSV', (shifted_h_channel_pil, s_channel, v_channel))
    final_rgb_image = final_hsv_image.convert('RGB')

    # If original image had alpha, re-apply it
    if alpha:
        if final_rgb_image.mode != 'RGBA':
             final_rgb_image.putalpha(alpha)
        # If original_mode was RGBA and we want to return RGBA
        # If original_mode was something else but we had alpha, it's a bit ambiguous
        # For now, if alpha existed, we assume RGBA output is desired or safe.

    # Attempt to convert back to original mode if it was not RGBA but had specific characteristics (e.g. P)
    # This can be tricky. For now, RGB or RGBA is the most stable output from this type of manipulation.
    # If original_mode was 'L', 'P', etc., direct conversion back might not be ideal after HSV.
    # Sticking to RGB/RGBA output primarily.
    if original_mode == 'RGBA' and final_rgb_image.mode != 'RGBA':
        # This case should be handled by putalpha already if alpha was present
        pass 
    elif original_mode != 'RGBA' and original_mode != 'RGB' and final_rgb_image.mode == 'RGB':
        # Consider if conversion back to original_mode is safe/desired
        # e.g. if original_mode was 'L', converting colorful RGB to 'L' is lossy.
        # For now, we output RGB or RGBA.
        pass 
        
    return final_rgb_image 