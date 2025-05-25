from PIL import Image
import numpy as np
import colorsys
from io import BytesIO
from ..core.pixel_attributes import PixelAttributes
import random
import bisect
from PIL import ImageFilter

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

def color_filter(image, filter_type='solid', color='#FF0000', blend_mode='overlay', opacity=0.5, 
                 gradient_color2='#0000FF', gradient_angle=0):
    """
    Apply a colored filter effect to simulate camera lens filters.
    
    Args:
        image (Image): PIL Image object to process.
        filter_type (str): Type of filter ('solid' or 'gradient').
        color (str): Primary filter color in hex format (e.g., '#FF0000').
        blend_mode (str): Blending mode ('overlay' or 'soft_light').
        opacity (float): Filter opacity (0.0 to 1.0).
        gradient_color2 (str): Secondary color for gradient filter in hex format.
        gradient_angle (int): Gradient rotation angle in degrees (0-360).
    
    Returns:
        Image: Processed image with color filter effect.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image, dtype=np.float32)
    height, width, _ = img_array.shape
    
    # Convert hex colors to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    color_rgb = hex_to_rgb(color)
    
    # Create filter layer
    if filter_type == 'solid':
        # Create solid color filter
        filter_layer = np.full((height, width, 3), color_rgb, dtype=np.float32)
    else:  # gradient
        gradient_color2_rgb = hex_to_rgb(gradient_color2)
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Convert angle to radians and create gradient direction
        angle_rad = np.radians(gradient_angle)
        
        # Calculate gradient based on rotated coordinates
        # Normalize coordinates to [-1, 1] range
        x_norm = (x_coords - width / 2) / (width / 2)
        y_norm = (y_coords - height / 2) / (height / 2)
        
        # Rotate coordinates
        x_rot = x_norm * np.cos(angle_rad) - y_norm * np.sin(angle_rad)
        
        # Create gradient values from -1 to 1
        gradient_values = x_rot
        
        # Normalize to [0, 1] range
        gradient_values = (gradient_values + 1) / 2
        gradient_values = np.clip(gradient_values, 0, 1)
        
        # Create gradient filter layer
        filter_layer = np.zeros((height, width, 3), dtype=np.float32)
        for i in range(3):
            filter_layer[:, :, i] = (
                gradient_values * color_rgb[i] + 
                (1 - gradient_values) * gradient_color2_rgb[i]
            )
    
    # Apply blend mode
    if blend_mode == 'overlay':
        # Overlay blend mode
        result = np.where(
            img_array < 128,
            2 * img_array * filter_layer / 255,
            255 - 2 * (255 - img_array) * (255 - filter_layer) / 255
        )
    else:  # soft_light
        # Soft Light blend mode
        img_norm = img_array / 255.0
        filter_norm = filter_layer / 255.0
        
        result = np.where(
            filter_norm <= 0.5,
            img_norm - (1 - 2 * filter_norm) * img_norm * (1 - img_norm),
            img_norm + (2 * filter_norm - 1) * (
                np.where(img_norm <= 0.25, 
                        ((16 * img_norm - 12) * img_norm + 4) * img_norm,
                        np.sqrt(img_norm)) - img_norm
            )
        )
        result = result * 255
    
    # Apply opacity
    result = img_array * (1 - opacity) + result * opacity
    
    # Ensure values are in valid range
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return Image.fromarray(result) 

def gaussian_blur(image, radius=5.0, sigma=None):
    """
    Apply Gaussian blur to an image.
    
    Args:
        image (Image): PIL Image object to process.
        radius (float): Blur radius in pixels (0.1 to 50.0).
        sigma (float, optional): Standard deviation for Gaussian kernel. 
                                If None, sigma = radius / 3.0 (common approximation).
    
    Returns:
        Image: Blurred image.
    """
    if image.mode not in ['RGB', 'RGBA', 'L']:
        image = image.convert('RGB')
    
    # If sigma is not provided, use the common approximation
    if sigma is None:
        sigma = radius / 3.0
    
    # Ensure minimum values to prevent errors
    radius = max(0.1, radius)
    sigma = max(0.1, sigma)
    
    # Apply Gaussian blur using PIL's ImageFilter
    # PIL's GaussianBlur uses radius parameter
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    return blurred_image 

def noise_effect(image, noise_type='film_grain', intensity=0.3, grain_size=1.0, 
                color_variation=0.2, noise_color='#FFFFFF', blend_mode='overlay', 
                pattern='random', seed=None):
    """
    Add various types of noise effects to an image.
    
    Args:
        image (Image): PIL Image object to process.
        noise_type (str): Type of noise ('film_grain', 'digital', 'colored', 'salt_pepper', 'gaussian').
        intensity (float): Overall noise intensity (0.0 to 1.0).
        grain_size (float): Size of noise particles (0.5 to 5.0).
        color_variation (float): Amount of color variation in noise (0.0 to 1.0).
        noise_color (str): Base color for colored noise in hex format.
        blend_mode (str): How to blend noise ('overlay', 'add', 'multiply', 'screen').
        pattern (str): Noise pattern ('random', 'perlin', 'cellular').
        seed (int, optional): Random seed for reproducible results.
    
    Returns:
        Image: Image with noise effect applied.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    img_array = np.array(image, dtype=np.float32)
    height, width, channels = img_array.shape
    
    # Convert hex color to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    base_color = hex_to_rgb(noise_color)
    
    # Generate base noise pattern
    if pattern == 'perlin':
        # Simple Perlin-like noise using multiple octaves
        noise_base = np.zeros((height, width))
        for octave in range(3):
            scale = 0.1 * (2 ** octave) * grain_size
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            noise_octave = np.sin(x_coords * scale) * np.cos(y_coords * scale)
            noise_base += noise_octave / (2 ** octave)
        noise_base = (noise_base + 1) / 2  # Normalize to 0-1
    elif pattern == 'cellular':
        # Cellular automata-like pattern
        noise_base = np.random.random((height, width))
        # Apply cellular automata rules
        for _ in range(2):
            kernel = np.ones((3, 3)) / 9
            from scipy import ndimage
            noise_base = ndimage.convolve(noise_base, kernel, mode='wrap')
            noise_base = (noise_base > 0.5).astype(float)
    else:  # random
        noise_base = np.random.random((height, width))
    
    # Scale noise by grain size
    if grain_size != 1.0:
        from scipy import ndimage
        # Resize noise pattern
        scale_factor = 1.0 / grain_size
        small_height = max(1, int(height * scale_factor))
        small_width = max(1, int(width * scale_factor))
        
        # Generate smaller noise and scale up
        if pattern == 'random':
            small_noise = np.random.random((small_height, small_width))
        else:
            small_noise = noise_base[:small_height, :small_width]
        
        # Resize back to original size
        noise_base = ndimage.zoom(small_noise, (height/small_height, width/small_width), order=1)
    
    # Create noise based on type
    if noise_type == 'film_grain':
        # Simulate film grain with luminance-dependent noise
        luminance = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
        luminance_norm = luminance / 255.0
        
        # More grain in mid-tones, less in shadows and highlights
        grain_mask = 4 * luminance_norm * (1 - luminance_norm)
        
        # Generate colored grain
        noise_r = (noise_base - 0.5) * intensity * grain_mask
        noise_g = (np.random.random((height, width)) - 0.5) * intensity * grain_mask * color_variation
        noise_b = (np.random.random((height, width)) - 0.5) * intensity * grain_mask * color_variation
        
        noise_array = np.stack([noise_r, noise_g, noise_b], axis=2) * 255
        
    elif noise_type == 'digital':
        # Sharp digital noise
        noise_base = (noise_base > (1 - intensity)).astype(float)
        noise_array = np.stack([noise_base] * 3, axis=2) * 255
        
        # Add color variation
        if color_variation > 0:
            for i in range(3):
                color_noise = (np.random.random((height, width)) - 0.5) * color_variation * 255
                noise_array[:,:,i] += color_noise
    
    elif noise_type == 'colored':
        # Colored noise based on base color
        noise_r = (noise_base - 0.5) * intensity * (base_color[0] / 255.0) * 255
        noise_g = (noise_base - 0.5) * intensity * (base_color[1] / 255.0) * 255
        noise_b = (noise_base - 0.5) * intensity * (base_color[2] / 255.0) * 255
        
        # Add color variation
        if color_variation > 0:
            noise_r += (np.random.random((height, width)) - 0.5) * color_variation * 255
            noise_g += (np.random.random((height, width)) - 0.5) * color_variation * 255
            noise_b += (np.random.random((height, width)) - 0.5) * color_variation * 255
        
        noise_array = np.stack([noise_r, noise_g, noise_b], axis=2)
    
    elif noise_type == 'salt_pepper':
        # Salt and pepper noise
        salt_pepper = np.random.random((height, width))
        salt_mask = salt_pepper > (1 - intensity/2)
        pepper_mask = salt_pepper < (intensity/2)
        
        noise_array = np.zeros((height, width, 3))
        noise_array[salt_mask] = 255  # Salt (white)
        noise_array[pepper_mask] = -255  # Pepper (black)
    
    else:  # gaussian
        # Gaussian noise
        noise_r = np.random.normal(0, intensity * 255, (height, width))
        noise_g = np.random.normal(0, intensity * 255 * color_variation, (height, width))
        noise_b = np.random.normal(0, intensity * 255 * color_variation, (height, width))
        
        noise_array = np.stack([noise_r, noise_g, noise_b], axis=2)
    
    # Apply blend mode
    if blend_mode == 'add':
        result = img_array + noise_array
    elif blend_mode == 'multiply':
        noise_norm = (noise_array + 255) / 510  # Normalize to 0-1 range
        result = img_array * noise_norm
    elif blend_mode == 'screen':
        img_norm = img_array / 255.0
        noise_norm = (noise_array + 255) / 510
        result = (1 - (1 - img_norm) * (1 - noise_norm)) * 255
    else:  # overlay (default)
        noise_norm = noise_array / 255.0
        img_norm = img_array / 255.0
        
        result = np.where(
            img_norm < 0.5,
            2 * img_norm * (noise_norm + 0.5),
            1 - 2 * (1 - img_norm) * (0.5 - noise_norm)
        ) * 255
    
    # Ensure values are in valid range
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return Image.fromarray(result) 

def chromatic_aberration(image, intensity=5.0, pattern='radial', red_shift_x=0.0, red_shift_y=0.0,
                        blue_shift_x=0.0, blue_shift_y=0.0, center_x=0.5, center_y=0.5,
                        falloff='quadratic', edge_enhancement=0.0, color_boost=1.0, seed=None):
    """
    Apply chromatic aberration effect to simulate lens color fringing.
    
    Args:
        image (Image): PIL Image object to process.
        intensity (float): Overall aberration intensity (0.0 to 50.0).
        pattern (str): Aberration pattern ('radial', 'linear', 'barrel', 'custom').
        red_shift_x (float): Manual red channel X displacement (-20.0 to 20.0).
        red_shift_y (float): Manual red channel Y displacement (-20.0 to 20.0).
        blue_shift_x (float): Manual blue channel X displacement (-20.0 to 20.0).
        blue_shift_y (float): Manual blue channel Y displacement (-20.0 to 20.0).
        center_x (float): Aberration center X position (0.0 to 1.0).
        center_y (float): Aberration center Y position (0.0 to 1.0).
        falloff (str): Distance falloff type ('linear', 'quadratic', 'cubic').
        edge_enhancement (float): Enhance edge contrast (0.0 to 1.0).
        color_boost (float): Boost color saturation (0.5 to 2.0).
        seed (int, optional): Random seed for pattern variations.
    
    Returns:
        Image: Image with chromatic aberration effect applied.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    img_array = np.array(image, dtype=np.float32)
    height, width, channels = img_array.shape
    
    # Extract color channels
    red_channel = img_array[:, :, 0]
    green_channel = img_array[:, :, 1]
    blue_channel = img_array[:, :, 2]
    
    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Normalize coordinates to [-1, 1] range centered on aberration center
    center_x_px = center_x * width
    center_y_px = center_y * height
    x_norm = (x_coords - center_x_px) / (width / 2)
    y_norm = (y_coords - center_y_px) / (height / 2)
    
    # Calculate distance from center
    distance = np.sqrt(x_norm**2 + y_norm**2)
    
    # Apply falloff function
    if falloff == 'linear':
        falloff_factor = distance
    elif falloff == 'cubic':
        falloff_factor = distance**3
    else:  # quadratic (default)
        falloff_factor = distance**2
    
    # Calculate displacement based on pattern
    if pattern == 'radial':
        # Classic radial chromatic aberration
        # Red channel moves outward, blue moves inward
        red_disp_x = x_norm * falloff_factor * intensity * 0.1
        red_disp_y = y_norm * falloff_factor * intensity * 0.1
        blue_disp_x = -x_norm * falloff_factor * intensity * 0.1
        blue_disp_y = -y_norm * falloff_factor * intensity * 0.1
        
    elif pattern == 'linear':
        # Linear aberration (like prism effect)
        red_disp_x = np.full_like(x_coords, intensity * 0.2, dtype=np.float32)
        red_disp_y = np.zeros_like(y_coords, dtype=np.float32)
        blue_disp_x = np.full_like(x_coords, -intensity * 0.2, dtype=np.float32)
        blue_disp_y = np.zeros_like(y_coords, dtype=np.float32)
        
    elif pattern == 'barrel':
        # Barrel distortion-like aberration
        angle = np.arctan2(y_norm, x_norm)
        radial_factor = falloff_factor * intensity * 0.1
        red_disp_x = np.cos(angle) * radial_factor * 1.2
        red_disp_y = np.sin(angle) * radial_factor * 1.2
        blue_disp_x = np.cos(angle) * radial_factor * 0.8
        blue_disp_y = np.sin(angle) * radial_factor * 0.8
        
    else:  # custom
        # Use manual displacement values with some distance modulation
        distance_mod = 1.0 + falloff_factor * 0.5
        red_disp_x = np.full_like(x_coords, red_shift_x * distance_mod, dtype=np.float32)
        red_disp_y = np.full_like(y_coords, red_shift_y * distance_mod, dtype=np.float32)
        blue_disp_x = np.full_like(x_coords, blue_shift_x * distance_mod, dtype=np.float32)
        blue_disp_y = np.full_like(y_coords, blue_shift_y * distance_mod, dtype=np.float32)
    
    # Add manual shifts to pattern-based displacement
    if pattern != 'custom':
        red_disp_x += red_shift_x
        red_disp_y += red_shift_y
        blue_disp_x += blue_shift_x
        blue_disp_y += blue_shift_y
    
    # Apply displacement to channels using bilinear interpolation
    def apply_displacement(channel, disp_x, disp_y):
        # Calculate new coordinates
        new_x = x_coords + disp_x
        new_y = y_coords + disp_y
        
        # Clip coordinates to image bounds
        new_x = np.clip(new_x, 0, width - 1)
        new_y = np.clip(new_y, 0, height - 1)
        
        # Get integer and fractional parts
        x0 = np.floor(new_x).astype(int)
        x1 = np.minimum(x0 + 1, width - 1)
        y0 = np.floor(new_y).astype(int)
        y1 = np.minimum(y0 + 1, height - 1)
        
        # Calculate interpolation weights
        wx = new_x - x0
        wy = new_y - y0
        
        # Bilinear interpolation
        interpolated = (
            channel[y0, x0] * (1 - wx) * (1 - wy) +
            channel[y0, x1] * wx * (1 - wy) +
            channel[y1, x0] * (1 - wx) * wy +
            channel[y1, x1] * wx * wy
        )
        
        return interpolated
    
    # Apply displacements
    red_displaced = apply_displacement(red_channel, red_disp_x, red_disp_y)
    blue_displaced = apply_displacement(blue_channel, blue_disp_x, blue_disp_y)
    
    # Green channel stays in place (or minimal displacement)
    green_displaced = green_channel.copy()
    
    # Combine channels
    result = np.stack([red_displaced, green_displaced, blue_displaced], axis=2)
    
    # Apply edge enhancement if requested
    if edge_enhancement > 0:
        from scipy import ndimage
        # Create edge detection kernel
        edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        
        # Apply edge detection to each channel
        for i in range(3):
            edges = ndimage.convolve(result[:, :, i], edge_kernel, mode='reflect')
            result[:, :, i] += edges * edge_enhancement * 0.1
    
    # Apply color boost
    if color_boost != 1.0:
        # Convert to HSV for saturation boost
        result_uint8 = np.clip(result, 0, 255).astype(np.uint8)
        result_pil = Image.fromarray(result_uint8)
        
        # Simple saturation boost by scaling color channels relative to luminance
        luminance = 0.299 * result[:, :, 0] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 2]
        for i in range(3):
            # Boost color relative to luminance
            color_diff = result[:, :, i] - luminance
            result[:, :, i] = luminance + color_diff * color_boost
    
    # Ensure values are in valid range
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return Image.fromarray(result) 

def vhs_effect(image, quality_preset='medium', scan_line_intensity=0.3, scan_line_spacing=2,
               static_intensity=0.2, static_type='white', vertical_hold_frequency=0.1,
               vertical_hold_intensity=5.0, color_bleeding=0.3, chroma_shift=0.2,
               tracking_errors=0.15, tape_wear=0.1, head_switching_noise=0.1,
               color_desaturation=0.3, brightness_variation=0.2, seed=None):
    """
    Apply comprehensive VHS tape effect with authentic analog video artifacts.
    
    Args:
        image (Image): PIL Image object to process.
        quality_preset (str): Overall quality preset ('high', 'medium', 'low', 'damaged').
        scan_line_intensity (float): Intensity of horizontal scan lines (0.0 to 1.0).
        scan_line_spacing (int): Spacing between scan lines in pixels (1 to 5).
        static_intensity (float): Amount of static noise (0.0 to 1.0).
        static_type (str): Type of static ('white', 'colored', 'mixed').
        vertical_hold_frequency (float): Frequency of vertical hold issues (0.0 to 1.0).
        vertical_hold_intensity (float): Intensity of line displacement (0.0 to 20.0).
        color_bleeding (float): Amount of color bleeding between channels (0.0 to 1.0).
        chroma_shift (float): Chroma/luma separation artifacts (0.0 to 1.0).
        tracking_errors (float): Frequency of tracking problems (0.0 to 1.0).
        tape_wear (float): Amount of tape wear and dropouts (0.0 to 1.0).
        head_switching_noise (float): Head switching noise bands (0.0 to 1.0).
        color_desaturation (float): Color desaturation amount (0.0 to 1.0).
        brightness_variation (float): Random brightness variations (0.0 to 1.0).
        seed (int, optional): Random seed for reproducible results.
    
    Returns:
        Image: Image with VHS effect applied.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Apply quality preset adjustments
    preset_multipliers = {
        'high': 0.3,
        'medium': 1.0,
        'low': 2.0,
        'damaged': 3.5
    }
    multiplier = preset_multipliers.get(quality_preset, 1.0)
    
    # Adjust parameters based on preset
    scan_line_intensity *= multiplier * 0.8
    static_intensity *= multiplier
    vertical_hold_frequency *= multiplier
    vertical_hold_intensity *= multiplier
    color_bleeding *= multiplier
    chroma_shift *= multiplier
    tracking_errors *= multiplier
    tape_wear *= multiplier
    head_switching_noise *= multiplier
    color_desaturation *= multiplier * 0.7
    brightness_variation *= multiplier * 0.6
    
    # Clamp values to valid ranges
    scan_line_intensity = min(1.0, scan_line_intensity)
    static_intensity = min(1.0, static_intensity)
    vertical_hold_frequency = min(1.0, vertical_hold_frequency)
    color_bleeding = min(1.0, color_bleeding)
    chroma_shift = min(1.0, chroma_shift)
    tracking_errors = min(1.0, tracking_errors)
    tape_wear = min(1.0, tape_wear)
    head_switching_noise = min(1.0, head_switching_noise)
    color_desaturation = min(1.0, color_desaturation)
    brightness_variation = min(1.0, brightness_variation)
    
    img_array = np.array(image, dtype=np.float32)
    height, width, channels = img_array.shape
    
    # 1. Apply color desaturation (VHS color degradation)
    if color_desaturation > 0:
        # Convert to grayscale and blend
        gray = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
        for i in range(3):
            img_array[:, :, i] = img_array[:, :, i] * (1 - color_desaturation) + gray * color_desaturation
    
    # 2. Apply chroma shift (Y/C separation)
    if chroma_shift > 0:
        shift_amount = int(chroma_shift * 3) + 1
        # Shift chroma channels slightly
        img_array[:, :, 0] = np.roll(img_array[:, :, 0], shift_amount, axis=1)  # Red shift right
        img_array[:, :, 2] = np.roll(img_array[:, :, 2], -shift_amount, axis=1)  # Blue shift left
    
    # 3. Apply color bleeding
    if color_bleeding > 0:
        from scipy import ndimage
        # Create bleeding effect by blurring individual channels differently
        blur_kernel = np.ones((1, 3)) / 3  # Horizontal blur
        for i in range(3):
            if np.random.random() < color_bleeding:
                img_array[:, :, i] = ndimage.convolve(img_array[:, :, i], blur_kernel, mode='wrap')
    
    # 4. Apply vertical hold issues (horizontal line displacement)
    if vertical_hold_frequency > 0 and vertical_hold_intensity > 0:
        num_glitches = int(height * vertical_hold_frequency * 0.1)
        for _ in range(num_glitches):
            glitch_line = np.random.randint(0, height)
            displacement = int((np.random.random() - 0.5) * vertical_hold_intensity * 2)
            if displacement != 0:
                # Displace the line horizontally
                img_array[glitch_line] = np.roll(img_array[glitch_line], displacement, axis=0)
    
    # 5. Apply tracking errors (horizontal bands of distortion)
    if tracking_errors > 0:
        num_bands = int(tracking_errors * 10) + 1
        for _ in range(num_bands):
            band_start = np.random.randint(0, height - 10)
            band_height = np.random.randint(2, 8)
            band_end = min(band_start + band_height, height)
            
            # Apply horizontal displacement to the band
            displacement = int((np.random.random() - 0.5) * 6)
            if displacement != 0:
                img_array[band_start:band_end] = np.roll(img_array[band_start:band_end], displacement, axis=1)
            
            # Add some noise to the band
            noise = np.random.normal(0, 10, (band_end - band_start, width, 3))
            img_array[band_start:band_end] += noise * tracking_errors
    
    # 5.5. Apply severe tracking errors (large displaced bands with color shifts)
    if tracking_errors > 0.3:  # Only apply for medium to high tracking error levels
        # Create larger, more dramatic tracking bands
        num_severe_bands = max(1, int(tracking_errors * 5))
        
        for _ in range(num_severe_bands):
            # Create larger bands (10-50 pixels high)
            band_start = np.random.randint(0, height - 20)
            band_height = np.random.randint(10, min(50, height - band_start))
            band_end = min(band_start + band_height, height)
            
            # Store original band data
            original_band = img_array[band_start:band_end].copy()
            
            # Apply large horizontal displacement (can be quite dramatic)
            large_displacement = int((np.random.random() - 0.5) * width * 0.3)  # Up to 30% of width
            
            if large_displacement != 0:
                # Displace the band
                displaced_band = np.roll(original_band, large_displacement, axis=1)
                
                # Apply color channel shifts to simulate bad sync
                color_shift_intensity = tracking_errors * 0.5
                
                # Randomly shift individual color channels
                if np.random.random() < color_shift_intensity:
                    # Red channel shift
                    red_shift = int((np.random.random() - 0.5) * 8)
                    displaced_band[:, :, 0] = np.roll(displaced_band[:, :, 0], red_shift, axis=1)
                
                if np.random.random() < color_shift_intensity:
                    # Blue channel shift (opposite direction often)
                    blue_shift = int((np.random.random() - 0.5) * 8)
                    displaced_band[:, :, 2] = np.roll(displaced_band[:, :, 2], blue_shift, axis=1)
                
                # Apply color bleeding/smearing effect
                if np.random.random() < 0.7:
                    # Horizontal blur to simulate color bleeding
                    from scipy import ndimage
                    blur_kernel = np.ones((1, 3)) / 3
                    for channel in range(3):
                        displaced_band[:, :, channel] = ndimage.convolve(
                            displaced_band[:, :, channel], blur_kernel, mode='wrap'
                        )
                
                # Apply brightness/contrast variations
                brightness_variation = (np.random.random() - 0.5) * 0.3 * tracking_errors
                contrast_variation = 1.0 + (np.random.random() - 0.5) * 0.4 * tracking_errors
                
                displaced_band = displaced_band * contrast_variation + brightness_variation * 255
                displaced_band = np.clip(displaced_band, 0, 255)
                
                # Blend the displaced band back with some transparency for realism
                blend_factor = 0.7 + np.random.random() * 0.3  # 70-100% opacity
                
                # Replace the original band with the processed version
                img_array[band_start:band_end] = (
                    displaced_band * blend_factor + 
                    original_band * (1 - blend_factor)
                )
                
                # Add some edge artifacts at band boundaries
                if band_start > 0:
                    # Add noise/distortion at top edge
                    edge_noise = np.random.normal(0, 15, (1, width, 3))
                    img_array[band_start:band_start+1] += edge_noise * tracking_errors * 0.5
                
                if band_end < height:
                    # Add noise/distortion at bottom edge
                    edge_noise = np.random.normal(0, 15, (1, width, 3))
                    img_array[band_end-1:band_end] += edge_noise * tracking_errors * 0.5
    
    # 6. Apply head switching noise (horizontal noise bands)
    if head_switching_noise > 0:
        # Add periodic horizontal noise bands
        band_frequency = 60  # Approximate scan lines where head switching occurs
        for y in range(0, height, band_frequency):
            if np.random.random() < head_switching_noise:
                noise_height = np.random.randint(1, 3)
                noise_end = min(y + noise_height, height)
                noise = np.random.normal(0, 30, (noise_end - y, width, 3))
                img_array[y:noise_end] += noise
    
    # 7. Apply tape wear (random dropouts)
    if tape_wear > 0:
        num_dropouts = int(tape_wear * width * height * 0.0001)
        for _ in range(num_dropouts):
            dropout_x = np.random.randint(0, width)
            dropout_y = np.random.randint(0, height)
            dropout_size = np.random.randint(1, 5)
            
            x_end = min(dropout_x + dropout_size, width)
            y_end = min(dropout_y + dropout_size, height)
            
            # Create dropout (dark spots or noise)
            if np.random.random() < 0.5:
                img_array[dropout_y:y_end, dropout_x:x_end] *= 0.1  # Dark dropout
            else:
                dropout_noise = np.random.normal(128, 50, (y_end - dropout_y, x_end - dropout_x, 3))
                img_array[dropout_y:y_end, dropout_x:x_end] = dropout_noise
    
    # 8. Apply static noise
    if static_intensity > 0:
        if static_type == 'white':
            noise = np.random.normal(0, static_intensity * 30, (height, width, 3))
        elif static_type == 'colored':
            noise = np.random.normal(0, static_intensity * 25, (height, width, 3))
            # Add color tint to noise
            noise[:, :, 0] *= 1.2  # More red noise
            noise[:, :, 2] *= 0.8  # Less blue noise
        else:  # mixed
            # Combine white and colored noise
            white_noise = np.random.normal(0, static_intensity * 20, (height, width, 3))
            colored_noise = np.random.normal(0, static_intensity * 15, (height, width, 3))
            colored_noise[:, :, 0] *= 1.3
            colored_noise[:, :, 2] *= 0.7
            noise = white_noise + colored_noise
        
        img_array += noise
    
    # 9. Apply brightness variation
    if brightness_variation > 0:
        # Create random brightness variations across the image
        variation_map = np.random.normal(1.0, brightness_variation * 0.1, (height, width))
        variation_map = np.clip(variation_map, 0.5, 1.5)
        for i in range(3):
            img_array[:, :, i] *= variation_map
    
    # 10. Apply scan lines (last to be most visible)
    if scan_line_intensity > 0:
        for y in range(0, height, scan_line_spacing):
            # Create scan line effect
            scan_line_darkness = 1.0 - scan_line_intensity
            img_array[y] *= scan_line_darkness
            
            # Add slight horizontal blur to scan lines for authenticity
            if scan_line_spacing > 1 and y + 1 < height:
                img_array[y + 1] *= (1.0 - scan_line_intensity * 0.3)
    
    # Ensure values are in valid range
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array) 

def sharpen_effect(image, method='unsharp_mask', intensity=1.0, radius=1.0, threshold=0,
                   edge_enhancement=0.0, high_pass_radius=3.0, custom_kernel=None):
    """
    Apply various sharpening effects to an image.
    
    Args:
        image (Image): PIL Image object to process.
        method (str): Sharpening method ('unsharp_mask', 'high_pass', 'edge_enhance', 'custom').
        intensity (float): Sharpening intensity/amount (0.0 to 5.0).
        radius (float): Radius for blur operations in unsharp mask (0.1 to 10.0).
        threshold (int): Threshold for unsharp mask (0 to 255).
        edge_enhancement (float): Additional edge enhancement (0.0 to 2.0).
        high_pass_radius (float): Radius for high-pass filter (1.0 to 10.0).
        custom_kernel (str): Custom convolution kernel type ('laplacian', 'sobel', 'prewitt').
    
    Returns:
        Image: Sharpened image.
    """
    if image.mode not in ['RGB', 'RGBA', 'L']:
        image = image.convert('RGB')
    
    # Ensure parameters are in valid ranges
    intensity = max(0.0, min(5.0, intensity))
    radius = max(0.1, min(10.0, radius))
    threshold = max(0, min(255, threshold))
    edge_enhancement = max(0.0, min(2.0, edge_enhancement))
    high_pass_radius = max(1.0, min(10.0, high_pass_radius))
    
    img_array = np.array(image, dtype=np.float32)
    
    if method == 'unsharp_mask':
        # Classic unsharp mask sharpening
        # 1. Create blurred version
        blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
        blurred_array = np.array(blurred, dtype=np.float32)
        
        # 2. Calculate difference (mask)
        mask = img_array - blurred_array
        
        # 3. Apply threshold if specified
        if threshold > 0:
            # Only sharpen where the difference exceeds threshold
            mask_magnitude = np.sqrt(np.sum(mask**2, axis=2, keepdims=True))
            threshold_mask = mask_magnitude > threshold
            mask = mask * threshold_mask
        
        # 4. Add scaled mask back to original
        result = img_array + mask * intensity
        
    elif method == 'high_pass':
        # High-pass filter sharpening
        # Create a strong blur and subtract from original
        heavily_blurred = image.filter(ImageFilter.GaussianBlur(radius=high_pass_radius))
        heavily_blurred_array = np.array(heavily_blurred, dtype=np.float32)
        
        # High-pass = original - low-pass
        high_pass = img_array - heavily_blurred_array
        
        # Add high-pass back to original with intensity scaling
        result = img_array + high_pass * intensity
        
    elif method == 'edge_enhance':
        # Use PIL's built-in edge enhancement as base
        if intensity <= 1.0:
            # Use smooth edge enhancement for subtle effects
            enhanced = image.filter(ImageFilter.EDGE_ENHANCE)
        else:
            # Use more aggressive edge enhancement
            enhanced = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        enhanced_array = np.array(enhanced, dtype=np.float32)
        
        # Blend between original and enhanced based on intensity
        blend_factor = min(1.0, intensity)
        result = img_array * (1 - blend_factor) + enhanced_array * blend_factor
        
        # If intensity > 1.0, apply additional sharpening
        if intensity > 1.0:
            extra_intensity = intensity - 1.0
            # Apply unsharp mask for additional sharpening
            blurred = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(radius=1.0)
            )
            blurred_array = np.array(blurred, dtype=np.float32)
            mask = result - blurred_array
            result = result + mask * extra_intensity
    
    elif method == 'custom':
        # Custom convolution kernel sharpening
        from scipy import ndimage
        
        if custom_kernel == 'laplacian':
            # Laplacian kernel for edge detection/sharpening
            kernel = np.array([
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]
            ], dtype=np.float32)
        elif custom_kernel == 'sobel':
            # Sobel-based sharpening (combine X and Y)
            sobel_x = np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ], dtype=np.float32)
            sobel_y = np.array([
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ], dtype=np.float32)
            # Combine Sobel X and Y for omnidirectional edge detection
            kernel = sobel_x + sobel_y
        elif custom_kernel == 'prewitt':
            # Prewitt operator
            prewitt_x = np.array([
                [-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1]
            ], dtype=np.float32)
            prewitt_y = np.array([
                [-1, -1, -1],
                [0, 0, 0],
                [1, 1, 1]
            ], dtype=np.float32)
            kernel = prewitt_x + prewitt_y
        else:  # Default to a simple sharpening kernel
            kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ], dtype=np.float32)
        
        # Apply convolution to each channel
        if len(img_array.shape) == 3:  # Color image
            result = np.zeros_like(img_array)
            for i in range(img_array.shape[2]):
                convolved = ndimage.convolve(img_array[:, :, i], kernel, mode='reflect')
                result[:, :, i] = img_array[:, :, i] + convolved * intensity
        else:  # Grayscale
            convolved = ndimage.convolve(img_array, kernel, mode='reflect')
            result = img_array + convolved * intensity
    
    else:
        # Fallback to simple sharpening
        result = img_array
    
    # Apply additional edge enhancement if requested
    if edge_enhancement > 0:
        from scipy import ndimage
        # Use Laplacian for edge detection
        edge_kernel = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        
        if len(result.shape) == 3:  # Color image
            for i in range(result.shape[2]):
                edges = ndimage.convolve(result[:, :, i], edge_kernel, mode='reflect')
                result[:, :, i] += edges * edge_enhancement
        else:  # Grayscale
            edges = ndimage.convolve(result, edge_kernel, mode='reflect')
            result += edges * edge_enhancement
    
    # Ensure values are in valid range
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return Image.fromarray(result) 