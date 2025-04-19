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
    elif manipulation_type == 'negative':
        # Create a negative by inverting all channels
        r = r.point(lambda i: 255 - i)
        g = g.point(lambda i: 255 - i)
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
                        color_theme='full-spectrum', decay_factor=0.0):
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
    
    Returns:
        Image: Processed image with color shift expansion effect.
    """
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
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
    
    # Create distance maps for each seed point
    distance_maps = []
    for point in seed_points:
        # Create a new distance map for each seed point
        distance_map = np.zeros((height, width), dtype=float)
        
        # Assign distances based on expansion type
        x0, y0 = point
        for y in range(height):
            for x in range(width):
                if expansion_type == 'square':
                    # Chebyshev distance (L∞ norm)
                    d = max(abs(x - x0), abs(y - y0))
                elif expansion_type == 'diamond':
                    # Manhattan distance (L1 norm)
                    d = abs(x - x0) + abs(y - y0)
                else:  # circle
                    # Euclidean distance (L2 norm)
                    d = np.sqrt((x - x0)**2 + (y - y0)**2)
                
                # Apply the distance
                distance_map[y, x] = d
                
        distance_maps.append(distance_map)
    
    # Process each pixel
    for y in range(height):
        for x in range(width):
            # Get the original pixel color
            original_r, original_g, original_b = image_np[y, x]
            
            # Convert to HSV for easier manipulation
            h, s, v = colorsys.rgb_to_hsv(original_r / 255.0, original_g / 255.0, original_b / 255.0)
            
            # Find the closest seed point and its distance
            closest_idx = 0
            min_dist = float('inf')
            
            # Find weighted influences from all points based on their distances
            total_influence = 0
            influences = []
            
            for i, distance_map in enumerate(distance_maps):
                distance = distance_map[y, x]
                
                # Check if this is the closest point
                if distance < min_dist:
                    min_dist = distance
                    closest_idx = i
                
                # Calculate influence based on distance and decay
                if decay_factor > 0:
                    # With decay, influence drops off with distance
                    influence = max(0.0, 1.0 - (decay_factor * distance / max_distance))
                else:
                    # Without decay, we use an inverse square relationship
                    influence = 1.0 / (1.0 + (distance / 50.0)**2)
                
                # Store the influence and add to total
                influences.append(influence)
                total_influence += influence
            
            # If no significant influence, keep original color
            if total_influence < 0.001:
                output_np[y, x] = image_np[y, x]
                continue
            
            # Normalize influences so they sum to 1
            influences = [inf / total_influence for inf in influences]
            
            # Calculate the weighted blend of all seed colors
            blend_r, blend_g, blend_b = 0, 0, 0
            for i, influence in enumerate(influences):
                if i < len(seed_colors):  # Safety check
                    seed_r, seed_g, seed_b = seed_colors[i]
                    blend_r += seed_r * influence
                    blend_g += seed_g * influence
                    blend_b += seed_b * influence
            
            # Convert the blend to HSV
            blend_h, blend_s, blend_v = colorsys.rgb_to_hsv(
                blend_r / 255.0, blend_g / 255.0, blend_b / 255.0)
            
            # Apply the shift amount to control the intensity of the effect
            # The higher the shift_amount, the more of the seed colors show through
            # Scale to provide good results in the 1-20 range
            shift_weight = min(0.85, shift_amount / 12.0)
            
            # Blend original and seed colors based on shift_weight
            final_h = h * (1 - shift_weight) + blend_h * shift_weight
            final_s = s * (1 - shift_weight) + (blend_s + saturation_boost) * shift_weight
            final_v = v * (1 - shift_weight) + (blend_v + value_boost) * shift_weight
            
            # Ensure saturation and value are in valid range
            final_s = min(1.0, max(0.0, final_s))
            final_v = min(1.0, max(0.0, final_v))
            
            # Convert back to RGB
            final_r, final_g, final_b = colorsys.hsv_to_rgb(final_h, final_s, final_v)
            
            # Store the final color
            output_np[y, x] = [
                int(final_r * 255), 
                int(final_g * 255), 
                int(final_b * 255)
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
        image = image.convert("RGB")
    
    # Split the image into its R, G, B channels
    r, g, b = image.split()
    
    # Create a mapping of input pixel values to posterized values
    # Formula: convert input [0-255] to [0-(levels-1)] then back to [0-255]
    lut = [int(((i * (levels - 1)) / 255) + 0.5) * (255 // (levels - 1)) for i in range(256)]
    
    # Apply the lookup table to each channel
    new_r = r.point(lut)
    new_g = g.point(lut)
    new_b = b.point(lut)
    
    # Merge the channels back into an RGB image
    new_image = Image.merge("RGB", (new_r, new_g, new_b))
    
    return new_image

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
    
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert image to NumPy array
    arr = np.array(image)
    
    # Convert RGB to HSV manually without scikit-image dependency
    r, g, b = arr[:, :, 0] / 255.0, arr[:, :, 1] / 255.0, arr[:, :, 2] / 255.0
    
    # RGB to HSV conversion
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    
    # Avoid division by zero
    delta = maxc - minc + 1e-10
    s = delta / (maxc + 1e-10)
    
    # Initialize h to zeros
    h = np.zeros_like(s)
    
    # Calculate hue
    rc = (maxc - r) / delta
    gc = (maxc - g) / delta
    bc = (maxc - b) / delta
    
    mask_r = (r == maxc)
    h[mask_r] = bc[mask_r] - gc[mask_r]
    
    mask_g = (g == maxc)
    h[mask_g] = 2.0 + rc[mask_g] - bc[mask_g]
    
    mask_b = (b == maxc)
    h[mask_b] = 4.0 + gc[mask_b] - rc[mask_b]
    
    h = (h / 6.0) % 1.0
    
    # Convert hue to degrees (0-360)
    H = h * 360.0
    
    # Compute the curve parameter p
    p = (C - 180) / 180.0
    
    # Compute the shift amount S for each pixel
    S_shift = A * np.exp(p * (H / 360.0 - 0.5))
    
    # Compute the new hue H'
    H_new = (H + S_shift) % 360.0
    
    # Convert back to [0, 1] range
    h_new = H_new / 360.0
    
    # HSV to RGB conversion
    i = np.floor(h_new * 6.0)
    f = h_new * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    
    # Initialize RGB arrays
    r_new = np.zeros_like(h_new)
    g_new = np.zeros_like(h_new)
    b_new = np.zeros_like(h_new)
    
    # Convert based on hue sector
    i = i.astype(int) % 6
    
    mask = (i == 0)
    r_new[mask], g_new[mask], b_new[mask] = v[mask], t[mask], p[mask]
    
    mask = (i == 1)
    r_new[mask], g_new[mask], b_new[mask] = q[mask], v[mask], p[mask]
    
    mask = (i == 2)
    r_new[mask], g_new[mask], b_new[mask] = p[mask], v[mask], t[mask]
    
    mask = (i == 3)
    r_new[mask], g_new[mask], b_new[mask] = p[mask], q[mask], v[mask]
    
    mask = (i == 4)
    r_new[mask], g_new[mask], b_new[mask] = t[mask], p[mask], v[mask]
    
    mask = (i == 5)
    r_new[mask], g_new[mask], b_new[mask] = v[mask], p[mask], q[mask]
    
    # Stack and convert back to uint8
    rgb_new = np.stack([r_new, g_new, b_new], axis=2)
    out_arr = (rgb_new * 255).astype(np.uint8)
    
    # Create a new PIL image
    new_img = Image.fromarray(out_arr)
    
    return new_img 