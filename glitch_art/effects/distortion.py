from PIL import Image
import numpy as np
import random
import cv2
import colorsys
import math
import noise
from PIL import ImageChops

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

def perlin_noise_displacement(image, scale=100, intensity=30, octaves=6, persistence=0.5, lacunarity=2.0):
    """
    Applies a Perlin noise-based displacement to the image.

    Args:
        image (PIL.Image): PIL Image to process.
        scale (float): Scale of the Perlin noise.
        intensity (float): Maximum displacement in pixels.
        octaves (int): Number of layers of noise.
        persistence (float): Amplitude of each octave.
        lacunarity (float): Frequency of each octave.
    
    Returns:
        PIL.Image: Displaced PIL Image.
    """
    width, height = image.size
    image_np = np.array(image)
    displaced_image = np.zeros_like(image_np)

    # Generate Perlin noise for both x and y displacements
    perlin_x = np.zeros((height, width))
    perlin_y = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            perlin_x[y][x] = noise.pnoise2(
                x / scale,
                y / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=width,
                repeaty=height,
                base=0
            )
            perlin_y[y][x] = noise.pnoise2(
                (x + 100) / scale,  # Offset to generate different noise
                (y + 100) / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=width,
                repeaty=height,
                base=0
            )

    # Normalize noise to range [-1, 1]
    perlin_x = perlin_x / np.max(np.abs(perlin_x))
    perlin_y = perlin_y / np.max(np.abs(perlin_y))

    # Apply displacement
    # Vectorized approach for better performance
    displacement_x = (perlin_x * intensity).astype(int)
    displacement_y = (perlin_y * intensity).astype(int)

    # Create coordinate grids
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    # Calculate source coordinates with displacement
    src_x = np.clip(x_coords + displacement_x, 0, width - 1)
    src_y = np.clip(y_coords + displacement_y, 0, height - 1)

    # Apply displacement
    displaced_image = image_np[src_y, src_x]

    return Image.fromarray(displaced_image)

def geometric_distortion(image, scale=50.0, octaves=4, distortion_amount=20.0, distortion_type='opposite'):
    """
    Apply channel-specific geometric distortion using Perlin noise.
    
    Args:
        image (PIL.Image): Input RGB image.
        scale (float): Noise frequency (50.0 for broad patterns, lower for more detail).
        octaves (int): Noise detail level (1-8, higher = more detail).
        distortion_amount (float): Max pixel displacement (1.0-50.0).
        distortion_type (str): Distortion pattern ('opposite', 'radial', 'circular', 'random').
    
    Returns:
        PIL.Image: Distorted image with channel-specific warping.
    """
    # Convert PIL image to OpenCV format (BGR)
    cv_image = np.array(image)
    if cv_image.shape[2] == 4:  # Handle RGBA
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2RGB)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    height, width = cv_image.shape[:2]

    # Generate base Perlin noise maps for x and y displacements
    nx = generate_noise_map((height, width), scale, octaves, base=0)
    ny = generate_noise_map((height, width), scale, octaves, base=1)
    
    # Optional: additional noise map for radial or random patterns
    nz = generate_noise_map((height, width), scale, octaves, base=2)

    # Create coordinate grid
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Define displacement maps based on distortion type
    if distortion_type == 'opposite':
        # Opposite displacement for different channels
        r_map_x = (x + nx * distortion_amount).astype(np.float32)
        r_map_y = (y + ny * distortion_amount).astype(np.float32)
        g_map_x = (x - nx * distortion_amount).astype(np.float32)  # Opposite x direction
        g_map_y = (y + ny * distortion_amount).astype(np.float32)
        b_map_x = (x + nx * distortion_amount).astype(np.float32)
        b_map_y = (y - ny * distortion_amount).astype(np.float32)  # Opposite y direction
    
    elif distortion_type == 'radial':
        # Radial displacement pattern (outward/inward)
        center_x, center_y = width // 2, height // 2
        dx = x - center_x
        dy = y - center_y
        # Normalize distance from center
        dist = np.sqrt(dx**2 + dy**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        norm_dist = dist / max_dist
        # Calculate displacement direction
        angle = np.arctan2(dy, dx)
        # Red channel: outward displacement
        r_map_x = (x + np.cos(angle) * norm_dist * nx * distortion_amount).astype(np.float32)
        r_map_y = (y + np.sin(angle) * norm_dist * ny * distortion_amount).astype(np.float32)
        # Green channel: inward displacement
        g_map_x = (x - np.cos(angle) * norm_dist * nx * distortion_amount).astype(np.float32)
        g_map_y = (y - np.sin(angle) * norm_dist * ny * distortion_amount).astype(np.float32)
        # Blue channel: circular displacement
        b_map_x = (x + np.sin(angle) * norm_dist * nx * distortion_amount).astype(np.float32)
        b_map_y = (y - np.cos(angle) * norm_dist * ny * distortion_amount).astype(np.float32)
    
    elif distortion_type == 'circular':
        # Circular/swirl displacement
        center_x, center_y = width // 2, height // 2
        dx = x - center_x
        dy = y - center_y
        # Normalize distance from center
        dist = np.sqrt(dx**2 + dy**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        norm_dist = dist / max_dist
        # Calculate angle for swirl
        angle = np.arctan2(dy, dx)
        # Apply swirl with different strengths per channel
        swirl_factor = distortion_amount * 0.01
        r_map_x = (x + np.cos(angle + nx * swirl_factor) * dist * 0.05).astype(np.float32)
        r_map_y = (y + np.sin(angle + nx * swirl_factor) * dist * 0.05).astype(np.float32)
        g_map_x = (x + np.cos(angle - ny * swirl_factor) * dist * 0.05).astype(np.float32)
        g_map_y = (y + np.sin(angle - ny * swirl_factor) * dist * 0.05).astype(np.float32)
        b_map_x = (x + np.cos(angle + nz * swirl_factor) * dist * 0.05).astype(np.float32)
        b_map_y = (y + np.sin(angle + nz * swirl_factor) * dist * 0.05).astype(np.float32)
    
    else:  # 'random'
        # Random independent displacement for each channel
        r_map_x = (x + nx * distortion_amount).astype(np.float32)
        r_map_y = (y + ny * distortion_amount).astype(np.float32)
        # Generate additional noise maps for other channels
        nx2 = generate_noise_map((height, width), scale, octaves, base=3)
        ny2 = generate_noise_map((height, width), scale, octaves, base=4)
        nx3 = generate_noise_map((height, width), scale, octaves, base=5)
        ny3 = generate_noise_map((height, width), scale, octaves, base=6)
        g_map_x = (x + nx2 * distortion_amount).astype(np.float32)
        g_map_y = (y + ny2 * distortion_amount).astype(np.float32)
        b_map_x = (x + nx3 * distortion_amount).astype(np.float32)
        b_map_y = (y + ny3 * distortion_amount).astype(np.float32)

    # Split image into BGR channels
    b, g, r = cv2.split(cv_image)

    # Warp each channel using remap
    r_warped = cv2.remap(r, r_map_x, r_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    g_warped = cv2.remap(g, g_map_x, g_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    b_warped = cv2.remap(b, b_map_x, b_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Merge warped channels
    warped_image = cv2.merge([b_warped, g_warped, r_warped])

    # Convert back to RGB and return as PIL image
    warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(warped_image)

def generate_voronoi_seeds(image, num_seeds):
    """
    Generate random seed points for the Voronoi diagram.
    
    Args:
        image (PIL.Image): Input image.
        num_seeds (int): Number of seed points.
    
    Returns:
        np.ndarray: Array of seed points (shape: (num_seeds, 2)).
    """
    width, height = image.size
    seeds = np.random.randint(0, min(width, height), size=(num_seeds, 2))
    seeds[:, 0] = seeds[:, 0] % width
    seeds[:, 1] = seeds[:, 1] % height
    return seeds

def voronoi_distortion(image, num_seeds=100, distortion_amount=20.0):
    """
    Apply Voronoi-based geometric distortion to each color channel.
    
    Args:
        image (PIL.Image): Input RGB image.
        num_seeds (int): Number of Voronoi seeds (50-500).
        distortion_amount (float): Maximum displacement amount (1.0-50.0 pixels).
    
    Returns:
        PIL.Image: Distorted image with Voronoi-based warping.
    """
    # Convert PIL image to OpenCV format (BGR)
    cv_image = np.array(image)
    if cv_image.shape[2] == 4:  # Handle RGBA
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2RGB)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    height, width = cv_image.shape[:2]

    # Try to import scipy.spatial for Voronoi computation
    try:
        from scipy.spatial import Voronoi
        from scipy.ndimage import distance_transform_edt as edt
    except ImportError:
        raise ImportError("scipy is required for Voronoi distortion. Install with: pip install scipy")

    # Generate Voronoi seeds
    seeds = generate_voronoi_seeds(image, num_seeds)

    # Compute Voronoi diagram
    vor = Voronoi(seeds)

    # Create coordinate grid
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    pixel_coords = np.stack([x.flatten(), y.flatten()], axis=1)

    # Assign each pixel to its nearest seed (Voronoi region)
    distances = np.sqrt(((pixel_coords[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2))
    nearest_seed_indices = np.argmin(distances, axis=1)

    # Compute displacement maps
    displacement_x = np.zeros((height, width), dtype=np.float32)
    displacement_y = np.zeros((height, width), dtype=np.float32)
    for i in range(num_seeds):
        mask = (nearest_seed_indices == i).reshape(height, width)
        dist_map = edt(~mask)
        seed_x, seed_y = seeds[i]
        dx = seed_x - x
        dy = seed_y - y
        displacement_x += dx * dist_map
        displacement_y += dy * dist_map

    # Normalize displacement
    max_disp = np.max(np.sqrt(displacement_x**2 + displacement_y**2))
    if max_disp > 0:  # Avoid division by zero
        displacement_x /= max_disp
        displacement_y /= max_disp

    # Create channel-specific displacement maps
    r_map_x = (x + displacement_x * distortion_amount).astype(np.float32)
    r_map_y = (y + displacement_y * distortion_amount).astype(np.float32)
    g_map_x = (x - displacement_x * distortion_amount).astype(np.float32)
    g_map_y = (y + displacement_y * distortion_amount).astype(np.float32)
    b_map_x = (x + displacement_x * distortion_amount).astype(np.float32)
    b_map_y = (y - displacement_y * distortion_amount).astype(np.float32)

    # Split image into BGR channels
    b, g, r = cv2.split(cv_image)

    # Warp each channel
    r_warped = cv2.remap(r, r_map_x, r_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    g_warped = cv2.remap(g, g_map_x, g_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    b_warped = cv2.remap(b, b_map_x, b_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Merge warped channels
    warped_image = cv2.merge([b_warped, g_warped, r_warped])

    # Convert back to RGB and return as PIL image
    warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(warped_image)

def ripple_effect(image, num_droplets=5, amplitude=10, frequency=0.1, decay=0.01, distortion_type="color_shift", distortion_params={}):
    """
    Apply a ripple effect to the image, simulating water droplets.
    
    Args:
        image (Image): PIL Image object to process.
        num_droplets (int): Number of ripple sources.
        amplitude (float): Maximum pixel displacement.
        frequency (float): Ripple frequency.
        decay (float): How quickly ripples fade with distance.
        distortion_type (str): Type of distortion to apply ('color_shift', 'displacement').
        distortion_params (dict): Additional parameters for the distortion.
    
    Returns:
        Image: Processed image with ripple effect.
    """
    # Convert PIL Image to OpenCV format (BGR)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image)
    cv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Get image dimensions
    height, width = cv_image.shape[:2]
    
    # Create coordinate grids for x and y
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    
    # Initialize displacement maps
    dx = np.zeros((height, width), dtype=np.float32)
    dy = np.zeros((height, width), dtype=np.float32)
    
    # Generate random droplet centers
    droplet_centers = [(np.random.randint(0, width), np.random.randint(0, height)) for _ in range(num_droplets)]
    
    # Compute cumulative displacement from all droplets
    for cx, cy in droplet_centers:
        # Calculate distance from droplet center
        dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        # Prevent division by zero
        dist[dist < 1e-6] = 1e-6
        # Calculate displacement magnitude (sinusoidal ripple with exponential decay)
        mag = amplitude * np.sin(frequency * dist) * np.exp(-decay * dist)
        # Compute radial displacement components
        dx_droplet = mag * (x_coords - cx) / dist
        dy_droplet = mag * (y_coords - cy) / dist
        # Add to total displacement
        dx += dx_droplet
        dy += dy_droplet
    
    # Apply distortion based on type
    if distortion_type == "color_shift":
        # Extract parameters for color shift (default factors create slight channel separation)
        factor_b = distortion_params.get("factor_b", 1.0)
        factor_g = distortion_params.get("factor_g", 1.1)
        factor_r = distortion_params.get("factor_r", 0.9)
        
        # Split image into BGR channels
        b, g, r = cv2.split(cv_image)
        
        # Create displacement maps for each channel
        map_x_b = (x_coords - dx * factor_b).astype(np.float32)
        map_y_b = (y_coords - dy * factor_b).astype(np.float32)
        map_x_g = (x_coords - dx * factor_g).astype(np.float32)
        map_y_g = (y_coords - dy * factor_g).astype(np.float32)
        map_x_r = (x_coords - dx * factor_r).astype(np.float32)
        map_y_r = (y_coords - dy * factor_r).astype(np.float32)
        
        # Warp each channel separately
        b_warped = cv2.remap(b, map_x_b, map_y_b, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        g_warped = cv2.remap(g, map_x_g, map_y_g, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        r_warped = cv2.remap(r, map_x_r, map_y_r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Merge warped channels
        warped_image = cv2.merge([b_warped, g_warped, r_warped])
    else:
        # Default warping for other distortion types
        map_x = (x_coords - dx).astype(np.float32)
        map_y = (y_coords - dy).astype(np.float32)
        warped_image = cv2.remap(cv_image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        if distortion_type == "pixelation":
            # Compute displacement magnitude
            mag = np.sqrt(dx**2 + dy**2)
            # Normalize magnitude for blending (max_mag controls sensitivity)
            max_mag = distortion_params.get("max_mag", 10.0)
            blend_factor = np.clip(mag / max_mag, 0, 1)
            
            # Create pixelated version
            scale = distortion_params.get("scale", 10)
            small = cv2.resize(warped_image, (width // scale, height // scale), interpolation=cv2.INTER_NEAREST)
            pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # Blend pixelated image with warped image based on displacement magnitude
            blend_factor = blend_factor[:, :, np.newaxis]  # Add channel dimension for broadcasting
            warped_image = ((1 - blend_factor) * warped_image + blend_factor * pixelated).astype(np.uint8)
        elif distortion_type != "none":
            # Handle unknown distortion types gracefully
            print(f"Unknown distortion type: {distortion_type}. Applying default warping only.")
    
    # Convert back to RGB and return as PIL image
    warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(warped_image)

def pixel_scatter(image, direction, select_by, min_val, max_val):
    """
    Apply pixel scattering effect across an image based on pixel properties.
    
    Args:
        image (Image): PIL Image object to process.
        direction (str): 'horizontal' or 'vertical'.
        select_by (str): Pixel attribute to select by ('hue', 'saturation', 'luminance', 'red', 'green', 'blue', 'contrast').
        min_val (float): Minimum value for selection range.
        max_val (float): Maximum value for selection range.
    
    Returns:
        Image: Processed image with pixel scattering effect.
    """
    # Convert to RGB mode if the image is not already in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    width, height = image.size
    
    # Get all the pixels
    pixels = list(image.getdata())
    result_image = Image.new('RGB', (width, height))
    modified_pixels = []
    
    # Define attribute functions 
    # For different pixel properties
    def rgb_to_hue(pixel):
        r, g, b = [c / 255.0 for c in pixel[:3]]
        h, _, _ = colorsys.rgb_to_hsv(r, g, b)
        return h * 360  # Convert to 0-360 range
    
    def rgb_to_saturation(pixel):
        r, g, b = [c / 255.0 for c in pixel[:3]]
        _, s, _ = colorsys.rgb_to_hsv(r, g, b)
        return s * 100  # Convert to 0-100 range
    
    def rgb_to_luminance(pixel):
        r, g, b = [c / 255.0 for c in pixel[:3]]
        _, _, v = colorsys.rgb_to_hsv(r, g, b)
        return v * 100  # Convert to 0-100 range
    
    def calculate_contrast(pixel):
        return max(pixel[:3]) - min(pixel[:3])
    
    # Choose the selection function based on the select_by parameter
    if select_by == 'hue':
        selection_func = rgb_to_hue
    elif select_by == 'saturation':
        selection_func = rgb_to_saturation
    elif select_by == 'luminance':
        selection_func = rgb_to_luminance
    elif select_by == 'red':
        selection_func = lambda p: p[0]
    elif select_by == 'green':
        selection_func = lambda p: p[1]
    elif select_by == 'blue':
        selection_func = lambda p: p[2]
    elif select_by == 'contrast':
        selection_func = calculate_contrast
    else:
        # Default to luminance
        selection_func = rgb_to_luminance
    
    if direction == 'horizontal':
        # First pass: Identify pixels to scatter
        for y in range(height):
            row = []
            selected_pixels = []
            for x in range(width):
                idx = y * width + x
                pixel = pixels[idx]
                value = selection_func(pixel)
                if min_val <= value <= max_val:
                    # Save selected pixels
                    selected_pixels.append(pixel)
                    # Add placeholder
                    row.append(None)
                else:
                    # Keep unselected pixels in place
                    row.append(pixel)
            
            # Randomize the selected pixels
            random.shuffle(selected_pixels)
            
            # Second pass: Replace placeholders with shuffled pixels
            pixel_index = 0
            for x in range(width):
                if row[x] is None:
                    if pixel_index < len(selected_pixels):
                        row[x] = selected_pixels[pixel_index]
                        pixel_index += 1
                    else:
                        # If we run out of selected pixels (shouldn't happen)
                        row[x] = (0, 0, 0)  # Black as a fallback
            
            # Add the row to the modified pixels
            modified_pixels.extend(row)
    
    elif direction == 'vertical':
        # Initialize columns for vertical scattering
        columns = [[] for _ in range(width)]
        selected_columns = [[] for _ in range(width)]
        selected_masks = [[] for _ in range(width)]
        
        # First pass: Identify pixels to scatter, organize by columns
        for y in range(height):
            for x in range(width):
                idx = y * width + x
                pixel = pixels[idx]
                value = selection_func(pixel)
                
                if min_val <= value <= max_val:
                    # Mark for scatter
                    selected_columns[x].append(pixel)
                    selected_masks[x].append(True)
                    columns[x].append(None)  # Placeholder
                else:
                    # Keep in place
                    selected_masks[x].append(False)
                    columns[x].append(pixel)
        
        # Shuffle selected pixels in each column
        for x in range(width):
            random.shuffle(selected_columns[x])
        
        # Second pass: Replace placeholders with shuffled pixels
        for x in range(width):
            pixel_index = 0
            for y in range(height):
                if selected_masks[x][y]:
                    if pixel_index < len(selected_columns[x]):
                        columns[x][y] = selected_columns[x][pixel_index]
                        pixel_index += 1
                    else:
                        # Fallback (shouldn't happen)
                        columns[x][y] = (0, 0, 0)
        
        # Convert columns back to row-major format
        for y in range(height):
            for x in range(width):
                modified_pixels.append(columns[x][y])
    
    # Update the result image with modified pixels
    result_image.putdata(modified_pixels)
    return result_image

# Offset Effect
def offset_effect(image, offset_x=0, offset_y=0, unit_x='pixels', unit_y='pixels'):
    """
    Apply an offset effect to the image, shifting it horizontally and vertically.
    
    This function accepts offset values either as absolute pixels or as percentages of the image dimensions.
    The offset is applied with a wrap-around, so pixels that move off one edge reappear on the opposite edge.
    
    Args:
        image (PIL.Image): The input image to process.
        offset_x (float): Horizontal offset. If unit_x is 'percentage', this is interpreted as a percentage of the image width.
        offset_y (float): Vertical offset. If unit_y is 'percentage', this is interpreted as a percentage of the image height.
        unit_x (str): Unit for horizontal offset: 'pixels' (default) or 'percentage'.
        unit_y (str): Unit for vertical offset: 'pixels' (default) or 'percentage'.
    
    Returns:
        PIL.Image: The offset image.
    """
    width, height = image.size

    if unit_x == 'percentage':
        offset_px = int(width * (offset_x / 100.0))
    else:
        offset_px = int(offset_x)

    if unit_y == 'percentage':
        offset_py = int(height * (offset_y / 100.0))
    else:
        offset_py = int(offset_y)
        
    return ImageChops.offset(image, offset_px, offset_py)

# Slice Shuffle Effect
def slice_shuffle(image, slice_count, orientation, seed=None):
    """
    Apply the Slice Shuffle effect by dividing the image into a specified number of slices and shuffling them randomly.

    Args:
        image (PIL.Image): Input image to process.
        slice_count (int): Number of slices (must be between 4 and 128).
        orientation (str): Either 'rows' (to shuffle horizontal slices) or 'columns' (to shuffle vertical slices).
        seed (int, optional): Optional random seed for reproducibility.

    Returns:
        PIL.Image: Image with shuffled slices.
    """
    image_np = np.array(image)

    # Set seed if provided
    if seed is not None:
        random.seed(seed)

    # Split image into slices along the specified axis
    if orientation == 'rows':
        slices = np.array_split(image_np, slice_count, axis=0)
        random.shuffle(slices)
        shuffled_np = np.vstack(slices)
    elif orientation == 'columns':
        slices = np.array_split(image_np, slice_count, axis=1)
        random.shuffle(slices)
        shuffled_np = np.hstack(slices)
    else:
        # If orientation is unrecognized, return the original image
        return image

    return Image.fromarray(shuffled_np)

# Slice Offset Effect
def slice_offset(image, slice_count, max_offset, orientation, offset_mode='random', sine_frequency=0.1, seed=None):
    """
    Apply the Slice Offset effect by dividing the image into a specified number of slices 
    and offsetting each slice horizontally or vertically using either random offsets or a sine wave pattern.

    Args:
        image (PIL.Image): Input image to process.
        slice_count (int): Number of slices (must be between 4 and 128).
        max_offset (int): Maximum offset in pixels (positive or negative, up to 512).
        orientation (str): Either 'rows' (horizontal slices with horizontal offsets) 
                           or 'columns' (vertical slices with vertical offsets).
        offset_mode (str): Either 'random' (random offsets) or 'sine' (sine wave pattern).
        sine_frequency (float): Frequency of the sine wave when using sine mode (0.01-1.0).
        seed (int, optional): Optional random seed for reproducibility when using random mode.

    Returns:
        PIL.Image: Image with offset slices.
    """
    # Convert to NumPy array
    image_np = np.array(image)
    height, width = image_np.shape[:2]
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Generate offsets for each slice
    if offset_mode == 'sine':
        # Generate sine wave offsets
        slice_indices = np.arange(slice_count)
        # Scale the sine wave to have amplitude of max_offset
        offsets = (np.sin(2 * np.pi * sine_frequency * slice_indices) * max_offset).astype(int)
    else:  # random mode (default)
        # Generate random offsets, between -max_offset and max_offset
        offsets = np.random.randint(-max_offset, max_offset + 1, slice_count)
    
    # Process based on orientation
    if orientation == 'rows':
        # Split into horizontal slices, offset horizontally
        slices = np.array_split(image_np, slice_count, axis=0)
        result = np.zeros_like(image_np)
        
        # Process each slice
        for i, slice_data in enumerate(slices):
            slice_height = slice_data.shape[0]
            y_start = i * slice_height
            y_end = min(y_start + slice_height, height)
            
            # Apply horizontal offset to each row
            offset_x = offsets[i]
            
            # For each row in this slice, shift it horizontally
            for y in range(y_start, y_end):
                row = image_np[y]
                if offset_x > 0:
                    # Shift right with wraparound
                    result[y] = np.roll(row, offset_x, axis=0)
                elif offset_x < 0:
                    # Shift left with wraparound
                    result[y] = np.roll(row, offset_x, axis=0)
                else:
                    # No offset
                    result[y] = row
            
    elif orientation == 'columns':
        # Split into vertical slices, offset vertically
        slices = np.array_split(image_np, slice_count, axis=1)
        result = np.zeros_like(image_np)
        
        # Process each slice
        for i, slice_data in enumerate(slices):
            slice_width = slice_data.shape[1]
            x_start = i * slice_width
            x_end = min(x_start + slice_width, width)
            
            # Apply vertical offset to each column
            offset_y = offsets[i]
            
            # Extract this slice
            slice_columns = image_np[:, x_start:x_end]
            
            # Apply vertical offset with wraparound
            if offset_y != 0:
                shifted_columns = np.roll(slice_columns, offset_y, axis=0)
            else:
                shifted_columns = slice_columns
            
            # Place the shifted slice back into the result
            result[:, x_start:x_end] = shifted_columns
    
    else:
        # If orientation is unrecognized, return the original image
        return image

    return Image.fromarray(result)

# Slice Reduction Effect
def slice_reduction(image, slice_count, reduction_value, orientation):
    """
    Apply the Slice Reduction effect by dividing the image into a specified number of slices
    and reordering them based on a reduction value.

    Args:
        image (PIL.Image): Input image to process.
        slice_count (int): Number of slices (must be between 16 and 256).
        reduction_value (int): The reduction value (must be between 2 and 8) that determines
                              how slices are reordered.
        orientation (str): Either 'rows' (to process horizontal slices) or 'columns' (to process vertical slices).

    Returns:
        PIL.Image: Image with reordered slices.
    """
    # Convert to NumPy array
    image_np = np.array(image)
    
    # Split image into slices along the specified axis
    if orientation == 'rows':
        slices = np.array_split(image_np, slice_count, axis=0)
    elif orientation == 'columns':
        slices = np.array_split(image_np, slice_count, axis=1)
    else:
        # If orientation is unrecognized, return the original image
        return image
    
    # Create new order of slices based on reduction value
    new_order = []
    for offset in range(reduction_value):
        new_order.extend(range(offset, slice_count, reduction_value))
    
    # Reorder slices according to new_order
    reordered_slices = [slices[i] for i in new_order]
    
    # Recombine the slices
    if orientation == 'rows':
        result_np = np.vstack(reordered_slices)
    else:  # columns
        result_np = np.hstack(reordered_slices)
    
    return Image.fromarray(result_np) 