from PIL import Image
import numpy as np
import random
import cv2
import colorsys
import math
from PIL import ImageChops
from ..core.pixel_attributes import PixelAttributes
from ..utils.helpers import generate_noise_map

def pixel_drift(image, direction='down', num_bands=10, intensity=1.0):
    """
    Apply a pixel drift effect, shifting pixels in a specified direction with variable amounts
    to create a more glitchy effect. This version mimics the original PIL-based per-pixel logic.
    
    Args:
        image (Image): PIL Image object to process.
        direction (str): Direction of drift ('up', 'down', 'left', 'right').
        num_bands (int): Number of bands to divide the image into (more bands = more variation).
        intensity (float): Multiplier for the drift amount (higher = more extreme drift).
    
    Returns:
        Image: Processed image with drifted pixels.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    result_array = np.zeros_like(img_array) 
    
    base_drift = int(20 * intensity)
    
    # Ensure band_height/width are at least 1 to avoid division by zero if num_bands is very large or zero
    band_height = max(1, height // num_bands if num_bands > 0 else height)
    band_width = max(1, width // num_bands if num_bands > 0 else width)
    
    if direction == 'down' or direction == 'up':
        for y in range(height):
            band_index = y // band_height
            current_base_drift_for_row = base_drift
            if band_index % 3 == 0:
                current_base_drift_for_row = base_drift
            elif band_index % 3 == 1:
                current_base_drift_for_row = base_drift * 2
            else:
                current_base_drift_for_row = base_drift // 2
            
            drift_amount_for_row = current_base_drift_for_row
            if random.random() < 0.1: # 10% chance of a random shift for the entire row
                drift_amount_for_row = random.randint(5, int(max(5, 50 * intensity)))
                
            for x_coord in range(width): # Iterate through each pixel in the row
                if direction == 'down':
                    # Add height before modulo to ensure positive operand if y - drift_amount_for_row is negative
                    source_y = (y - drift_amount_for_row + height) % height 
                else:  # up
                    source_y = (y + drift_amount_for_row + height) % height
                result_array[y, x_coord] = img_array[source_y, x_coord]
    
    elif direction == 'left' or direction == 'right':
        for y_coord in range(height): # Iterate through each row
            for x in range(width):    # Iterate through each pixel in the row
                band_index = x // band_width
                current_base_drift_for_pixel = base_drift
                if band_index % 3 == 0:
                    current_base_drift_for_pixel = base_drift
                elif band_index % 3 == 1:
                    current_base_drift_for_pixel = base_drift * 2
                else:
                    current_base_drift_for_pixel = base_drift // 2
                
                drift_amount_for_pixel = current_base_drift_for_pixel
                if random.random() < 0.1: # 10% chance of a random shift for this pixel
                    drift_amount_for_pixel = random.randint(5, int(max(5, 50 * intensity)))
                    
                if direction == 'right':
                    # Add width before modulo for potentially negative result of x - drift_amount_for_pixel
                    source_x = (x - drift_amount_for_pixel + width) % width
                else:  # left
                    source_x = (x + drift_amount_for_pixel + width) % width
                result_array[y_coord, x] = img_array[y_coord, source_x]
    else:
        # Fallback if direction is invalid (should be caught by form validation)
        result_array = np.copy(img_array)
        
    return Image.fromarray(result_array)

def perlin_noise_displacement(image, scale=44.0, intensity=25, octaves=3, seed=None, mode='color_shift', displacement_type='vector_field'):
    """
    Applies a Perlin noise-based displacement to the image.

    Args:
        image (PIL.Image): PIL Image to process.
        scale (float): Scale of the Perlin noise.
        intensity (float): Maximum displacement in pixels.
        octaves (int): Number of layers of noise.
        persistence (float): Amplitude of each octave.
        lacunarity (float): Frequency of each octave.
        seed (int, optional): Seed for the Perlin noise generator. Defaults to None.
    
    Returns:
        PIL.Image: Displaced PIL Image.
    """
    width, height = image.size
    image_np = np.array(image)
    displaced_image = np.zeros_like(image_np)

    # Generate noise fields
    if seed is not None:
        base_seed = seed
    else:
        # If no seed, use different fixed bases for some variation, or could use random.randint
        base_seed = random.randint(0, 10000) # Ensure this random is seeded if global seed is None but determinism is desired from no-seed state

    if displacement_type == 'vector_field':
        # Use the imported generate_noise_map
        nx = generate_noise_map((height, width), scale, octaves, base=base_seed + 0)
        ny = generate_noise_map((height, width), scale, octaves, base=base_seed + 1)
        # For color shift, an optional third noise map for value/channel manipulation
        nz = generate_noise_map((height, width), scale, octaves, base=base_seed + 2) if mode == 'color_shift' else None
    elif displacement_type == 'multi_layered':
        nx = generate_noise_map((height, width), scale, octaves, base=base_seed + 0)
        ny = generate_noise_map((height, width), scale, octaves, base=base_seed + 1)
        nx2 = generate_noise_map((height, width), scale * 0.5, octaves + 1, base=base_seed + 3) # Finer detail
        ny2 = generate_noise_map((height, width), scale * 0.5, octaves + 1, base=base_seed + 4)
        nx3 = generate_noise_map((height, width), scale * 0.25, octaves + 2, base=base_seed + 5) # Even finer
        ny3 = generate_noise_map((height, width), scale * 0.25, octaves + 2, base=base_seed + 6)
        nz = None # Not typically used in multi_layered in the same way, but could be added
    else: # Default to 'simple' or handle as error
        nx = generate_noise_map((height, width), scale, octaves, base=base_seed + 0)
        ny = np.zeros_like(nx) # Simple horizontal displacement only
        nz = None

    # Normalize noise to range [-1, 1] (already done by helper)
    # nx = (nx - np.min(nx)) / (np.max(nx) - np.min(nx)) * 2 - 1
    # ny = (ny - np.min(ny)) / (np.max(ny) - np.min(ny)) * 2 - 1
    # if nz is not None:
    #     nz = (nz - np.min(nz)) / (np.max(nz) - np.min(nz)) * 2 - 1
    # if displacement_type == 'multi_layered':
    #     nx2 = (nx2 - np.min(nx2)) / (np.max(nx2) - np.min(nx2)) * 2 - 1
    #     ny2 = (ny2 - np.min(ny2)) / (np.max(ny2) - np.min(ny2)) * 2 - 1
    #     nx3 = (nx3 - np.min(nx3)) / (np.max(nx3) - np.min(nx3)) * 2 - 1
    #     ny3 = (ny3 - np.min(ny3)) / (np.max(ny3) - np.min(ny3)) * 2 - 1

    # Create coordinate grid
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    # Calculate source coordinates with displacement
    src_x_float = x_coords + nx * intensity
    src_y_float = y_coords + ny * intensity

    # Clip to image boundaries
    src_x_clipped = np.clip(src_x_float, 0, width - 1)
    src_y_clipped = np.clip(src_y_float, 0, height - 1)

    # Convert to integer for indexing
    src_x = src_x_clipped.astype(np.int32)
    src_y = src_y_clipped.astype(np.int32)

    # Apply displacement
    # Create an empty array for the displaced image
    displaced_image_data = np.zeros_like(image_np)
    # Map pixels from source to destination
    # This loop is a basic way to handle it; more advanced mapping like cv2.remap could be used
    # but direct indexing after int conversion should work.
    displaced_image_data[y_coords, x_coords] = image_np[src_y, src_x]

    return Image.fromarray(displaced_image_data)

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

def ripple_effect(image, num_droplets=5, amplitude=10, frequency=0.1, decay=0.01, distortion_type="color_shift", distortion_params={}, seed=None):
    """
    Apply a ripple effect to the image, simulating water droplets.
    
    Args:
        image (Image): PIL Image object to process.
        num_droplets (int): Number of ripple sources.
        amplitude (float): Maximum pixel displacement.
        frequency (float): Ripple frequency.
        decay (float): How quickly ripples fade with distance.
        distortion_type (str): Type of distortion to apply ('color_shift', 'displacement', 'pixelation', 'none').
        distortion_params (dict): Additional parameters for the distortion.
        seed (int, optional): Seed for random number generation. Defaults to None.
    
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

    # Set seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed) # Ensure both numpy and python random are seeded if used
    
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
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    height, width, channels = img_array.shape
    result_array = np.copy(img_array) # Start with a copy
    
    # Selection function setup (as previously refactored)
    if select_by == 'hue':
        selection_func = PixelAttributes.hue
    elif select_by == 'saturation':
        selection_func = lambda p: PixelAttributes.saturation(p) * 100.0
    elif select_by == 'luminance':
        selection_func = lambda p: PixelAttributes.luminance(p) * 100.0
    elif select_by == 'red':
        selection_func = lambda p: p[0]
    elif select_by == 'green':
        selection_func = lambda p: p[1]
    elif select_by == 'blue':
        selection_func = lambda p: p[2]
    elif select_by == 'contrast':
        selection_func = PixelAttributes.contrast
    else: 
        selection_func = lambda p: PixelAttributes.luminance(p) * 100.0

    if direction == 'horizontal':
        for y in range(height):
            row_data_np = img_array[y, :, :]
            row_pixel_tuples = [tuple(p) for p in row_data_np]
            
            selected_pixels_in_row = []
            original_values_in_row = [None] * width # Store original pixel or placeholder
            selection_mask_for_row = [False] * width

            for x, pixel_tuple in enumerate(row_pixel_tuples):
                value = selection_func(pixel_tuple)
                if min_val <= value <= max_val:
                    selected_pixels_in_row.append(pixel_tuple) # Store as tuple
                    selection_mask_for_row[x] = True
                    # original_values_in_row[x] remains None (placeholder)
                else:
                    original_values_in_row[x] = pixel_tuple # Store original if not selected
            
            if not selected_pixels_in_row: # No pixels selected in this row
                result_array[y, :, :] = row_data_np # Original row remains
                continue

            random.shuffle(selected_pixels_in_row)
            
            new_row_list = []
            shuffled_idx = 0
            for x in range(width):
                if selection_mask_for_row[x]: # This was a selected pixel
                    if shuffled_idx < len(selected_pixels_in_row):
                        new_row_list.append(selected_pixels_in_row[shuffled_idx])
                        shuffled_idx += 1
                    else: # Should not happen if logic is correct
                        new_row_list.append(original_values_in_row[x] if original_values_in_row[x] is not None else (0,0,0)) 
                else: # Not selected, use original value stored
                    new_row_list.append(original_values_in_row[x])
            
            result_array[y, :, :] = np.array(new_row_list, dtype=img_array.dtype).reshape(width, channels)

    elif direction == 'vertical':
        for x in range(width):
            col_data_np = img_array[:, x, :]
            col_pixel_tuples = [tuple(p) for p in col_data_np]
            
            selected_pixels_in_col = []
            original_values_in_col = [None] * height
            selection_mask_for_col = [False] * height

            for y, pixel_tuple in enumerate(col_pixel_tuples):
                value = selection_func(pixel_tuple)
                if min_val <= value <= max_val:
                    selected_pixels_in_col.append(pixel_tuple)
                    selection_mask_for_col[y] = True
                else:
                    original_values_in_col[y] = pixel_tuple

            if not selected_pixels_in_col:
                result_array[:, x, :] = col_data_np
                continue

            random.shuffle(selected_pixels_in_col)

            new_col_list = []
            shuffled_idx = 0
            for y in range(height):
                if selection_mask_for_col[y]:
                    if shuffled_idx < len(selected_pixels_in_col):
                        new_col_list.append(selected_pixels_in_col[shuffled_idx])
                        shuffled_idx += 1
                    else:
                        new_col_list.append(original_values_in_col[y] if original_values_in_col[y] is not None else (0,0,0))
                else:
                    new_col_list.append(original_values_in_col[y])
            
            result_array[:, x, :] = np.array(new_col_list, dtype=img_array.dtype).reshape(height, channels)
    
    return Image.fromarray(result_array)

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
def slice_shuffle(image, count, orientation, seed=None):
    """
    Apply the Slice Shuffle effect by dividing the image into a specified number of slices and shuffling them randomly.

    Args:
        image (PIL.Image): Input image to process.
        count (int): Number of slices (must be between 4 and 128).
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
        slices = np.array_split(image_np, count, axis=0)
        random.shuffle(slices)
        shuffled_np = np.vstack(slices)
    elif orientation == 'columns':
        slices = np.array_split(image_np, count, axis=1)
        random.shuffle(slices)
        shuffled_np = np.hstack(slices)
    else:
        # If orientation is unrecognized, return the original image
        return image

    return Image.fromarray(shuffled_np)

# Slice Offset Effect
def slice_offset(image, count, max_offset, orientation, offset_mode='random', sine_frequency=0.1, seed=None):
    """
    Apply the Slice Offset effect by dividing the image into a specified number of slices 
    and offsetting each slice horizontally or vertically using either random offsets or a sine wave pattern.

    Args:
        image (PIL.Image): Input image to process.
        count (int): Number of slices (must be between 4 and 128).
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
        slice_indices = np.arange(count)
        # Scale the sine wave to have amplitude of max_offset
        offsets = (np.sin(2 * np.pi * sine_frequency * slice_indices) * max_offset).astype(int)
    else:  # random mode (default)
        # Generate random offsets, between -max_offset and max_offset
        offsets = np.random.randint(-max_offset, max_offset + 1, count)
    
    # Process based on orientation
    if orientation == 'rows':
        # Split into horizontal slices, offset horizontally
        slices = np.array_split(image_np, count, axis=0)
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
        slices = np.array_split(image_np, count, axis=1)
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
def slice_reduction(image, count, reduction_value, orientation):
    """
    Apply the Slice Reduction effect by dividing the image into a specified number of slices
    and reordering them based on a reduction value.

    Args:
        image (PIL.Image): Input image to process.
        count (int): Number of slices (must be between 16 and 256).
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
        slices = np.array_split(image_np, count, axis=0)
    elif orientation == 'columns':
        slices = np.array_split(image_np, count, axis=1)
    else:
        # If orientation is unrecognized, return the original image
        return image
    
    # Create new order of slices based on reduction value
    new_order = []
    for offset in range(reduction_value):
        new_order.extend(range(offset, count, reduction_value))
    
    # Reorder slices according to new_order
    reordered_slices = [slices[i] for i in new_order]
    
    # Recombine the slices
    if orientation == 'rows':
        result_np = np.vstack(reordered_slices)
    else:  # columns
        result_np = np.hstack(reordered_slices)
    
    return Image.fromarray(result_np)

def block_shuffle(image, block_width, block_height, seed=None):
    """
    Divide the image into blocks of the given size, shuffle them randomly, and reassemble.
    Handles non-square and partial edge blocks. Optionally uses a seed for reproducibility.
    Args:
        image (PIL.Image): Input image.
        block_width (int): Width of each block.
        block_height (int): Height of each block.
        seed (int, optional): Random seed for reproducibility.
    Returns:
        PIL.Image: Image with shuffled blocks.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img_w, img_h = image.size
    image_np = np.array(image)
    
    source_blocks_for_shuffling = [] # Stores blocks, potentially padded to full block_width/height
    positions = [] # Top-left coordinates of each block slot in the original image grid

    num_channels = image_np.shape[2] if image_np.ndim == 3 else 0

    # Collect all blocks, padding them to be full size for the shuffle pool
    for y_start in range(0, img_h, block_height):
        for x_start in range(0, img_w, block_width):
            # Extract the original block (could be partial if at edge)
            original_block_data = image_np[y_start : min(y_start + block_height, img_h), 
                                           x_start : min(x_start + block_width, img_w)]
            orig_h, orig_w = original_block_data.shape[:2]

            # Create a full-sized padded block (e.g. with zeros/black)
            if num_channels > 0:
                padded_block_np = np.zeros((block_height, block_width, num_channels), dtype=image_np.dtype)
                padded_block_np[:orig_h, :orig_w, :] = original_block_data
            else: # Grayscale
                padded_block_np = np.zeros((block_height, block_width), dtype=image_np.dtype)
                padded_block_np[:orig_h, :orig_w] = original_block_data
            
            source_blocks_for_shuffling.append(padded_block_np)
            positions.append((y_start, x_start))

    # Shuffle the source_blocks_for_shuffling (by shuffling their indices)
    indices = list(range(len(source_blocks_for_shuffling)))
    random.shuffle(indices)

    # Create a new array for the output
    output = np.zeros_like(image_np) # Initialize with zeros, will be fully painted

    # Place shuffled blocks into their target positions
    for i in range(len(positions)):
        y_target, x_target = positions[i]
        
        # Get the (full-sized, padded) block that will go into this position
        # The actual source content for this block came from blocks[indices[i]] originally, 
        # but source_blocks_for_shuffling[indices[i]] is its padded version.
        block_to_place_padded = source_blocks_for_shuffling[indices[i]]
        
        # Determine the actual dimensions of the target slot in the output image
        # (this slot can be partial if it's at the image edge)
        actual_target_slot_h = min(block_height, img_h - y_target)
        actual_target_slot_w = min(block_width, img_w - x_target)
        
        # Crop the (full-sized, padded) source block to fit into the actual_target_slot dimensions
        cropped_block_for_slot = block_to_place_padded[:actual_target_slot_h, :actual_target_slot_w]
        
        # Place the cropped block into the output image
        output[y_target : y_target + actual_target_slot_h, 
               x_target : x_target + actual_target_slot_w] = cropped_block_for_slot

    return Image.fromarray(output)

def wave_distortion(image, wave_type='horizontal', amplitude=20.0, frequency=0.02, phase=0.0,
                   secondary_wave=False, secondary_amplitude=10.0, secondary_frequency=0.05,
                   secondary_phase=90.0, blend_mode='add', edge_behavior='wrap', 
                   interpolation='bilinear'):
    """
    Apply sine wave distortion to an image with comprehensive controls.
    
    Args:
        image (Image): PIL Image object to process.
        wave_type (str): Type of wave ('horizontal', 'vertical', 'both', 'diagonal', 'radial').
        amplitude (float): Wave amplitude in pixels (0.0 to 100.0).
        frequency (float): Wave frequency (0.001 to 0.1).
        phase (float): Wave phase offset in degrees (0.0 to 360.0).
        secondary_wave (bool): Enable secondary wave for complex patterns.
        secondary_amplitude (float): Secondary wave amplitude (0.0 to 100.0).
        secondary_frequency (float): Secondary wave frequency (0.001 to 0.1).
        secondary_phase (float): Secondary wave phase in degrees (0.0 to 360.0).
        blend_mode (str): How to combine waves ('add', 'multiply', 'max', 'interference').
        edge_behavior (str): How to handle edges ('wrap', 'clamp', 'reflect').
        interpolation (str): Interpolation method ('bilinear', 'nearest', 'bicubic').
    
    Returns:
        Image: Image with wave distortion applied.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image, dtype=np.float32)
    height, width, channels = img_array.shape
    
    # Convert phase from degrees to radians
    phase_rad = np.radians(phase)
    secondary_phase_rad = np.radians(secondary_phase)
    
    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Initialize displacement arrays
    dx = np.zeros((height, width), dtype=np.float32)
    dy = np.zeros((height, width), dtype=np.float32)
    
    # Calculate primary wave displacement
    if wave_type == 'horizontal':
        # Horizontal waves - vertical displacement based on x position
        primary_wave = np.sin(x_coords * frequency * 2 * np.pi + phase_rad) * amplitude
        dy += primary_wave
        
    elif wave_type == 'vertical':
        # Vertical waves - horizontal displacement based on y position
        primary_wave = np.sin(y_coords * frequency * 2 * np.pi + phase_rad) * amplitude
        dx += primary_wave
        
    elif wave_type == 'both':
        # Both horizontal and vertical waves
        horizontal_wave = np.sin(x_coords * frequency * 2 * np.pi + phase_rad) * amplitude
        vertical_wave = np.sin(y_coords * frequency * 2 * np.pi + phase_rad) * amplitude
        dy += horizontal_wave
        dx += vertical_wave
        
    elif wave_type == 'diagonal':
        # Diagonal wave pattern
        diagonal_coord = (x_coords + y_coords) / np.sqrt(2)
        primary_wave = np.sin(diagonal_coord * frequency * 2 * np.pi + phase_rad) * amplitude
        # Split displacement between x and y for diagonal effect
        dx += primary_wave * 0.707  # cos(45°)
        dy += primary_wave * 0.707  # sin(45°)
        
    elif wave_type == 'radial':
        # Radial waves from center
        center_x, center_y = width // 2, height // 2
        distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        primary_wave = np.sin(distance * frequency * 2 * np.pi + phase_rad) * amplitude
        
        # Calculate radial displacement
        angle = np.arctan2(y_coords - center_y, x_coords - center_x)
        dx += primary_wave * np.cos(angle)
        dy += primary_wave * np.sin(angle)
    
    # Add secondary wave if enabled
    if secondary_wave:
        if wave_type == 'horizontal':
            secondary_wave_pattern = np.sin(x_coords * secondary_frequency * 2 * np.pi + secondary_phase_rad) * secondary_amplitude
            if blend_mode == 'add':
                dy += secondary_wave_pattern
            elif blend_mode == 'multiply':
                dy *= (1 + secondary_wave_pattern / amplitude)
            elif blend_mode == 'max':
                dy = np.maximum(dy, secondary_wave_pattern)
            elif blend_mode == 'interference':
                # Create interference pattern
                dy += secondary_wave_pattern * np.cos(x_coords * frequency * np.pi)
                
        elif wave_type == 'vertical':
            secondary_wave_pattern = np.sin(y_coords * secondary_frequency * 2 * np.pi + secondary_phase_rad) * secondary_amplitude
            if blend_mode == 'add':
                dx += secondary_wave_pattern
            elif blend_mode == 'multiply':
                dx *= (1 + secondary_wave_pattern / amplitude)
            elif blend_mode == 'max':
                dx = np.maximum(dx, secondary_wave_pattern)
            elif blend_mode == 'interference':
                dx += secondary_wave_pattern * np.cos(y_coords * frequency * np.pi)
                
        elif wave_type == 'both':
            h_secondary = np.sin(x_coords * secondary_frequency * 2 * np.pi + secondary_phase_rad) * secondary_amplitude
            v_secondary = np.sin(y_coords * secondary_frequency * 2 * np.pi + secondary_phase_rad) * secondary_amplitude
            
            if blend_mode == 'add':
                dy += h_secondary
                dx += v_secondary
            elif blend_mode == 'multiply':
                dy *= (1 + h_secondary / amplitude)
                dx *= (1 + v_secondary / amplitude)
            elif blend_mode == 'interference':
                dy += h_secondary * np.cos(y_coords * frequency * np.pi)
                dx += v_secondary * np.cos(x_coords * frequency * np.pi)
                
        elif wave_type == 'diagonal':
            diagonal_coord = (x_coords + y_coords) / np.sqrt(2)
            secondary_wave_pattern = np.sin(diagonal_coord * secondary_frequency * 2 * np.pi + secondary_phase_rad) * secondary_amplitude
            
            if blend_mode == 'add':
                dx += secondary_wave_pattern * 0.707
                dy += secondary_wave_pattern * 0.707
            elif blend_mode == 'interference':
                dx += secondary_wave_pattern * 0.707 * np.cos(diagonal_coord * frequency * np.pi)
                dy += secondary_wave_pattern * 0.707 * np.cos(diagonal_coord * frequency * np.pi)
                
        elif wave_type == 'radial':
            center_x, center_y = width // 2, height // 2
            distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            secondary_wave_pattern = np.sin(distance * secondary_frequency * 2 * np.pi + secondary_phase_rad) * secondary_amplitude
            
            angle = np.arctan2(y_coords - center_y, x_coords - center_x)
            if blend_mode == 'add':
                dx += secondary_wave_pattern * np.cos(angle)
                dy += secondary_wave_pattern * np.sin(angle)
            elif blend_mode == 'interference':
                interference_factor = np.cos(distance * frequency * np.pi)
                dx += secondary_wave_pattern * np.cos(angle) * interference_factor
                dy += secondary_wave_pattern * np.sin(angle) * interference_factor
    
    # Calculate new coordinates
    new_x = x_coords + dx
    new_y = y_coords + dy
    
    # Handle edge behavior
    if edge_behavior == 'wrap':
        new_x = new_x % width
        new_y = new_y % height
    elif edge_behavior == 'clamp':
        new_x = np.clip(new_x, 0, width - 1)
        new_y = np.clip(new_y, 0, height - 1)
    elif edge_behavior == 'reflect':
        # Reflect coordinates at boundaries
        new_x = np.where(new_x < 0, -new_x, new_x)
        new_x = np.where(new_x >= width, 2 * (width - 1) - new_x, new_x)
        new_y = np.where(new_y < 0, -new_y, new_y)
        new_y = np.where(new_y >= height, 2 * (height - 1) - new_y, new_y)
        
        # Clamp after reflection to ensure bounds
        new_x = np.clip(new_x, 0, width - 1)
        new_y = np.clip(new_y, 0, height - 1)
    
    # Apply interpolation
    result_array = np.zeros_like(img_array)
    
    if interpolation == 'nearest':
        # Nearest neighbor interpolation
        new_x_int = np.round(new_x).astype(int)
        new_y_int = np.round(new_y).astype(int)
        
        # Ensure indices are within bounds
        new_x_int = np.clip(new_x_int, 0, width - 1)
        new_y_int = np.clip(new_y_int, 0, height - 1)
        
        result_array[y_coords, x_coords] = img_array[new_y_int, new_x_int]
        
    elif interpolation == 'bilinear':
        # Bilinear interpolation
        x0 = np.floor(new_x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(new_y).astype(int)
        y1 = y0 + 1
        
        # Clamp coordinates
        x0 = np.clip(x0, 0, width - 1)
        x1 = np.clip(x1, 0, width - 1)
        y0 = np.clip(y0, 0, height - 1)
        y1 = np.clip(y1, 0, height - 1)
        
        # Calculate interpolation weights
        wx = new_x - x0
        wy = new_y - y0
        
        # Perform bilinear interpolation for each channel
        for c in range(channels):
            result_array[y_coords, x_coords, c] = (
                img_array[y0, x0, c] * (1 - wx) * (1 - wy) +
                img_array[y0, x1, c] * wx * (1 - wy) +
                img_array[y1, x0, c] * (1 - wx) * wy +
                img_array[y1, x1, c] * wx * wy
            )
    
    elif interpolation == 'bicubic':
        # Simplified bicubic (using scipy if available, otherwise fallback to bilinear)
        try:
            from scipy import ndimage
            
            # Use scipy's map_coordinates for bicubic interpolation
            coordinates = np.array([new_y.ravel(), new_x.ravel()])
            
            for c in range(channels):
                mapped = ndimage.map_coordinates(
                    img_array[:, :, c], 
                    coordinates, 
                    order=3,  # Cubic interpolation
                    mode='wrap' if edge_behavior == 'wrap' else 'nearest',
                    prefilter=True
                )
                result_array[:, :, c] = mapped.reshape(height, width)
                
        except ImportError:
            # Fallback to bilinear if scipy not available
            x0 = np.floor(new_x).astype(int)
            x1 = x0 + 1
            y0 = np.floor(new_y).astype(int)
            y1 = y0 + 1
            
            x0 = np.clip(x0, 0, width - 1)
            x1 = np.clip(x1, 0, width - 1)
            y0 = np.clip(y0, 0, height - 1)
            y1 = np.clip(y1, 0, height - 1)
            
            wx = new_x - x0
            wy = new_y - y0
            
            for c in range(channels):
                result_array[y_coords, x_coords, c] = (
                    img_array[y0, x0, c] * (1 - wx) * (1 - wy) +
                    img_array[y0, x1, c] * wx * (1 - wy) +
                    img_array[y1, x0, c] * (1 - wx) * wy +
                    img_array[y1, x1, c] * wx * wy
                )
    
    # Ensure values are in valid range
    result_array = np.clip(result_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(result_array) 