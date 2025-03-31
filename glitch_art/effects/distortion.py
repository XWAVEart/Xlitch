from PIL import Image
import numpy as np
import random
import cv2
import colorsys
import math
import noise

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

def Ripple(image, num_droplets=5, amplitude=10, frequency=0.1, decay=0.01, distortion_type="color_shift", distortion_params={}):
    """
    Apply a ripple effect to the image, simulating water droplets, with optional distortion effects.

    Args:
        image (Image): PIL Image object to process.
        num_droplets (int): Number of droplet centers (default: 5).
        amplitude (float): Strength of the ripple displacement (default: 10).
        frequency (float): Frequency of the ripple waves (default: 0.1).
        decay (float): Decay rate of the ripple amplitude with distance (default: 0.01).
        distortion_type (str): Type of distortion to apply ("color_shift", "pixelation", "none"; default: "color_shift").
        distortion_params (dict): Additional parameters for the distortion effect (default: {}).

    Returns:
        Image: The distorted image with ripple effect applied.
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