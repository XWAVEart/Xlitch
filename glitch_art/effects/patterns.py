from PIL import Image, ImageDraw
import numpy as np
import math
import random
import colorsys
from scipy.spatial import Voronoi, cKDTree
from scipy.ndimage import distance_transform_edt as edt
from ..core.pixel_attributes import PixelAttributes

def voronoi_pixel_sort(image, num_cells=100, size_variation=0.5, sort_by='color', sort_order='clockwise', seed=None, orientation='horizontal', start_position='left'):
    """
    Applies a Voronoi-based pixel sorting effect to the image.

    Args:
        image (Image): The PIL Image to process.
        num_cells (int): Approximate number of Voronoi cells.
        size_variation (float): Variability in cell sizes (0 to 1).
        sort_by (str): Property to sort by ('color', 'brightness', 'hue').
        sort_order (str): Sorting order ('clockwise' or 'counter-clockwise').
        seed (int): Random seed for reproducibility.
        orientation (str): Direction for sorting within cells ('horizontal', 'vertical', 'radial', 'spiral').
        start_position (str): Starting position for the sort ('left', 'right', 'top', 'bottom', 'center').
    
    Returns:
        Image: The processed image.
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert image to NumPy array
    image_np = np.array(image)
    height, width, channels = image_np.shape

    # Generate seed points for Voronoi cells
    num_points = num_cells
    # Adjust for size variation
    variation = int(num_points * size_variation)
    point_counts = num_points + random.randint(-variation, variation)
    xs = np.random.randint(0, width, size=point_counts)
    ys = np.random.randint(0, height, size=point_counts)
    points = np.vstack((xs, ys)).T

    # Assign each pixel to the nearest seed point
    # Create a grid of pixel coordinates
    xv, yv = np.meshgrid(np.arange(width), np.arange(height))
    pixel_coords = np.column_stack((xv.ravel(), yv.ravel()))

    # Build a KD-Tree for efficient nearest-neighbor search
    tree = cKDTree(points)
    _, regions = tree.query(pixel_coords)

    # Reshape regions to the image shape
    regions = regions.reshape((height, width))

    # Define sorting functions based on PixelAttributes class
    sort_function = {
        'color': lambda p: np.sum(p[:3], dtype=int),
        'brightness': PixelAttributes.brightness,
        'hue': PixelAttributes.hue,
        'saturation': PixelAttributes.saturation,
        'luminance': PixelAttributes.luminance,
        'contrast': PixelAttributes.contrast,
        'red': lambda p: p[0],
        'green': lambda p: p[1],
        'blue': lambda p: p[2]
    }.get(sort_by, lambda p: np.sum(p[:3], dtype=int))

    # Process each cell
    for region_id in np.unique(regions):
        # Get the mask for the current region
        mask = regions == region_id

        # Get the coordinates of pixels in the cell
        ys_cell, xs_cell = np.where(mask)

        # Get the pixels in the cell
        pixels = image_np[ys_cell, xs_cell]

        # Skip regions with only one pixel
        if len(pixels) <= 1:
            continue

        # Compute the centroid of the cell
        centroid_x = np.mean(xs_cell)
        centroid_y = np.mean(ys_cell)

        # Sort the pixels based on the sorting property
        pixel_values = np.array([sort_function(p) for p in pixels])
        pixel_order = np.argsort(pixel_values)
        sorted_pixels = pixels[pixel_order]
        
        # Determine position ordering based on orientation parameter
        if orientation == 'horizontal':
            # Order pixels horizontally
            if start_position == 'left':
                # Left to right (default)
                position_order = np.argsort(xs_cell)
            elif start_position == 'right':
                # Right to left
                position_order = np.argsort(-xs_cell)
            elif start_position == 'center':
                # Order from center horizontally outward
                dist_from_center_x = np.abs(xs_cell - centroid_x)
                position_order = np.argsort(-dist_from_center_x) # Start from periphery
            else:
                # Default to left-to-right
                position_order = np.argsort(xs_cell)
                
        elif orientation == 'vertical':
            # Order pixels vertically
            if start_position == 'top':
                # Top to bottom
                position_order = np.argsort(ys_cell)
            elif start_position == 'bottom':
                # Bottom to top
                position_order = np.argsort(-ys_cell)
            elif start_position == 'center':
                # Order from center vertically outward
                dist_from_center_y = np.abs(ys_cell - centroid_y)
                position_order = np.argsort(-dist_from_center_y) # Start from periphery
            else:
                # Default to top-to-bottom
                position_order = np.argsort(ys_cell)
                
        elif orientation == 'radial':
            # Compute distance from centroid
            dist_from_center = np.sqrt((xs_cell - centroid_x)**2 + (ys_cell - centroid_y)**2)
            
            if start_position == 'center':
                # Order from center outward
                position_order = np.argsort(dist_from_center)
            else:
                # Order from edge inward
                position_order = np.argsort(-dist_from_center)
                
        elif orientation == 'spiral':
            # Calculate angle for each pixel relative to the centroid
            angles = np.arctan2(ys_cell - centroid_y, xs_cell - centroid_x)
            # Calculate distance from centroid
            dist_from_center = np.sqrt((xs_cell - centroid_x)**2 + (ys_cell - centroid_y)**2)
            
            # Combine angle and distance for a spiral ordering
            # Scale angles to [0, 10] for appropriate weight
            scaled_angles = (angles + np.pi) / (2 * np.pi) * 10
            # Scale distances to [0, 1] relative to max distance
            max_dist = np.max(dist_from_center) if np.max(dist_from_center) > 0 else 1
            scaled_dist = dist_from_center / max_dist
            
            # Combine for spiral: angle + weight*distance
            spiral_metric = scaled_angles + scaled_dist
            
            if sort_order == 'clockwise':
                position_order = np.argsort(spiral_metric)
            else:
                position_order = np.argsort(-spiral_metric)
                
        else:
            # Default: sort by angle around centroid (original behavior)
            angles = np.arctan2(ys_cell - centroid_y, xs_cell - centroid_x)
            if sort_order == 'clockwise':
                angles = -angles  # Reverse the angle for clockwise sorting
            position_order = np.argsort(angles)

        # Get the sorted coordinates based on position_order
        sorted_ys = ys_cell[position_order]
        sorted_xs = xs_cell[position_order]

        # Assign sorted pixels to positions
        image_np[sorted_ys, sorted_xs] = sorted_pixels

    # Reset random seeds
    if seed is not None:
        np.random.seed(None)
        random.seed(None)

    # Convert back to PIL Image
    processed_image = Image.fromarray(image_np)
    return processed_image

def masked_merge(image, secondary_image, mask_type='checkerboard', width=32, height=32, random_seed=None, stripe_width=16, stripe_angle=45, perlin_noise_scale=0.1, threshold=0.5, voronoi_cells=50):
    """
    Merge two images using a mask pattern.
    
    Args:
        image (PIL.Image): Primary image.
        secondary_image (PIL.Image): Secondary image to merge with.
        mask_type (str): Type of mask ('checkerboard', 'random_checkerboard', 'striped', 'gradient_striped', 'perlin', 'voronoi').
        width (int): Width of the squares/rectangles in the checkerboard mask pattern.
        height (int): Height of the squares/rectangles in the checkerboard mask pattern.
        random_seed (int): Seed for random generation (for 'random_checkerboard' mask type) or Perlin noise (for 'perlin' mask type)
        stripe_width (int): Width of stripes in pixels (for 'striped' and 'gradient_striped' mask types).
        stripe_angle (int): Angle of stripes in degrees (for 'striped' and 'gradient_striped' mask types).
        perlin_noise_scale (float): Scale of the perlin noise (for 'perlin' mask type).
        threshold (float): Threshold for the perlin noise (for 'perlin' mask type).
        voronoi_cells (int): Number of Voronoi cells for the voronoi mask type.
    
    Returns:
        PIL.Image: Merged image.
    """
    # Ensure images are the same size
    if image.size != secondary_image.size:
        secondary_image = secondary_image.resize(image.size, Image.LANCZOS)
    
    img_width, img_height = image.size
    
    # Create mask based on mask_type
    mask = Image.new('L', image.size, 0)
    
    if mask_type == 'checkerboard':
        # Draw a regular checkerboard pattern
        draw = ImageDraw.Draw(mask)
        
        for y in range(0, img_height, height):
            for x in range(0, img_width, width):
                if ((x // width) + (y // height)) % 2 == 0:
                    draw.rectangle([x, y, x + width - 1, y + height - 1], fill=255)
    
    elif mask_type == 'random_checkerboard':
        # Draw a random checkerboard pattern
        if random_seed is not None:
            random.seed(random_seed)
        
        draw = ImageDraw.Draw(mask)
        
        for y in range(0, img_height, height):
            for x in range(0, img_width, width):
                if random.random() > 0.5:
                    draw.rectangle([x, y, x + width - 1, y + height - 1], fill=255)
    
    elif mask_type == 'striped':
        # Draw stripes at the specified angle with consistent width
        img_width, img_height = image.size
        
        # Convert angle to radians
        angle_rad = math.radians(stripe_angle)
        
        # Calculate normal vector (perpendicular to stripe direction)
        nx = math.cos(angle_rad)
        ny = math.sin(angle_rad)
        
        # Create coordinate arrays
        y_coords, x_coords = np.mgrid[:img_height, :img_width]
        
        # Calculate the perpendicular distance for each pixel using vectorization
        distances = x_coords * nx + y_coords * ny
        
        # Create stripes with consistent width using modulo operation
        mask_array = np.zeros((img_height, img_width), dtype=np.uint8)
        mask_array[np.int32(distances / stripe_width) % 2 == 0] = 255
        
        # Convert the mask array to a PIL image
        mask = Image.fromarray(mask_array)
    
    elif mask_type == 'gradient_striped':
        # Draw gradient stripes at the specified angle
        img_width, img_height = image.size
        
        # Convert angle to radians
        angle_rad = math.radians(stripe_angle)
        
        # Calculate normal vector (perpendicular to stripe direction)
        nx = math.cos(angle_rad)
        ny = math.sin(angle_rad)
        
        # Create coordinate arrays
        y_coords, x_coords = np.mgrid[:img_height, :img_width]
        
        # Calculate the perpendicular distance for each pixel using vectorization
        distances = x_coords * nx + y_coords * ny
        
        # Create gradient stripes using cosine for smooth transitions
        mask_array = np.zeros((img_height, img_width), dtype=np.uint8)
        
        # Scale distances to appropriate frequency
        normalized_distances = 2 * np.pi * distances / (2 * stripe_width)
        
        # Generate smooth transitions with cosine
        mask_array = np.uint8(((np.cos(normalized_distances) + 1) / 2) * 255)
        
        # Convert the mask array to a PIL image
        mask = Image.fromarray(mask_array)
    
    elif mask_type == 'perlin':
        # Create a perlin noise pattern
        if random_seed is not None:
            np.random.seed(random_seed)
        
        try:
            import noise
            # Generate perlin noise
            noise_map = np.zeros((img_height, img_width), dtype=np.float32)
            for y in range(img_height):
                for x in range(img_width):
                    noise_map[y, x] = noise.pnoise2(x / (img_width * perlin_noise_scale), 
                                                   y / (img_height * perlin_noise_scale), 
                                                   octaves=6, 
                                                   base=random_seed or 42)
        except ImportError:
            # Fallback to random noise if noise module is not available
            print("Warning: noise module not found, using random noise instead")
            noise_map = np.random.rand(img_height, img_width)
        
        # Apply threshold to create binary mask
        mask_array = np.zeros((img_height, img_width), dtype=np.uint8)
        mask_array[noise_map > threshold] = 255
        
        # Convert the mask array to a PIL image
        mask = Image.fromarray(mask_array)
        
    elif mask_type == 'voronoi':
        # Create a Voronoi pattern with straight edges
        if random_seed is not None:
            np.random.seed(random_seed)
        else:
            np.random.seed(42)  # Default seed for reproducibility
        
        # Generate random points for Voronoi cells
        num_points = voronoi_cells
        
        # Using a stratified approach for better point distribution
        # Divide image into grid sectors for more evenly distributed cells
        grid_size = int(np.sqrt(num_points)) + 1
        grid_width = img_width / grid_size
        grid_height = img_height / grid_size
        
        points = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) < num_points:  # Stop once we have enough points
                    # Add some randomness within each grid cell
                    x = (j + np.random.rand()) * grid_width
                    y = (i + np.random.rand()) * grid_height
                    points.append([x, y])
        
        # Convert to numpy array and shuffle to break any remaining grid patterns
        points = np.array(points)
        np.random.shuffle(points)
        points = points[:num_points]  # Ensure we have exactly num_points
        
        # Create a mask array
        mask_array = np.zeros((img_height, img_width), dtype=np.uint8)
        
        # Generate a random binary mask for each cell
        cell_mask = np.random.randint(0, 2, num_points) * 255
        
        # Pre-allocate arrays for vectorized operations
        y_coords, x_coords = np.mgrid[:img_height, :img_width]
        
        # Process the image more efficiently using vectorization
        # Creates a distance matrix of shape (height*width, num_points)
        # This is memory-intensive but much faster than pixel-by-pixel calculation
        for i in range(0, img_height, 128):  # Process in horizontal slices to conserve memory
            end_i = min(i + 128, img_height)
            slice_height = end_i - i
            
            # Extract pixel coordinates for this slice
            slice_y = y_coords[i:end_i, :]
            slice_x = x_coords[i:end_i, :]
            pixel_coords = np.column_stack((slice_x.ravel(), slice_y.ravel()))
            
            # Calculate distances from each pixel to all points
            # Mix of Manhattan and Euclidean for angular but not strictly grid-aligned cells
            # Lambda parameter controls the mix (0 = Manhattan, 1 = Euclidean)
            lambda_param = 0.7  # Adjust for more or less angular cells
            
            # Manhattan component
            manhattan_dist = np.sum(np.abs(pixel_coords[:, np.newaxis, :] - points[np.newaxis, :, :]), axis=2)
            
            # Euclidean component
            delta = pixel_coords[:, np.newaxis, :] - points[np.newaxis, :, :]
            euclidean_dist = np.sqrt(np.sum(delta**2, axis=2))
            
            # Weighted combination
            distances = (1 - lambda_param) * manhattan_dist + lambda_param * euclidean_dist
            
            # Find the closest point for each pixel
            closest_points = np.argmin(distances, axis=1)
            
            # Apply the cell assignments to the mask
            mask_array[i:end_i, :].flat = cell_mask[closest_points]
        
        # Convert the mask array to a PIL image
        mask = Image.fromarray(mask_array)
    
    # Merge the images using the mask
    merged_image = Image.composite(image, secondary_image, mask)
    
    return merged_image

def concentric_shapes(image, num_points=5, shape_type='circle', thickness=3, spacing=10,
                      rotation_angle=0, darken_step=0, color_shift=0):
    """
    Generates concentric shapes from random points in the image.

    Args:
        image (Image): The PIL Image to process.
        num_points (int): Number of random pixels to select.
        shape_type (str): Type of shape ('square', 'circle', 'hexagon', 'triangle').
        thickness (int): Thickness of the shapes in pixels.
        spacing (int): Spacing between shapes in pixels.
        rotation_angle (int): Incremental rotation angle in degrees for each subsequent shape.
        darken_step (int): Amount to darken the color for each subsequent shape (0-255).
        color_shift (int): Amount to shift the hue for each shape (0-360 degrees).
    
    Returns:
        Image: The processed image.
    """
    width, height = image.size
    image = image.convert('RGBA')  # Ensure image has an alpha channel

    # Create a base image to draw on
    base_image = image.copy()

    # Select random points
    xs = np.random.randint(0, width, size=num_points)
    ys = np.random.randint(0, height, size=num_points)
    points = list(zip(xs, ys))

    # For each point
    for x0, y0 in points:
        # Get the color of the pixel
        original_color = base_image.getpixel((x0, y0))

        # Initialize variables
        current_size = spacing
        current_rotation = 0  # Initialize cumulative rotation

        # Initialize HSV color from the original color
        r, g, b = original_color[:3]
        h_original, s_original, v_original = rgb_to_hsv(r, g, b)
        current_hue = h_original  # Start with the original hue

        max_dimension = max(width, height) * 1.5  # Set a maximum size to prevent infinite loops

        while current_size < max_dimension:
            # Adjust the hue for the current shape
            if color_shift != 0:
                current_hue = (current_hue + color_shift) % 360
            # Convert HSV back to RGB
            current_color = hsv_to_rgb(current_hue, s_original, v_original)

            # Darken the color if darken_step is set
            if darken_step != 0:
                current_color = darken_color(current_color, darken_step)

            # Create a shape image to draw the shape
            shape_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            shape_draw = ImageDraw.Draw(shape_image)

            # Calculate the points of the shape
            if shape_type == 'circle':
                bbox = [x0 - current_size, y0 - current_size, x0 + current_size, y0 + current_size]
                shape_draw.ellipse(bbox, outline=current_color, width=thickness)
                shape_bbox = bbox
            else:
                if shape_type == 'square':
                    half_size = current_size
                    points_list = [
                        (x0 - half_size, y0 - half_size),
                        (x0 + half_size, y0 - half_size),
                        (x0 + half_size, y0 + half_size),
                        (x0 - half_size, y0 + half_size)
                    ]
                elif shape_type == 'triangle':
                    half_size = current_size
                    height_triangle = half_size * math.sqrt(3)
                    points_list = [
                        (x0, y0 - 2 * half_size / math.sqrt(3)),
                        (x0 - half_size, y0 + height_triangle / 3),
                        (x0 + half_size, y0 + height_triangle / 3)
                    ]
                elif shape_type == 'hexagon':
                    half_size = current_size
                    points_list = []
                    for angle in range(0, 360, 60):
                        rad = math.radians(angle + current_rotation)
                        px = x0 + half_size * math.cos(rad)
                        py = y0 + half_size * math.sin(rad)
                        points_list.append((px, py))
                else:
                    print(f"Unsupported shape type: {shape_type}")
                    return base_image.convert('RGB')

                # Apply cumulative rotation
                if current_rotation != 0 and shape_type != 'hexagon':
                    points_list = rotate_points(points_list, (x0, y0), current_rotation)

                # Draw the shape
                shape_draw.polygon(points_list, outline=current_color, width=thickness)

                # Calculate the bounding box of the shape
                xs_list = [p[0] for p in points_list]
                ys_list = [p[1] for p in points_list]
                shape_bbox = [min(xs_list), min(ys_list), max(xs_list), max(ys_list)]

            # Check if the shape is completely outside the image bounds
            if (shape_bbox[2] < 0 or shape_bbox[0] > width or
                    shape_bbox[3] < 0 or shape_bbox[1] > height):
                break

            # Composite the shape onto the base image
            base_image = Image.alpha_composite(base_image, shape_image)

            # Update variables
            current_size += spacing + thickness

            # Increment the rotation angle
            current_rotation += rotation_angle

    return base_image.convert('RGB')

def rgb_to_hsv(r, g, b):
    """
    Convert RGB color to HSV.

    Args:
        r (int): Red component (0-255)
        g (int): Green component (0-255)
        b (int): Blue component (0-255)

    Returns:
        tuple: (hue, saturation, value) in degrees (0-360), and 0-1 ranges
    """
    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    return h * 360, s, v

def hsv_to_rgb(h, s, v):
    """
    Convert HSV color to RGB.
    
    Args:
        h (float): Hue in degrees (0-360)
        s (float): Saturation (0-1)
        v (float): Value (0-1)
        
    Returns:
        tuple: (r, g, b) values in 0-255 range
    """
    r, g, b = colorsys.hsv_to_rgb(h/360, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))

def darken_color(color, amount):
    """
    Darken a color by a specified amount.
    
    Args:
        color (tuple): RGB color tuple
        amount (int): Amount to darken (0-255)
        
    Returns:
        tuple: Darkened RGB color
    """
    r, g, b = color
    return (
        max(0, r - amount),
        max(0, g - amount),
        max(0, b - amount)
    )

def rotate_points(points, center, angle_degrees):
    """
    Rotate a list of points around a center point by a specified angle.
    
    Args:
        points (list): List of (x, y) tuples
        center (tuple): (x, y) center point for rotation
        angle_degrees (float): Rotation angle in degrees
        
    Returns:
        list: Rotated points
    """
    angle_rad = math.radians(angle_degrees)
    cx, cy = center
    rotated_points = []
    
    for x, y in points:
        # Translate point to origin
        tx = x - cx
        ty = y - cy
        
        # Rotate point
        rx = tx * math.cos(angle_rad) - ty * math.sin(angle_rad)
        ry = tx * math.sin(angle_rad) + ty * math.cos(angle_rad)
        
        # Translate back
        rotated_points.append((rx + cx, ry + cy))
    
    return rotated_points

def color_shift_expansion(image, num_points=5, shift_amount=5, expansion_type='square', mode='xtreme', 
                        saturation_boost=0.0, value_boost=0.0, pattern_type='random', 
                        color_theme='full-spectrum', decay_factor=0.0):
    """
    Creates vibrant color transformations expanding from seed points across the image.

    Args:
        image (Image): The PIL Image to process.
        num_points (int): Number of seed points to generate.
        shift_amount (float): Intensity of the color effect.
        expansion_type (str): Type of expansion ('square', 'cross', 'circular').
        mode (str): Parameter kept for backward compatibility.
        saturation_boost (float): Amount to boost saturation (0.0-1.0).
        value_boost (float): Amount to boost value/brightness (0.0-1.0).
        pattern_type (str): Pattern for seedpoints ('random', 'grid', 'radial', 'spiral').
        color_theme (str): Color theme to use ('full-spectrum', 'warm', 'cool', 'complementary', 'analogous').
        decay_factor (float): Controls how the effect fades with distance (0.0-1.0). Higher values make the effect more
                         concentrated around seed points. Uses linear decay relative to image diagonal.
    
    Returns:
        Image: The processed image.
    """
    width, height = image.size
    image = image.convert('RGB')
    image_np = np.array(image)
    
    # Create a blank canvas for our output
    output_np = np.zeros_like(image_np)
    
    # Generate seed points based on pattern type
    seed_points = []
    if pattern_type == 'grid':
        # Create an evenly spaced grid of points
        cols = max(2, int(math.sqrt(num_points)))
        rows = max(2, num_points // cols)
        x_step = width // cols
        y_step = height // rows
        for i in range(rows):
            for j in range(cols):
                x = j * x_step + x_step // 2
                y = i * y_step + y_step // 2
                if x < width and y < height:
                    seed_points.append((x, y))
    elif pattern_type == 'radial':
        # Create points in concentric circles
        center_x, center_y = width // 2, height // 2
        num_circles = min(5, num_points // 4)
        points_per_circle = max(4, num_points // num_circles)
        for circle in range(1, num_circles + 1):
            radius = (min(width, height) // 2) * (circle / num_circles)
            for i in range(points_per_circle):
                angle = (2 * math.pi * i) / points_per_circle
                x = int(center_x + radius * math.cos(angle))
                y = int(center_y + radius * math.sin(angle))
                if 0 <= x < width and 0 <= y < height:
                    seed_points.append((x, y))
    elif pattern_type == 'spiral':
        # Create points in a spiral pattern
        center_x, center_y = width // 2, height // 2
        max_radius = min(width, height) // 2
        for i in range(num_points):
            # Spiral formula: r = a * theta
            theta = 3 * i / num_points * 2 * math.pi
            r = (i / num_points) * max_radius
            x = int(center_x + r * math.cos(theta))
            y = int(center_y + r * math.sin(theta))
            if 0 <= x < width and 0 <= y < height:
                seed_points.append((x, y))
    else:  # random
        # Generate random points
        xs = np.random.randint(0, width, size=num_points)
        ys = np.random.randint(0, height, size=num_points)
        seed_points = list(zip(xs, ys))
    
    # Define the base colors for each theme
    base_colors = []
    if color_theme == 'warm':
        # Vibrant warm colors: reds, oranges, yellows, pinks
        base_colors = [
            (255, 0, 0),     # Red
            (255, 128, 0),   # Orange
            (255, 255, 0),   # Yellow
            (255, 0, 128),   # Pink
            (255, 128, 128)  # Light red
        ]
    elif color_theme == 'cool':
        # Vibrant cool colors: blues, greens, purples
        base_colors = [
            (0, 0, 255),     # Blue
            (0, 255, 255),   # Cyan
            (128, 0, 255),   # Purple
            (0, 255, 128),   # Mint
            (128, 255, 255)  # Light blue
        ]
    elif color_theme == 'complementary':
        # Complementary color pairs
        hue = random.uniform(0, 1)
        hue_comp = (hue + 0.5) % 1.0
        
        r1, g1, b1 = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        r2, g2, b2 = colorsys.hsv_to_rgb(hue_comp, 1.0, 1.0)
        
        base_colors = [
            (int(r1 * 255), int(g1 * 255), int(b1 * 255)),
            (int(r2 * 255), int(g2 * 255), int(b2 * 255))
        ]
    elif color_theme == 'analogous':
        # Analogous colors (30° apart)
        base_hue = random.uniform(0, 1)
        base_colors = []
        for i in range(5):
            h = (base_hue + i * 0.083) % 1.0  # ~30° steps (0.083 = 30/360)
            r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
            base_colors.append((int(r * 255), int(g * 255), int(b * 255)))
    else:  # full-spectrum
        # Full spectrum of vibrant colors
        for i in range(num_points):
            h = i / num_points
            r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
            base_colors.append((int(r * 255), int(g * 255), int(b * 255)))
    
    # Assign a color to each seed point
    seed_colors = []
    for i in range(num_points):
        color_idx = i % len(base_colors)
        seed_colors.append(base_colors[color_idx])
    
    # Calculate the maximum possible distance (diagonal of the image)
    max_distance = math.sqrt(width**2 + height**2)
    
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
                    # Manhattan distance (L1 norm)
                    d = abs(x - x0) + abs(y - y0)
                elif expansion_type == 'cross':
                    # Modified Manhattan - stricter cross pattern
                    d = max(abs(x - x0), abs(y - y0)) * 1.5 + min(abs(x - x0), abs(y - y0)) * 0.5
                else:  # circular
                    # Euclidean distance (L2 norm)
                    d = math.sqrt((x - x0)**2 + (y - y0)**2)
                
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
                seed_r, seed_g, seed_b = seed_colors[i]
                blend_r += seed_r * influence
                blend_g += seed_g * influence
                blend_b += seed_b * influence
            
            # Convert the blend to HSV
            blend_h, blend_s, blend_v = colorsys.rgb_to_hsv(
                blend_r / 255.0, blend_g / 255.0, blend_b / 255.0)
            
            # Apply the shift amount to control the intensity of the effect
            # The higher the shift_amount, the more of the seed colors show through
            # Scale to provide good results in the 1-10 range
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