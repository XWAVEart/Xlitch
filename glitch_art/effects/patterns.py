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
    # Keep the original sort_function map for PixelAttributes methods
    pa_sort_functions = {
        'brightness': PixelAttributes.brightness,
        'hue': PixelAttributes.hue,
        'saturation': PixelAttributes.saturation,
        'luminance': PixelAttributes.luminance,
        'contrast': PixelAttributes.contrast
    }

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
        if sort_by == 'color':
            pixel_values = np.sum(pixels, axis=1, dtype=int)
        elif sort_by == 'red':
            pixel_values = pixels[:, 0]
        elif sort_by == 'green':
            pixel_values = pixels[:, 1]
        elif sort_by == 'blue':
            pixel_values = pixels[:, 2]
        elif sort_by in pa_sort_functions:
            # Use list comprehension for PixelAttributes methods
            selected_pa_func = pa_sort_functions[sort_by]
            pixel_values = np.array([selected_pa_func(p) for p in pixels])
        else: # Default to color sum if sort_by is unknown
            pixel_values = np.sum(pixels, axis=1, dtype=int)
            
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

def masked_merge(image, secondary_image, mask_type='checkerboard', width=32, height=32, random_seed=None, stripe_width=16, stripe_angle=45, perlin_noise_scale=0.1, threshold=0.5, voronoi_cells=50, perlin_octaves=1, circle_origin='center', gradient_direction='up', triangle_size=32):
    """
    Merge two images using a mask pattern.
    
    Args:
        image (PIL.Image): Primary image.
        secondary_image (PIL.Image): Secondary image to merge with.
        mask_type (str): Type of mask ('checkerboard', 'random_checkerboard', 'striped', 
                       'gradient_striped', 'linear_gradient_striped', 'perlin', 'voronoi', 
                       'concentric_rectangles', 'concentric_circles', 'random_triangles').
        width (int): Width of the squares/rectangles in the checkerboard mask pattern, or band thickness for concentric rectangles/circles.
        height (int): Height of the squares/rectangles in the checkerboard mask pattern.
        random_seed (int): Seed for random generation (for 'random_checkerboard', 'random_triangles', 'perlin', 'voronoi' mask types).
        stripe_width (int): Width of stripes in pixels (for 'striped', 'gradient_striped', 'linear_gradient_striped' mask types).
        stripe_angle (int): Angle of stripes in degrees (for 'striped', 'gradient_striped', 'linear_gradient_striped' mask types).
        perlin_noise_scale (float): Scale of the perlin noise (for 'perlin' mask type).
        threshold (float): Threshold for the perlin noise (for 'perlin' mask type).
        voronoi_cells (int): Number of Voronoi cells for the voronoi mask type.
        perlin_octaves (int): Number of octaves for Perlin noise (1=smooth curves, higher=more detailed).
        circle_origin (str): Origin for concentric circles ('center', 'top-left', 'top-right', 'bottom-left', 'bottom-right').
        gradient_direction (str): Direction for linear gradient stripes ('up' for 0->255, 'down' for 255->0).
        triangle_size (int): Side length for the 'random_triangles' mask type.

    Returns:
        PIL.Image: Merged image.
    """
    # Ensure images are the same size
    if image.size != secondary_image.size:
        secondary_image = secondary_image.resize(image.size, Image.LANCZOS)
    
    img_width, img_height = image.size
    
    # Create mask based on mask_type
    mask_array = np.zeros((img_height, img_width), dtype=np.uint8)

    yy_grid, xx_grid = np.mgrid[0:img_height, 0:img_width]

    if mask_type == 'checkerboard':
        # Draw a regular checkerboard pattern using NumPy
        # Ensure width and height are at least 1 to avoid division by zero or empty patterns
        safe_width = max(1, width)
        safe_height = max(1, height)
        mask_array = (((xx_grid // safe_width) + (yy_grid // safe_height)) % 2 == 0) * 255
        mask_array = mask_array.astype(np.uint8)
        mask = Image.fromarray(mask_array, mode='L')
    
    elif mask_type == 'random_checkerboard':
        # Draw a random checkerboard pattern using NumPy
        if random_seed is not None:
            np.random.seed(random_seed) # Use NumPy's random for consistency
        
        safe_width = max(1, width)
        safe_height = max(1, height)

        num_blocks_y = (img_height + safe_height - 1) // safe_height
        num_blocks_x = (img_width + safe_width - 1) // safe_width
        
        block_choices = np.random.randint(0, 2, size=(num_blocks_y, num_blocks_x), dtype=np.uint8)
        
        # Create the full mask by repeating blocks using np.kron
        mask_array_full = np.kron(block_choices, np.ones((safe_height, safe_width), dtype=np.uint8)) * 255
        
        # Crop to the exact image dimensions
        mask_array = mask_array_full[:img_height, :img_width]
        mask = Image.fromarray(mask_array, mode='L')
    
    elif mask_type == 'striped':
        # Draw stripes at the specified angle with consistent width
        img_width, img_height = image.size
        
        # Convert angle to radians
        angle_rad = math.radians(stripe_angle)
        
        # Calculate normal vector (perpendicular to stripe direction)
        nx = math.cos(angle_rad)
        ny = math.sin(angle_rad)
        
        # Calculate the perpendicular distance for each pixel using vectorization
        distances = xx_grid * nx + yy_grid * ny
        
        # Create stripes with consistent width using modulo operation
        safe_stripe_width = max(1, stripe_width)
        mask_array = np.zeros((img_height, img_width), dtype=np.uint8)
        mask_array[np.int32(distances / safe_stripe_width) % 2 == 0] = 255
        
        # Convert the mask array to a PIL image
        mask = Image.fromarray(mask_array, mode='L')
    
    elif mask_type == 'gradient_striped':
        # Draw gradient stripes using a sawtooth wave for linear alpha blend within stripes
        img_width, img_height = image.size
        
        # Convert angle to radians
        angle_rad = math.radians(stripe_angle)
        
        # Calculate normal vector (perpendicular to stripe direction)
        nx = math.cos(angle_rad)
        ny = math.sin(angle_rad)
        
        # Calculate the perpendicular distance for each pixel using vectorization
        distances = xx_grid * nx + yy_grid * ny
        
        # Ensure stripe_width is at least 1 to avoid division by zero
        safe_stripe_width = max(1, stripe_width)
        
        # Calculate band index (0, 1, 2, 3...)
        band_index = np.int32(distances // safe_stripe_width)
        
        # Calculate position within the stripe (0.0 to 1.0)
        # Distance from the start of the current stripe
        start_of_stripe = band_index * safe_stripe_width
        pos_in_stripe = (distances - start_of_stripe) / safe_stripe_width
        pos_in_stripe = np.clip(pos_in_stripe, 0.0, 1.0) # Ensure it stays within 0-1

        # Apply sawtooth gradient based on even/odd bands
        even_bands = (band_index % 2 == 0)
        odd_bands = ~even_bands

        # For even bands (e.g., 0, 2, 4...), gradient 0 -> 255 (secondary -> primary)
        mask_array = np.zeros((img_height, img_width), dtype=np.uint8)
        mask_array[even_bands] = (pos_in_stripe[even_bands] * 255).astype(np.uint8)

        # For odd bands (e.g., 1, 3, 5...), gradient 255 -> 0 (primary -> secondary)
        mask_array[odd_bands] = ((1.0 - pos_in_stripe[odd_bands]) * 255).astype(np.uint8)
        
        # Convert the mask array to a PIL image
        mask = Image.fromarray(mask_array, mode='L')
    
    elif mask_type == 'linear_gradient_striped':
        # Draw stripes with a consistent linear gradient (sawtooth across whole image)
        img_width, img_height = image.size
        
        # Convert angle to radians
        angle_rad = math.radians(stripe_angle)
        
        # Calculate normal vector
        nx = math.cos(angle_rad)
        ny = math.sin(angle_rad)
        
        # Create coordinate arrays
        y_coords, x_coords = np.mgrid[:img_height, :img_width]
        
        # Calculate perpendicular distance
        distances = x_coords * nx + y_coords * ny
        
        # Ensure stripe_width is at least 1
        safe_stripe_width = max(1, stripe_width)
        
        # Calculate position within the repeating pattern (0.0 to 1.0)
        # Use modulo arithmetic for a repeating sawtooth wave
        pos_in_pattern = (distances % safe_stripe_width) / safe_stripe_width

        # Apply the gradient based on direction
        if gradient_direction == 'down':
            # Ramp down from 255 to 0
            mask_array = ((1.0 - pos_in_pattern) * 255).astype(np.uint8)
        else: # Default to 'up'
            # Ramp up from 0 to 255
            mask_array = (pos_in_pattern * 255).astype(np.uint8)
        
        # Convert the mask array to a PIL image
        mask = Image.fromarray(mask_array)
    
    elif mask_type == 'perlin':
        # Create a perlin noise pattern
        noise_module_found = False
        try:
            import noise
            noise_module_found = True
        except ImportError:
            print("Warning: noise module not found for Perlin mask, using random noise fallback.")
            # Fallback to random noise
            if random_seed is not None:
                np.random.seed(random_seed)
            perlin_mask_array = (np.random.rand(img_height, img_width) > threshold).astype(np.uint8) * 255
            mask = Image.fromarray(perlin_mask_array, mode='L')
            # No need to fall through to composite if this path is taken and mask is set.
            # The structure implies 'mask' should be set by one of the branches.
        
        if noise_module_found:
            noise_map_values = np.zeros((img_height, img_width), dtype=np.float32)
            
            # The 'base' parameter in noise.pnoise2 acts as a seed offset.
            # If random_seed is None, use 0, otherwise use the int value of random_seed.
            base_for_noise_module = int(random_seed) if random_seed is not None else 0

            # Note: Calling pnoise2 in a loop is standard for the 'noise' library to fill a map.
            # This can be slow for very large images or high octaves.
            for y_idx in range(img_height):
                for x_idx in range(img_width):
                    noise_map_values[y_idx, x_idx] = noise.pnoise2(
                        y_idx * perlin_noise_scale,
                        x_idx * perlin_noise_scale,
                        octaves=perlin_octaves,
                        persistence=0.5, # Default persistence
                        lacunarity=2.0,  # Default lacunarity
                        repeatx=1024,    # Default repeat
                        repeaty=1024,    # Default repeat
                        base=base_for_noise_module # Seed for the noise instance
                    )
            
            # Normalize the noise map from its actual range (approx -1 to 1 for pnoise2) to [0, 1]
            min_val = np.min(noise_map_values)
            max_val = np.max(noise_map_values)
            
            if max_val == min_val: # Avoid division by zero if noise_map is flat
                # If flat, make all pixels either 0 or 255 based on threshold comparison with the flat value
                # e.g. if noise_map is all 0.5 and threshold is 0.4, all should be 255.
                # if noise_map is all 0.3 and threshold is 0.4, all should be 0.
                normalized_noise_map = np.ones_like(noise_map_values) if min_val > threshold else np.zeros_like(noise_map_values)

            else:
                normalized_noise_map = (noise_map_values - min_val) / (max_val - min_val)
            
            perlin_mask_array = (normalized_noise_map > threshold).astype(np.uint8) * 255
            mask = Image.fromarray(perlin_mask_array, mode='L')
        # If noise_module_found is false, the 'except' block should have already set 'mask' and returned or skipped this.
        # However, the original structure was if/elif, so this 'if noise_module_found' implies
        # 'mask' might not be set if the import fails and the 'except' block doesn't create it.
        # The provided snippet had the except block printing and then continuing, which means it would fall through.
        # My change to the except block makes it create a random mask and set `mask`.
    
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
    
    elif mask_type == 'concentric_rectangles':
        # No seed or height needed for this type
        # width parameter determines the thickness of the rectangle bands
        img_width, img_height = image.size
        mask = Image.new('L', (img_width, img_height), 0) # Start with a black background
        draw = ImageDraw.Draw(mask)
        
        # Calculate aspect ratio
        aspect_ratio = img_width / img_height
        
        x_offset = 0
        y_offset = 0
        fill_value = 255 # Start with white
        
        # The 'width' parameter now directly corresponds to the band thickness
        band_thickness = max(1, width) # Ensure at least 1 pixel thickness
        
        while x_offset < img_width / 2 and y_offset < img_height / 2:
            # Calculate rectangle width and height for this step based on aspect ratio and band thickness
            # This ensures bands are uniform visually
            rect_width = band_thickness
            rect_height = band_thickness / aspect_ratio if aspect_ratio > 0 else band_thickness
            
            # Adjust heights if aspect ratio makes height smaller than 1 pixel
            if rect_height < 1:
                rect_height = 1
                rect_width = aspect_ratio # Adjust width proportionally
                
            # Ensure integers
            rect_width = max(1, int(round(rect_width)))
            rect_height = max(1, int(round(rect_height)))
                
            # Draw the outer rectangle for the current band
            draw.rectangle([x_offset, y_offset, img_width - x_offset, img_height - y_offset],
                             fill=fill_value)
            
            # Update offsets and fill value for the next rectangle
            x_offset += rect_width # Increment by the calculated width for this dimension
            y_offset += rect_height # Increment by the calculated height for this dimension
            fill_value = 255 - fill_value # Alternate between black and white
    
    elif mask_type == 'concentric_circles':
        # width parameter determines the thickness of the circle bands
        img_width, img_height = image.size
        # mask = Image.new('L', (img_width, img_height), 0) # Start with a black background
        # draw = ImageDraw.Draw(mask)

        # Determine the center coordinates based on circle_origin
        if circle_origin == 'top-left':
            cx, cy = 0.0, 0.0 # Use floats for precision in distance calcs
        elif circle_origin == 'top-right':
            cx, cy = float(img_width -1), 0.0
        elif circle_origin == 'bottom-left':
            cx, cy = 0.0, float(img_height -1)
        elif circle_origin == 'bottom-right':
            cx, cy = float(img_width -1), float(img_height -1)
        else: # default to center
            cx, cy = (img_width - 1) / 2.0, (img_height - 1) / 2.0

        # The 'width' parameter now directly corresponds to the band thickness
        band_thickness = max(1, width) # Ensure at least 1 pixel thickness

        # yy_grid, xx_grid are already defined globally in the function
        # Calculate distance of each pixel from the center (cx, cy)
        distances = np.sqrt((xx_grid - cx)**2 + (yy_grid - cy)**2)
        
        # Determine band index for each pixel
        # Integer division by band_thickness groups pixels into bands
        band_indices = (distances // band_thickness).astype(int)
        
        # Alternate fill based on band index (even/odd)
        # Pixels in even bands get 255, odd bands get 0
        # To make the outermost band white (like the original ImageDraw loop that starts with fill_value=255 and draws from max_radius down)
        # we need to consider the maximum band index. Or, more simply, adjust the % 2 logic.
        # If (band_index % 2 == 0) is white, then the band at distance 0 (band_index 0) is white.
        # The original code's fill logic started white for the largest circle and alternated inwards.
        # Let's find max_band_index to correctly mimic the alternating pattern starting from outside.
        # max_dist_val = np.max(distances)
        # max_band_idx = int(max_dist_val // band_thickness)
        # mask_array = ((max_band_idx - band_indices) % 2 == 0) * 255
        # A simpler way: (band_indices % 2) will give 0 for first band, 1 for second etc.
        # If we want the 0-distance band to be, say, black, and next white: (band_indices % 2 == 1) * 255
        # The original loop: current_radius = max_radius, fill_value = 255 (white). Draws. Then current_radius -= band_thickness, fill_value = 0 (black).
        # So, the largest bands are white. The bands closest to the origin could be black or white depending on how many bands there are.
        # ( (distances // band_thickness).astype(int) % 2 == 0 ) * 255 -> band 0 (closest) is white
        # ( (distances // band_thickness).astype(int) % 2 == 1 ) * 255 -> band 0 (closest) is black
        # To match the original logic (outermost is white):
        # We need to know the "parity" of the outermost band.
        # Let's use a slightly different approach: calculate number of bands from center to farthest point.
        # If total bands is odd, center band has same color as outermost. If even, different.
        # Simpler: (floor(dist / thickness)) mod 2. This gives 0,1,0,1... from center.
        # If we want white as the "first" drawn band (largest radius), then flip the logic for pixels
        # farther away.
        # The key is that `draw.ellipse(..., fill=fill_value)` draws the *current* band.
        # `fill_value` starts at 255. `current_radius` starts at `max_radius`.
        # So band from `max_radius - band_thickness` to `max_radius` is 255.
        # Band from `max_radius - 2*band_thickness` to `max_radius - band_thickness` is 0.
        # This means `floor( (max_radius - distance) / band_thickness ) % 2 == 0` should be 255.
        
        # Re-calculate max_radius as it was done in the original loop for consistency
        corners = [(0,0), (img_width-1,0), (0,img_height-1), (img_width-1,img_height-1)] # Use actual pixel indices
        max_r_val = 0
        for corner_x, corner_y in corners:
            dist_sq = (corner_x - cx)**2 + (corner_y - cy)**2
            max_r_val = max(max_r_val, math.sqrt(dist_sq))

        # Mask where (max_radius - distance) / thickness, floored, is even.
        # This makes the outermost band (distance closest to max_radius) have index 0, hence white.
        # And the next band inwards (distance further from max_radius) have index 1, hence black.
        mask_values = (np.floor((max_r_val - distances) / band_thickness) % 2 == 0)
        mask_array = mask_values.astype(np.uint8) * 255
        
        mask = Image.fromarray(mask_array, mode='L')

    elif mask_type == 'random_triangles':
        # Draw a random pattern of equilateral triangles
        if random_seed is not None:
            random.seed(random_seed)
        
        img_width, img_height = image.size
        mask = Image.new('L', (img_width, img_height), 0) # Start black
        draw = ImageDraw.Draw(mask)
        
        # Calculate triangle dimensions
        side = max(2, triangle_size) # Ensure minimum size
        tri_height = math.sqrt(3) / 2 * side
        half_side = side / 2
        
        # Calculate grid steps
        step_x = side
        step_y = tri_height
        
        # Determine number of rows/cols needed (add extra to cover edges)
        cols = int(math.ceil(img_width / step_x)) + 1
        rows = int(math.ceil(img_height / step_y)) + 1
        
        for row in range(rows):
            for col in range(cols):
                # Calculate top-left corner of the rhombus formed by two triangles
                start_x = col * step_x
                start_y = row * step_y
                
                # Shift every other row horizontally for tiling
                if row % 2 == 1:
                    start_x -= half_side
                    
                # Define vertices for the two triangles in the rhombus
                p1 = (start_x, start_y)                           # Top vertex
                p2 = (start_x + side, start_y)                  # Top right vertex (for rhombus)
                p3 = (start_x + half_side, start_y + tri_height) # Bottom vertex
                p4 = (start_x - half_side, start_y + tri_height) # Bottom left vertex (for next rhombus)
                
                # Triangle 1 (pointing down)
                triangle1 = [p1, p2, p3]
                if random.random() > 0.5:
                    draw.polygon(triangle1, fill=255)
                
                # Triangle 2 (pointing up)
                triangle2 = [p1, p3, p4]
                if random.random() > 0.5:
                    draw.polygon(triangle2, fill=255)

    # Merge the images using the mask
    merged_image = Image.composite(image, secondary_image, mask)
    
    return merged_image

def concentric_shapes(image, num_points=5, shape_type='circle', thickness=3, spacing=10,
                      rotation_angle=0, darken_step=0, color_shift=0, seed=None):
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
        seed (int, optional): Seed for random point generation. Defaults to None.
    
    Returns:
        Image: The processed image.
    """
    width, height = image.size
    image = image.convert('RGBA')  # Ensure image has an alpha channel

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed) # Also seed python's random if it's used anywhere internally by helpers

    # Create a base image to draw on
    base_image = image.copy() # image is already RGBA
    draw_on_base = ImageDraw.Draw(base_image)

    # Select random points
    xs = np.random.randint(0, width, size=num_points)
    ys = np.random.randint(0, height, size=num_points)
    points = list(zip(xs, ys))

    # For each point
    for x0, y0 in points:
        # Get the color of the pixel
        original_color_rgba = base_image.getpixel((x0, y0)) # This is (R,G,B,A)
        original_alpha = original_color_rgba[3]

        # Initialize variables
        current_size = spacing
        current_rotation = 0  # Initialize cumulative rotation

        # Initialize HSV color from the original color RGB components
        r_orig, g_orig, b_orig = original_color_rgba[:3]
        h_temp, s_original, v_original = colorsys.rgb_to_hsv(r_orig / 255.0, g_orig / 255.0, b_orig / 255.0)
        h_original = h_temp * 360.0
        current_hue = h_original  # Start with the original hue

        max_dimension = max(width, height) * 1.5  # Set a maximum size to prevent infinite loops

        while current_size < max_dimension:
            # Adjust the hue for the current shape
            if color_shift != 0:
                current_hue = (current_hue + color_shift) % 360
            
            # Convert HSV back to RGB
            r_temp, g_temp, b_temp = colorsys.hsv_to_rgb(current_hue / 360.0, s_original, v_original)
            # Current color with original alpha for drawing
            current_color_rgb = (int(r_temp * 255), int(g_temp * 255), int(b_temp * 255))
            current_color_rgba_for_draw = current_color_rgb + (original_alpha,)

            # Darken the color if darken_step is set (darken RGB, keep alpha)
            if darken_step != 0:
                current_color_rgba_for_draw = darken_color(current_color_rgba_for_draw, darken_step)

            # Calculate the points of the shape
            if shape_type == 'circle':
                bbox = [x0 - current_size, y0 - current_size, x0 + current_size, y0 + current_size]
                draw_on_base.ellipse(bbox, outline=current_color_rgba_for_draw, width=thickness)
                shape_bbox = bbox # Bounding box for visibility check
            elif shape_type == 'square':
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
                    rad = math.radians(angle + current_rotation) # Hexagon points already consider rotation
                    px = x0 + half_size * math.cos(rad)
                    py = y0 + half_size * math.sin(rad)
                    points_list.append((px, py))
            else:
                # print(f"Unsupported shape type: {shape_type}") # Or log a warning
                # Consider logging: import logging; logging.warning(f"Unsupported shape type: {shape_type} in concentric_shapes")
                # If an unsupported shape is encountered for a point, skip drawing for this point's iteration and continue to next shape size or next point.
                # However, the current loop structure will break out of the while loop for this point only if we 'continue'.
                # To prevent processing further shapes for this point if it's unsupported, we can break from the inner while current_size < max_dimension loop.
                # For simplicity, if we encounter an unsupported shape, we will simply not draw it for this iteration.
                # The function will still return the image processed so far.
                # A more robust solution might involve raising an error or ensuring all form options map to supported types.
                points_list = [] # Ensure points_list is empty so nothing is drawn
                # break # This would break the while loop for the current point's shapes

            # Apply cumulative rotation if not already handled (e.g. for hexagon)
            if current_rotation != 0 and shape_type not in ['hexagon']:
                points_list = rotate_points(points_list, (x0, y0), current_rotation)

            # Draw the shape if points_list is not empty
            if points_list:
                draw_on_base.polygon(points_list, outline=current_color_rgba_for_draw, width=thickness)

                # Calculate the bounding box of the shape
                xs_list = [p[0] for p in points_list]
                ys_list = [p[1] for p in points_list]
                shape_bbox = [min(xs_list), min(ys_list), max(xs_list), max(ys_list)]
            else: # If points_list is empty (e.g. unsupported shape), create a dummy bbox to allow loop to continue/break correctly
                shape_bbox = [x0,y0,x0,y0] # A single point, won't cause premature break unless point itself is out of bounds

            # Check if the shape is completely outside the image bounds
            if (shape_bbox[2] < 0 or shape_bbox[0] > width or
                    shape_bbox[3] < 0 or shape_bbox[1] > height):
                break # Stop drawing shapes for this point if they go out of bounds

            # Update variables
            current_size += spacing + thickness

            # Increment the rotation angle
            current_rotation += rotation_angle

    return base_image.convert('RGB')

def darken_color(color_rgba, amount):
    """
    Darken an RGBA color by a specified amount, preserving alpha.
    
    Args:
        color_rgba (tuple): RGBA color tuple (R, G, B, A)
        amount (int): Amount to darken RGB components (0-255)
        
    Returns:
        tuple: Darkened RGBA color tuple
    """
    r, g, b, a = color_rgba
    return (
        max(0, r - amount),
        max(0, g - amount),
        max(0, b - amount),
        a  # Preserve alpha
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
                        color_theme='full-spectrum', decay_factor=0.0, seed=None):
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
        seed (int): Random seed for reproducibility.
    
    Returns:
        Image: The processed image.
    """
    width, height = image.size
    image = image.convert('RGB')
    image_np = np.array(image)
    
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Create a blank canvas for our output
    output_np = np.zeros_like(image_np)
    
    # Parameter clamping (added for robustness, consider defining valid ranges clearly)
    num_points = max(1, min(100, num_points)) # Example range, adjust as needed
    shift_amount = max(0.0, min(20.0, shift_amount))
    saturation_boost = max(0.0, min(1.0, saturation_boost))
    value_boost = max(0.0, min(1.0, value_boost))
    decay_factor = max(0.0, min(1.0, decay_factor))

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
    if not base_colors: # Ensure base_colors is not empty before modulo
        # Fallback or default color if color_theme logic somehow results in empty base_colors
        # This might happen if num_points is 0 for 'full-spectrum' before this loop,
        # though num_points is clamped > 0. Still, good for robustness.
        if num_points > 0:
             base_colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255))] 
        else: # Should not happen
            return Image.fromarray(image_np) # Or original image

    for i in range(num_points):
        color_idx = i % len(base_colors)
        seed_colors.append(base_colors[color_idx])
    
    # Calculate the maximum possible distance (diagonal of the image)
    max_distance = math.sqrt(width**2 + height**2)
    
    # Create distance maps for each seed point (vectorized)
    yy_grid, xx_grid = np.mgrid[0:height, 0:width]
    distance_maps_list = []

    if not seed_points: # Handle case of no seed points (e.g. if num_points was 0 and clamping failed)
        if seed is not None: # Reset seed if it was set
            np.random.seed(None)
            random.seed(None)
        return Image.fromarray(image_np) # Return original or current state

    for point_coords in seed_points:
        x0, y0 = point_coords
        if expansion_type == 'square': # In patterns.py, 'square' was Manhattan
            dist_map = np.abs(xx_grid - x0) + np.abs(yy_grid - y0)
        elif expansion_type == 'cross':
            # Modified Manhattan - stricter cross pattern
            # d = max(abs(x - x0), abs(y - y0)) * 1.5 + min(abs(x - x0), abs(y - y0)) * 0.5
            # Vectorized: term1 = np.maximum(np.abs(xx_grid - x0), np.abs(yy_grid - y0))
            #             term2 = np.minimum(np.abs(xx_grid - x0), np.abs(yy_grid - y0))
            #             dist_map = term1 * 1.5 + term2 * 0.5 
            # Simpler interpretation or alternative for cross might be needed if above is too complex or slow
            # For now, let's use Chebyshev for cross as a placeholder, it creates square-like influence zones
            # This might need to be revisited for true 'cross' shape.
            dist_map_abs_x = np.abs(xx_grid - x0)
            dist_map_abs_y = np.abs(yy_grid - y0)
            dist_map = np.maximum(dist_map_abs_x, dist_map_abs_y) # Chebyshev distance for now for 'cross'
            # A true cross might be better made by masking conditions, e.g. (abs(x-x0) < W or abs(y-y0) < H)
        else:  # 'circular' (default)
            dist_map = np.sqrt((xx_grid - x0)**2 + (yy_grid - y0)**2)
        distance_maps_list.append(dist_map)
    
    distance_maps_stack = np.stack(distance_maps_list, axis=0) # (num_points, height, width)
    seed_colors_np = np.array(seed_colors, dtype=np.float32) # (num_points, 3)

    # Process each pixel (main loop - further optimization below)
    for y in range(height):
        for x in range(width):
            original_r, original_g, original_b = image_np[y, x]
            h, s, v = colorsys.rgb_to_hsv(original_r / 255.0, original_g / 255.0, original_b / 255.0)
            
            pixel_distances = distance_maps_stack[:, y, x] # Shape: (num_points,)
            
            # closest_idx = np.argmin(pixel_distances) # Not explicitly used in blending logic later
            # min_dist = pixel_distances[closest_idx] # Not explicitly used
            
            influences = np.zeros(len(seed_points), dtype=float)
            if decay_factor > 0:
                influences = np.maximum(0.0, 1.0 - (decay_factor * pixel_distances / max_distance))
            else:
                influences = 1.0 / (1.0 + (pixel_distances / 50.0)**2)

            total_influence = np.sum(influences)
            
            if total_influence < 0.001 or len(seed_colors_np) == 0:
                output_np[y, x] = image_np[y, x]
                continue
            
            normalized_influences = influences / total_influence
            
            blend_rgb = np.sum(normalized_influences[:, np.newaxis] * seed_colors_np, axis=0)
            blend_r, blend_g, blend_b = blend_rgb[0], blend_rgb[1], blend_rgb[2]
            
            blend_h, blend_s, blend_v = colorsys.rgb_to_hsv(
                np.clip(blend_r / 255.0, 0, 1),
                np.clip(blend_g / 255.0, 0, 1),
                np.clip(blend_b / 255.0, 0, 1)
            )
            
            shift_weight = min(0.85, shift_amount / 12.0) 
            
            final_h = h * (1 - shift_weight) + blend_h * shift_weight
            final_s = s * (1 - shift_weight) + (blend_s + saturation_boost) * shift_weight
            final_v = v * (1 - shift_weight) + (blend_v + value_boost) * shift_weight
            
            final_s = np.clip(final_s, 0.0, 1.0) # Using np.clip for consistency
            final_v = np.clip(final_v, 0.0, 1.0)
            final_h = final_h % 1.0
            
            final_r_float, final_g_float, final_b_float = colorsys.hsv_to_rgb(final_h, final_s, final_v)
            
            output_np[y, x] = [
                int(final_r_float * 255),
                int(final_g_float * 255),
                int(final_b_float * 255)
            ]
    
    if seed is not None: # Reset seed
        np.random.seed(None)
        random.seed(None)

    # Convert back to PIL Image
    processed_image = Image.fromarray(output_np.astype(np.uint8))
    return processed_image 