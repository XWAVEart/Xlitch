from PIL import Image
import numpy as np
import random
from scipy.spatial import cKDTree
# import colorsys # Not directly used after refactor
from ..core.pixel_attributes import PixelAttributes
# from scipy.stats import mode # Alternative for most_common, but np.unique is fine

def pixelate_by_attribute(image, pixel_width=8, pixel_height=8, attribute='color', num_bins=100):
    """ # num_bins is currently unused, consider removing or implementing its use if intended
    Apply pixelation grouping similar values from the specified attribute.
    
    Args:
        image (Image): PIL Image object to process.
        pixel_width (int): Width of each pixelated block.
        pixel_height (int): Height of each pixelated block.
        attribute (str): Attribute to use for pixel grouping ('color', 'brightness', 'hue', 'saturation', 'value').
        num_bins (int): Number of value groups to create (currently unused).
    
    Returns:
        Image: Processed image with pixelation effect.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    height, width, _ = img_array.shape
    result_array = np.zeros_like(img_array)
    
    # Define the attribute function based on the selected attribute
    if attribute == 'color':
        # Handled specially in the loop
        pass 
    elif attribute == 'brightness':
        attr_func = PixelAttributes.brightness
    elif attribute == 'hue':
        attr_func = PixelAttributes.hue
    elif attribute == 'saturation':
        attr_func = PixelAttributes.saturation
    elif attribute == 'luminance' or attribute == 'value': # 'value' is often used for luminance in HSV context
        attr_func = PixelAttributes.luminance
    else:
        # Default to brightness if the attribute is not recognized
        attr_func = PixelAttributes.brightness
    
    # Process each pixel block
    for y in range(0, height, pixel_height):
        for x in range(0, width, pixel_width):
            block_y_end = min(y + pixel_height, height)
            block_x_end = min(x + pixel_width, width)
            
            current_block_slice = img_array[y:block_y_end, x:block_x_end]
            if current_block_slice.size == 0:
                continue

            fill_color = None
            
            if attribute == 'color':
                # Reshape block to (num_pixels_in_block, 3)
                flat_block = current_block_slice.reshape(-1, 3)
                if flat_block.shape[0] == 0: continue # Skip empty blocks if any
                
                # Find the most common color
                # np.unique returns unique rows and their counts
                unique_colors, counts = np.unique(flat_block, axis=0, return_counts=True)
                most_common_color_idx = np.argmax(counts)
                fill_color = unique_colors[most_common_color_idx]
            else:
                # Reshape block to (num_pixels_in_block, 3)
                flat_block = current_block_slice.reshape(-1, 3)
                if flat_block.shape[0] == 0: continue # Skip empty blocks

                # Calculate attribute for each pixel in the block
                # This part is not fully vectorized due to PixelAttributes taking tuples
                attr_values = np.array([attr_func(tuple(p)) for p in flat_block])
                
                if attr_values.size == 0: # Should be caught by flat_block.shape[0] == 0
                    # Fallback or skip, e.g., use average color of block or first pixel
                    if flat_block.shape[0] > 0:
                         fill_color = flat_block[0] # Fallback to first pixel color
                    else:
                        continue # Should not happen if previous checks are fine
                else:
                    avg_attr = np.mean(attr_values)
                    
                    # Find the pixel with the attribute value closest to the average
                    closest_idx = np.argmin(np.abs(attr_values - avg_attr))
                    fill_color = flat_block[closest_idx]
            
            if fill_color is not None:
                result_array[y:block_y_end, x:block_x_end] = fill_color
            else: # Should not happen if logic above ensures fill_color is set
                # As a robust fallback, copy original block if no fill_color determined
                result_array[y:block_y_end, x:block_x_end] = current_block_slice

    return Image.fromarray(result_array) 

def voronoi_pixelate(image, num_cells=50, attribute='color', seed=None):
    """
    Apply Voronoi-based pixelation effect to the image.
    
    Args:
        image (Image): PIL Image object to process.
        num_cells (int): Number of Voronoi cells to generate.
        attribute (str): Attribute to use for cell coloring ('color', 'brightness', 'hue', 'saturation', 'luminance').
        seed (int): Random seed for reproducible cell generation.
    
    Returns:
        Image: Processed image with Voronoi pixelation effect.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    img_array = np.array(image)
    height, width, _ = img_array.shape
    result_array = np.zeros_like(img_array)
    
    # Generate seed points for Voronoi cells using stratified approach for better distribution
    grid_size = int(np.sqrt(num_cells)) + 1
    grid_width = width / grid_size
    grid_height = height / grid_size
    
    points = []
    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) < num_cells:
                # Add some randomness within each grid cell
                x = (j + np.random.rand()) * grid_width
                y = (i + np.random.rand()) * grid_height
                # Ensure points are within image bounds
                x = max(0, min(width - 1, x))
                y = max(0, min(height - 1, y))
                points.append([x, y])
    
    # Convert to numpy array and shuffle to break grid patterns
    points = np.array(points)
    np.random.shuffle(points)
    points = points[:num_cells]  # Ensure we have exactly num_cells
    
    # Create a grid of pixel coordinates
    xv, yv = np.meshgrid(np.arange(width), np.arange(height))
    pixel_coords = np.column_stack((xv.ravel(), yv.ravel()))
    
    # Build a KD-Tree for efficient nearest-neighbor search
    tree = cKDTree(points)
    _, regions = tree.query(pixel_coords)
    
    # Reshape regions to the image shape
    regions = regions.reshape((height, width))
    
    # Define the attribute function based on the selected attribute
    if attribute == 'color':
        # Handled specially in the loop
        attr_func = None
    elif attribute == 'brightness':
        attr_func = PixelAttributes.brightness
    elif attribute == 'hue':
        attr_func = PixelAttributes.hue
    elif attribute == 'saturation':
        attr_func = PixelAttributes.saturation
    elif attribute == 'luminance' or attribute == 'value':
        attr_func = PixelAttributes.luminance
    else:
        # Default to brightness if the attribute is not recognized
        attr_func = PixelAttributes.brightness
    
    # Process each Voronoi cell
    for region_id in np.unique(regions):
        # Get the mask for the current region
        mask = regions == region_id
        
        # Get the coordinates of pixels in the cell
        ys_cell, xs_cell = np.where(mask)
        
        # Get the pixels in the cell
        pixels = img_array[ys_cell, xs_cell]
        
        # Skip regions with no pixels
        if len(pixels) == 0:
            continue
        
        fill_color = None
        
        if attribute == 'color':
            # Find the most common color in the cell
            if len(pixels) == 1:
                fill_color = pixels[0]
            else:
                unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
                most_common_color_idx = np.argmax(counts)
                fill_color = unique_colors[most_common_color_idx]
        else:
            # Calculate attribute for each pixel in the cell
            attr_values = np.array([attr_func(tuple(p)) for p in pixels])
            
            if attr_values.size == 0:
                # Fallback to first pixel color
                fill_color = pixels[0] if len(pixels) > 0 else [0, 0, 0]
            else:
                avg_attr = np.mean(attr_values)
                
                # Find the pixel with the attribute value closest to the average
                closest_idx = np.argmin(np.abs(attr_values - avg_attr))
                fill_color = pixels[closest_idx]
        
        # Fill the entire cell with the determined color
        if fill_color is not None:
            result_array[ys_cell, xs_cell] = fill_color
        else:
            # Fallback: keep original pixels
            result_array[ys_cell, xs_cell] = pixels
    
    # Reset random seeds
    if seed is not None:
        np.random.seed(None)
        random.seed(None)
    
    return Image.fromarray(result_array) 