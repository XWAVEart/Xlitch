from PIL import Image
import numpy as np
import colorsys
from ..core.pixel_attributes import PixelAttributes

def pixelate_by_attribute(image, pixel_width=8, pixel_height=8, attribute='color', num_bins=100):
    """
    Apply pixelation grouping similar values from the specified attribute.
    
    Args:
        image (Image): PIL Image object to process.
        pixel_width (int): Width of each pixelated block.
        pixel_height (int): Height of each pixelated block.
        attribute (str): Attribute to use for pixel grouping ('color', 'brightness', 'hue', 'saturation', 'value').
        num_bins (int): Number of value groups to create.
    
    Returns:
        Image: Processed image with pixelation effect.
    """
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Create a copy of the image
    im = image.copy()
    width, height = im.size
    
    # Create a new image with the same size
    result = Image.new('RGB', (width, height))
    
    # Define the attribute function based on the selected attribute
    if attribute == 'color':
        # For color, we'll use the most common color in the block
        def attr_func(pixel):
            return pixel
    elif attribute == 'brightness':
        def attr_func(pixel):
            return PixelAttributes.brightness(pixel)
    elif attribute == 'hue':
        def attr_func(pixel):
            return PixelAttributes.hue(pixel)
    elif attribute == 'saturation':
        def attr_func(pixel):
            return PixelAttributes.saturation(pixel)
    elif attribute == 'luminance' or attribute == 'value':
        def attr_func(pixel):
            return PixelAttributes.luminance(pixel)
    else:
        # Default to brightness if the attribute is not recognized
        def attr_func(pixel):
            return PixelAttributes.brightness(pixel)
    
    # Process each pixel block
    for y in range(0, height, pixel_height):
        for x in range(0, width, pixel_width):
            # Define the block boundaries
            block_right = min(x + pixel_width, width)
            block_bottom = min(y + pixel_height, height)
            
            # Get all pixels in the block
            block_pixels = []
            for by in range(y, block_bottom):
                for bx in range(x, block_right):
                    pixel = im.getpixel((bx, by))
                    block_pixels.append(pixel)
            
            if attribute == 'color':
                # Find the most common color in the block
                color_counts = {}
                for pixel in block_pixels:
                    pixel_str = str(pixel)
                    if pixel_str in color_counts:
                        color_counts[pixel_str] += 1
                    else:
                        color_counts[pixel_str] = 1
                
                # Get the most common color
                most_common_color_str = max(color_counts, key=color_counts.get)
                most_common_color = eval(most_common_color_str)
                
                # Fill the block with the most common color
                for by in range(y, block_bottom):
                    for bx in range(x, block_right):
                        result.putpixel((bx, by), most_common_color)
            else:
                # Calculate the average attribute value for the block
                attr_values = [attr_func(pixel) for pixel in block_pixels]
                avg_attr = sum(attr_values) / len(attr_values)
                
                # Find a representative pixel with the closest attribute value
                closest_pixel = min(block_pixels, key=lambda p: abs(attr_func(p) - avg_attr))
                
                # Fill the block with the representative color
                for by in range(y, block_bottom):
                    for bx in range(x, block_right):
                        result.putpixel((bx, by), closest_pixel)
    
    return result 