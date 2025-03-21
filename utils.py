from PIL import Image, ImageColor, ImageDraw
import os
import random
import numpy as np
import cv2
import colorsys
import math
import heapq
import noise  # Import the noise module for Perlin noise
from scipy.spatial import cKDTree
from io import BytesIO

# Centralized pixel attribute calculations
class PixelAttributes:
    """Central module for all pixel attribute calculations to avoid duplicates."""
    
    @staticmethod
    def brightness(pixel):
        """
        Calculate the perceived brightness of a pixel using the luminosity formula.
        
        Args:
            pixel (tuple): RGB pixel values.
        
        Returns:
            float: Brightness value.
        """
        return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]
    
    @staticmethod
    def hue(pixel):
        """
        Calculate the hue of a pixel.
        
        Args:
            pixel (tuple): RGB pixel values.
        
        Returns:
            int or float: Hue value (0-360 or 0-1 depending on context).
        """
        # Handle both PIL and NumPy contexts
        if isinstance(pixel, (list, tuple)) and len(pixel) >= 3:
            # Check if values are in 0-1 range (likely NumPy normalized context)
            if all(0 <= p <= 1 for p in pixel[:3]):
                r, g, b = pixel[:3]
                h, _, _ = colorsys.rgb_to_hsv(r, g, b)
                return h
            else:
                # Standard PIL context with 0-255 values
                r, g, b = [p/255.0 for p in pixel[:3]]
                h, _, _ = colorsys.rgb_to_hsv(r, g, b)
                return h * 360

        # Fall back to ImageColor for complex cases
        return ImageColor.getcolor(f"#{pixel[0]:02x}{pixel[1]:02x}{pixel[2]:02x}", "HSV")[0]
    
    @staticmethod
    def saturation(pixel):
        """
        Calculate the saturation of a pixel.
        
        Args:
            pixel (tuple): RGB pixel values.
        
        Returns:
            int or float: Saturation value (0-100 or 0-1 depending on context).
        """
        # Handle both PIL and NumPy contexts
        if isinstance(pixel, (list, tuple)) and len(pixel) >= 3:
            # Check if values are in 0-1 range (likely NumPy normalized context)
            if all(0 <= p <= 1 for p in pixel[:3]):
                r, g, b = pixel[:3]
                _, s, _ = colorsys.rgb_to_hsv(r, g, b)
                return s
            else:
                # Standard PIL context with 0-255 values
                r, g, b = [p/255.0 for p in pixel[:3]]
                _, s, _ = colorsys.rgb_to_hsv(r, g, b)
                return s

        # Fall back to ImageColor for complex cases
        return ImageColor.getcolor(f"#{pixel[0]:02x}{pixel[1]:02x}{pixel[2]:02x}", "HSV")[1]
    
    @staticmethod
    def luminance(pixel):
        """
        Calculate the luminance (value in HSV) of a pixel.
        
        Args:
            pixel (tuple): RGB pixel values.
        
        Returns:
            int or float: Luminance value (0-100 or 0-1 depending on context).
        """
        # Handle both PIL and NumPy contexts
        if isinstance(pixel, (list, tuple)) and len(pixel) >= 3:
            # Check if values are in 0-1 range (likely NumPy normalized context)
            if all(0 <= p <= 1 for p in pixel[:3]):
                r, g, b = pixel[:3]
                _, _, v = colorsys.rgb_to_hsv(r, g, b)
                return v
            else:
                # Standard PIL context with 0-255 values
                r, g, b = [p/255.0 for p in pixel[:3]]
                _, _, v = colorsys.rgb_to_hsv(r, g, b)
                return v

        # Fall back to ImageColor for complex cases
        return ImageColor.getcolor(f"#{pixel[0]:02x}{pixel[1]:02x}{pixel[2]:02x}", "HSV")[2]
    
    @staticmethod
    def contrast(pixel):
        """
        Calculate a contrast value based on the difference between max and min RGB values.
        
        Args:
            pixel (tuple): RGB pixel values.
        
        Returns:
            int: Contrast value.
        """
        return max(pixel[:3]) - min(pixel[:3])
    
    @staticmethod
    def color_sum(pixel):
        """
        Calculate the sum of RGB values.
        
        Args:
            pixel (tuple): RGB pixel values.
        
        Returns:
            int: Sum of RGB values.
        """
        return int(pixel[0]) + int(pixel[1]) + int(pixel[2])

# Replace all individual function implementations with references to the centralized class
def brightness(pixel):
    """
    Calculate the perceived brightness of a pixel using the luminosity formula.
    
    Args:
        pixel (tuple): RGB pixel values.
    
    Returns:
        float: Brightness value.
    """
    return PixelAttributes.brightness(pixel)

def hue(pixel):
    """
    Calculate the hue of a pixel.
    
    Args:
        pixel (tuple): RGB pixel values.
    
    Returns:
        int: Hue value (0-360).
    """
    return PixelAttributes.hue(pixel)

def saturation(pixel):
    """
    Calculate the saturation of a pixel.
    
    Args:
        pixel (tuple): RGB pixel values.
    
    Returns:
        int: Saturation value (0-100).
    """
    return PixelAttributes.saturation(pixel)

def luminance(pixel):
    """
    Calculate the luminance (value in HSV) of a pixel.
    
    Args:
        pixel (tuple): RGB pixel values.
    
    Returns:
        int: Luminance value (0-100).
    """
    return PixelAttributes.luminance(pixel)

def contrast(pixel):
    """
    Calculate a contrast value based on the difference between max and min RGB values.
    
    Args:
        pixel (tuple): RGB pixel values.
    
    Returns:
        int: Contrast value.
    """
    return PixelAttributes.contrast(pixel)

def pixel_sorting(image, direction, chunk_size, sort_by, starting_corner=None):
    """
    Apply pixel sorting to the image in chunks based on a specified property.
    
    Args:
        image (Image): PIL Image object to process.
        direction (str): 'horizontal', 'vertical', or 'diagonal' sorting direction.
        chunk_size (str): Chunk dimensions as 'widthxheight' (e.g., '32x32').
        sort_by (str): Property to sort by ('color', 'brightness', 'hue', 'red', 'green', 'blue', 
                       'saturation', 'luminance', 'contrast').
        starting_corner (str, optional): Starting corner for corner-to-corner or diagonal sorting.
    
    Returns:
        Image: Processed image with sorted pixels.
    """
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Handle corner-to-corner sorting if starting_corner is provided and direction isn't diagonal
    if starting_corner and direction != 'diagonal':
        return pixel_sorting_corner_to_corner(
            image, 
            chunk_size, 
            sort_by, 
            starting_corner, 
            direction == 'horizontal'
        )
    
    # Handle diagonal sorting
    if direction == 'diagonal':
        if not starting_corner:
            starting_corner = 'top-left'  # Default corner if not specified
        return diagonal_pixel_sort(image, chunk_size, sort_by, starting_corner)
    
    # Handle horizontal and vertical sorting
    sort_function = {
        'color': lambda p: sum(p[:3]),
        'brightness': brightness,
        'hue': hue,
        'red': lambda p: p[0],     # Sort by red channel only
        'green': lambda p: p[1],   # Sort by green channel only
        'blue': lambda p: p[2],     # Sort by blue channel only
        'saturation': saturation,  # Sort by color saturation
        'luminance': luminance,    # Sort by luminance (value in HSV)
        'contrast': contrast       # Sort by contrast (max-min RGB)
    }.get(sort_by, lambda p: sum(p[:3]))  # Default to sum of RGB if invalid

    pixels = list(image.getdata())
    width, height = image.size
    chunk_width, chunk_height = map(int, chunk_size.split('x'))
    
    # Create result image and get pixel access object for direct pixel manipulation
    result_image = Image.new(image.mode, image.size)
    
    # Calculate how many full chunks we can fit and the remainder
    num_chunks_x = width // chunk_width
    num_chunks_y = height // chunk_height
    remainder_x = width % chunk_width
    remainder_y = height % chunk_height
    
    # Process standard chunks (full sized)
    for chunk_row in range(num_chunks_y):
        for chunk_col in range(num_chunks_x):
            # Extract chunk pixels
            chunk_pixels = []
            start_y = chunk_row * chunk_height
            start_x = chunk_col * chunk_width
            
            for y in range(start_y, start_y + chunk_height):
                for x in range(start_x, start_x + chunk_width):
                    chunk_pixels.append(pixels[y * width + x])
            
            # Sort chunk
            if direction == 'horizontal':
                sorted_chunk = sorted(chunk_pixels, key=sort_function)
                
                # Place sorted pixels back in horizontal order
                for i, pixel in enumerate(sorted_chunk):
                    x = start_x + (i % chunk_width)
                    y = start_y + (i // chunk_width)
                    result_image.putpixel((x, y), pixel)
            else:  # vertical
                # Reshape the chunk for column-wise sorting
                chunk_2d = []
                for i in range(0, len(chunk_pixels), chunk_width):
                    chunk_2d.append(chunk_pixels[i:i + chunk_width])
                
                # For each column in the chunk
                for x in range(chunk_width):
                    # Extract column
                    column = [chunk_2d[y][x] for y in range(chunk_height)]
                    # Sort column
                    sorted_column = sorted(column, key=sort_function)
                    
                    # Place sorted column back
                    for y, pixel in enumerate(sorted_column):
                        result_image.putpixel((start_x + x, start_y + y), pixel)
    
    # Process right edge (if there's a remainder in width)
    if remainder_x > 0:
        for chunk_row in range(num_chunks_y):
            # Extract the right edge chunk
            start_y = chunk_row * chunk_height
            start_x = num_chunks_x * chunk_width
            
            edge_chunk = []
            for y in range(start_y, start_y + chunk_height):
                for x in range(start_x, start_x + remainder_x):
                    edge_chunk.append(pixels[y * width + x])
            
            # Sort the edge chunk
            if direction == 'horizontal':
                sorted_edge = sorted(edge_chunk, key=sort_function)
                
                # Place sorted pixels back
                for i, pixel in enumerate(sorted_edge):
                    x = start_x + (i % remainder_x)
                    y = start_y + (i // remainder_x)
                    result_image.putpixel((x, y), pixel)
            else:  # vertical
                # Reshape for column-wise sorting
                edge_2d = []
                for i in range(0, len(edge_chunk), remainder_x):
                    edge_2d.append(edge_chunk[i:i + remainder_x])
                
                # For each column in the edge chunk
                for x in range(remainder_x):
                    # Extract column
                    column = [edge_2d[y][x] for y in range(chunk_height)]
                    # Sort column
                    sorted_column = sorted(column, key=sort_function)
                    
                    # Place sorted column back
                    for y, pixel in enumerate(sorted_column):
                        result_image.putpixel((start_x + x, start_y + y), pixel)
    
    # Process bottom edge (if there's a remainder in height)
    if remainder_y > 0:
        for chunk_col in range(num_chunks_x):
            # Extract the bottom edge chunk
            start_y = num_chunks_y * chunk_height
            start_x = chunk_col * chunk_width
            
            edge_chunk = []
            for y in range(start_y, start_y + remainder_y):
                for x in range(start_x, start_x + chunk_width):
                    edge_chunk.append(pixels[y * width + x])
            
            # Sort the edge chunk
            if direction == 'horizontal':
                sorted_edge = sorted(edge_chunk, key=sort_function)
                
                # Place sorted pixels back
                for i, pixel in enumerate(sorted_edge):
                    x = start_x + (i % chunk_width)
                    y = start_y + (i // chunk_width)
                    result_image.putpixel((x, y), pixel)
            else:  # vertical
                # Reshape for column-wise sorting
                edge_2d = []
                for i in range(0, len(edge_chunk), chunk_width):
                    edge_2d.append(edge_chunk[i:i + chunk_width])
                
                # For each column in the edge chunk
                for x in range(chunk_width):
                    # Extract column
                    column = [row[x] for row in edge_2d if x < len(row)]
                    # Sort column
                    sorted_column = sorted(column, key=sort_function)
                    
                    # Place sorted column back
                    for y, pixel in enumerate(sorted_column):
                        result_image.putpixel((start_x + x, start_y + y), pixel)
    
    # Process bottom-right corner (if there's remainder in both width and height)
    if remainder_x > 0 and remainder_y > 0:
        # Extract the corner chunk
        start_y = num_chunks_y * chunk_height
        start_x = num_chunks_x * chunk_width
        
        corner_chunk = []
        for y in range(start_y, start_y + remainder_y):
            for x in range(start_x, start_x + remainder_x):
                corner_chunk.append(pixels[y * width + x])
        
        # Sort the corner chunk
        if direction == 'horizontal':
            sorted_corner = sorted(corner_chunk, key=sort_function)
            
            # Place sorted pixels back
            for i, pixel in enumerate(sorted_corner):
                x = start_x + (i % remainder_x)
                y = start_y + (i // remainder_x)
                result_image.putpixel((x, y), pixel)
        else:  # vertical
            # Reshape for column-wise sorting
            corner_2d = []
            for i in range(0, len(corner_chunk), remainder_x):
                corner_2d.append(corner_chunk[i:i + remainder_x])
            
            # For each column in the corner chunk
            for x in range(remainder_x):
                # Extract column
                column = [corner_2d[y][x] for y in range(len(corner_2d)) if x < len(corner_2d[y])]
                # Sort column
                sorted_column = sorted(column, key=sort_function)
                
                # Place sorted column back
                for y, pixel in enumerate(sorted_column):
                    result_image.putpixel((start_x + x, start_y + y), pixel)
    
    return result_image

def pixel_sorting_corner_to_corner(image, chunk_size, sort_by, corner, horizontal):
    """
    Apply pixel sorting starting from a specified corner, either horizontally or vertically.
    
    Args:
        image (Image): PIL Image object to process.
        chunk_size (str): Chunk dimensions as 'widthxheight' (e.g., '32x32').
        sort_by (str): Property to sort by ('color', 'brightness', 'hue', 'red', 'green', 'blue',
                       'saturation', 'luminance', 'contrast').
        corner (str): Starting corner ('top-left', 'top-right', 'bottom-left', 'bottom-right').
        horizontal (bool): True for horizontal sorting, False for vertical.
    
    Returns:
        Image: Processed image with corner-to-corner sorting.
    """
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    sort_function = {
        'color': lambda p: sum(p[:3]),
        'brightness': brightness,
        'hue': hue,
        'red': lambda p: p[0],     # Sort by red channel only
        'green': lambda p: p[1],   # Sort by green channel only
        'blue': lambda p: p[2],     # Sort by blue channel only
        'saturation': saturation,  # Sort by color saturation
        'luminance': luminance,    # Sort by luminance (value in HSV)
        'contrast': contrast       # Sort by contrast (max-min RGB)
    }.get(sort_by, lambda p: sum(p[:3]))

    # Create a result image to work with
    result_image = Image.new(image.mode, image.size)
    width, height = image.size
    chunk_width, chunk_height = map(int, chunk_size.split('x'))
    
    # Get pixel data as a 2D array
    pixels = list(image.getdata())
    pixels_2d = []
    for y in range(height):
        row = []
        for x in range(width):
            row.append(pixels[y * width + x])
        pixels_2d.append(row)
    
    # Initialize result_image with a copy of the original image data
    # This ensures no black regions if any pixels are missed
    for y in range(height):
        for x in range(width):
            result_image.putpixel((x, y), pixels_2d[y][x])
    
    # Calculate how many full chunks and the remainder
    full_chunks_x = width // chunk_width
    full_chunks_y = height // chunk_height
    remainder_x = width % chunk_width
    remainder_y = height % chunk_height
    
    # Create step values based on the corner
    x_step = chunk_width
    y_step = chunk_height
    
    # Determine chunk processing start points and ranges based on corner
    if corner == 'top-left':
        x_start_val = 0
        y_start_val = 0
        x_end_val = width
        y_end_val = height
        x_step = chunk_width
        y_step = chunk_height
    elif corner == 'top-right':
        x_start_val = width - chunk_width
        y_start_val = 0
        x_end_val = -chunk_width
        y_end_val = height
        x_step = -chunk_width
        y_step = chunk_height
    elif corner == 'bottom-left':
        x_start_val = 0
        y_start_val = height - chunk_height
        x_end_val = width
        y_end_val = -chunk_height
        x_step = chunk_width
        y_step = -chunk_height
    elif corner == 'bottom-right':
        x_start_val = width - chunk_width
        y_start_val = height - chunk_height
        x_end_val = -chunk_width
        y_end_val = -chunk_height
        x_step = -chunk_width
        y_step = -chunk_height
    
    # Create ranges based on the corner - ensure we include all pixels
    x_range = list(range(x_start_val, x_end_val, x_step))
    y_range = list(range(y_start_val, y_end_val, y_step))
    
    # Handle edge cases where ranges might be empty
    if not x_range and x_step > 0:
        x_range = [0]
    elif not x_range and x_step < 0:
        x_range = [width - chunk_width]
        
    if not y_range and y_step > 0:
        y_range = [0]
    elif not y_range and y_step < 0:
        y_range = [height - chunk_height]
    
    # Process each chunk
    for y_start in y_range:
        for x_start in x_range:
            # Calculate chunk boundaries
            y_end = min(y_start + abs(y_step), height) if y_step > 0 else max(y_start + y_step, 0)
            x_end = min(x_start + abs(x_step), width) if x_step > 0 else max(x_start + x_step, 0)
            
            # Make sure we're not going out of bounds
            if x_start < 0: x_start = 0
            if y_start < 0: y_start = 0
            if x_end > width: x_end = width
            if y_end > height: y_end = height
            if x_end <= 0: x_end = 1
            if y_end <= 0: y_end = 1
            
            # Ensure chunks always process from lower to higher indices
            x_start, x_end = min(x_start, x_end), max(x_start, x_end)
            y_start, y_end = min(y_start, y_end), max(y_start, y_end)
            
            # Skip empty chunks
            if x_start == x_end or y_start == y_end:
                continue
            
            # Extract all pixels in this chunk
            chunk_pixels = []
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    chunk_pixels.append(pixels_2d[y][x])
            
            # Sort the chunk pixels
            sorted_pixels = sorted(chunk_pixels, key=sort_function)
            
            # Reverse the order if needed based on corner
            if (corner in ['bottom-left', 'bottom-right'] and horizontal) or \
               (corner in ['top-right', 'bottom-right'] and not horizontal):
                sorted_pixels = sorted_pixels[::-1]
            
            # Put the sorted pixels back
            pixel_index = 0
            if horizontal:
                # Sort horizontally within chunk
                for y in range(y_start, y_end):
                    for x in range(x_start, x_end):
                        if pixel_index < len(sorted_pixels):
                            result_image.putpixel((x, y), sorted_pixels[pixel_index])
                            pixel_index += 1
            else:
                # Sort vertically within chunk
                for x in range(x_start, x_end):
                    for y in range(y_start, y_end):
                        if pixel_index < len(sorted_pixels):
                            result_image.putpixel((x, y), sorted_pixels[pixel_index])
                            pixel_index += 1
    
    # Ensure we process any remaining edge pixels not covered by the chunks
    
    # Adjust x_range and y_range for remainder handling
    x_range_rem = []
    y_range_rem = []
    
    # Add remainder chunks based on image dimensions and chunk size
    if remainder_x > 0:
        if x_step > 0:
            x_range_rem.append(width - remainder_x)
        else:
            x_range_rem.append(0)
    
    if remainder_y > 0:
        if y_step > 0:
            y_range_rem.append(height - remainder_y)
        else:
            y_range_rem.append(0)
    
    # Process right/left edge
    for y_start in y_range:
        for x_start in x_range_rem:
            # Calculate appropriate bounds
            y_end = min(y_start + abs(y_step), height) if y_step > 0 else max(y_start + y_step, 0)
            x_end = x_start + remainder_x if x_step > 0 else x_start + chunk_width
            
            # Ensure valid boundaries
            if x_start < 0: x_start = 0
            if y_start < 0: y_start = 0
            if x_end > width: x_end = width
            if y_end > height: y_end = height
            if x_end <= 0: x_end = 1
            if y_end <= 0: y_end = 1
            
            # Ensure chunks always process from lower to higher indices
            x_start, x_end = min(x_start, x_end), max(x_start, x_end)
            y_start, y_end = min(y_start, y_end), max(y_start, y_end)
            
            # Skip empty chunks
            if x_start == x_end or y_start == y_end:
                continue
            
            # Extract all pixels in this chunk
            chunk_pixels = []
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    chunk_pixels.append(pixels_2d[y][x])
            
            # Sort the chunk pixels
            sorted_pixels = sorted(chunk_pixels, key=sort_function)
            
            # Reverse the order if needed based on corner
            if (corner in ['bottom-left', 'bottom-right'] and horizontal) or \
               (corner in ['top-right', 'bottom-right'] and not horizontal):
                sorted_pixels = sorted_pixels[::-1]
            
            # Put the sorted pixels back
            pixel_index = 0
            if horizontal:
                # Sort horizontally within chunk
                for y in range(y_start, y_end):
                    for x in range(x_start, x_end):
                        if pixel_index < len(sorted_pixels):
                            result_image.putpixel((x, y), sorted_pixels[pixel_index])
                            pixel_index += 1
            else:
                # Sort vertically within chunk
                for x in range(x_start, x_end):
                    for y in range(y_start, y_end):
                        if pixel_index < len(sorted_pixels):
                            result_image.putpixel((x, y), sorted_pixels[pixel_index])
                            pixel_index += 1
    
    # Process bottom/top edge
    for x_start in x_range:
        for y_start in y_range_rem:
            # Calculate appropriate bounds
            x_end = min(x_start + abs(x_step), width) if x_step > 0 else max(x_start + x_step, 0)
            y_end = y_start + remainder_y if y_step > 0 else y_start + chunk_height
            
            # Ensure valid boundaries
            if x_start < 0: x_start = 0
            if y_start < 0: y_start = 0
            if x_end > width: x_end = width
            if y_end > height: y_end = height
            if x_end <= 0: x_end = 1
            if y_end <= 0: y_end = 1
            
            # Ensure chunks always process from lower to higher indices
            x_start, x_end = min(x_start, x_end), max(x_start, x_end)
            y_start, y_end = min(y_start, y_end), max(y_start, y_end)
            
            # Skip empty chunks
            if x_start == x_end or y_start == y_end:
                continue
            
            # Extract all pixels in this chunk
            chunk_pixels = []
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    chunk_pixels.append(pixels_2d[y][x])
            
            # Sort the chunk pixels
            sorted_pixels = sorted(chunk_pixels, key=sort_function)
            
            # Reverse the order if needed based on corner
            if (corner in ['bottom-left', 'bottom-right'] and horizontal) or \
               (corner in ['top-right', 'bottom-right'] and not horizontal):
                sorted_pixels = sorted_pixels[::-1]
            
            # Put the sorted pixels back
            pixel_index = 0
            if horizontal:
                # Sort horizontally within chunk
                for y in range(y_start, y_end):
                    for x in range(x_start, x_end):
                        if pixel_index < len(sorted_pixels):
                            result_image.putpixel((x, y), sorted_pixels[pixel_index])
                            pixel_index += 1
            else:
                # Sort vertically within chunk
                for x in range(x_start, x_end):
                    for y in range(y_start, y_end):
                        if pixel_index < len(sorted_pixels):
                            result_image.putpixel((x, y), sorted_pixels[pixel_index])
                            pixel_index += 1
    
    # Process corner edge (if there's remainder in both width and height)
    if remainder_x > 0 and remainder_y > 0:
        # Determine corner position based on sort direction
        if corner == 'top-left':
            x_start = width - remainder_x
            y_start = height - remainder_y
        elif corner == 'top-right':
            x_start = 0
            y_start = height - remainder_y
        elif corner == 'bottom-left':
            x_start = width - remainder_x
            y_start = 0
        elif corner == 'bottom-right':
            x_start = 0
            y_start = 0
        
        x_end = x_start + remainder_x
        y_end = y_start + remainder_y
        
        # Ensure valid boundaries
        if x_start < 0: x_start = 0
        if y_start < 0: y_start = 0
        if x_end > width: x_end = width
        if y_end > height: y_end = height
        
        # Extract all pixels in this chunk
        chunk_pixels = []
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                chunk_pixels.append(pixels_2d[y][x])
        
        # Sort the chunk pixels
        sorted_pixels = sorted(chunk_pixels, key=sort_function)
        
        # Reverse the order if needed based on corner
        if (corner in ['bottom-left', 'bottom-right'] and horizontal) or \
           (corner in ['top-right', 'bottom-right'] and not horizontal):
            sorted_pixels = sorted_pixels[::-1]
        
        # Put the sorted pixels back
        pixel_index = 0
        if horizontal:
            # Sort horizontally within chunk
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    if pixel_index < len(sorted_pixels):
                        result_image.putpixel((x, y), sorted_pixels[pixel_index])
                        pixel_index += 1
        else:
            # Sort vertically within chunk
            for x in range(x_start, x_end):
                for y in range(y_start, y_end):
                    if pixel_index < len(sorted_pixels):
                        result_image.putpixel((x, y), sorted_pixels[pixel_index])
                        pixel_index += 1
    
    return result_image

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

def double_expose(image, secondary_image, blend_mode='classic', opacity=0.5):
    """
    Apply a double expose effect by blending two images with various blend modes.
    
    Args:
        image (Image): Primary PIL Image object to process.
        secondary_image (Image): Secondary image to blend with the primary image.
        blend_mode (str): Blending mode to use ('classic', 'screen', 'multiply', 'overlay', 'difference').
        opacity (float): Opacity of the secondary image (0.0 to 1.0).
    
    Returns:
        Image: Processed image with double expose effect.
    """
    # Convert both images to RGB mode if they have alpha channels or are in different modes
    if image.mode != 'RGB':
        image = image.convert('RGB')
    if secondary_image.mode != 'RGB':
        secondary_image = secondary_image.convert('RGB')
        
    secondary_image = secondary_image.resize(image.size)

    # Convert to numpy arrays for more advanced blending
    img1 = np.array(image).astype(float)
    img2 = np.array(secondary_image).astype(float)
    
    # Apply the selected blend mode
    if blend_mode == 'classic':
        # Simple alpha blending
        blended = img1 * (1 - opacity) + img2 * opacity
    elif blend_mode == 'screen':
        # Screen blend mode: 1 - (1 - img1) * (1 - img2)
        blended = 255 - (255 - img1) * (255 - img2) / 255
        blended = img1 * (1 - opacity) + blended * opacity
    elif blend_mode == 'multiply':
        # Multiply blend mode: img1 * img2
        blended = img1 * img2 / 255
        blended = img1 * (1 - opacity) + blended * opacity
    elif blend_mode == 'overlay':
        # Overlay blend mode
        mask = img1 > 127.5
        blended = np.zeros_like(img1)
        blended[mask] = 255 - (255 - 2 * (img1[mask] - 127.5)) * (255 - img2[mask]) / 255
        blended[~mask] = (2 * img1[~mask]) * img2[~mask] / 255
        blended = img1 * (1 - opacity) + blended * opacity
    elif blend_mode == 'difference':
        # Difference blend mode: |img1 - img2|
        blended = np.abs(img1 - img2)
        blended = img1 * (1 - opacity) + blended * opacity
    elif blend_mode == 'color_dodge':
        # Color dodge blend mode: img1 / (1 - img2)
        blended = np.zeros_like(img1)
        mask = img2 < 255
        blended[mask] = np.minimum(255, img1[mask] / (1 - img2[mask] / 255))
        blended[~mask] = 255
        blended = img1 * (1 - opacity) + blended * opacity
    else:
        # Default to classic blend if mode not recognized
        blended = img1 * (1 - opacity) + img2 * opacity
    
    # Clip values to valid range and convert back to uint8
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # Create a new image from the blended array
    return Image.fromarray(blended)

def data_mosh_blocks(image, num_operations=64, max_block_size=50, block_movement='swap',
                color_swap='random', invert='never', shift='never', flip='never', seed=None):
    """
    Applies data moshing effects to an image for glitch art purposes.

    Args:
        image (Image): PIL Image object to be manipulated.
        num_operations (int): Number of manipulation operations to perform (default: 64).
        max_block_size (int): Maximum size for block width and height (default: 50).
        block_movement (str): 'swap' to swap blocks or 'in_place' to modify blocks in place (default: 'swap').
        color_swap (str): 'never', 'always', or 'random' to control color channel swapping (default: 'random').
        invert (str): 'never', 'always', or 'random' to control color inversion (default: 'never').
        shift (str): 'never', 'always', or 'random' to control channel value shifting (default: 'never').
        flip (str): 'never', 'vertical', 'horizontal', or 'random' to control block flipping (default: 'never').
        seed (int or None): Random seed for reproducibility (default: None).

    Returns:
        Image: PIL Image object with the applied glitch effects.
    """
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    # Set random seeds for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Convert PIL Image to NumPy array
    image_np = np.array(image)
    height, width, channels = image_np.shape

    # Perform the specified number of operations
    for _ in range(num_operations):
        # Determine block dimensions (non-square blocks)
        block_width = random.randint(1, min(max_block_size, width))
        block_height = random.randint(1, min(max_block_size, height))

        # Select first block position
        x1 = random.randint(0, width - block_width)
        y1 = random.randint(0, height - block_height)

        if block_movement == 'swap':
            # Select second block position, ensuring it's different from the first
            attempts = 0
            max_attempts = 10
            while attempts < max_attempts:
                x2 = random.randint(0, width - block_width)
                y2 = random.randint(0, height - block_height)
                if (x2, y2) != (x1, y1):
                    break
                attempts += 1
            else:
                # Could not find a different position, skip this operation
                continue

            # Swap the blocks
            block1 = np.copy(image_np[y1:y1 + block_height, x1:x1 + block_width, :])
            block2 = np.copy(image_np[y2:y2 + block_height, x2:x2 + block_width, :])
            image_np[y1:y1 + block_height, x1:x1 + block_width, :] = block2
            image_np[y2:y2 + block_height, x2:x2 + block_width, :] = block1

            # Define regions for transformation (both swapped blocks)
            regions = [
                (y1, y1 + block_height, x1, x1 + block_width),
                (y2, y2 + block_height, x2, x2 + block_width)
            ]
        elif block_movement == 'in_place':
            # Define region for transformation (single block)
            regions = [(y1, y1 + block_height, x1, x1 + block_width)]
        else:
            raise ValueError("block_movement must be 'swap' or 'in_place'")

        # Apply transformations to each region
        for region in regions:
            y_start, y_end, x_start, x_end = region

            # Color channel swapping
            if color_swap == 'always' or (color_swap == 'random' and random.random() < 0.5):
                swap_choice = random.choice(['red-green', 'red-blue', 'green-blue'])
                if swap_choice == 'red-green':
                    image_np[y_start:y_end, x_start:x_end, [0, 1]] = \
                        image_np[y_start:y_end, x_start:x_end, [1, 0]]
                elif swap_choice == 'red-blue':
                    image_np[y_start:y_end, x_start:x_end, [0, 2]] = \
                        image_np[y_start:y_end, x_start:x_end, [2, 0]]
                elif swap_choice == 'green-blue':
                    image_np[y_start:y_end, x_start:x_end, [1, 2]] = \
                        image_np[y_start:y_end, x_start:x_end, [2, 1]]

            # Color inversion
            if invert == 'always' or (invert == 'random' and random.random() < 0.5):
                image_np[y_start:y_end, x_start:x_end, :] = \
                    255 - image_np[y_start:y_end, x_start:x_end, :]

            # Channel value shifting
            if shift == 'always' or (shift == 'random' and random.random() < 0.5):
                offsets = np.random.randint(-50, 51, size=(3,))
                temp_block = image_np[y_start:y_end, x_start:x_end, :].astype(np.int16)
                temp_block += offsets
                image_np[y_start:y_end, x_start:x_end, :] = \
                    np.clip(temp_block, 0, 255).astype(np.uint8)

            # Block flipping
            if flip != 'never':
                if flip == 'random':
                    flip_type = random.choice(['vertical', 'horizontal'])
                else:
                    flip_type = flip
                if flip_type == 'vertical':
                    image_np[y_start:y_end, x_start:x_end, :] = \
                        np.flipud(image_np[y_start:y_end, x_start:x_end, :])
                elif flip_type == 'horizontal':
                    image_np[y_start:y_end, x_start:x_end, :] = \
                        np.fliplr(image_np[y_start:y_end, x_start:x_end, :])

    # Convert back to PIL Image and return
    moshed_image = Image.fromarray(image_np)
    return moshed_image

def perlin_noise_replacement(image, secondary_image, noise_scale=0.1, threshold=0.5, seed=None):
    """
    Replace pixels in the primary image with pixels from a secondary image based on Perlin noise.

    Args:
        image (Image): Primary PIL Image object to process.
        secondary_image (Image): Secondary PIL Image for replacement pixels.
        noise_scale (float): Scale of Perlin noise.
        threshold (float): Noise threshold for replacement (0 to 1).
        seed (int, optional): Seed for the Perlin noise generator. If None, a random pattern is generated.

    Returns:
        Image: Processed image with noise-based pixel replacement.
    """
    # Try to import noise packages, with fallbacks
    noise_module = None
    try:
        # Try the noise package first
        from noise import pnoise2
        noise_module = 'noise'
    except ImportError:
        try:
            # Try noise-python as a fallback
            from noise_python import snoise2
            noise_module = 'noise-python'
        except ImportError:
            raise ImportError("Either 'noise' or 'noise-python' package is required for Perlin Merge. Install with: pip install noise noise-python")
    
    # Convert both images to RGB mode if they have alpha channels or are in different modes
    if image.mode != 'RGB':
        image = image.convert('RGB')
    if secondary_image.mode != 'RGB':
        secondary_image = secondary_image.convert('RGB')
    
    # Resize secondary image to match primary image
    secondary_image = secondary_image.resize(image.size)
    image_array = np.array(image)
    secondary_array = np.array(secondary_image)
    
    # Generate Perlin noise map for the entire image
    noise_map = np.zeros((image.height, image.width))
    for i in range(image.height):
        for j in range(image.width):
            # Use the appropriate noise function based on which package is available
            if noise_module == 'noise':
                from noise import pnoise2
                # Use seed as base parameter if provided
                if seed is not None:
                    noise_map[i, j] = pnoise2(i * noise_scale, j * noise_scale, base=seed)
                else:
                    noise_map[i, j] = pnoise2(i * noise_scale, j * noise_scale)
            else:  # noise-python
                from noise_python import snoise2
                # Use seed as octaves parameter if provided (not ideal but works as a seed)
                if seed is not None:
                    noise_map[i, j] = snoise2(i * noise_scale, j * noise_scale, octaves=seed % 10 + 1)
                else:
                    noise_map[i, j] = snoise2(i * noise_scale, j * noise_scale)
    
    # Normalize noise map to [0, 1]
    noise_map = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
    
    # Replace pixels where noise exceeds threshold
    mask = noise_map > threshold
    image_array[mask] = secondary_array[mask]
    
    return Image.fromarray(image_array)

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

def bit_manipulation(image, chunk_size=1):
    """
    Apply a bit manipulation effect by inverting bytes of the image data in chunks.
    
    Args:
        image (Image): PIL Image object to process.
        chunk_size (int): Size of chunks to process together (1=every other byte, 2=every other 2 bytes, etc.)
    
    Returns:
        Image: Processed image with bit-level glitches.
    """
    # Convert image to RGB mode if it's not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Get image data as bytes
    image_bytes = bytearray(image.tobytes())
    
    # Manipulate bytes in chunks
    chunk_size = max(1, chunk_size)  # Ensure chunk_size is at least 1
    chunk_total = chunk_size * 2  # Total size of a chunk pair (manipulated + skipped)
    
    for i in range(0, len(image_bytes), chunk_total):
        # Manipulate 'chunk_size' bytes, then skip 'chunk_size' bytes
        for j in range(chunk_size):
            if i + j < len(image_bytes):
                image_bytes[i + j] = image_bytes[i + j] ^ 0xFF  # XOR with 0xFF to invert bits
    
    # Create a new image from the manipulated bytes
    manipulated_image = Image.frombytes('RGB', image.size, bytes(image_bytes))
    return manipulated_image

def spiral_coords(size):
    """
    Generate coordinates in spiral order starting from the center of a square chunk.
    
    Args:
        size (int): Size of the square chunk.
        
    Yields:
        tuple: (x, y) coordinates in spiral order.
    """
    x, y = size // 2, size // 2  # Start at the center
    yield (x, y)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
    steps = 1
    dir_idx = 0
    while steps < size:
        for _ in range(2):  # Two sides per step increase (e.g., right then down)
            dx, dy = directions[dir_idx % 4]
            for _ in range(steps):
                x += dx
                y += dy
                if 0 <= x < size and 0 <= y < size:
                    yield (x, y)
            dir_idx += 1
        steps += 1

def spiral_sort(image, chunk_size=32, order='lightest-to-darkest'):
    """
    Apply a spiral sort effect, arranging pixels in a spiral pattern based on brightness.
    
    Args:
        image (Image): PIL Image object to process.
        chunk_size (int): Size of square chunks to process.
        order (str): 'lightest-to-darkest' or 'darkest-to-lightest'.
    
    Returns:
        Image: Processed image with spiral-sorted pixels.
    """
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert PIL image to numpy array
    np_image = np.array(image)
    
    # Get image dimensions
    height, width, _ = np_image.shape
    
    # Adjust chunk_size if needed to ensure it divides evenly into the image
    if height % chunk_size != 0 or width % chunk_size != 0:
        # Find the largest chunk size that divides evenly
        for i in range(chunk_size, 0, -1):
            if height % i == 0 and width % i == 0:
                chunk_size = i
                break
    
    # Split the image into chunks
    chunks = []
    for y in range(0, height, chunk_size):
        for x in range(0, width, chunk_size):
            if y + chunk_size <= height and x + chunk_size <= width:
                chunks.append(np_image[y:y+chunk_size, x:x+chunk_size])
    
    # Sort each chunk
    sorted_chunks = []
    for chunk in chunks:
        # Flatten the chunk and calculate luminance
        flattened_chunk = chunk.reshape(-1, chunk.shape[-1])
        luminance = np.mean(flattened_chunk, axis=-1)
        sorted_indices = np.argsort(luminance)
        
        # Reverse order if needed
        if order == 'darkest-to-lightest':
            sorted_indices = sorted_indices[::-1]
        
        # Create a new chunk with pixels arranged in a spiral
        sorted_chunk = np.zeros_like(chunk)
        spiral_order = list(spiral_coords(chunk_size))
        
        # Place pixels in spiral order
        for idx, coord in zip(sorted_indices, spiral_order):
            pixel_y, pixel_x = divmod(idx, chunk_size)
            sorted_chunk[coord[0], coord[1]] = chunk[pixel_y, pixel_x]
        
        sorted_chunks.append(sorted_chunk)
    
    # Recombine chunks into the final image
    result = np.zeros_like(np_image)
    chunk_idx = 0
    for y in range(0, height, chunk_size):
        for x in range(0, width, chunk_size):
            if y + chunk_size <= height and x + chunk_size <= width:
                result[y:y+chunk_size, x:x+chunk_size] = sorted_chunks[chunk_idx]
                chunk_idx += 1
    
    # Convert back to PIL image
    return Image.fromarray(result)

def spiral_sort_2(image, chunk_size=64, sort_by='brightness', reverse=False):
    """
    Apply spiral sorting starting from the center of each chunk, with pixels sorted by the specified property.
    
    Args:
        image (Image): PIL Image object to process.
        chunk_size (int): Size of square chunks to process.
        sort_by (str): Property to sort by ('color', 'brightness', 'hue', 'red', 'green', 'blue',
                       'saturation', 'luminance', 'contrast').
        reverse (bool): Whether to reverse the sort order.
        
    Returns:
        Image: Processed image with spiral sorting.
    """
    # Convert PIL Image to NumPy array (RGB)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    
    # Get image dimensions
    height, width, channels = img_array.shape
    
    # Calculate padding needed to make dimensions multiples of chunk_size
    pad_y = (chunk_size - (height % chunk_size)) % chunk_size
    pad_x = (chunk_size - (width % chunk_size)) % chunk_size
    
    # Pad the image if necessary using edge padding (better for visual continuity)
    if pad_y != 0 or pad_x != 0:
        img_array = np.pad(img_array, ((0, pad_y), (0, pad_x), (0, 0)), mode='edge')
        padded_height, padded_width = img_array.shape[:2]
    else:
        padded_height, padded_width = height, width
    
    # Map sort_by parameter to the appropriate function from our centralized class
    sort_function = {
        'color': PixelAttributes.color_sum,
        'brightness': PixelAttributes.brightness,
        'hue': PixelAttributes.hue,
        'red': lambda p: p[0],
        'green': lambda p: p[1],
        'blue': lambda p: p[2],
        'saturation': PixelAttributes.saturation,
        'luminance': PixelAttributes.luminance,
        'contrast': PixelAttributes.contrast
    }.get(sort_by, PixelAttributes.brightness)  # Default to brightness if invalid choice
    
    # Calculate number of chunks
    num_chunks_y = padded_height // chunk_size
    num_chunks_x = padded_width // chunk_size
    
    # Generate spiral coordinates once
    spiral_coords_list = list(spiral_coords(chunk_size))
    total_pixels = chunk_size * chunk_size
    
    # Sort each chunk
    sorted_chunks = []
    for y in range(num_chunks_y):
        for x in range(num_chunks_x):
            # Extract chunk
            chunk = img_array[y*chunk_size:(y+1)*chunk_size, x*chunk_size:(x+1)*chunk_size]
            
            # Flatten the chunk and calculate sort values
            flattened_chunk = chunk.reshape(-1, channels)
            sort_values = np.array([sort_function(p) for p in flattened_chunk])
            
            # Sort pixels based on the sort values
            sorted_indices = np.argsort(sort_values)
            if reverse:
                sorted_indices = sorted_indices[::-1]
            
            # Create a new chunk with pixels arranged in a spiral
            sorted_chunk = np.zeros_like(chunk)
            
            # Place sorted pixels in spiral order
            for idx, (row, col) in zip(sorted_indices[:total_pixels], spiral_coords_list):
                sorted_chunk[row, col] = flattened_chunk[idx]
            
            sorted_chunks.append(sorted_chunk)
    
    # Recombine chunks into the final image
    result = np.zeros((padded_height, padded_width, channels), dtype=img_array.dtype)
    chunk_idx = 0
    for y in range(num_chunks_y):
        for x in range(num_chunks_x):
            result[y*chunk_size:(y+1)*chunk_size, x*chunk_size:(x+1)*chunk_size] = sorted_chunks[chunk_idx]
            chunk_idx += 1
    
    # Crop to original size if padded
    if pad_y != 0 or pad_x != 0:
        result = result[:height, :width]
    
    # Convert back to PIL image
    return Image.fromarray(result)

def generate_output_filename(original_filename, effect, settings):
    """
    Generate a descriptive filename for the processed image.
    
    Args:
        original_filename (str): Original image filename.
        effect (str): Name of the applied effect.
        settings (str): String representing effect settings.
    
    Returns:
        str: New filename with effect and settings appended.
    """
    base, ext = os.path.splitext(original_filename)
    return f"{base}_{effect}_{settings}{ext}"

def full_frame_sort(image, direction='vertical', sort_by='brightness', reverse=False):
    """
    Apply full-frame pixel sorting in the specified direction.
    
    Args:
        image (Image): PIL Image object to process.
        direction (str): Direction of sorting ('vertical', 'horizontal').
        sort_by (str): Property to sort by ('color', 'brightness', 'hue', 'red', 'green', 'blue',
                       'saturation', 'luminance', 'contrast').
        reverse (bool): Whether to reverse the sort order.
    
    Returns:
        Image: Processed image with full-frame sorting.
    """
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Create a new image with the same size as the input image
    width, height = image.size
    sorted_im = Image.new(image.mode, image.size)
    
    # Define the sort function based on the sort_by parameter
    sort_function = {
        'color': PixelAttributes.color_sum,
        'brightness': PixelAttributes.brightness,
        'hue': PixelAttributes.hue,
        'red': lambda p: p[0],     # Sort by red channel only
        'green': lambda p: p[1],   # Sort by green channel only
        'blue': lambda p: p[2],     # Sort by blue channel only
        'saturation': PixelAttributes.saturation,  # Sort by color saturation
        'luminance': PixelAttributes.luminance,    # Sort by luminance (value in HSV)
        'contrast': PixelAttributes.contrast       # Sort by contrast (max-min RGB)
    }.get(sort_by, PixelAttributes.brightness)
    
    if direction == 'vertical':
        # Sort each column from top to bottom
        for x in range(width):
            # Get the pixels in the current column
            column_pixels = [(image.getpixel((x, y)), y) for y in range(height)]
            
            # Sort the pixels by the specified criteria
            column_pixels.sort(key=lambda item: sort_function(item[0]), reverse=reverse)
            
            # Set the pixels in the current column of the output image
            for new_y, (pixel, _) in enumerate(column_pixels):
                sorted_im.putpixel((x, new_y), pixel)
    
    elif direction == 'horizontal':
        # Sort each row from left to right
        for y in range(height):
            # Get the pixels in the current row
            row_pixels = [(image.getpixel((x, y)), x) for x in range(width)]
            
            # Sort the pixels by the specified criteria
            row_pixels.sort(key=lambda item: sort_function(item[0]), reverse=reverse)
            
            # Set the pixels in the current row of the output image
            for new_x, (pixel, _) in enumerate(row_pixels):
                sorted_im.putpixel((new_x, y), pixel)
    
    return sorted_im

def polar_sorting(image, chunk_size, sort_by='angle', reverse=False):
    """
    Apply polar sorting to an image by sorting pixels within chunks based on polar coordinates.

    Args:
        image (Image): PIL Image object to process.
        chunk_size (int): Size of square chunks (e.g., 32 for 32x32 chunks).
        sort_by (str): 'angle' to sort by angle, 'radius' to sort by distance from center.
        reverse (bool): Whether to reverse the sort order.

    Returns:
        Image: Processed image with polar-sorted pixels.
    """
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    width, height = image.size
    # Split image into chunks
    chunks = []
    for y in range(0, height, chunk_size):
        for x in range(0, width, chunk_size):
            # Calculate actual chunk dimensions (handle edge cases)
            actual_width = min(chunk_size, width - x)
            actual_height = min(chunk_size, height - y)
            if actual_width > 0 and actual_height > 0:
                chunks.append((image.crop((x, y, x + actual_width, y + actual_height)), (x, y)))

    sorted_image = Image.new('RGB', image.size)
    
    for chunk, (x_offset, y_offset) in chunks:
        chunk_width, chunk_height = chunk.size
        chunk_array = np.array(chunk)
        
        # Calculate polar coordinates relative to chunk center
        cx, cy = chunk_width // 2, chunk_height // 2
        y_coords, x_coords = np.mgrid[:chunk_height, :chunk_width]
        angles = np.arctan2(y_coords - cy, x_coords - cx)  # Angle from center
        radii = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)  # Distance from center
        
        # Create a mapping of original positions to sorted positions
        positions = []
        for y in range(chunk_height):
            for x in range(chunk_width):
                pixel = chunk_array[y, x]
                angle = angles[y, x]
                radius = radii[y, x]
                sort_value = angle if sort_by == 'angle' else radius
                positions.append(((y, x), sort_value, pixel))
        
        # Sort by the chosen coordinate
        positions.sort(key=lambda p: p[1], reverse=reverse)
        
        # Create sorted chunk
        sorted_chunk = np.zeros_like(chunk_array)
        for i, ((orig_y, orig_x), _, pixel) in enumerate(positions):
            new_y = i // chunk_width
            new_x = i % chunk_width
            if new_y < chunk_height and new_x < chunk_width:
                sorted_chunk[new_y, new_x] = pixel
        
        # Convert back to PIL Image and paste into the result
        sorted_chunk_img = Image.fromarray(sorted_chunk)
        sorted_image.paste(sorted_chunk_img, (x_offset, y_offset))
    
    return sorted_image

def perlin_noise_sorting(image, chunk_size=32, noise_scale=0.1, direction='horizontal', reverse=False, seed=None):
    """
    Apply Perlin noise-based sorting to an image by using noise values to sort pixels in chunks.

    Args:
        image (Image): PIL Image object to process.
        chunk_size (int or str): Size of chunks. Can be an integer for square chunks or a string 'widthxheight'.
        noise_scale (float): Scale of Perlin noise (higher = more detailed noise).
        direction (str): 'horizontal' or 'vertical' sorting direction.
        reverse (bool): Whether to reverse the sort order.
        seed (int, optional): Seed for the Perlin noise generator. If None, a random pattern is generated.

    Returns:
        Image: Processed image with noise-sorted pixels.
    """
    # Try to import noise packages, with fallbacks
    noise_module = None
    try:
        # Try the noise package first
        from noise import pnoise2
        noise_module = 'noise'
    except ImportError:
        try:
            # Try noise-python as a fallback
            from noise_python import snoise2
            noise_module = 'noise-python'
        except ImportError:
            raise ImportError("Either 'noise' or 'noise-python' package is required for Perlin noise sorting. Install with: pip install noise noise-python")
    
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    width, height = image.size
    
    # Parse chunk size
    if isinstance(chunk_size, str) and 'x' in chunk_size:
        # Format: 'widthxheight'
        chunk_width, chunk_height = map(int, chunk_size.split('x'))
    else:
        # Square chunks
        chunk_width = chunk_height = int(chunk_size)
    
    # Handle special case for full image sorting
    if chunk_width >= width and chunk_height >= height:
        chunk_width, chunk_height = width, height
    
    # Split image into chunks
    chunks = []
    for y in range(0, height, chunk_height):
        for x in range(0, width, chunk_width):
            # Calculate actual chunk dimensions (handle edge cases)
            actual_width = min(chunk_width, width - x)
            actual_height = min(chunk_height, height - y)
            if actual_width > 0 and actual_height > 0:
                chunks.append((image.crop((x, y, x + actual_width, y + actual_height)), (x, y)))

    sorted_image = Image.new('RGB', image.size)
    
    for chunk, (x_offset, y_offset) in chunks:
        chunk_width_actual, chunk_height_actual = chunk.size
        chunk_array = np.array(chunk)
        
        # Generate Perlin noise map for the chunk
        noise_map = np.zeros((chunk_height_actual, chunk_width_actual))
        for i in range(chunk_height_actual):
            for j in range(chunk_width_actual):
                # Add some variation based on chunk position for more interesting results
                x_noise = (j + x_offset) * noise_scale
                y_noise = (i + y_offset) * noise_scale
                
                # Use the appropriate noise function based on which package is available
                if noise_module == 'noise':
                    from noise import pnoise2
                    # Use seed as base parameter if provided
                    if seed is not None:
                        noise_map[i, j] = pnoise2(x_noise, y_noise, base=seed)
                    else:
                        noise_map[i, j] = pnoise2(x_noise, y_noise)
                else:  # noise-python
                    from noise_python import snoise2
                    # Use seed as octaves parameter if provided (not ideal but works as a seed)
                    if seed is not None:
                        noise_map[i, j] = snoise2(x_noise, y_noise, octaves=seed % 10 + 1)
                    else:
                        noise_map[i, j] = snoise2(x_noise, y_noise)
        
        if direction == 'horizontal':
            # Sort each row by noise values
            sorted_chunk = np.zeros_like(chunk_array)
            for i in range(chunk_height_actual):
                row_pixels = [(j, chunk_array[i, j]) for j in range(chunk_width_actual)]
                row_noise = noise_map[i, :]
                
                # Sort pixels by noise values
                noise_pixel_pairs = list(zip(row_noise, row_pixels))
                noise_pixel_pairs.sort(key=lambda x: x[0], reverse=reverse)
                
                # Place sorted pixels back into the row
                for new_j, (_, (_, pixel)) in enumerate(noise_pixel_pairs):
                    sorted_chunk[i, new_j] = pixel
        else:  # vertical
            # Sort each column by noise values
            sorted_chunk = np.zeros_like(chunk_array)
            for j in range(chunk_width_actual):
                col_pixels = [(i, chunk_array[i, j]) for i in range(chunk_height_actual)]
                col_noise = noise_map[:, j]
                
                # Sort pixels by noise values
                noise_pixel_pairs = list(zip(col_noise, col_pixels))
                noise_pixel_pairs.sort(key=lambda x: x[0], reverse=reverse)
                
                # Place sorted pixels back into the column
                for new_i, (_, (_, pixel)) in enumerate(noise_pixel_pairs):
                    sorted_chunk[new_i, j] = pixel
        
        # Convert back to PIL Image and paste into the result
        sorted_chunk_img = Image.fromarray(sorted_chunk)
        sorted_image.paste(sorted_chunk_img, (x_offset, y_offset))
    
    return sorted_image

def perlin_full_frame_sort(image, noise_scale=0.01, sort_by='brightness', reverse=False, seed=None):
    """
    Apply full-frame pixel sorting controlled by Perlin noise.
    
    Args:
        image (Image): PIL Image object to process.
        noise_scale (float): Scale of Perlin noise (higher = more detailed noise).
        sort_by (str): Property to sort by ('color', 'brightness', 'hue', 'red', 'green', 'blue',
                       'saturation', 'luminance', 'contrast').
        reverse (bool): Whether to reverse the sort order.
        seed (int, optional): Seed for the Perlin noise generator. If None, a random pattern is generated.
    
    Returns:
        Image: Processed image with Perlin noise-controlled full-frame sorting.
    """
    # Try to import noise packages, with fallbacks
    noise_module = None
    try:
        # Try the noise package first
        from noise import pnoise2
        noise_module = 'noise'
    except ImportError:
        try:
            # Try noise-python as a fallback
            from noise_python import snoise2
            noise_module = 'noise-python'
        except ImportError:
            raise ImportError("Either 'noise' or 'noise-python' package is required for Perlin noise sorting. Install with: pip install noise noise-python")
    
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Create a new image with the same size as the input image
    width, height = image.size
    sorted_im = Image.new(image.mode, image.size)
    
    # Define the sort function based on the sort_by parameter
    sort_function = {
        'color': PixelAttributes.color_sum,
        'brightness': PixelAttributes.brightness,
        'hue': PixelAttributes.hue,
        'red': lambda p: p[0],     # Sort by red channel only
        'green': lambda p: p[1],   # Sort by green channel only
        'blue': lambda p: p[2],     # Sort by blue channel only
        'saturation': PixelAttributes.saturation,  # Sort by color saturation
        'luminance': PixelAttributes.luminance,    # Sort by luminance (value in HSV)
        'contrast': PixelAttributes.contrast       # Sort by contrast (max-min RGB)
    }.get(sort_by, PixelAttributes.brightness)  # Default to brightness if invalid choice
    
    # Generate Perlin noise map for the entire image
    noise_map = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            # Use the appropriate noise function based on which package is available
            if noise_module == 'noise':
                from noise import pnoise2
                # Use seed as base parameter if provided
                if seed is not None:
                    noise_map[y, x] = pnoise2(x * noise_scale, y * noise_scale, base=seed)
                else:
                    noise_map[y, x] = pnoise2(x * noise_scale, y * noise_scale)
            else:  # noise-python
                from noise_python import snoise2
                # Use seed as octaves parameter if provided (not ideal but works as a seed)
                if seed is not None:
                    noise_map[y, x] = snoise2(x * noise_scale, y * noise_scale, octaves=seed % 10 + 1)
                else:
                    noise_map[y, x] = snoise2(x * noise_scale, y * noise_scale)
    
    # Normalize noise map to [0, 1]
    noise_map = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
    
    # Sort each column based on Perlin noise values
    for x in range(width):
        # Get the pixels in the current column
        column_pixels = [(image.getpixel((x, y)), y, noise_map[y, x]) for y in range(height)]
        
        # Sort the pixels by Perlin noise value first, then by the specified criteria
        column_pixels.sort(key=lambda item: (item[2], sort_function(item[0])), reverse=reverse)
        
        # Set the pixels in the current column of the output image
        for new_y, (pixel, _, _) in enumerate(column_pixels):
            sorted_im.putpixel((x, new_y), pixel)
    
    return sorted_im

def pixelate_by_mode(image, pixel_width=8, pixel_height=8, attribute='color', num_bins=100):
    """
    Pixelate an image by creating larger blocks of pixels based on various attributes.
    
    Args:
        image (Image): PIL Image object to process.
        pixel_width (int): Width of each pixel block.
        pixel_height (int): Height of each pixel block.
        attribute (str): Attribute to use for determining pixel color ('color', 'brightness', 'hue', etc.).
        num_bins (int): Number of bins to use for quantizing attribute values.
    
    Returns:
        Image: A new pixelated image.
    """
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
    elif attribute == 'luminance':
        def attr_func(pixel):
            return PixelAttributes.luminance(pixel)
    
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

def concentric_shapes(image, num_points=5, shape_type='circle', thickness=3, spacing=10,
                      rotation_angle=0, darken_step=0, color_shift=0):
    """
    Generates concentric shapes from random points in the image.

    :param image: The PIL Image to process.
    :param num_points: Number of random pixels to select.
    :param shape_type: Type of shape ('square', 'circle', 'hexagon', 'triangle').
    :param thickness: Thickness of the shapes in pixels.
    :param spacing: Spacing between shapes in pixels.
    :param rotation_angle: Incremental rotation angle in degrees for each subsequent shape.
    :param darken_step: Amount to darken the color for each subsequent shape (0-255).
    :param color_shift: Amount to shift the hue for each shape (0-360 degrees).
    :return: The processed image.
    """
    from PIL import ImageDraw

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

        # No need to update previous_image or use ImageChops.difference

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

def color_shift_expansion(image, num_points=5, shift_amount=5, expansion_type='square', mode='xtreme'):
    """
    Expands color shifts from seed points across the image, shifting the existing pixel colors.

    :param image: The PIL Image to process.
    :param num_points: Number of seed points to generate.
    :param shift_amount: Amount to shift the hue per unit distance.
    :param expansion_type: Type of expansion ('square', 'cross', 'circular').
    :param mode: 'classic' or 'xtreme' mode.
    :return: The processed image.
    """
    width, height = image.size
    image = image.convert('RGB')
    image_np = np.array(image)

    # Convert image to HSV
    image_hsv = np.zeros((height, width, 3), dtype=float)
    for y in range(height):
        for x in range(width):
            r, g, b = image_np[y, x]
            h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            image_hsv[y, x] = [h * 360, s, v]

    # Initialize assigned pixels mask
    assigned = np.full((height, width), False, dtype=bool)

    # Generate seed points
    xs = np.random.randint(0, width, size=num_points)
    ys = np.random.randint(0, height, size=num_points)
    seed_points = list(zip(xs, ys))

    # Initialize distance map with infinity
    distance_map = np.full((height, width), np.inf)
    # Initialize shift direction map
    shift_direction_map = np.full((height, width), '', dtype=object)
    # Create a queue for BFS
    queue = []

    for idx, (x0, y0) in enumerate(seed_points):
        # Randomly choose shift direction for each seed point
        shift_direction = random.choice(['add', 'subtract'])
        # Initialize seed point
        assigned[y0, x0] = True
        distance_map[y0, x0] = 0
        shift_direction_map[y0, x0] = shift_direction
        # Enqueue the seed point
        heapq.heappush(queue, (0, x0, y0))

    # Define neighbor offsets
    if expansion_type == 'square':
        neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                            (0, -1),          (0, 1),
                            (1, -1),  (1, 0), (1, 1)]
    elif expansion_type == 'cross':
        neighbor_offsets = [(-1, 0),
                            (0, -1),        (0, 1),
                            (1, 0)]
    elif expansion_type == 'circular':
        neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                            (0, -1),          (0, 1),
                            (1, -1),  (1, 0), (1, 1)]
    else:
        print(f"Invalid expansion type: {expansion_type}. Defaulting to square.")
        neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                            (0, -1),          (0, 1),
                            (1, -1),  (1, 0), (1, 1)]

    # Perform BFS
    while queue:
        current_distance, x, y = heapq.heappop(queue)
        current_distance = distance_map[y, x]
        shift_direction = shift_direction_map[y, x]

        for dx, dy in neighbor_offsets:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                new_distance = current_distance + np.hypot(dx, dy) if expansion_type == 'circular' else current_distance + 1
                if not assigned[ny, nx] or new_distance < distance_map[ny, nx]:
                    assigned[ny, nx] = True
                    distance_map[ny, nx] = new_distance
                    shift_direction_map[ny, nx] = shift_direction
                    heapq.heappush(queue, (new_distance, nx, ny))

    # Apply hue shifts
    for y in range(height):
        for x in range(width):
            existing_h, existing_s, existing_v = image_hsv[y, x]
            shift_direction = shift_direction_map[y, x]
            distance = distance_map[y, x]
            if mode == 'xtreme':
                hue_shift = shift_amount * distance
            else:  # classic
                hue_shift = shift_amount
            if shift_direction == 'add':
                new_hue = (existing_h + hue_shift) % 360
            else:
                new_hue = (existing_h - hue_shift) % 360
            image_hsv[y, x] = [new_hue, existing_s, existing_v]

    # Convert back to RGB
    for y in range(height):
        for x in range(width):
            h, s, v = image_hsv[y, x]
            r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
            image_np[y, x] = [int(r * 255), int(g * 255), int(b * 255)]

    # Convert back to PIL Image
    processed_image = Image.fromarray(image_np.astype(np.uint8))
    return processed_image

def perlin_noise_displacement(image, scale=100, intensity=30, octaves=6, persistence=0.5, lacunarity=2.0):
    """
    Applies a Perlin noise-based displacement to the image.

    :param image: PIL Image to process.
    :param scale: Scale of the Perlin noise.
    :param intensity: Maximum displacement in pixels.
    :param octaves: Number of layers of noise.
    :param persistence: Amplitude of each octave.
    :param lacunarity: Frequency of each octave.
    :return: Displaced PIL Image.
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

def load_image(file_path):
    """
    Load an image from the specified file path.
    
    Args:
        file_path (str): Path to the image file.
    
    Returns:
        Image or None: The loaded PIL Image object, or None if loading fails.
    """
    try:
        return Image.open(file_path)
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

def voronoi_pixel_sort(image, num_cells=100, size_variation=0.5, sort_by='color', sort_order='clockwise', seed=None, orientation='horizontal', start_position='left'):
    """
    Applies a Voronoi-based pixel sorting effect to the image.

    :param image: The PIL Image to process.
    :param num_cells: Approximate number of Voronoi cells.
    :param size_variation: Variability in cell sizes (0 to 1).
    :param sort_by: Property to sort by ('color', 'brightness', 'hue').
    :param sort_order: Sorting order ('clockwise' or 'counter-clockwise').
    :param seed: Random seed for reproducibility.
    :param orientation: Direction for sorting within cells ('horizontal', 'vertical', 'radial', 'spiral').
    :param start_position: Starting position for the sort ('left', 'right', 'top', 'bottom', 'center').
    :return: The processed image.
    """
    import numpy as np
    from scipy.spatial import cKDTree

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

def pixel_scatter(image, direction, select_by, min_val, max_val):
    """
    Scatter selected pixels horizontally or vertically based on a specified criterion.

    Args:
        image (PIL.Image): The input image.
        direction (str): 'horizontal' to scatter within rows, 'vertical' to scatter within columns.
        select_by (str): Criterion for selecting pixels ('brightness', 'red', 'green', 'blue', 'hue', 
                         'saturation', 'luminance', 'contrast').
        min_val (float): Minimum value for selection (e.g., 0-255 for brightness, 0-360 for hue).
        max_val (float): Maximum value for selection.

    Returns:
        PIL.Image: The image with scattered pixels.
    """
    # Convert image to RGB mode if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    # Convert image to NumPy array
    arr = np.array(image)
    height, width, _ = arr.shape

    # Define selection function based on select_by parameter
    if select_by == 'brightness':
        # Calculate brightness using the standard formula
        brightness = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        mask = (brightness >= min_val) & (brightness <= max_val)
    elif select_by == 'red':
        mask = (arr[:, :, 0] >= min_val) & (arr[:, :, 0] <= max_val)
    elif select_by == 'green':
        mask = (arr[:, :, 1] >= min_val) & (arr[:, :, 1] <= max_val)
    elif select_by == 'blue':
        mask = (arr[:, :, 2] >= min_val) & (arr[:, :, 2] <= max_val)
    elif select_by == 'hue':
        # Compute hue for each pixel
        def rgb_to_hue(pixel):
            r, g, b = pixel / 255.0
            h, _, _ = colorsys.rgb_to_hsv(r, g, b)
            return h * 360  # Hue in degrees
        
        # Apply function to each pixel
        hue = np.apply_along_axis(rgb_to_hue, 2, arr)
        mask = (hue >= min_val) & (hue <= max_val)
    elif select_by == 'saturation':
        # Compute saturation for each pixel
        def rgb_to_saturation(pixel):
            r, g, b = pixel / 255.0
            _, s, _ = colorsys.rgb_to_hsv(r, g, b)
            return s * 100  # Scale to 0-100
        
        saturation = np.apply_along_axis(rgb_to_saturation, 2, arr)
        mask = (saturation >= min_val) & (saturation <= max_val)
    elif select_by == 'luminance':
        # Compute luminance (value in HSV) for each pixel
        def rgb_to_luminance(pixel):
            r, g, b = pixel / 255.0
            _, _, v = colorsys.rgb_to_hsv(r, g, b)
            return v * 100  # Scale to 0-100
        
        luminance = np.apply_along_axis(rgb_to_luminance, 2, arr)
        mask = (luminance >= min_val) & (luminance <= max_val)
    elif select_by == 'contrast':
        # Compute contrast for each pixel
        def calculate_contrast(pixel):
            return max(pixel) - min(pixel)
        
        contrast = np.apply_along_axis(calculate_contrast, 2, arr)
        mask = (contrast >= min_val) & (contrast <= max_val)
    else:
        raise ValueError("Invalid select_by. Choose 'brightness', 'red', 'green', 'blue', 'hue', 'saturation', 'luminance', or 'contrast'.")

    # Perform scattering
    if direction == 'horizontal':
        for y in range(height):
            selected_xs = np.where(mask[y, :])[0]
            if len(selected_xs) > 0:  # Only process if there are selected pixels
                for x in selected_xs:
                    x_prime = np.random.randint(0, width)
                    arr[y, x], arr[y, x_prime] = arr[y, x_prime].copy(), arr[y, x].copy()
    elif direction == 'vertical':
        for x in range(width):
            selected_ys = np.where(mask[:, x])[0]
            if len(selected_ys) > 0:  # Only process if there are selected pixels
                for y in selected_ys:
                    y_prime = np.random.randint(0, height)
                    arr[y, x], arr[y_prime, x] = arr[y_prime, x].copy(), arr[y, x].copy()
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'.")

    # Convert back to PIL Image
    return Image.fromarray(arr)

def contrast(pixel):
    """Calculate the contrast of a pixel"""
    return max(pixel[:3]) - min(pixel[:3])

def diagonal_pixel_sort(image, chunk_size, sort_by, corner):
    """
    Apply pixel sorting diagonally within each chunk, starting from a specified corner.
    
    Args:
        image (Image): PIL Image object to process.
        chunk_size (str): Chunk dimensions as 'widthxheight' (e.g., '32x32').
        sort_by (str): Property to sort by ('color', 'brightness', 'hue', etc.).
        corner (str): Starting corner ('top-left', 'top-right', 'bottom-left', 'bottom-right').
    
    Returns:
        Image: Processed image with diagonally sorted pixels.
    """
    # Convert to RGB mode if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Define sort function based on sort_by parameter
    sort_function = {
        'color': lambda p: sum(p[:3]),
        'brightness': brightness,
        'hue': hue,
        'red': lambda p: p[0],
        'green': lambda p: p[1],
        'blue': lambda p: p[2],
        'saturation': saturation,
        'luminance': luminance,
        'contrast': contrast
    }.get(sort_by, lambda p: sum(p[:3]))
    
    # Create a new image for the result
    result_image = Image.new(image.mode, image.size)
    width, height = image.size
    chunk_width, chunk_height = map(int, chunk_size.split('x'))
    
    # Get pixel data
    pixels = list(image.getdata())
    pixels_2d = []
    for y in range(height):
        row = []
        for x in range(width):
            row.append(pixels[y * width + x])
        pixels_2d.append(row)
    
    # Calculate chunks
    num_chunks_x = width // chunk_width
    num_chunks_y = height // chunk_height
    remainder_x = width % chunk_width
    remainder_y = height % chunk_height
    
    # Process each chunk
    for chunk_row in range(num_chunks_y):
        for chunk_col in range(num_chunks_x):
            start_y = chunk_row * chunk_height
            start_x = chunk_col * chunk_width
            
            # Get diagonal lines within this chunk based on corner
            diagonals = []
            
            if corner == 'top-left':
                # Get all diagonals from top-left to bottom-right
                for offset in range(chunk_width + chunk_height - 1):
                    diagonal = []
                    for d in range(max(offset - chunk_height + 1, 0), min(offset + 1, chunk_width)):
                        x = start_x + d
                        y = start_y + offset - d
                        if (0 <= y < start_y + chunk_height and 
                            0 <= x < start_x + chunk_width):
                            diagonal.append(pixels_2d[y][x])
                    diagonals.append(diagonal)
            
            elif corner == 'top-right':
                # Get all diagonals from top-right to bottom-left
                for offset in range(chunk_width + chunk_height - 1):
                    diagonal = []
                    for d in range(max(offset - chunk_height + 1, 0), min(offset + 1, chunk_width)):
                        x = start_x + chunk_width - 1 - d
                        y = start_y + offset - d
                        if (0 <= y < start_y + chunk_height and 
                            start_x <= x < start_x + chunk_width):
                            diagonal.append(pixels_2d[y][x])
                    diagonals.append(diagonal)
            
            elif corner == 'bottom-left':
                # Get all diagonals from bottom-left to top-right
                for offset in range(chunk_width + chunk_height - 1):
                    diagonal = []
                    for d in range(max(offset - chunk_height + 1, 0), min(offset + 1, chunk_width)):
                        x = start_x + d
                        y = start_y + chunk_height - 1 - (offset - d)
                        if (start_y <= y < start_y + chunk_height and 
                            0 <= x < start_x + chunk_width):
                            diagonal.append(pixels_2d[y][x])
                    diagonals.append(diagonal)
            
            elif corner == 'bottom-right':
                # Get all diagonals from bottom-right to top-left
                for offset in range(chunk_width + chunk_height - 1):
                    diagonal = []
                    for d in range(max(offset - chunk_height + 1, 0), min(offset + 1, chunk_width)):
                        x = start_x + chunk_width - 1 - d
                        y = start_y + chunk_height - 1 - (offset - d)
                        if (start_y <= y < start_y + chunk_height and 
                            start_x <= x < start_x + chunk_width):
                            diagonal.append(pixels_2d[y][x])
                    diagonals.append(diagonal)
            
            # Sort each diagonal
            sorted_diagonals = []
            for diagonal in diagonals:
                if diagonal:  # Only sort non-empty diagonals
                    sorted_diagonal = sorted(diagonal, key=sort_function)
                    sorted_diagonals.append(sorted_diagonal)
                else:
                    sorted_diagonals.append([])
            
            # Place the sorted diagonals back
            if corner == 'top-left':
                for i, diagonal in enumerate(sorted_diagonals):
                    idx = 0
                    for d in range(max(i - chunk_height + 1, 0), min(i + 1, chunk_width)):
                        x = start_x + d
                        y = start_y + i - d
                        if (0 <= y < start_y + chunk_height and 
                            0 <= x < start_x + chunk_width and 
                            idx < len(diagonal)):
                            result_image.putpixel((x, y), diagonal[idx])
                            idx += 1
            
            elif corner == 'top-right':
                for i, diagonal in enumerate(sorted_diagonals):
                    idx = 0
                    for d in range(max(i - chunk_height + 1, 0), min(i + 1, chunk_width)):
                        x = start_x + chunk_width - 1 - d
                        y = start_y + i - d
                        if (0 <= y < start_y + chunk_height and 
                            start_x <= x < start_x + chunk_width and 
                            idx < len(diagonal)):
                            result_image.putpixel((x, y), diagonal[idx])
                            idx += 1
            
            elif corner == 'bottom-left':
                for i, diagonal in enumerate(sorted_diagonals):
                    idx = 0
                    for d in range(max(i - chunk_height + 1, 0), min(i + 1, chunk_width)):
                        x = start_x + d
                        y = start_y + chunk_height - 1 - (i - d)
                        if (start_y <= y < start_y + chunk_height and 
                            0 <= x < start_x + chunk_width and 
                            idx < len(diagonal)):
                            result_image.putpixel((x, y), diagonal[idx])
                            idx += 1
            
            elif corner == 'bottom-right':
                for i, diagonal in enumerate(sorted_diagonals):
                    idx = 0
                    for d in range(max(i - chunk_height + 1, 0), min(i + 1, chunk_width)):
                        x = start_x + chunk_width - 1 - d
                        y = start_y + chunk_height - 1 - (i - d)
                        if (start_y <= y < start_y + chunk_height and 
                            start_x <= x < start_x + chunk_width and 
                            idx < len(diagonal)):
                            result_image.putpixel((x, y), diagonal[idx])
                            idx += 1
    
    # Handle remainder chunks on right edge
    if remainder_x > 0:
        for chunk_row in range(num_chunks_y):
            start_y = chunk_row * chunk_height
            start_x = num_chunks_x * chunk_width
            
            # Apply the same diagonal sorting but with adjusted width
            process_remainder_chunk(result_image, pixels_2d, start_x, start_y, 
                                   remainder_x, chunk_height, corner, sort_function)
    
    # Handle remainder chunks on bottom edge
    if remainder_y > 0:
        for chunk_col in range(num_chunks_x):
            start_y = num_chunks_y * chunk_height
            start_x = chunk_col * chunk_width
            
            # Apply the same diagonal sorting but with adjusted height
            process_remainder_chunk(result_image, pixels_2d, start_x, start_y, 
                                   chunk_width, remainder_y, corner, sort_function)
    
    # Handle corner remainder (if both remainders are non-zero)
    if remainder_x > 0 and remainder_y > 0:
        start_x = num_chunks_x * chunk_width
        start_y = num_chunks_y * chunk_height
        
        # Process the corner remainder
        process_remainder_chunk(result_image, pixels_2d, start_x, start_y, 
                               remainder_x, remainder_y, corner, sort_function)
    
    return result_image

def process_remainder_chunk(result_image, pixels_2d, start_x, start_y, width, height, corner, sort_function):
    """Helper function to process remainder chunks with diagonal sorting"""
    # Get diagonal lines within this chunk based on corner
    diagonals = []
    
    if corner == 'top-left':
        # Get all diagonals from top-left to bottom-right
        for offset in range(width + height - 1):
            diagonal = []
            for d in range(max(offset - height + 1, 0), min(offset + 1, width)):
                x = start_x + d
                y = start_y + offset - d
                if (0 <= y < start_y + height and 
                    0 <= x < start_x + width):
                    diagonal.append(pixels_2d[y][x])
            diagonals.append(diagonal)
    
    elif corner == 'top-right':
        # Get all diagonals from top-right to bottom-left
        for offset in range(width + height - 1):
            diagonal = []
            for d in range(max(offset - height + 1, 0), min(offset + 1, width)):
                x = start_x + width - 1 - d
                y = start_y + offset - d
                if (0 <= y < start_y + height and 
                    start_x <= x < start_x + width):
                    diagonal.append(pixels_2d[y][x])
            diagonals.append(diagonal)
    
    elif corner == 'bottom-left':
        # Get all diagonals from bottom-left to top-right
        for offset in range(width + height - 1):
            diagonal = []
            for d in range(max(offset - height + 1, 0), min(offset + 1, width)):
                x = start_x + d
                y = start_y + height - 1 - (offset - d)
                if (start_y <= y < start_y + height and 
                    0 <= x < start_x + width):
                    diagonal.append(pixels_2d[y][x])
            diagonals.append(diagonal)
    
    elif corner == 'bottom-right':
        # Get all diagonals from bottom-right to top-left
        for offset in range(width + height - 1):
            diagonal = []
            for d in range(max(offset - height + 1, 0), min(offset + 1, width)):
                x = start_x + width - 1 - d
                y = start_y + height - 1 - (offset - d)
                if (start_y <= y < start_y + height and 
                    start_x <= x < start_x + width):
                    diagonal.append(pixels_2d[y][x])
            diagonals.append(diagonal)
    
    # Sort each diagonal
    sorted_diagonals = []
    for diagonal in diagonals:
        if diagonal:  # Only sort non-empty diagonals
            sorted_diagonal = sorted(diagonal, key=sort_function)
            sorted_diagonals.append(sorted_diagonal)
        else:
            sorted_diagonals.append([])
    
    # Place the sorted diagonals back
    if corner == 'top-left':
        for i, diagonal in enumerate(sorted_diagonals):
            idx = 0
            for d in range(max(i - height + 1, 0), min(i + 1, width)):
                x = start_x + d
                y = start_y + i - d
                if (0 <= y < start_y + height and 
                    0 <= x < start_x + width and 
                    idx < len(diagonal)):
                    result_image.putpixel((x, y), diagonal[idx])
                    idx += 1
    
    elif corner == 'top-right':
        for i, diagonal in enumerate(sorted_diagonals):
            idx = 0
            for d in range(max(i - height + 1, 0), min(i + 1, width)):
                x = start_x + width - 1 - d
                y = start_y + i - d
                if (0 <= y < start_y + height and 
                    start_x <= x < start_x + width and 
                    idx < len(diagonal)):
                    result_image.putpixel((x, y), diagonal[idx])
                    idx += 1
    
    elif corner == 'bottom-left':
        for i, diagonal in enumerate(sorted_diagonals):
            idx = 0
            for d in range(max(i - height + 1, 0), min(i + 1, width)):
                x = start_x + d
                y = start_y + height - 1 - (i - d)
                if (start_y <= y < start_y + height and 
                    0 <= x < start_x + width and 
                    idx < len(diagonal)):
                    result_image.putpixel((x, y), diagonal[idx])
                    idx += 1
    
    elif corner == 'bottom-right':
        for i, diagonal in enumerate(sorted_diagonals):
            idx = 0
            for d in range(max(i - height + 1, 0), min(i + 1, width)):
                x = start_x + width - 1 - d
                y = start_y + height - 1 - (i - d)
                if (start_y <= y < start_y + height and 
                    start_x <= x < start_x + width and 
                    idx < len(diagonal)):
                    result_image.putpixel((x, y), diagonal[idx])
                    idx += 1
