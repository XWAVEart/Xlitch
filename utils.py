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
from scipy.spatial import Voronoi
from scipy.ndimage import distance_transform_edt as edt

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

def pixel_sorting(image, direction, chunk_size, sort_by, starting_corner=None, sort_order='ascending'):
    """
    Apply pixel sorting to an image by chunks, sorting pixels within each chunk based on a specific property.
    
    Args:
        image (Image): PIL Image object to process.
        direction (str): 'horizontal', 'vertical', or 'diagonal' sort direction.
        chunk_size (str): Chunk dimensions as 'widthxheight' (e.g., '32x32').
        sort_by (str): Property to sort by ('color', 'brightness', 'hue', 'red', 'green', 'blue',
                       'saturation', 'luminance', 'contrast').
        starting_corner (str, optional): Corner to start from for diagonal sorting
                                        ('top-left', 'top-right', 'bottom-left', 'bottom-right').
                                        Required only when direction is 'diagonal'.
        sort_order (str): 'ascending' (low to high) or 'descending' (high to low). Default is 'ascending'.
    
    Returns:
        Image: Processed image with sorted pixels.
    """
    # Convert to RGB mode if the image has an alpha channel or is in a different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Handle special case for diagonal sorting
    if direction == 'diagonal':
        if not starting_corner:
            raise ValueError("starting_corner is required for diagonal sorting")
        # Determine if horizontal is True based on starting corner
        horizontal = starting_corner in ['top-left', 'bottom-left']
        # Pass the sort order parameter to the diagonal sorting function
        reverse = (sort_order == 'descending')
        return pixel_sorting_corner_to_corner(image, chunk_size, sort_by, starting_corner, horizontal)
    
    # Helper functions for calculating pixel properties
    def brightness(pixel):
        return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]
    
    def hue(pixel):
        r, g, b = pixel[:3]
        r, g, b = r/255.0, g/255.0, b/255.0
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        if max_val == min_val:
            return 0  # Achromatic (gray)
        
        d = max_val - min_val
        if max_val == r:
            h = (g - b) / d + (6 if g < b else 0)
        elif max_val == g:
            h = (b - r) / d + 2
        else:
            h = (r - g) / d + 4
        
        h /= 6
        return h * 360  # Convert to degrees
    
    def saturation(pixel):
        r, g, b = pixel[:3]
        r, g, b = r/255.0, g/255.0, b/255.0
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        if max_val == 0:
            return 0
        return (max_val - min_val) / max_val
    
    def luminance(pixel):
        r, g, b = pixel[:3]
        r, g, b = r/255.0, g/255.0, b/255.0
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    
    def contrast(pixel):
        r, g, b = pixel[:3]
        return max(r, g, b) - min(r, g, b)
    
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
    
    # Determine whether to reverse the sort based on sort_order
    reverse = (sort_order == 'descending')
    
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
            
            # Sort chunk with the appropriate order
            if direction == 'horizontal':
                sorted_chunk = sorted(chunk_pixels, key=sort_function, reverse=reverse)
                
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
                    # Sort column with the appropriate order
                    sorted_column = sorted(column, key=sort_function, reverse=reverse)
                    
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
            
            # Sort the edge chunk with the appropriate order
            if direction == 'horizontal':
                sorted_edge = sorted(edge_chunk, key=sort_function, reverse=reverse)
                
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
                    column = [row[x] for row in edge_2d if x < len(row)]
                    # Sort column with the appropriate order
                    sorted_column = sorted(column, key=sort_function, reverse=reverse)
                    
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
            
            # Sort the edge chunk with the appropriate order
            if direction == 'horizontal':
                sorted_edge = sorted(edge_chunk, key=sort_function, reverse=reverse)
                
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
                    # Sort column with the appropriate order
                    sorted_column = sorted(column, key=sort_function, reverse=reverse)
                    
                    # Place sorted column back
                    for y, pixel in enumerate(sorted_column):
                        result_image.putpixel((start_x + x, start_y + y), pixel)
    
    # Process bottom-right corner (if there's a remainder in both width and height)
    if remainder_x > 0 and remainder_y > 0:
        # Extract the corner chunk
        start_y = num_chunks_y * chunk_height
        start_x = num_chunks_x * chunk_width
        
        corner_chunk = []
        for y in range(start_y, start_y + remainder_y):
            for x in range(start_x, start_x + remainder_x):
                corner_chunk.append(pixels[y * width + x])
        
        # Sort the corner chunk with the appropriate order
        if direction == 'horizontal':
            sorted_corner = sorted(corner_chunk, key=sort_function, reverse=reverse)
            
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
                column = [row[x] for row in corner_2d if x < len(row)]
                # Sort column with the appropriate order
                sorted_column = sorted(column, key=sort_function, reverse=reverse)
                
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

def color_shift_expansion(image, num_points=5, shift_amount=5, expansion_type='square', mode='xtreme', 
                        saturation_boost=0.0, value_boost=0.0, pattern_type='random', 
                        color_theme='full-spectrum', decay_factor=0.0):
    """
    Creates vibrant color transformations expanding from seed points across the image.

    :param image: The PIL Image to process.
    :param num_points: Number of seed points to generate.
    :param shift_amount: Intensity of the color effect.
    :param expansion_type: Type of expansion ('square', 'cross', 'circular').
    :param mode: Parameter kept for backward compatibility.
    :param saturation_boost: Amount to boost saturation (0.0-1.0).
    :param value_boost: Amount to boost value/brightness (0.0-1.0).
    :param pattern_type: Pattern for seedpoints ('random', 'grid', 'radial', 'spiral').
    :param color_theme: Color theme to use ('full-spectrum', 'warm', 'cool', 'complementary', 'analogous').
    :param decay_factor: Controls how the effect fades with distance (0.0-1.0). Higher values make the effect more
                         concentrated around seed points. Uses linear decay relative to image diagonal.
    :return: The processed image.
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
        img = Image.open(file_path)
        # Automatically resize large images for better performance
        return resize_image_if_needed(img)
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

def resize_image_if_needed(image, max_width=1920, max_height=1920):
    """
    Resize an image if its longest dimension exceeds the specified maximum,
    maintaining the original aspect ratio.
    
    Args:
        image (Image): PIL Image object to check and possibly resize
        max_width (int): Maximum allowed width in pixels (default 1920)
        max_height (int): Maximum allowed height in pixels (default 1920)
        
    Returns:
        Image: Original image or resized version if dimensions exceeded limits
    """
    # Try to get max dimensions from Flask app config if available
    try:
        from flask import current_app
        if current_app and current_app.config:
            max_width = current_app.config.get('MAX_IMAGE_WIDTH', max_width)
            max_height = current_app.config.get('MAX_IMAGE_HEIGHT', max_height)
    except (ImportError, RuntimeError):
        pass  # Not in a Flask context or app not available
    
    # Get current image dimensions
    width, height = image.size
    
    # Check if resizing is needed (if either dimension exceeds max)
    if width <= max_width and height <= max_height:
        return image
    
    # Calculate ratio based on the longest dimension
    ratio = min(max_width / width, max_height / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    
    # Resize the image using high quality resampling
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    return resized_image

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

def diagonal_pixel_sort(image, chunk_size, sort_by, corner, sort_order='ascending'):
    """
    Apply pixel sorting diagonally within each chunk, starting from a specified corner.
    
    Args:
        image (Image): PIL Image object to process.
        chunk_size (str): Chunk dimensions as 'widthxheight' (e.g., '32x32').
        sort_by (str): Property to sort by ('color', 'brightness', 'hue', etc.).
        corner (str): Starting corner ('top-left', 'top-right', 'bottom-left', 'bottom-right').
        sort_order (str): 'ascending' (low to high) or 'descending' (high to low). Default is 'ascending'.
    
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
    
    # Determine whether to reverse the sort based on sort_order
    reverse = (sort_order == 'descending')
    
    # Process each chunk
    for chunk_y in range(num_chunks_y):
        for chunk_x in range(num_chunks_x):
            start_x = chunk_x * chunk_width
            start_y = chunk_y * chunk_height
            
            # Process diagonals
            if corner == 'top-left':
                # Top-left to bottom-right diagonals
                for diag in range(chunk_width + chunk_height - 1):
                    diagonal_pixels = []
                    
                    # Collect pixels along this diagonal
                    for i in range(diag + 1):
                        x = start_x + i
                        y = start_y + diag - i
                        
                        if (x < start_x + chunk_width and 
                            y < start_y + chunk_height and 
                            y >= start_y and x >= start_x):
                            diagonal_pixels.append(pixels_2d[y][x])
                    
                    # Sort diagonal pixels
                    sorted_pixels = sorted(diagonal_pixels, key=sort_function, reverse=reverse)
                    
                    # Place sorted pixels back on the diagonal
                    pixel_index = 0
                    for i in range(diag + 1):
                        x = start_x + i
                        y = start_y + diag - i
                        
                        if (x < start_x + chunk_width and 
                            y < start_y + chunk_height and 
                            y >= start_y and x >= start_x):
                            if pixel_index < len(sorted_pixels):
                                result_image.putpixel((x, y), sorted_pixels[pixel_index])
                                pixel_index += 1
            
            elif corner == 'top-right':
                # Top-right to bottom-left diagonals
                for diag in range(chunk_width + chunk_height - 1):
                    diagonal_pixels = []
                    
                    # Collect pixels along this diagonal
                    for i in range(diag + 1):
                        x = start_x + chunk_width - 1 - i
                        y = start_y + diag - i
                        
                        if (x >= start_x and 
                            y < start_y + chunk_height and 
                            y >= start_y and x < start_x + chunk_width):
                            diagonal_pixels.append(pixels_2d[y][x])
                    
                    # Sort diagonal pixels
                    sorted_pixels = sorted(diagonal_pixels, key=sort_function, reverse=reverse)
                    
                    # Place sorted pixels back on the diagonal
                    pixel_index = 0
                    for i in range(diag + 1):
                        x = start_x + chunk_width - 1 - i
                        y = start_y + diag - i
                        
                        if (x >= start_x and 
                            y < start_y + chunk_height and 
                            y >= start_y and x < start_x + chunk_width):
                            if pixel_index < len(sorted_pixels):
                                result_image.putpixel((x, y), sorted_pixels[pixel_index])
                                pixel_index += 1
            
            elif corner == 'bottom-left':
                # Bottom-left to top-right diagonals
                for diag in range(chunk_width + chunk_height - 1):
                    diagonal_pixels = []
                    
                    # Collect pixels along this diagonal
                    for i in range(diag + 1):
                        x = start_x + i
                        y = start_y + chunk_height - 1 - (diag - i)
                        
                        if (x < start_x + chunk_width and 
                            y >= start_y and 
                            y < start_y + chunk_height and x >= start_x):
                            diagonal_pixels.append(pixels_2d[y][x])
                    
                    # Sort diagonal pixels
                    sorted_pixels = sorted(diagonal_pixels, key=sort_function, reverse=reverse)
                    
                    # Place sorted pixels back on the diagonal
                    pixel_index = 0
                    for i in range(diag + 1):
                        x = start_x + i
                        y = start_y + chunk_height - 1 - (diag - i)
                        
                        if (x < start_x + chunk_width and 
                            y >= start_y and 
                            y < start_y + chunk_height and x >= start_x):
                            if pixel_index < len(sorted_pixels):
                                result_image.putpixel((x, y), sorted_pixels[pixel_index])
                                pixel_index += 1
            
            else:  # bottom-right
                # Bottom-right to top-left diagonals
                for diag in range(chunk_width + chunk_height - 1):
                    diagonal_pixels = []
                    
                    # Collect pixels along this diagonal
                    for i in range(diag + 1):
                        x = start_x + chunk_width - 1 - i
                        y = start_y + chunk_height - 1 - (diag - i)
                        
                        if (x >= start_x and 
                            y >= start_y and 
                            y < start_y + chunk_height and x < start_x + chunk_width):
                            diagonal_pixels.append(pixels_2d[y][x])
                    
                    # Sort diagonal pixels
                    sorted_pixels = sorted(diagonal_pixels, key=sort_function, reverse=reverse)
                    
                    # Place sorted pixels back on the diagonal
                    pixel_index = 0
                    for i in range(diag + 1):
                        x = start_x + chunk_width - 1 - i
                        y = start_y + chunk_height - 1 - (diag - i)
                        
                        if (x >= start_x and 
                            y >= start_y and 
                            y < start_y + chunk_height and x < start_x + chunk_width):
                            if pixel_index < len(sorted_pixels):
                                result_image.putpixel((x, y), sorted_pixels[pixel_index])
                                pixel_index += 1
    
    # Handle edge cases (remainder chunks)
    # This would involve similar code as above but for the edge chunks
    # For brevity, this could be handled in a future update if needed
    
    return result_image

def databend_image(image, intensity=0.1, preserve_header=True, seed=None):
    """
    Apply databending to an image by simulating binary data corruption through direct pixel manipulation.
    
    Args:
        image (PIL.Image): PIL Image object to apply databending to.
        intensity (float): Intensity of the effect (0.1 to 1.0).
        preserve_header (bool): Controls whether the top portion of the image is preserved.
        seed (int or None): Random seed for reproducible results.
    
    Returns:
        PIL.Image: The glitched image.
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Make a copy of the original image to work with
    img = image.copy().convert('RGB')
    width, height = img.size
    
    # Create a pixelated version for more stable glitching
    pixels = img.load()
    
    # Determine how much of the image to protect (if preserve_header is True)
    protected_height = int(height * 0.1) if preserve_header else 0
    
    # Apply different glitch effects based on intensity
    # Scale the number of operations based on intensity - new range is 0.1 to 1.0
    # At 0.1 intensity: ~5-10 operations
    # At 1.0 intensity: ~100-150 operations
    num_operations = int(intensity * 150)
    num_operations = max(5, min(150, num_operations))  # Ensure a reasonable range
    
    # Different databending-like effects
    operations = [
        'shift_rows',
        'shift_columns',
        'swap_channels',
        'xor_block',
        'repeat_block',
        'color_shift',
        'invert_block',
        'shift_channels',
        'block_offset',
    ]
    
    # Ensure minimal dimensions for operations
    if width < 10 or height < 10:
        # For very small images, just apply simple color effects
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                if random.random() < intensity:
                    # Apply random color effect
                    effect = random.choice(['invert', 'xor', 'shift'])
                    if effect == 'invert':
                        pixels[x, y] = (255 - r, 255 - g, 255 - b)
                    elif effect == 'xor':
                        xor_val = random.randint(1, 255)
                        pixels[x, y] = (r ^ xor_val, g ^ xor_val, b ^ xor_val)
                    elif effect == 'shift':
                        shift = random.randint(-50, 50)
                        pixels[x, y] = (
                            max(0, min(255, r + shift)),
                            max(0, min(255, g + shift)),
                            max(0, min(255, b + shift))
                        )
        return img
    
    for _ in range(num_operations):
        # Select a random operation
        operation = random.choice(operations)
        
        if operation == 'shift_rows':
            # Shift a chunk of rows horizontally
            # Ensure valid y range
            max_start_y = max(protected_height, height - 10)
            if protected_height >= max_start_y:
                continue  # Skip this operation if there's not enough height
            
            start_y = random.randint(protected_height, max_start_y)
            y_size = random.randint(1, max(1, min(20, height - start_y)))
            shift_amount = random.randint(-width // 2, width // 2)
            
            for y in range(start_y, min(start_y + y_size, height)):
                row = [pixels[x, y] for x in range(width)]
                # Shift the row
                shifted_row = row[-shift_amount:] + row[:-shift_amount] if shift_amount > 0 else row[abs(shift_amount):] + row[:abs(shift_amount)]
                # Write back
                for x in range(width):
                    pixels[x, y] = shifted_row[x]
                    
        elif operation == 'shift_columns':
            # Shift a chunk of columns vertically
            # Ensure valid x range
            max_start_x = max(0, width - 10)
            if max_start_x <= 0:
                continue  # Skip this operation if there's not enough width
                
            start_x = random.randint(0, max_start_x)
            x_size = random.randint(1, max(1, min(20, width - start_x)))
            shift_amount = random.randint(-height // 3, height // 3)
            
            for x in range(start_x, min(start_x + x_size, width)):
                col = [pixels[x, y] for y in range(protected_height, height)]
                # Shift the column
                if not col:  # Skip if column is empty
                    continue
                shifted_col = col[-shift_amount:] + col[:-shift_amount] if shift_amount > 0 else col[abs(shift_amount):] + col[:abs(shift_amount)]
                # Write back
                for i, y in enumerate(range(protected_height, height)):
                    if i < len(shifted_col):
                        pixels[x, y] = shifted_col[i]
                        
        elif operation == 'swap_channels':
            # Swap color channels in a random block
            max_start_x = max(0, width - 10)
            max_start_y = max(protected_height, height - 10)
            
            if max_start_x <= 0 or max_start_y <= protected_height:
                continue  # Skip if we don't have enough space
                
            start_x = random.randint(0, max_start_x)
            start_y = random.randint(protected_height, max_start_y)
            block_width = random.randint(1, max(1, min(100, width - start_x)))
            block_height = random.randint(1, max(1, min(100, height - start_y)))
            swap_type = random.choice(['rb', 'rg', 'gb'])
            
            for y in range(start_y, min(start_y + block_height, height)):
                for x in range(start_x, min(start_x + block_width, width)):
                    r, g, b = pixels[x, y]
                    if swap_type == 'rb':
                        pixels[x, y] = (b, g, r)
                    elif swap_type == 'rg':
                        pixels[x, y] = (g, r, b)
                    elif swap_type == 'gb':
                        pixels[x, y] = (r, b, g)
                        
        elif operation == 'xor_block':
            # Apply XOR to a block
            max_start_x = max(0, width - 10)
            max_start_y = max(protected_height, height - 10)
            
            if max_start_x <= 0 or max_start_y <= protected_height:
                continue  # Skip if we don't have enough space
                
            start_x = random.randint(0, max_start_x)
            start_y = random.randint(protected_height, max_start_y)
            block_width = random.randint(1, max(1, min(100, width - start_x)))
            block_height = random.randint(1, max(1, min(100, height - start_y)))
            xor_value = random.randint(1, 255)
            
            for y in range(start_y, min(start_y + block_height, height)):
                for x in range(start_x, min(start_x + block_width, width)):
                    r, g, b = pixels[x, y]
                    pixels[x, y] = ((r ^ xor_value) & 0xFF, (g ^ xor_value) & 0xFF, (b ^ xor_value) & 0xFF)
                    
        elif operation == 'repeat_block':
            # Repeat a block somewhere else
            max_src_x = max(0, width - 20)
            max_src_y = max(protected_height, height - 20)
            
            if max_src_x <= 0 or max_src_y <= protected_height:
                continue  # Skip if we don't have enough space
                
            src_x = random.randint(0, max_src_x)
            src_y = random.randint(protected_height, max_src_y)
            block_width = random.randint(1, max(1, min(50, width - src_x)))
            block_height = random.randint(1, max(1, min(50, height - src_y)))
            
            # Ensure destination has valid space
            max_dst_x = max(0, width - block_width)
            max_dst_y = max(protected_height, height - block_height)
            
            if max_dst_x < 0 or max_dst_y < protected_height:
                continue  # Skip if destination doesn't have enough space
                
            # Destination
            dst_x = random.randint(0, max_dst_x)
            dst_y = random.randint(protected_height, max_dst_y)
            
            # Copy block
            for y_offset in range(block_height):
                for x_offset in range(block_width):
                    if (src_y + y_offset < height and src_x + x_offset < width and 
                        dst_y + y_offset < height and dst_x + x_offset < width):
                        pixels[dst_x + x_offset, dst_y + y_offset] = pixels[src_x + x_offset, src_y + y_offset]
                        
        elif operation == 'color_shift':
            # Shift color values in a region
            max_start_x = max(0, width - 10)
            max_start_y = max(protected_height, height - 10)
            
            if max_start_x <= 0 or max_start_y <= protected_height:
                continue  # Skip if we don't have enough space
                
            start_x = random.randint(0, max_start_x)
            start_y = random.randint(protected_height, max_start_y)
            block_width = random.randint(1, max(1, min(200, width - start_x)))
            block_height = random.randint(1, max(1, min(200, height - start_y)))
            
            shift_r = random.randint(-50, 50)
            shift_g = random.randint(-50, 50)
            shift_b = random.randint(-50, 50)
            
            for y in range(start_y, min(start_y + block_height, height)):
                for x in range(start_x, min(start_x + block_width, width)):
                    r, g, b = pixels[x, y]
                    pixels[x, y] = (
                        max(0, min(255, r + shift_r)),
                        max(0, min(255, g + shift_g)),
                        max(0, min(255, b + shift_b))
                    )
                    
        elif operation == 'invert_block':
            # Invert colors in a block
            max_start_x = max(0, width - 10)
            max_start_y = max(protected_height, height - 10)
            
            if max_start_x <= 0 or max_start_y <= protected_height:
                continue  # Skip if we don't have enough space
                
            start_x = random.randint(0, max_start_x)
            start_y = random.randint(protected_height, max_start_y)
            block_width = random.randint(1, max(1, min(100, width - start_x)))
            block_height = random.randint(1, max(1, min(100, height - start_y)))
            
            for y in range(start_y, min(start_y + block_height, height)):
                for x in range(start_x, min(start_x + block_width, width)):
                    r, g, b = pixels[x, y]
                    pixels[x, y] = (255 - r, 255 - g, 255 - b)
                    
        elif operation == 'shift_channels':
            # Shift RGB channels independently
            max_start_x = max(0, width - 10)
            max_start_y = max(protected_height, height - 10)
            
            if max_start_x <= 0 or max_start_y <= protected_height:
                continue  # Skip if we don't have enough space
                
            start_x = random.randint(0, max_start_x)
            start_y = random.randint(protected_height, max_start_y)
            block_width = random.randint(1, max(1, min(300, width - start_x)))
            block_height = random.randint(1, max(1, min(100, height - start_y)))
            
            # Skip if block is too small
            if block_width < 4 or block_height < 2:
                continue
                
            # Extract the block for each channel
            r_channel = []
            g_channel = []
            b_channel = []
            
            for y in range(start_y, min(start_y + block_height, height)):
                r_row = []
                g_row = []
                b_row = []
                for x in range(start_x, min(start_x + block_width, width)):
                    r, g, b = pixels[x, y]
                    r_row.append(r)
                    g_row.append(g)
                    b_row.append(b)
                r_channel.append(r_row)
                g_channel.append(g_row)
                b_channel.append(b_row)
            
            # Ensure we have data in the channels
            if not r_channel or not r_channel[0]:
                continue
                
            # Get channel lengths for safe shifting
            row_length = len(r_channel[0])
            
            # Determine safe shift amounts to avoid empty ranges
            max_shift = max(1, row_length // 4)
            
            # Shift each channel differently
            r_shift = random.randint(-max_shift, max_shift)
            g_shift = random.randint(-max_shift, max_shift)
            b_shift = random.randint(-max_shift, max_shift)
            
            # Apply shifts
            for y_idx in range(len(r_channel)):
                r_row = r_channel[y_idx]
                g_row = g_channel[y_idx]
                b_row = b_channel[y_idx]
                
                # Only shift if rows have data
                if not r_row or not g_row or not b_row:
                    continue
                    
                # Apply shifts safely
                if abs(r_shift) >= len(r_row):
                    r_shift = r_shift % len(r_row)
                if abs(g_shift) >= len(g_row):
                    g_shift = g_shift % len(g_row)
                if abs(b_shift) >= len(b_row):
                    b_shift = b_shift % len(b_row)
                
                r_row_shifted = r_row[-r_shift:] + r_row[:-r_shift] if r_shift > 0 else r_row[abs(r_shift):] + r_row[:abs(r_shift)]
                g_row_shifted = g_row[-g_shift:] + g_row[:-g_shift] if g_shift > 0 else g_row[abs(g_shift):] + g_row[:abs(g_shift)]
                b_row_shifted = b_row[-b_shift:] + b_row[:-b_shift] if b_shift > 0 else b_row[abs(b_shift):] + b_row[:abs(b_shift)]
                
                for x_idx in range(len(r_row)):
                    x = start_x + x_idx
                    y = start_y + y_idx
                    if y < height and x < width and x_idx < len(r_row_shifted) and x_idx < len(g_row_shifted) and x_idx < len(b_row_shifted):
                        pixels[x, y] = (r_row_shifted[x_idx], g_row_shifted[x_idx], b_row_shifted[x_idx])
                        
        elif operation == 'block_offset':
            # Create offset block effect
            if random.random() < 0.7:  # Horizontal offset
                # Ensure valid y range for protected height
                if height <= protected_height + 20:
                    continue  # Skip if not enough vertical space
                    
                max_y_start = max(protected_height, height - 20)
                if protected_height >= max_y_start:
                    continue  # Skip if not enough space
                    
                y_start = random.randint(protected_height, max_y_start)
                height_offset = random.randint(1, max(1, min(50, height - y_start)))
                x_offset = random.randint(-width // 3, width // 3) if width > 3 else 1
                
                for y in range(y_start, min(y_start + height_offset, height)):
                    for x in range(width):
                        src_x = (x - x_offset) % width
                        pixels[x, y] = pixels[src_x, y]
            else:  # Vertical offset
                # Ensure valid x range
                max_x_start = max(0, width - 20)
                if max_x_start <= 0:
                    continue  # Skip if not enough horizontal space
                    
                x_start = random.randint(0, max_x_start)
                width_offset = random.randint(1, max(1, min(50, width - x_start)))
                
                # Ensure vertical space considering protected height
                if height <= protected_height:
                    continue  # Skip if all vertical space is protected
                    
                y_offset = random.randint(-max(1, (height - protected_height) // 3), max(1, (height - protected_height) // 3))
                
                for x in range(x_start, min(x_start + width_offset, width)):
                    for y in range(protected_height, height):
                        # Make sure we stay within bounds
                        if height <= protected_height:
                            continue
                        src_y = protected_height + ((y - protected_height - y_offset) % max(1, (height - protected_height)))
                        pixels[x, y] = pixels[x, src_y]
    
    # Add scan lines for more glitch aesthetic
    if random.random() < 0.3 and height > protected_height + 5:
        scan_line_spacing = random.randint(4, min(20, max(4, height // 10)))
        scan_line_offset = random.randint(0, scan_line_spacing - 1) if scan_line_spacing > 1 else 0
        
        for y in range(protected_height, height):
            if scan_line_spacing > 0 and (y + scan_line_offset) % scan_line_spacing == 0:
                for x in range(width):
                    r, g, b = pixels[x, y]
                    pixels[x, y] = (min(255, int(r * 1.2)), min(255, int(g * 1.2)), min(255, int(b * 1.2)))
    
    # Add some noisy lines
    if random.random() < 0.4 and height > protected_height and width > 0:
        num_lines = random.randint(1, max(3, min(20, height // 20)))
        for _ in range(num_lines):
            if height <= protected_height:
                continue  # Skip if protected height is the whole image
                
            y = random.randint(protected_height, height - 1)
            length = random.randint(max(1, width // 5), width) if width > 5 else width
            start_x = random.randint(0, max(0, width - length))
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            for x in range(start_x, min(start_x + length, width)):
                if x < width:
                    pixels[x, y] = color
    
    # Add noise in some blocks
    if random.random() < 0.5 and width > 10 and height > protected_height + 10:
        noise_blocks = random.randint(1, max(1, min(5, (width * height) // 10000)))
        for _ in range(noise_blocks):
            max_start_x = max(0, width - 50)
            max_start_y = max(protected_height, height - 50)
            
            if max_start_x < 0 or max_start_y <= protected_height:
                continue  # Skip if not enough space
                
            start_x = random.randint(0, max_start_x)
            start_y = random.randint(protected_height, max_start_y)
            block_width = random.randint(1, max(1, min(100, width - start_x)))
            block_height = random.randint(1, max(1, min(30, height - start_y)))
            noise_intensity = random.uniform(0.2, 0.8)
            
            for y in range(start_y, min(start_y + block_height, height)):
                for x in range(start_x, min(start_x + block_width, width)):
                    if random.random() < noise_intensity:
                        r, g, b = pixels[x, y]
                        noise = random.randint(-50, 50)
                        pixels[x, y] = (
                            max(0, min(255, r + noise)),
                            max(0, min(255, g + noise)),
                            max(0, min(255, b + noise))
                        )
    
    return img

def find_next_from_index(data, pattern, start_index):
    """Helper function to find the next occurrence of a byte pattern"""
    for i in range(start_index, len(data) - len(pattern) + 1):
        if data[i:i+len(pattern)] == pattern:
            return i
    return -1

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
        
        # Generate perlin noise
        noise = generate_noise_map((img_width, img_height), perlin_noise_scale, 6, random_seed or 42)
        
        # Apply threshold to create binary mask
        mask_array = np.zeros((img_height, img_width), dtype=np.uint8)
        mask_array[noise > threshold] = 255
        
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
