import colorsys
import math

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
    import numpy as np
    try:
        import noise
    except ImportError:
        raise ImportError("The 'noise' package is required. Install with: pip install noise")
    
    height, width = shape
    noise_map = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            noise_map[y, x] = noise.pnoise2(x / scale, y / scale, octaves=octaves, base=base)
    return noise_map

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
    import numpy as np
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
    import numpy as np
    if direction == 'horizontal':
        return np.fliplr(channel)
    elif direction == 'vertical':
        return np.flipud(channel)
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'.") 