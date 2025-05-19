from PIL import ImageColor
import colorsys

# Centralized pixel attribute calculations
class PixelAttributes:
    """
    Central module for all pixel attribute calculations.
    All methods expect pixel input as a 3-element tuple (R, G, B)
    with integer values in the 0-255 range.
    """
    
    @staticmethod
    def brightness(pixel):
        """
        Calculate the perceived brightness of a pixel using the luminosity formula.
        Input: (R, G, B) tuple, 0-255.
        Output: float, 0-255.0 range.
        """
        # pixel is expected to be (R,G,B) tuple with 0-255 int values
        return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]
    
    @staticmethod
    def hue(pixel):
        """
        Calculate the hue of a pixel.
        Input: (R, G, B) tuple, 0-255.
        Output: float, 0-360.0 range.
        """
        # pixel is expected to be (R,G,B) tuple with 0-255 int values
        # Normalize to 0-1 for colorsys
        r, g, b = [p/255.0 for p in pixel[:3]]
        h, _, _ = colorsys.rgb_to_hsv(r, g, b)
        return h * 360.0 # Scale to 0-360

    @staticmethod
    def saturation(pixel):
        """
        Calculate the saturation of a pixel.
        Input: (R, G, B) tuple, 0-255.
        Output: float, 0-1.0 range.
        """
        # pixel is expected to be (R,G,B) tuple with 0-255 int values
        # Normalize to 0-1 for colorsys
        r, g, b = [p/255.0 for p in pixel[:3]]
        _, s, _ = colorsys.rgb_to_hsv(r, g, b)
        return s # Directly returns 0-1.0

    @staticmethod
    def luminance(pixel):
        """
        Calculate the luminance (value in HSV) of a pixel.
        Input: (R, G, B) tuple, 0-255.
        Output: float, 0-1.0 range.
        """
        # pixel is expected to be (R,G,B) tuple with 0-255 int values
        # Normalize to 0-1 for colorsys
        r, g, b = [p/255.0 for p in pixel[:3]]
        _, _, v = colorsys.rgb_to_hsv(r, g, b)
        return v # Directly returns 0-1.0
    
    @staticmethod
    def contrast(pixel):
        """
        Calculate a contrast value based on the difference between max and min RGB values.
        Input: (R, G, B) tuple, 0-255.
        Output: int, 0-255 range.
        """
        # pixel is expected to be (R,G,B) tuple with 0-255 int values
        return max(pixel[:3]) - min(pixel[:3])
    
    @staticmethod
    def color_sum(pixel):
        """
        Calculate the sum of RGB values.
        Input: (R, G, B) tuple, 0-255.
        Output: int, 0-765 range.
        """
        # pixel is expected to be (R,G,B) tuple with 0-255 int values
        return int(pixel[0]) + int(pixel[1]) + int(pixel[2]) 