from PIL import ImageColor
import colorsys

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