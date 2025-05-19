from PIL import Image
import logging

logger = logging.getLogger(__name__)

def load_image(file_path, max_width_config=None, max_height_config=None):
    """
    Load an image from the specified file path.
    
    Args:
        file_path (str): Path to the image file.
        max_width_config (int, optional): Max width from app config.
        max_height_config (int, optional): Max height from app config.
    
    Returns:
        Image or None: The loaded PIL Image object, or None if loading fails.
    """
    try:
        img = Image.open(file_path)
        # Pass config values; resize_image_if_needed will use its defaults if None
        return resize_image_if_needed(img, max_width=max_width_config, max_height=max_height_config)
    except Exception as e:
        logger.error(f"Error loading image {file_path}: {e}")
        return None

def resize_image_if_needed(image, max_width=1920, max_height=1920):
    """
    Resize an image if its longest dimension exceeds the specified maximum,
    maintaining the original aspect ratio.
    
    Args:
        image (Image): PIL Image object to check and possibly resize
        max_width (int, optional): Maximum allowed width in pixels. Defaults to 1920.
                                 If None, 1920 is used.
        max_height (int, optional): Maximum allowed height in pixels. Defaults to 1920.
                                  If None, 1920 is used.
        
    Returns:
        Image: Original image or resized version if dimensions exceeded limits
    """
    # Use provided max_width/max_height or fall back to internal defaults if they are None
    effective_max_width = max_width if max_width is not None else 1920
    effective_max_height = max_height if max_height is not None else 1920
    
    # Get current image dimensions
    width, height = image.size
    
    # Check if resizing is needed (if either dimension exceeds max)
    if width <= effective_max_width and height <= effective_max_height:
        return image
    
    # Calculate ratio based on the longest dimension
    ratio = min(effective_max_width / width, effective_max_height / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    
    # Resize the image using high quality resampling
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    return resized_image

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
    import os
    base, ext = os.path.splitext(original_filename)
    return f"{base}_{effect}_{settings}{ext}" 