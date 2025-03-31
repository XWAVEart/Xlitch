from PIL import Image

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