from PIL import Image
import numpy as np

def double_expose(image, secondary_image, blend_mode='classic', opacity=0.5):
    """
    Blend two images together with the specified blend mode and opacity.
    
    Args:
        image (Image): Primary PIL Image object.
        secondary_image (Image): Secondary PIL Image object to blend with the primary.
        blend_mode (str): Blending mode ('classic', 'screen', 'multiply', 'overlay', 'difference', 'color_dodge').
        opacity (float): Opacity of the second image (0.0 to 1.0).
    
    Returns:
        Image: Blended result.
    """
    # Ensure the secondary image is the same size as the primary
    if image.size != secondary_image.size:
        secondary_image = secondary_image.resize(image.size)
    
    # Convert both images to RGB mode if they have alpha channels or are in different modes
    if image.mode != 'RGB':
        image = image.convert('RGB')
    if secondary_image.mode != 'RGB':
        secondary_image = secondary_image.convert('RGB')
    
    # Convert to numpy arrays for faster processing
    img1 = np.array(image).astype(np.float32)
    img2 = np.array(secondary_image).astype(np.float32)
    
    # Ensure opacity is within valid range
    opacity = max(0.0, min(1.0, opacity))
    
    # Apply the selected blend mode
    if blend_mode == 'classic':
        # Simple weighted average
        result = (1 - opacity) * img1 + opacity * img2
    
    elif blend_mode == 'screen':
        # Screen blend mode: 1 - (1-a)*(1-b)
        # Normalize to 0-1 range for calculation
        img1_norm = img1 / 255.0
        img2_norm = img2 / 255.0
        screen = 1.0 - ((1.0 - img1_norm) * (1.0 - img2_norm * opacity))
        result = screen * 255.0
    
    elif blend_mode == 'multiply':
        # Multiply blend mode
        # Normalize to 0-1 range for calculation
        img1_norm = img1 / 255.0
        img2_norm = img2 / 255.0
        # Apply opacity to second image
        img2_blend = img2_norm * opacity + (1 - opacity)
        multiply = img1_norm * img2_blend
        result = multiply * 255.0
    
    elif blend_mode == 'overlay':
        # Overlay blend mode
        # Normalize to 0-1 range for calculation
        img1_norm = img1 / 255.0
        img2_norm = img2 / 255.0
        
        # Overlay formula
        overlay = np.zeros_like(img1_norm)
        # For values < 0.5 in the base layer
        mask = img1_norm < 0.5
        overlay[mask] = 2 * img1_norm[mask] * (img2_norm[mask] * opacity + (1 - opacity))
        # For values >= 0.5 in the base layer
        mask = ~mask
        overlay[mask] = 1 - 2 * (1 - img1_norm[mask]) * (1 - (img2_norm[mask] * opacity + (1 - opacity)))
        
        result = overlay * 255.0
    
    elif blend_mode == 'difference':
        # Difference blend mode: |img1 - img2|
        # Normalize to 0-1 range for calculation
        img1_norm = img1 / 255.0
        img2_norm = img2 / 255.0
        
        # Apply opacity to second image
        img2_blend = img2_norm * opacity
        
        # Difference formula: |a - b|
        difference = np.abs(img1_norm - img2_blend)
        
        result = difference * 255.0
    
    elif blend_mode == 'color_dodge' or blend_mode == 'dodge':  # Support both names for compatibility
        # Color Dodge blend mode
        # Normalize to 0-1 range for calculation
        img1_norm = img1 / 255.0
        img2_norm = img2 / 255.0
        
        # Adjust second image with opacity
        img2_blend = img2_norm * opacity
        
        # Color Dodge formula: base / (1 - blend)
        dodge = np.zeros_like(img1_norm)
        divisor = 1.0 - img2_blend
        
        # Handle division by zero (or near zero)
        mask = divisor < 0.001
        dodge[mask] = 1.0  # White for areas that would divide by zero
        
        # Regular calculation for safe areas
        mask = ~mask
        dodge[mask] = img1_norm[mask] / divisor[mask]
        dodge = np.clip(dodge, 0.0, 1.0)  # Clamp to valid range
        
        result = dodge * 255.0
    
    elif blend_mode == 'burn':
        # Color Burn blend mode
        # Normalize to 0-1 range for calculation
        img1_norm = img1 / 255.0
        img2_norm = img2 / 255.0
        
        # Adjust second image with opacity
        img2_blend = 1.0 - (1.0 - img2_norm) * opacity
        
        # Color Burn formula: 1 - (1 - base) / blend
        burn = np.zeros_like(img1_norm)
        divisor = img2_blend
        
        # Handle division by zero (or near zero)
        mask = divisor < 0.001
        burn[mask] = 0.0  # Black for areas that would divide by zero
        
        # Regular calculation for safe areas
        mask = ~mask
        burn[mask] = 1.0 - (1.0 - img1_norm[mask]) / divisor[mask]
        burn = np.clip(burn, 0.0, 1.0)  # Clamp to valid range
        
        result = burn * 255.0
    
    elif blend_mode == 'hard_light':
        # Hard Light blend mode (similar to overlay but with layers swapped)
        # Normalize to 0-1 range for calculation
        img1_norm = img1 / 255.0
        img2_norm = img2 / 255.0
        
        # Adjust second image with opacity
        img2_blend = img2_norm * opacity + (1 - opacity)
        
        # Hard Light formula
        hard_light = np.zeros_like(img1_norm)
        # For values < 0.5 in the second layer
        mask = img2_blend < 0.5
        hard_light[mask] = 2 * img1_norm[mask] * img2_blend[mask]
        # For values >= 0.5 in the second layer
        mask = ~mask
        hard_light[mask] = 1 - 2 * (1 - img1_norm[mask]) * (1 - img2_blend[mask])
        
        result = hard_light * 255.0
    
    else:
        # Default to classic blend if an invalid mode is specified
        result = (1 - opacity) * img1 + opacity * img2
    
    # Convert result back to uint8 and clip to valid range
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Create and return the final image
    return Image.fromarray(result) 