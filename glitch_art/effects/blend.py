from PIL import Image
import numpy as np

def double_expose(image, secondary_image, blend_mode='classic', opacity=0.5):
    """
    Blend two images together with the specified blend mode and opacity.
    
    Args:
        image (Image): Primary PIL Image object.
        secondary_image (Image): Secondary PIL Image object to blend with the primary.
        blend_mode (str): Blending mode ('classic', 'screen', 'multiply', 'overlay', 'difference', 'color_dodge', 'burn', 'hard_light', 'subtract', 'add', 'darken_only', 'lighten_only', 'exclusion').
        opacity (float): Opacity of the blend effect (0.0 to 1.0). 0 means primary image, 1 means full effect.
    
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
    img1_raw = np.array(image).astype(np.float32)
    img2_raw = np.array(secondary_image).astype(np.float32)
    
    # Ensure opacity is within valid range
    opacity = max(0.0, min(1.0, opacity))

    # Normalize to 0-1 range for calculation
    img1_norm = img1_raw / 255.0
    img2_norm = img2_raw / 255.0
    
    pure_blend_result_norm = np.zeros_like(img1_norm)

    if blend_mode == 'classic':
        # Simple weighted average - this mode already incorporates opacity directly
        result = (1 - opacity) * img1_raw + opacity * img2_raw
        result = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(result)
    
    elif blend_mode == 'screen':
        # Screen blend mode: 1 - (1-a)*(1-b)
        pure_blend_result_norm = 1.0 - (1.0 - img1_norm) * (1.0 - img2_norm)
    
    elif blend_mode == 'multiply':
        # Multiply blend mode: a * b
        pure_blend_result_norm = img1_norm * img2_norm
    
    elif blend_mode == 'overlay':
        # Overlay formula
        mask = img1_norm < 0.5
        pure_blend_result_norm[mask] = 2 * img1_norm[mask] * img2_norm[mask]
        mask = ~mask
        pure_blend_result_norm[mask] = 1.0 - 2 * (1.0 - img1_norm[mask]) * (1.0 - img2_norm[mask])
    
    elif blend_mode == 'difference':
        # Difference blend mode: |a - b|
        pure_blend_result_norm = np.abs(img1_norm - img2_norm)
    
    elif blend_mode == 'color_dodge' or blend_mode == 'dodge':
        # Color Dodge formula: base / (1 - blend)
        divisor = 1.0 - img2_norm
        # Handle division by zero (or near zero)
        mask_zero_divisor = divisor < 0.001
        pure_blend_result_norm[mask_zero_divisor] = 1.0  # White for areas that would divide by zero
        
        mask_safe_divisor = ~mask_zero_divisor
        # Ensure divisor[mask_safe_divisor] is not zero before division
        safe_divisors = divisor[mask_safe_divisor]
        # Additional check for divisors that became zero after masking (shouldn't happen if < 0.001 check is robust)
        safe_divisors[safe_divisors < 0.001] = 0.001 # Prevent division by exact zero if any slipped through

        pure_blend_result_norm[mask_safe_divisor] = img1_norm[mask_safe_divisor] / safe_divisors
        pure_blend_result_norm = np.clip(pure_blend_result_norm, 0.0, 1.0)
            
    elif blend_mode == 'burn':
        # Color Burn formula: 1 - (1 - base) / blend
        divisor = img2_norm
        # Handle division by zero (or near zero)
        mask_zero_divisor = divisor < 0.001
        pure_blend_result_norm[mask_zero_divisor] = 0.0 # Black for areas that would divide by zero

        mask_safe_divisor = ~mask_zero_divisor
        safe_divisors = divisor[mask_safe_divisor]
        safe_divisors[safe_divisors < 0.001] = 0.001

        pure_blend_result_norm[mask_safe_divisor] = 1.0 - (1.0 - img1_norm[mask_safe_divisor]) / safe_divisors
        pure_blend_result_norm = np.clip(pure_blend_result_norm, 0.0, 1.0)

    elif blend_mode == 'hard_light':
        # Hard Light formula (like overlay, but layers swapped for condition)
        mask = img2_norm < 0.5
        pure_blend_result_norm[mask] = 2 * img1_norm[mask] * img2_norm[mask]
        mask = ~mask
        pure_blend_result_norm[mask] = 1.0 - 2 * (1.0 - img1_norm[mask]) * (1.0 - img2_norm[mask])
    
    elif blend_mode == 'subtract':
        pure_blend_result_norm = img1_norm - img2_norm
        # Result is clipped to [0, 1] later by the opacity application or final clip

    elif blend_mode == 'add' or blend_mode == 'linear_dodge':
        pure_blend_result_norm = img1_norm + img2_norm
        # Result is clipped to [0, 1] later

    elif blend_mode == 'darken_only':
        pure_blend_result_norm = np.minimum(img1_norm, img2_norm)

    elif blend_mode == 'lighten_only':
        pure_blend_result_norm = np.maximum(img1_norm, img2_norm)

    elif blend_mode == 'exclusion':
        pure_blend_result_norm = img1_norm + img2_norm - 2 * img1_norm * img2_norm
    
    else:
        # Default to img1_norm if blend mode is unknown, so opacity fades to original
        pure_blend_result_norm = img1_norm 

    # Apply opacity: final_pixel = (1-opacity)*base_pixel + opacity*effect_pixel
    # Here, img1_raw and pure_blend_result_norm*255.0 are the terms
    final_result_norm = (1.0 - opacity) * img1_norm + opacity * pure_blend_result_norm
    
    # Convert result back to uint8 and clip to valid range
    result = np.clip(final_result_norm * 255.0, 0, 255).astype(np.uint8)
    
    # Create and return the final image
    return Image.fromarray(result) 