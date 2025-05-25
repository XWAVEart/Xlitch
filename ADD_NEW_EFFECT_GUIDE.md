# Guide: Adding a New Effect to the Glitch Art App (Post-Consolidation)

This document outlines the step-by-step process for adding a new effect to the refactored Glitch Art application. The new architecture emphasizes modularity, with separate form classes for each effect, dynamic loading of UI components, and **consolidated effect interfaces** for related functionality.

## Architecture Overview

The application now uses a **two-tier effect system**:

1. **Individual Effects** - Specific implementations in their respective modules
2. **Consolidated Effects** - Unified interfaces that group related effects (e.g., `advanced_pixel_sorting`, `slice_block_manipulation`)

### When to Use Each Approach:

- **Add to Existing Consolidated Effect**: If your new effect fits into an existing category (pixel sorting, slice/block manipulation)
- **Create Individual Effect**: For standalone effects that don't fit existing categories
- **Create New Consolidated Effect**: When you have multiple related effects that should be grouped together

## Option 1: Adding to an Existing Consolidated Effect

### Example: Adding a New Pixel Sorting Algorithm

If you want to add a new pixel sorting method, add it to the existing `advanced_pixel_sorting` consolidated effect:

#### 1.1. Implement the Core Function

Add your sorting algorithm to `glitch_art/effects/sorting.py`:

```python
# In glitch_art/effects/sorting.py

def my_new_sort(image, chunk_width, chunk_height, sort_by='brightness', reverse=False, **kwargs):
    """
    Your new sorting algorithm implementation.
    
    Args:
        image (Image): PIL Image object
        chunk_width (int): Width of sorting chunks
        chunk_height (int): Height of sorting chunks
        sort_by (str): Attribute to sort by
        reverse (bool): Sort order
        **kwargs: Additional parameters specific to your algorithm
    
    Returns:
        Image: Processed image
    """
    # Your implementation here
    pass
```

#### 1.2. Update the Consolidated Interface

Modify `glitch_art/effects/consolidated.py` to include your new algorithm:

```python
# In glitch_art/effects/consolidated.py

def advanced_pixel_sorting(image, sorting_method, **kwargs):
    """Consolidated interface for all pixel sorting methods."""
    
    # ... existing methods ...
    
    elif sorting_method == 'my_new_sort':
        from .sorting import my_new_sort
        return my_new_sort(image, **kwargs)
    
    # ... rest of function ...
```

#### 1.3. Update the Form

Add your new method to `AdvancedPixelSortingForm` in `forms.py`:

```python
# In forms.py, within AdvancedPixelSortingForm

sorting_method = SelectField('Sorting Method', choices=[
    # ... existing choices ...
    ('my_new_sort', 'My New Sort Algorithm'),
], default='chunk', validators=[DataRequired()])

# Add any specific fields for your algorithm
my_new_sort_param = IntegerField('My New Sort Parameter', 
                                default=10, 
                                validators=[Optional(), NumberRange(min=1, max=100)])
```

#### 1.4. Update Parameter Extraction

Add parameter extraction in `app.py`:

```python
# In app.py, within the advanced_pixel_sorting elif block

elif sorting_method == 'my_new_sort':
    kwargs['chunk_width'] = effect_specific_form.chunk_width.data
    kwargs['chunk_height'] = effect_specific_form.chunk_height.data
    kwargs['my_new_sort_param'] = effect_specific_form.my_new_sort_param.data
    kwargs['sort_by'] = effect_specific_form.sort_by.data
    kwargs['reverse'] = effect_specific_form.reverse.data
```

## Option 2: Creating a Standalone Individual Effect

### 2.1. Choose the Appropriate Module

Create or add to the appropriate module based on your effect's nature:

- `glitch_art/effects/distortion.py` - Spatial distortions and displacement
- `glitch_art/effects/color.py` - Color manipulations and filters
- `glitch_art/effects/sorting.py` - Pixel sorting algorithms
- `glitch_art/effects/noise.py` - Noise-based effects
- `glitch_art/effects/patterns.py` - Pattern generation and masks
- `glitch_art/effects/glitch.py` - Data corruption and bit manipulation
- `glitch_art/effects/pixelate.py` - Pixelation and quantization
- `glitch_art/effects/blend.py` - Image blending and compositing
- `glitch_art/effects/contour.py` - Edge detection and contours
- `glitch_art/effects/consolidated.py` - **New consolidated interfaces**

### 2.2. Implement the Effect Function

```python
# Example: In glitch_art/effects/color.py

def my_color_effect(image, intensity=50, color_mode='rgb', **kwargs):
    """
    Apply a custom color effect to the image.
    
    Args:
        image (PIL.Image.Image): Input PIL Image object
        intensity (int): Effect intensity (0-100)
        color_mode (str): Color space to work in
        **kwargs: Additional parameters
    
    Returns:
        PIL.Image.Image: Processed image
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Your effect implementation here
    
    return processed_image
```

### 2.3. Update Module Exports

Add your function to `glitch_art/effects/__init__.py`:

```python
# In glitch_art/effects/__init__.py

from .color import (
    # ... existing imports ...
    my_color_effect,
)

__all__ = [
    # ... existing exports ...
    'my_color_effect',
]
```

### 2.4. Create Form Class

```python
# In forms.py

class MyColorEffectForm(FlaskForm):
    """Form for My Color Effect parameters."""
    intensity = IntegerField('Effect Intensity',
                           default=50,
                           validators=[DataRequired(), NumberRange(min=0, max=100)])
    color_mode = SelectField('Color Mode', choices=[
        ('rgb', 'RGB'),
        ('hsv', 'HSV'),
        ('lab', 'LAB')
    ], default='rgb', validators=[DataRequired()])
```

### 2.5. Add to Main Form and Mapping

```python
# In forms.py, within ImageProcessForm

effect = SelectField('Effect', choices=[
    # ... existing effects ...
    ('my_color_effect', 'My Color Effect'),
], validators=[DataRequired()])

# In app.py, add to EFFECT_FORM_MAP
EFFECT_FORM_MAP = {
    # ... existing mappings ...
    'my_color_effect': MyColorEffectForm,
}
```

### 2.6. Add Processing Logic

```python
# In app.py, within the main processing block

elif selected_effect_key == 'my_color_effect':
    intensity = effect_specific_form.intensity.data
    color_mode = effect_specific_form.color_mode.data
    
    logger.debug(f"My Color Effect params: intensity={intensity}, mode={color_mode}")
    
    processed_image = my_color_effect(image, 
                                    intensity=intensity, 
                                    color_mode=color_mode)
    
    settings = f"color_{intensity}_{color_mode}"
```

## Option 3: Creating a New Consolidated Effect

### When to Create a Consolidated Effect:

- You have multiple related algorithms/methods
- Users would benefit from a unified interface
- The effects share common parameters
- You want to allow easy switching between variants

### 3.1. Create the Consolidated Interface

```python
# In glitch_art/effects/consolidated.py

def my_consolidated_effect(image, method, **kwargs):
    """
    Consolidated interface for my family of related effects.
    
    Args:
        image (Image): PIL Image object
        method (str): Which specific method to use
        **kwargs: Method-specific parameters
    
    Returns:
        Image: Processed image
    """
    if method == 'variant_a':
        from .my_module import variant_a_function
        return variant_a_function(image, **kwargs)
    elif method == 'variant_b':
        from .my_module import variant_b_function
        return variant_b_function(image, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
```

### 3.2. Create Consolidated Form

```python
# In forms.py

class MyConsolidatedEffectForm(FlaskForm):
    """Consolidated form for my family of effects."""
    
    method = SelectField('Method', choices=[
        ('variant_a', 'Variant A'),
        ('variant_b', 'Variant B'),
    ], default='variant_a', validators=[DataRequired()])
    
    # Common parameters
    intensity = IntegerField('Intensity', default=50, 
                           validators=[DataRequired(), NumberRange(min=0, max=100)])
    
    # Method-specific parameters (shown/hidden via JavaScript)
    variant_a_param = IntegerField('Variant A Parameter', default=10,
                                 validators=[Optional(), NumberRange(min=1, max=50)])
    variant_b_param = FloatField('Variant B Parameter', default=1.5,
                               validators=[Optional(), NumberRange(min=0.1, max=5.0)])
```

## Current Consolidated Effects

The application currently includes these consolidated effects:

### 1. Advanced Pixel Sorting (`advanced_pixel_sorting`)
**Location**: `glitch_art/effects/consolidated.py`  
**Form**: `AdvancedPixelSortingForm`  
**Methods**: `chunk`, `full_frame`, `spiral`, `polar`, `wrapped`, `perlin_noise`, `voronoi`

**Adding New Sorting Method**:
1. Implement in `sorting.py`
2. Add to `advanced_pixel_sorting()` function
3. Add choice to form's `sorting_method` field
4. Add parameter extraction in `app.py`

### 2. Slice & Block Manipulation (`slice_block_manipulation`)
**Location**: `glitch_art/effects/consolidated.py`  
**Form**: `SliceBlockManipulationForm`  
**Methods**: `slice_shuffle`, `slice_offset`, `slice_reduction`, `block_shuffle`

**Adding New Manipulation**:
1. Implement in `distortion.py`
2. Add to `slice_block_manipulation()` function
3. Add choice to form's `manipulation_type` field
4. Add parameter extraction in `app.py`

## Best Practices

### 1. Parameter Consistency
- Use consistent parameter names across related effects
- Provide sensible defaults
- Include proper validation ranges

### 2. Error Handling
```python
def my_effect(image, param1, param2=None):
    try:
        # Effect implementation
        return processed_image
    except Exception as e:
        logger.error(f"Error in my_effect: {e}")
        return image  # Return original on error
```

### 3. Documentation
- Include comprehensive docstrings
- Document parameter ranges and effects
- Provide usage examples

### 4. Testing
- Test with various image sizes and formats
- Verify parameter validation
- Check edge cases and error conditions

## Form Field Conditional Logic

For complex forms with conditional fields, add JavaScript to `templates/index.html`:

```javascript
// In the setupConditionalFields function
else if (effectKey === 'my_consolidated_effect') {
    const methodSelect = $(getFieldGroupId('method') + ' select');
    const variantAFields = $(getFieldGroupId('variant_a_param'));
    const variantBFields = $(getFieldGroupId('variant_b_param'));
    
    function toggle() {
        const method = methodSelect.val();
        variantAFields.toggle(method === 'variant_a');
        variantBFields.toggle(method === 'variant_b');
    }
    
    methodSelect.off('change.conditionalMyEffect').on('change.conditionalMyEffect', toggle);
    toggle();
}
```

## Troubleshooting

### Common Issues:
1. **Import Errors**: Ensure all functions are properly exported in `__init__.py`
2. **Form Validation**: Check field validators and default values
3. **Parameter Mismatch**: Verify parameter names match between form and function
4. **Missing Choices**: Ensure new methods are added to form choice lists
5. **JavaScript Errors**: Check browser console for conditional field logic issues

### Debugging Tips:
- Use `logger.debug()` to trace parameter values
- Check Flask logs for detailed error messages
- Verify form data with browser developer tools
- Test individual effect functions in isolation

This guide reflects the current consolidated architecture and provides clear paths for extending the application's functionality. 