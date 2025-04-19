# Guide: Adding a New Effect to the Glitch Art App

This document outlines the step-by-step process for adding a new effect to the Glitch Art application.

## 1. Implement the Core Effect Function

First, decide which module the effect belongs in based on its nature:

- `glitch_art/effects/distortion.py` - For effects that manipulate pixel positions
- `glitch_art/effects/color.py` - For effects that manipulate colors
- `glitch_art/effects/sorting.py` - For pixel sorting effects
- `glitch_art/effects/noise.py` - For noise-based effects
- `glitch_art/effects/patterns.py` - For pattern generation
- `glitch_art/effects/glitch.py` - For classic glitch/databending effects
- `glitch_art/effects/pixelate.py` - For pixelation effects
- `glitch_art/effects/blend.py` - For blend/composite effects

### Example implementation:

```python
def my_new_effect(image, param1, param2, optional_param=None):
    """
    Description of what this effect does.
    
    Args:
        image (PIL.Image): Input image to process.
        param1 (type): Description of param1.
        param2 (type): Description of param2.
        optional_param (type, optional): Description of optional parameter.
    
    Returns:
        PIL.Image: Processed image.
    """
    # Convert to NumPy array if needed
    import numpy as np
    image_np = np.array(image)
    
    # Process the image
    # ...
    
    # Return the processed image
    from PIL import Image
    return Image.fromarray(processed_image_np)
```

## 2. Update the Form in `forms.py`

### 2.1. Add the effect to the choices in the main effect SelectField

```python
effect = SelectField('Effect', choices=[
    # ... existing effects ...
    ('my_new_effect', 'My New Effect')
], validators=[DataRequired()])
```

### 2.2. Add form fields for the effect's parameters

```python
# My New Effect Fields
my_effect_param1 = IntegerField('Parameter 1', 
                           validators=[Optional(), NumberRange(min=1, max=100)])
my_effect_param2 = SelectField('Parameter 2', choices=[
    ('option1', 'Option 1'), 
    ('option2', 'Option 2')
], default='option1', validators=[Optional()])
my_effect_optional_param = FloatField('Optional Parameter',
                                   validators=[Optional()])
```

### 2.3. Update the form validation method

Add a branch for your new effect in the `validate()` method:

```python
def validate(self, extra_validators=None):
    # ... existing code ...
    
    elif effect == 'my_new_effect':
        if not self.my_effect_param1.data:
            self.my_effect_param1.errors = ['Parameter 1 is required for My New Effect']
            return False
        if not self.my_effect_param2.data:
            self.my_effect_param2.errors = ['Parameter 2 is required for My New Effect']
            return False
        # Optional param doesn't need validation since it's optional
    
    # ... existing code ...
    return True
```

## 3. Update the Request Handler in `app.py`

### 3.1. Import the effect function

```python
from glitch_art.effects.chosen_module import my_new_effect
```

### 3.2. Add a branch for the new effect in the route handler

```python
elif effect == 'my_new_effect':
    # Get parameters from form
    param1 = form.my_effect_param1.data
    param2 = form.my_effect_param2.data
    optional_param = form.my_effect_optional_param.data
    
    # Log parameters
    logger.debug(f"My new effect params: param1={param1}, param2={param2}, optional_param={optional_param}")
    
    # Process the image
    processed_image = my_new_effect(image, param1, param2, optional_param)
    
    # Create settings string for output filename
    settings = f"myeffect_{param1}_{param2}"
    if optional_param:
        settings += f"_{optional_param}"
```

## 4. Update the HTML Template (`templates/index.html`)

### 4.1. Add a form section for the new effect

Add a div to hold your effect's fields (make sure it has a unique ID):

```html
<div id="my-new-effect-options" style="display: none;">
    <div class="form-group">
        {{ form.my_effect_param1.label }} {{ form.my_effect_param1(size=10, id='my_effect_param1') }}
        <small class="form-text text-muted">Help text if needed</small>
    </div>
    <div class="form-group">
        {{ form.my_effect_param2.label }} {{ form.my_effect_param2(id='my_effect_param2') }}
    </div>
    <div class="form-group">
        {{ form.my_effect_optional_param.label }} {{ form.my_effect_optional_param(size=10, id='my_effect_optional_param') }}
        <small class="form-text text-muted">Optional parameter</small>
    </div>
</div>
```

### 4.2. Update the JavaScript to show/hide your effect options

```javascript
var myNewEffectOptions = document.getElementById("my-new-effect-options");

function toggleEffectOptions() {
    // Hide all effect option sections
    offsetOptions.style.display = "none";
    sliceShuffleOptions.style.display = "none";
    myNewEffectOptions.style.display = "none";
    // ... hide other effects' option divs
    
    // Show only the selected effect's options
    if(effectField.value === "offset") {
        offsetOptions.style.display = "block";
    } else if(effectField.value === "slice_shuffle") {
        sliceShuffleOptions.style.display = "block";
    } else if(effectField.value === "my_new_effect") {
        myNewEffectOptions.style.display = "block";
    }
    // ... handle other effects
}

// Make sure this event listener is set up already
effectField.addEventListener("change", toggleEffectOptions);
```

## 5. Test Your New Effect

1. Start the application
2. Upload an image
3. Select your new effect from the dropdown
4. Verify that the appropriate parameter fields appear
5. Enter valid values for your parameters
6. Submit the form and verify that your effect is applied correctly

## Troubleshooting Checklist

If you encounter issues, check:

- Did you add the effect to the `__init__.py` file if required for import?
- Are all required parameters being passed to your effect function?
- Is the effect function properly converting between NumPy arrays and PIL Images?
- Is the validation logic correct for your form fields?
- Have you added your effect's options div to the JavaScript toggle function?
- Check for typos in field names and IDs
- Look at the application logs for specific error messages

## Example: Complete Effect Implementation

For reference, here's how a complete effect called "Slice Shuffle" might be implemented:

### In `glitch_art/effects/distortion.py`:
```python
def slice_shuffle(image, slice_count, orientation, seed=None):
    """
    Apply the Slice Shuffle effect by dividing the image into a specified number of slices and shuffling them randomly.

    Args:
        image (PIL.Image): Input image to process.
        slice_count (int): Number of slices (must be between 4 and 128).
        orientation (str): Either 'rows' (to shuffle horizontal slices) or 'columns' (to shuffle vertical slices).
        seed (int, optional): Optional random seed for reproducibility.

    Returns:
        PIL.Image: Image with shuffled slices.
    """
    import numpy as np
    image_np = np.array(image)

    # Set seed if provided
    if seed is not None:
        random.seed(seed)

    # Split image into slices along the specified axis
    if orientation == 'rows':
        slices = np.array_split(image_np, slice_count, axis=0)
        random.shuffle(slices)
        shuffled_np = np.vstack(slices)
    elif orientation == 'columns':
        slices = np.array_split(image_np, slice_count, axis=1)
        random.shuffle(slices)
        shuffled_np = np.hstack(slices)
    else:
        # If orientation is unrecognized, return the original image
        return image

    from PIL import Image
    return Image.fromarray(shuffled_np)
```

### In `forms.py`:
```python
# Slice Shuffle Effect Fields
slice_count = IntegerField('Slice Count', validators=[Optional(), NumberRange(min=4, max=128, message="Slice count must be between 4 and 128")])
slice_orientation = SelectField('Slice Orientation', choices=[('rows', 'Rows'), ('columns', 'Columns')], default='rows', validators=[Optional()])
slice_seed = IntegerField('Slice Shuffle Seed (optional)', validators=[Optional()])

# In the validate method:
elif effect == 'slice_shuffle':
    # For the slice shuffle effect, slice_count and slice_orientation are required
    if self.slice_count.data is None:
        self.slice_count.errors = ['Slice count is required for Slice Shuffle effect']
        return False
    if self.slice_orientation.data is None:
        self.slice_orientation.errors = ['Slice orientation is required for Slice Shuffle effect']
        return False
```

### In `app.py`:
```python
from glitch_art.effects.distortion import slice_shuffle

# In the route handler:
elif effect == 'slice_shuffle':
    # process slice shuffle effect
    slice_count = form.slice_count.data
    orientation = form.slice_orientation.data
    seed = form.slice_seed.data
    processed_image = slice_shuffle(image, slice_count, orientation, seed if seed != 0 else None)
    settings = f"slice_{slice_count}_{orientation}" + (f"_{seed}" if seed else "")
```

### In `templates/index.html`:
```html
<div id="slice-shuffle-options" style="display: none;">
    <div class="form-group">
        {{ form.slice_count.label }} {{ form.slice_count(size=10, id='slice_count') }}
        <small class="form-text text-muted">Enter a value between 4 and 128</small>
    </div>
    <div class="form-group">
        {{ form.slice_orientation.label }} {{ form.slice_orientation(id='slice_orientation') }}
    </div>
    <div class="form-group">
        {{ form.slice_seed.label }} {{ form.slice_seed(size=10, id='slice_seed') }}
        <small class="form-text text-muted">Optional - leave empty for random results each time</small>
    </div>
</div>

<!-- JavaScript -->
<script>
    var effectField = document.getElementById("effect");
    var offsetOptions = document.getElementById("offset-options");
    var sliceShuffleOptions = document.getElementById("slice-shuffle-options");

    function toggleEffectOptions() {
        // Hide all option sections first
        offsetOptions.style.display = "none";
        sliceShuffleOptions.style.display = "none";

        // Show the appropriate section based on selected effect
        if(effectField.value === "offset") {
            offsetOptions.style.display = "block";
        } else if(effectField.value === "slice_shuffle") {
            sliceShuffleOptions.style.display = "block";
        }
    }

    effectField.addEventListener("change", toggleEffectOptions);
    toggleEffectOptions(); // initial check on page load
</script>
```

By following these steps, you should be able to successfully implement a new effect in the Glitch Art application. 