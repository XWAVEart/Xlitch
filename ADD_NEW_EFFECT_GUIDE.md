# Guide: Adding a New Effect to the Glitch Art App (Post-Refactor)

This document outlines the step-by-step process for adding a new effect to the refactored Glitch Art application. The new architecture emphasizes modularity, with separate form classes for each effect and dynamic loading of UI components.

## 1. Implement the Core Effect Function

First, create or identify the Python module for your effect based on its nature:

- `glitch_art/effects/distortion.py` - For effects that manipulate pixel positions.
- `glitch_art/effects/color.py` - For effects that manipulate colors.
- `glitch_art/effects/sorting.py` - For pixel sorting effects.
- `glitch_art/effects/noise.py` - For noise-based effects.
- `glitch_art/effects/patterns.py` - For pattern generation.
- `glitch_art/effects/glitch.py` - For classic glitch/databending effects.
- `glitch_art/effects/pixelate.py` - For pixelation effects.
- `glitch_art/effects/blend.py` - For blending/compositing images.
- `glitch_art/effects/contour.py` - For contour and edge detection effects.

Define your effect function within the chosen module.

### Example: `my_new_effect.py` (e.g., in `glitch_art/effects/`)

```python
# In e.g., glitch_art/effects/custom.py (or an existing relevant module)
from PIL import Image
import numpy as np
import random # if needed

def my_awesome_effect(image, strength: int, use_random_seed: bool = False, seed: int = None):
    """
    Applies an awesome new effect to the image.

    Args:
        image (PIL.Image.Image): Input PIL Image object.
        strength (int): The intensity of the awesome effect.
        use_random_seed (bool, optional): Whether to use a specific seed for randomness.
        seed (int, optional): The seed value if use_random_seed is True.

    Returns:
        PIL.Image.Image: The processed PIL Image.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB') # Ensure RGB for consistency

    image_np = np.array(image)
    
    if use_random_seed and seed is not None:
        np.random.seed(seed)
        random.seed(seed) # Seed Python's random too if used directly

    # --- Your awesome effect logic here ---
    # Example: Simple inversion based on strength
    if strength > 50:
        processed_image_np = 255 - image_np
    else:
        processed_image_np = image_np.copy() # Make a copy if modifying in place
        # Add some noise if strength is low
        noise = np.random.randint(0, strength // 2 if strength > 0 else 1, 
                                  size=image_np.shape, dtype=image_np.dtype)
        processed_image_np = np.clip(image_np + noise, 0, 255)
    # --- End of effect logic ---
    
    return Image.fromarray(processed_image_np.astype(np.uint8))

```

### 1.2. Add to `glitch_art/effects/__init__.py`

Ensure your new effect function is importable from the `glitch_art.effects` package. Add it to the `__all__` list and import it in `glitch_art/effects/__init__.py`.

```python
# In glitch_art/effects/__init__.py
# ... other imports ...
from .custom import my_awesome_effect # Assuming it's in custom.py

__all__ = [
    # ... other effects ...
    'my_awesome_effect',
]
```

## 2. Create a Specific Form Class in `forms.py`

For each new effect, create a dedicated form class that inherits from `FlaskForm`. This class will define the UI fields for your effect's parameters.

### 2.1. Define the Form Class

```python
# In forms.py

from flask_wtf import FlaskForm
from wtforms import IntegerField, BooleanField # Add other fields as needed (SelectField, FloatField, etc.)
from wtforms.validators import DataRequired, Optional, NumberRange

# You can use existing mixins for common fields:
# from .mixins import SeedMixin, SecondaryImageMixin, SortingFieldsMixin 

class MyAwesomeEffectForm(FlaskForm): # If using SeedMixin: MyAwesomeEffectForm(FlaskForm, SeedMixin)
    """Form for My Awesome Effect parameters."""
    strength = IntegerField(
        'Effect Strength',
        default=25,
        validators=[DataRequired(), NumberRange(min=0, max=100, message="Strength must be 0-100.")]
    )
    # If your effect uses a seed, and you want to expose it via a standardized field,
    # inherit from SeedMixin and the 'seed' field will be automatically available.
    # For this example, let's add a boolean toggle and a conditional seed field if not using SeedMixin.
    use_random_seed = BooleanField('Use Specific Seed', default=False, validators=[Optional()])
    custom_seed = IntegerField('Seed Value', validators=[Optional(), NumberRange(min=1, max=99999)])
    
    # Add other fields as needed for your effect's parameters.
    # Example:
    # mode = SelectField('Mode', choices=[('fast', 'Fast'), ('quality', 'Quality')], default='fast')

    # If using SeedMixin, you don't need 'use_random_seed' or 'custom_seed' here;
    # The SeedMixin provides a standard 'seed' field (IntegerField with Optional validator).
    # Your effect function would then just check if `seed is not None`.
```

### 2.2. Add New Effect to `ImageProcessForm` Choices

In `forms.py`, add your new effect to the `choices` list of the `effect` `SelectField` within the main `ImageProcessForm`. The first element of the tuple is the effect key (used in `app.py` and URLs), and the second is the display name.

```python
# In forms.py, within class ImageProcessForm:

    effect = SelectField('Effect', choices=[
        # ... existing effects ...
        ('my_awesome_effect_key', 'My Awesome Effect') # Use a unique key
    ], validators=[DataRequired()])
```

## 3. Update Application Logic in `app.py`

### 3.1. Import Your New Effect Function and Form Class

At the top of `app.py`:

```python
# In app.py

# ... other effect imports ...
from glitch_art.effects.custom import my_awesome_effect # Adjust module if different

# ... other form imports ...
from forms import MyAwesomeEffectForm # Your new form class
```

### 3.2. Map Effect Key to Your Form Class in `EFFECT_FORM_MAP`

Add an entry to the `EFFECT_FORM_MAP` dictionary. This dictionary is crucial for the application to know which form class corresponds to which effect key.

```python
# In app.py

EFFECT_FORM_MAP = {
    # ... existing effect_key: FormClass mappings ...
    'my_awesome_effect_key': MyAwesomeEffectForm, # Must match the key from ImageProcessForm.effect.choices
}
```

### 3.3. Add Dispatch Logic for Your Effect

In the main `index()` route within `app.py`, add an `elif` block to handle your new effect. This block will be executed when your effect is selected and the form is submitted.

```python
# In app.py, inside the index() route, within the POST request handling block,
# after `if effect_specific_form.validate():` and inside the `try:` block:

                    # ... existing elif blocks for other effects ...
                    
                    elif selected_effect_key == 'my_awesome_effect_key':
                        # Parameters are retrieved from the automatically instantiated 
                        # and validated effect_specific_form (which is an instance of MyAwesomeEffectForm here).
                        
                        strength_val = effect_specific_form.strength.data
                        use_seed_flag = effect_specific_form.use_random_seed.data # From our example form
                        seed_val = effect_specific_form.custom_seed.data       # From our example form
                        
                        # If using SeedMixin, you'd access it like:
                        # seed_val = effect_specific_form.seed.data 
                        # And your effect function would just check `if seed is not None`.
                        # The 'use_random_seed' flag might become redundant or handled differently.

                        logger.debug(f"My Awesome Effect params: strength={strength_val}, use_seed={use_seed_flag}, seed={seed_val}")
                        
                        # Call your effect function
                        # Adapt parameters based on your actual effect function signature
                        if use_seed_flag and seed_val is not None:
                             processed_image = my_awesome_effect(image, 
                                                                 strength=strength_val, 
                                                                 use_random_seed=True, 
                                                                 seed=seed_val)
                        else:
                             processed_image = my_awesome_effect(image, 
                                                                 strength=strength_val, 
                                                                 use_random_seed=False, 
                                                                 seed=None) # Or just my_awesome_effect(image, strength_val) if defaults handle it

                        # Create a descriptive settings string for the output filename
                        settings = f"awesome_{strength_val}"
                        if use_seed_flag and seed_val is not None:
                            settings += f"_s{seed_val}"
                        # Add other parameters to settings string as needed
                    
                    # ... other elif blocks ...
```

## 4. HTML Template and JavaScript (Usually No Changes Needed!)

Due to the refactored dynamic form loading system, you typically **do not need to make direct changes** to `templates/index.html` or its associated JavaScript to add the fields for your new effect.

-   **Form Field Rendering:** The `templates/_effect_form_fields.html` partial template automatically iterates through the fields of the `effect_specific_form` (which will be an instance of `MyAwesomeEffectForm` when your effect is selected) and renders them.
    -   Ensure your fields in `MyAwesomeEffectForm` have appropriate labels (e.g., `IntegerField('Effect Strength', ...)`).
    -   Field descriptions/help text can be added via the `description` argument in WTForms fields or manually in `_effect_form_fields.html` if more complex markup is needed (though this is less common now).
-   **Dynamic Loading:** The JavaScript in `index.html` (specifically the `loadEffectFields` function) handles fetching and injecting the HTML for your specific form's fields by making an AJAX call to `/get-effect-form/<effect_key>`. This endpoint uses `EFFECT_FORM_MAP` to get your form class and `_effect_form_fields.html` to render it.

### When HTML/JS Changes Might Be Needed:

-   **Complex Conditional Logic *Within* Your Specific Form:** If fields in `MyAwesomeEffectForm` need to show/hide based on *other fields within that same form*, you might need to add custom JavaScript. This JavaScript could be:
    -   Included directly in `_effect_form_fields.html` within a `<script>` tag that only runs when your form is loaded.
    -   Attached globally in `index.html` using event delegation, targeting elements by IDs/classes you define in your form fields.
-   **Highly Custom Field Rendering:** If the standard rendering in `_effect_form_fields.html` is insufficient for a particular field in your new form, you might need to adjust the partial template with conditional logic for your specific field's name or type.

## 5. Test Your New Effect Thoroughly

1.  **Start the application:** `python app.py`
2.  **Open in browser:** `http://localhost:8080` (or your configured port).
3.  **Upload an image.**
4.  **Select your new effect** ("My Awesome Effect") from the main "Effect" dropdown.
5.  **Verify Form Display:**
    -   Confirm that the fields defined in `MyAwesomeEffectForm` (e.g., "Effect Strength", "Use Specific Seed", "Seed Value") appear correctly.
    -   Check default values and validators (e.g., try submitting without a required field, or with a value out of range).
6.  **Enter valid parameters.**
7.  **Submit the form.**
8.  **Verify Output:**
    -   Check that the image is processed as expected by `my_awesome_effect`.
    -   Examine the generated filename to ensure your `settings` string is correctly formed.
    -   Look for any errors in the browser console or the Flask application logs in your terminal.

## 6. Troubleshooting Checklist (Post-Refactor)

-   **Effect Function:**
    -   Is your effect function correctly defined and placed in a module under `glitch_art/effects/`?
    -   Did you add your effect function to `glitch_art/effects/__init__.py` and its `__all__` list?
    -   Does it handle `PIL.Image` input and return a `PIL.Image`?
    -   Is it correctly converting to/from NumPy arrays if used internally?
-   **Forms (`forms.py`):**
    -   Did you create a new, separate form class (e.g., `MyAwesomeEffectForm`) for your effect?
    -   Are all fields (e.g., `IntegerField`, `SelectField`) correctly defined with appropriate validators (`DataRequired`, `Optional`, `NumberRange`, etc.) and labels in this specific form class?
    -   Did you add your effect's unique key and display name to the `choices` of the `effect` field in the main `ImageProcessForm`?
-   **Application Logic (`app.py`):**
    -   Did you import your new effect function (e.g., `my_awesome_effect`) and your new specific form class (e.g., `MyAwesomeEffectForm`)?
    -   Did you add an entry to `EFFECT_FORM_MAP` mapping your effect's string key to your new form class? (e.g., `'my_awesome_effect_key': MyAwesomeEffectForm`). Ensure the key matches `ImageProcessForm`.
    -   In the `index()` route, does the `elif selected_effect_key == 'my_awesome_effect_key':` block correctly retrieve parameters from `effect_specific_form.your_field_name.data`?
    -   Are you passing the parameters to your effect function in the correct order and with the correct names?
    -   Is the `settings` string for the filename being generated correctly?
-   **General:**
    -   Check for typos in effect keys, function names, class names, field names, and dictionary keys. These must match exactly between `forms.py`, `app.py`, and your effect module.
    -   Review the Flask application logs (terminal output) for any error messages or tracebacks. These are usually very helpful.
    -   Use your browser's developer tools (Network tab for AJAX requests to `/get-effect-form/`, Console for JavaScript errors) if the UI is not behaving as expected.

This updated guide should provide a clearer path for integrating new effects into the refactored application structure. 