# Glitch Art App - Codebase Refactoring Plan

This document outlines the steps to refactor the Glitch Art Flask application for improved modularity, maintainability, and clarity.

## Phase 1: Refactor Form Handling and Application Dispatch Logic

This phase addresses the monolithic form structure and the centralized effect dispatch in `app.py`.

**1.1. Deconstruct `ImageProcessForm` (`forms.py`)**
    - **Goal:** Replace the single large form with smaller, effect-specific forms and reusable components.
    - **Tasks:**
        - **1.1.1. Define Base/Mixin Classes:** (Completed)
            - Create `SortingFieldsMixin` for common sorting parameters (`sort_by`, `reverse_sort`).
            - Create `SeedMixin` for a common `seed` parameter.
            - Create `SecondaryImageMixin` for `secondary_image` upload.
        - **1.1.2. Create Effect-Specific Form Classes:** (Completed)
            - For each effect listed in the main `effect` dropdown:
                - Create a new `FlaskForm` subclass (e.g., `PixelSortChunkForm`, `ColorChannelForm`, etc.).
                - Move the relevant fields from the old `ImageProcessForm` into this new class.
                - Inherit from mixin classes (`SortingFieldsMixin`, `SeedMixin`, `SecondaryImageMixin`) where applicable.
                - Standardize field names (e.g., `chunk_width` -> `width`, `bit_random_seed` -> `seed` via mixin).
                - Update field types where appropriate (e.g., SelectFields with 'true'/'false' choices to `BooleanField`).
                - Ensure validators (`DataRequired`, `Optional`, `NumberRange`, custom validators) are correctly applied within each specific form.
        - **1.1.3. Slim Down Main `ImageProcessForm`:** (Completed)
            - The main `ImageProcessForm` now only retains: `primary_image`, `effect`, and `submit`.
            - Removed all other effect-specific fields and the custom `validate()` method.
        - **1.1.4. Preserve Custom Validators:** (Completed)
            - Kept utility validators (e.g., `validate_chunk_size`, `validate_multiple_of_8`, `validate_is_odd`) at the top of `forms.py` for use by new specific forms.

**1.2. Refactor `app.py` (Main `index` Route)**
    - **Goal:** Decouple effect dispatch from the route logic using a registry and dynamic form handling.
    - **Tasks:**
        - **1.2.1. Create `EFFECT_FORM_MAP`:** (Completed)
            - Defined a Python dictionary in `app.py` mapping effect ID strings to their corresponding new effect-specific form classes from `forms.py`.
        - **1.2.2. Modify `index` Route Logic:** (Completed)
            - Route first validates the slimmed-down `ImageProcessForm`.
            - Uses `EFFECT_FORM_MAP` to instantiate the specific effect form class.
            - Validates the specific effect form.
            - If valid, prepares arguments and calls the processing function.
            - Handles AJAX responses and template rendering, passing the main form and the instantiated specific effect form to the template.
        - **1.2.3. Update Effect Function Signatures in `glitch_art.effects/`:** (Completed during testing in 1.4)
            - Modified Python functions to accept parameters matching form fields, notably adding `seed` where appropriate.

**1.3. Update `templates/index.html` for Dynamic Form Rendering** (Completed)
    - **Goal:** Dynamically load and display form fields for the selected effect without a page reload.
    - **Tasks:**
        - **1.3.1. Modify `index.html` Structure:** (Completed)
            - Removed the large static block of HTML for all effect fields.
            - Added a container `div` (e.g., `#effect-specific-fields-container`) for dynamic content.
            - Used `{% include '_effect_form_fields.html' %}` for server-side rendering on initial load/POST error.
        - **1.3.2. Implement AJAX for Dynamic Form Loading:** (Completed)
            - Created a new partial template `templates/_effect_form_fields.html` to render fields for any specific form.
            - Implemented a new `/get-effect-form/<effect_key>` endpoint in `app.py` that:
                - Takes an `effect_key`.
                - Instantiates the corresponding specific form.
                - Renders `_effect_form_fields.html` with this form.
                - Returns the HTML as a JSON response.
            - JavaScript in `index.html`:
                - Added an event listener to the `#effect` dropdown.
                - On change, calls a `loadEffectFields(effectKey)` function.
                - `loadEffectFields` makes an AJAX GET request to `/get-effect-form/<effectKey>`.
                - On success, injects the returned HTML into the `#effect-specific-fields-container`.
                - Updates URL query parameter (`?effect=...`) for deep linking.
        - **1.3.3. Ensure AJAX Submission Handles New Structure:** (Completed)
            - Updated AJAX form submission logic in `index.html` to correctly serialize and display validation errors for both the main form and the dynamically loaded specific form fields.

**1.4. Test and Refine Dynamic Form Behavior** (Completed)
    - **Goal:** Ensure all dynamic form functionalities work correctly, image processing is successful with new forms, and identify/fix bugs.
    - **Issues Resolved During Testing:**
        - **Initial `forms.py` Errors:**
            - `NameError`: `SecondaryImageMixin` was defined after `DoubleExposeForm` which tried to use it. Corrected order of definitions.
            - `TypeError` (MRO): `SecondaryImageMixin` incorrectly inherited from `FlaskForm`, causing issues when `DoubleExposeForm` inherited from both. Removed `FlaskForm` inheritance from mixin.
        - **Template Rendering Errors in `_effect_form_fields.html`:**
            - `AttributeError`: Issue with `BooleanField` rendering (e.g., `field.label.text`). Simplified to `{{ field() }}` and `{{ field.label }}`.
            - `TemplateSyntaxError`: Unescaped backslashes in `id` attributes. Removed offending backslashes.
        - **Effect Function and Parameter Mismatches:**
            - `TypeError: unexpected keyword argument 'seed'`: Several effect functions (`perlin_noise_displacement`, `ripple_effect`, `color_shift_expansion`, `concentric_shapes`, `contour_effect`) were missing the `seed` parameter in their signatures. Updated functions to include `seed=None` and use it.
        - **Form Validation Issues:**
            - `DataRequired` error for optional fields: `value_boost` in `ColorShiftExpansionForm` and `darken_step` in `ConcentricShapesForm` were incorrectly marked as required. Changed validators to `Optional()`.
            - `FileRequired` error for `DoubleExposeForm` and `MaskedMergeForm`: `FileField`s were not being populated correctly because the specific form was instantiated with `EffectFormClass(request.form)` instead of `EffectFormClass()`. Corrected instantiation in `app.py` to allow Flask-WTF to handle `request.files`.
        - **Performance Issues:**
            - `contour_effect` hanging: The `LineGrid` used for intersection checking was initialized with `cell_size=100`, potentially causing performance issues with dense contours. Changed to `cell_size=25`.
        - **Default Value Adjustments:**
            - `BitManipulationForm`: Changed default for `offset` from 0 to 1.

## Phase 2: Modularize Effect Implementations

This phase focuses on improving the internals of the image processing logic.

**2.1. Standardize Effect Function Parameters & Conventions** (Completed)
    - **Goal:** Ensure consistency across all effect functions.
    - **Tasks:**
        - **2.1.1. Sort Order:** (Completed)
            - Reviewed usage of boolean `reverse` parameter for sort order. Existing `SortingFieldsMixin` and specific form implementations (e.g., `PolarSortForm`, `PerlinNoiseSortForm`) handle this well. `voronoi_pixel_sort` maintains its string-based `sort_order` ('clockwise'/'counter-clockwise') as a justified exception due to its specific directional sorting logic.
        - **2.1.2. Naming:** (Completed)
            - Reviewed standardization of parameter names for `seed`, `width`/`height`, `intensity`, etc. These were largely addressed during Phase 1.
            - Specific standardizations applied in this step:
                - **Slice Effects (`slice_shuffle`, `slice_offset`, `slice_reduction`):** In `glitch_art/effects/distortion.py`, the function parameter `slice_count` was changed to `count` to align with the form field name (`count`) and the variable name used in `app.py`.
                - **Concentric Shapes Effect (`concentric_shapes`):** Corrected keyword argument in `app.py` call from `color_shift_amount` to `color_shift` to match function definition in `glitch_art/effects/patterns.py`.

**2.2. Refine Effect Modules (e.g., `glitch_art/effects/sorting.py`)**
    - **Goal:** Reduce duplication, improve clarity, and enhance performance.
    - **Tasks:**
        - **2.2.1. DRY (Don't Repeat Yourself):** (Completed for `sorting.py`)
            - **`glitch_art/effects/sorting.py` specific changes:**
                - **Centralized `_SORT_FUNCTIONS`:** Defined the sort function mapping dictionary once at the module level and updated all relevant sorting functions (`pixel_sorting`, `pixel_sorting_corner_to_corner`, `full_frame_sort`, `spiral_sort_2`, `polar_sorting`) to use this shared dictionary. Maintained original default fallbacks for `full_frame_sort` and `spiral_sort_2`.
                - **Refactored `pixel_sorting()` Chunk Logic:** Introduced a helper function `_process_and_sort_chunk` to handle the extraction, sorting, and placement of pixels for a single chunk. The main `pixel_sorting` function was rewritten to use this helper, significantly reducing repetitive code for full-sized, right-edge, bottom-edge, and corner-remainder chunks.
                - **Removed Redundant `diagonal_pixel_sort`:** The `diagonal_pixel_sort` function was identified as redundant because its functionality is covered by `pixel_sorting` (when `sort_mode='diagonal'`) which internally calls `pixel_sorting_corner_to_corner`. The `diagonal_pixel_sort` function has been removed.
            - Centralize common logic, like the `sort_function` dictionary in `sorting.py` (define once and reuse).
            - Refactor verbose logic (e.g., `pixel_sorting` chunk processing) into helper functions.
        - **2.2.2. Remove Redundancy:** (Partially completed for `sorting.py`)
            - **`glitch_art/effects/sorting.py` specific changes:**
                - Removed the `diagonal_pixel_sort` function as its functionality was covered by `pixel_sorting` with `sort_mode='diagonal'`.
                - Removed the `spiral_sort` function as it was superseded by the more flexible and currently used `spiral_sort_2`.
            - Identify and remove any deprecated or unused functions (e.g., `spiral_sort` if fully replaced by `spiral_sort_2`).
        - **2.2.3. Performance (Ongoing):** (Partially addressed for `pixel_sorting` in `sorting.py`)
            - **`glitch_art/effects/sorting.py` - `pixel_sorting` function:**
                - Refactored `pixel_sorting` and its helper function (renamed to `_process_and_sort_chunk_np`) to use NumPy arrays for image data representation and manipulation.
                - Chunk extraction and placement are now done using efficient NumPy slicing instead of per-pixel PIL operations or list iterations from a flat list.
                - The core sorting logic still converts NumPy pixel data within a chunk to a list of tuples to remain compatible with the existing `_SORT_FUNCTIONS` (which expect tuples), then sorts this list, and converts back to a NumPy array for placement.
                - This significantly improves performance for data handling aspects while maintaining compatibility for sort key generation.
            - **`glitch_art/effects/sorting.py` - `full_frame_sort` function:**
                - Refactored to use NumPy arrays for image data, column/row extraction via slicing, and result assembly. Conversion to list of tuples for `sort_function` compatibility is maintained.
            - **`glitch_art/effects/sorting.py` - `spiral_sort_2` function:**
                - Reviewed and made minor adjustments for explicit tuple conversion before calling `sort_function`, enhancing consistency. Loop variables also renamed for clarity.
            - **`glitch_art/effects/sorting.py` - `polar_sorting` function:**
                - Refactored to use NumPy arrays throughout, from initial image conversion to chunk processing via slicing, using `np.argsort` for sorting based on polar coordinates, and final image assembly.
            - **`glitch_art/effects/sorting.py` - `pixel_sorting_corner_to_corner` function (diagonal sort path):**
                - Refactored to use NumPy arrays for image data, chunk extraction via slicing, and result assembly. Conversion to list of tuples for `sort_function` compatibility is maintained. Logic for iterating chunks from different corners and reshaping sorted data based on the `horizontal` flag was adapted for NumPy.
            - Where appropriate and beneficial, explore replacing pixel-by-pixel Python loops with vectorized NumPy operations.

**2.3. Improve `glitch_art/core/pixel_attributes.py`** (Completed)
    - **Goal:** Make pixel attribute calculations more robust and consistent.
    - **Tasks:**
        - **2.3.1. Input Consistency:** (Completed)
            - Standardized all methods in `PixelAttributes` to expect input as a 3-element Python tuple `(R, G, B)` where R, G, B are integers in the `0-255` range.
            - Removed internal logic for handling/guessing `0-1` scaled inputs.
        - **2.3.2. Output Normalization:** (Completed)
            - Standardized output ranges:
                - `brightness`: `0-255.0` (float)
                - `hue`: `0-360.0` (float)
                - `saturation`: `0-1.0` (float)
                - `luminance` (HSV Value): `0-1.0` (float)
                - `contrast`: `0-255` (int)
                - `color_sum`: `0-765` (int)
            - Updated all docstrings to reflect these standardized input and output specifications.
        - **2.3.3. Review Fallbacks:** (Completed)
            - Removed the `ImageColor.getcolor` fallback logic from `hue`, `saturation`, and `luminance` methods, as standardized input makes them unnecessary and errors on incorrect input are preferred.

**2.4. Decouple `glitch_art/core/image_utils.py`** (Completed)
    - **Goal:** Remove Flask dependencies from core utilities.
    - **Tasks:**
        - **2.4.1. `resize_image_if_needed`:** (Completed)
            - Removed the direct Flask `current_app.config` access from `resize_image_if_needed`.
            - The function now takes `max_width` and `max_height` as direct arguments, with internal defaults if they are not provided (or are `None`).
            - Modified `load_image` to accept `max_width_config` and `max_height_config` and pass them to `resize_image_if_needed`.
            - Updated `app.py` to pass `app.config.get('MAX_IMAGE_WIDTH')` and `app.config.get('MAX_IMAGE_HEIGHT')` to `load_image` calls.
        - **2.4.2. Logging:** (Completed)
            - Changed `print()` error messages in `load_image` to use `logger.error()`.
            - Added `import logging` and `logger = logging.getLogger(__name__)` to `image_utils.py`.

## Phase 3: Consolidate Core Logic & Refine Effects

    - **Goal:** Further reduce duplication, improve clarity, and enhance performance in remaining effect modules. Acknowledge that significant consolidation occurred in Phase 2.
    - **Tasks (Targeted Review & Refinement):**
        - **3.1. Review `glitch_art/effects/distortion.py`:** (Completed)
            - **3.1.1. `pixel_scatter` - Use `PixelAttributes`:** (Completed)
                - Removed local helper functions for hue, saturation, luminance, and contrast.
                - Updated `selection_func` logic to use standardized methods from `PixelAttributes`, applying scaling for saturation/luminance to match form expectations.
            - **3.1.2. `pixel_drift` - NumPy Performance Refactor:** (Completed)
                - Converted main pixel manipulation logic to use NumPy arrays (image to array at start, slicing for row/column assignments, array to image at end), removing `getdata()`/`putdata()`.
            - **3.1.3. `pixel_scatter` - NumPy Performance Refactor:** (Completed)
                - Converted main pixel manipulation loops (both horizontal and vertical scatter) to use NumPy arrays. This includes extracting rows/columns via slicing, processing, and assigning back to a NumPy result array.
            - **3.1.4. `generate_noise_map` Review:** (Completed)
                - Reviewed `generate_noise_map`. Decided to keep it local to `distortion.py` for now as its uses are confined to this module. Will reconsider moving to core utilities if other modules require it later.
        - **3.2. Review other key effect modules (e.g., `color.py`, `patterns.py`) for similar consolidation opportunities.** (Ongoing)
            - **3.2.1. Review `glitch_art/effects/color.py`:** (Completed)
                - **`color_channel_manipulation` - NumPy Refactor:** (Completed) Converted to use NumPy for direct channel manipulation, replacing PIL's `split`, `merge`, and `point` methods.
                - **`posterize` - NumPy Refactor:** (Completed) Replaced PIL's LUT and channel-wise `point` operations with direct NumPy array calculations.
                - **`curved_hue_shift` - Refactor HSV Handling:** (Completed) Replaced manual NumPy-based HSV conversion with PIL's `image.convert('HSV')` and `image.convert('RGB')`, while retaining NumPy for H-channel manipulation. Improved RGBA handling.
                - **`color_shift_expansion` - Partial NumPy Refactor:** (Completed)
                    - Vectorized distance map generation using `np.mgrid` and NumPy array operations.
                    - Optimized parts of the main pixel loop by using NumPy for influence calculations and weighted blending of seed colors.
                    - Kept per-pixel `colorsys` conversions due to complexity of full vectorization for this specific effect's logic.
            - **3.2.2. Review `glitch_art/effects/patterns.py`:** (Completed)
                - **`color_shift_expansion` (patterns version) - NumPy Refactor & Seeding:** (Completed) Added `seed` parameter, vectorized distance map generation, and applied NumPy optimizations to its main loop for influence calculations and color blending.
                - **Local HSV Helpers Removal:** (Completed) Removed local `rgb_to_hsv` and `hsv_to_rgb` utility functions. Updated `concentric_shapes` to use `colorsys` directly for HSV conversions with appropriate scaling.
                - **`darken_color` Update:** (Completed) Modified to handle RGBA tuples, preserving the alpha channel, in support of `concentric_shapes` refactoring.
                - **`concentric_shapes` - Direct Drawing Refactor:** (Completed) Refactored to draw shapes directly onto a copy of the input image (converted to RGBA) using `ImageDraw.Draw()`. This avoids creating intermediate `shape_image` instances and performing repeated `Image.alpha_composite` calls. Corrected drawing calls to use the main `ImageDraw` object and appropriate RGBA color for outlines.
                - **`masked_merge` - NumPy & Logic Refinements:** (Completed)
                    - **`'checkerboard'` & `'random_checkerboard'`:** Already refactored to NumPy in a previous step, confirmed correct.
                    - **`'striped'`, `'gradient_striped'`, `'linear_gradient_striped'`:** Reviewed, already using NumPy effectively.
                    - **`'perlin'`:** Refined seeding for `noise.pnoise2` (using `base` parameter) and improved normalization logic for the noise map. Confirmed fallback to random noise if `noise` module is not found.
                    - **`'voronoi'`:** Reviewed, already using NumPy and `cKDTree` effectively for cell generation and assignment.
                    - **`'concentric_circles'`:** Refactored mask generation to use NumPy array operations (calculating distances from center and determining fill based on band index) instead of iterative `ImageDraw.ellipse` calls.
                    - **`'concentric_rectangles'` & `'random_triangles'`:** Reviewed; kept existing `ImageDraw` implementations as NumPy vectorization was less straightforward or beneficial for their specific logic.
                - **`voronoi_pixel_sort` - Partial NumPy Optimization:** (Completed) Optimized pixel value calculation for basic sort types ('color', 'red', 'green', 'blue') using direct NumPy array operations.

## Phase 4: Frontend Enhancements

## Phase 5: Review and Finalize

## Task Tracking

- **Phase 1: Refactor Form Handling and Application Dispatch Logic**
    - [X] 1.1. Deconstruct `ImageProcessForm` (`forms.py`)
        - [X] 1.1.1. Define Base/Mixin Classes
        - [X] 1.1.2. Create Effect-Specific Form Classes (All initial effects completed)
        - [X] 1.1.3. Slim Down Main `ImageProcessForm`
        - [X] 1.1.4. Preserve Custom Validators
    - [X] 1.2. Refactor `app.py` (Main `index` Route)
        - [X] 1.2.1. Create `EFFECT_FORM_MAP`
        - [X] 1.2.2. Modify `index` Route Logic for dynamic form instantiation and parameter handling
        - [X] 1.2.3. Update Effect Function Signatures (as needed during testing)
    - [X] 1.3. Update `templates/index.html` for Dynamic Form Rendering
        - [X] 1.3.1. Modify `index.html` for dynamic field container and partial include
        - [X] 1.3.2. Implement AJAX to fetch/render specific form HTML (`/get-effect-form/`, `_effect_form_fields.html`)
        - [X] 1.3.3. Ensure AJAX submission handles new dynamic form structure and errors
    - [X] 1.4. Test and Refine Dynamic Form Behavior (All listed sub-issues resolved)
- **Phase 2: Modularize Effect Implementations** (Completed)
    - [X] 2.1. Standardize Effect Function Parameters & Conventions
    - [X] 2.2. Refine Effect Modules (sorting.py)
    - [X] 2.3. Improve `glitch_art/core/pixel_attributes.py`
    - [X] 2.4. Decouple `glitch_art/core/image_utils.py`
- **Phase 3: Consolidate Core Logic & Refine Effects** (Ongoing)
    - [X] 3.1. Review `glitch_art/effects/distortion.py`
        - [X] 3.1.1. `pixel_scatter` - Use `PixelAttributes`
        - [X] 3.1.2. `pixel_drift` - NumPy Refactor
        - [X] 3.1.3. `pixel_scatter` - NumPy Refactor
        - [X] 3.1.4. `generate_noise_map` Review
    - [ ] 3.2. Review other key effect modules
        - [X] 3.2.1. Review `glitch_art/effects/color.py`
        - [X] 3.2.2. Review `glitch_art/effects/patterns.py`
- **Phase 4: Frontend Enhancements** (Next if Phase 3 concludes)
- **Phase 5: Review and Finalize** 