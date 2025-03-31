# Glitch Art App Refactoring Plan

## Completed Steps

1. ✅ Created the modular package structure 
   ```
   glitch_art/
   ├── __init__.py
   ├── core/
   │   ├── __init__.py
   │   ├── pixel_attributes.py
   │   └── image_utils.py
   ├── effects/
   │   ├── __init__.py
   │   └── sorting.py (started)
   └── utils/
       ├── __init__.py
       └── helpers.py
   ```

2. ✅ Implemented core modules:
   - PixelAttributes class in pixel_attributes.py
   - Image loading/resizing in image_utils.py
   - Basic utility functions in helpers.py

3. ✅ Set up backward compatibility in __init__.py to import from original utils.py
   - This allows for incremental migration without breaking the app

4. ✅ Updated app.py to import from the new package structure
   - This ensures the app works with both the old and new structure during transition

5. ✅ Created a test file to verify basic functionality

## Next Steps

1. Complete implementation of effects modules:

   - **sorting.py**:
     - Complete pixel_sorting() implementation
     - Add pixel_sorting_corner_to_corner()
     - Add full_frame_sort()
     - Add spiral_sort() and spiral_sort_2()
     - Add polar_sorting()
     - Add functions for diagonal pixel sort

   - **color.py**:
     - Add color_channel_manipulation()
     - Add split_and_shift_channels()
     - Add histogram_glitch()

   - **distortion.py**:
     - Add pixel_drift()
     - Add perlin_noise_displacement()
     - Add geometric_distortion()
     - Add voronoi_distortion()

   - **glitch.py**:
     - Add databend_image()
     - Add simulate_jpeg_artifacts()
     - Add bit_manipulation()

   - **patterns.py**:
     - Add voronoi_pixel_sort()
     - Add masked_merge()
     - Add concentric_shapes()
     - Add Ripple()

   - **noise.py**:
     - Add perlin_noise_sorting()
     - Add perlin_full_frame_sort()
     - Add generate_noise_map() (move from helpers)

2. Extend test suite:
   - Add tests for each module
   - Create test images
   - Verify all effects work as expected

3. Update package imports:
   - Gradually remove imports from original utils.py
   - Replace with imports from specific modules

4. Documentation:
   - Add docstrings to all functions
   - Update README with examples for each effect type
   - Create a migration guide for anyone using the original utils.py

## Migration Strategy

1. **Incremental Approach**:
   - Refactor one module at a time
   - Run tests after each module is completed
   - Keep backward compatibility through the top-level __init__.py

2. **Testing and Verification**:
   - After each module is completed, run the web app to ensure effects still work
   - Verify that images processed with new code match those from original code

3. **Final Steps**:
   - Once all modules are refactored, remove the backward compatibility imports
   - Update all file references to use the new structure
   - Remove the original utils.py file
   - Ensure all tests pass with the new structure

## Performance Improvements

During refactoring, implement these optimizations:

1. Replace pixel-by-pixel loops with vectorized NumPy operations where possible
2. Add caching for repeated calculations
3. Optimize memory usage by avoiding unnecessary array copies
4. Add optional multithreading for computationally intensive effects
5. Add progress reporting for long-running operations 