from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import os
import traceback
import logging
import sys
import numpy as np

# Configure logging immediately
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Ensure logs go to stdout for Heroku log capture
)
logger = logging.getLogger(__name__)

# Log startup information
logger.info("Starting application...")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current directory: {os.getcwd()}")
logger.info(f"Directory contents: {os.listdir('.')}")

try:
    # Import form module
    from forms import ImageProcessForm
    logger.info("Successfully imported forms module")

    # Try importing the glitch_art package
    logger.info("Attempting to import glitch_art modules...")
    
    # Import effect functions directly from their modules for improved maintainability
    from glitch_art.effects.sorting import pixel_sorting, full_frame_sort, spiral_sort_2, polar_sorting
    from glitch_art.effects.color import color_channel_manipulation, split_and_shift_channels, histogram_glitch, color_shift_expansion, posterize, curved_hue_shift
    from glitch_art.effects.distortion import pixel_drift, perlin_noise_displacement, pixel_scatter, ripple_effect, offset_effect, slice_shuffle, slice_offset, slice_reduction, geometric_distortion
    from glitch_art.effects.glitch import bit_manipulation, databend_image, simulate_jpeg_artifacts, data_mosh_blocks
    from glitch_art.effects.patterns import voronoi_pixel_sort, masked_merge, concentric_shapes
    from glitch_art.effects.noise import perlin_noise_sorting, perlin_full_frame_sort
    from glitch_art.effects.pixelate import pixelate_by_attribute
    from glitch_art.effects.blend import double_expose
    # Import core utilities
    from glitch_art.core.image_utils import load_image, generate_output_filename, resize_image_if_needed
    from glitch_art.effects import contour_effect
    from glitch_art.effects.distortion import block_shuffle
    
    logger.info("Successfully imported all glitch_art modules")

    # Import all specific form classes
    from forms import (
        PixelSortChunkForm, ColorChannelForm, BitManipulationForm, DoubleExposeForm,
        FullFrameSortForm, PixelDriftForm, DatabendingForm, PolarSortForm,
        SpiralSort2Form, JpegArtifactsForm, PerlinFullFrameForm, VoronoiSortForm,
        PerlinNoiseSortForm, PixelScatterForm, PerlinDisplacementForm, RippleEffectForm,
        RGBChannelShiftForm, HistogramGlitchForm, ColorShiftExpansionForm,
        PixelateForm, ConcentricShapesForm, PosterizeForm, CurvedHueShiftForm,
        MaskedMergeForm, OffsetEffectForm, SliceShuffleForm, SliceOffsetForm,
        SliceReductionForm, ContourForm, BlockShuffleForm, DataMoshBlocksForm
    )
    logger.info("Successfully imported all specific effect form classes")

except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    logger.error(traceback.format_exc())
    
    # Check if the glitch_art package exists
    if os.path.exists('glitch_art'):
        logger.info("glitch_art directory exists, listing contents...")
        logger.info(f"glitch_art contents: {os.listdir('glitch_art')}")
        
        if os.path.exists('glitch_art/effects'):
            logger.info(f"effects directory contents: {os.listdir('glitch_art/effects')}")
        
        if os.path.exists('glitch_art/core'):
            logger.info(f"core directory contents: {os.listdir('glitch_art/core')}")
    else:
        logger.error("glitch_art directory does not exist")

from flask_wtf.csrf import CSRFProtect
import random
import secrets
from PIL import Image

# --- Effect Form Mapping ---
EFFECT_FORM_MAP = {
    'pixel_sort_chunk': PixelSortChunkForm,
    'color_channel': ColorChannelForm,
    'bit_manipulation': BitManipulationForm,
    'double_expose': DoubleExposeForm,
    'full_frame_sort': FullFrameSortForm,
    'pixel_drift': PixelDriftForm,
    'databend': DatabendingForm,
    'polar_sort': PolarSortForm,
    'spiral_sort_2': SpiralSort2Form,
    'jpeg_artifacts': JpegArtifactsForm,
    'perlin_full_frame': PerlinFullFrameForm,
    'voronoi_sort': VoronoiSortForm,
    'perlin_noise_sort': PerlinNoiseSortForm,
    'pixel_scatter': PixelScatterForm,
    'perlin_displacement': PerlinDisplacementForm,
    'ripple': RippleEffectForm, # Key is 'ripple' from main form
    'channel_shift': RGBChannelShiftForm, # Key is 'channel_shift' from main form
    'histogram_glitch': HistogramGlitchForm,
    'color_shift_expansion': ColorShiftExpansionForm,
    'pixelate': PixelateForm,
    'concentric_shapes': ConcentricShapesForm,
    'posterize': PosterizeForm,
    'curved_hue_shift': CurvedHueShiftForm,
    'masked_merge': MaskedMergeForm,
    'offset': OffsetEffectForm, # Key is 'offset' from main form
    'slice_shuffle': SliceShuffleForm,
    'slice_offset': SliceOffsetForm,
    'slice_reduction': SliceReductionForm,
    'contour': ContourForm,
    'block_shuffle': BlockShuffleForm,
    'data_mosh_blocks': DataMoshBlocksForm,
    # Add a mapping for the old 'spiral_sort' if it needs to be handled gracefully
    # or ensure it's removed from the main form's choices if it's fully replaced by spiral_sort_2
}
logger.info("EFFECT_FORM_MAP created.")

app = Flask(__name__)
# Generate a secure random key for development, use environment variable in production
if os.environ.get('FLASK_ENV') == 'production':
    if not os.environ.get('SECRET_KEY'):
        raise RuntimeError('SECRET_KEY environment variable must be set in production mode')
    app.secret_key = os.environ.get('SECRET_KEY')
else:
    # Development mode - generate a random secret key
    app.secret_key = secrets.token_hex(32)  # 32 bytes = 256 bits of randomness
    print("Warning: Using randomly generated secret key for development. Sessions won't persist between restarts.")

# Check if running on Heroku
is_heroku = os.environ.get('IS_HEROKU', False)

# Set upload and processed folders
if is_heroku:
    # Use tmp directory on Heroku (note: this is ephemeral)
    app.config['UPLOAD_FOLDER'] = '/tmp/uploads/'
    app.config['PROCESSED_FOLDER'] = '/tmp/processed/'
else:
    # Local development
    app.config['UPLOAD_FOLDER'] = 'uploads/'
    app.config['PROCESSED_FOLDER'] = 'processed/'

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['WTF_CSRF_ENABLED'] = True  # Enable CSRF protection
app.config['WTF_CSRF_TIME_LIMIT'] = 3600  # Extend CSRF token validity to 1 hour
app.config['WTF_CSRF_CHECK_DEFAULT'] = False  # We'll check it manually for AJAX
app.config['MAX_IMAGE_WIDTH'] = 1920  # Maximum image width before resizing
app.config['MAX_IMAGE_HEIGHT'] = 1920  # Maximum image height before resizing - set to same as width for longest dimension resize

# Ensure upload and processed directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Import CSRF protection after app is created
csrf = CSRFProtect(app)

# Add CSRF exempt for AJAX requests
@csrf.exempt
def csrf_exempt(view):
    return view

@app.route('/', methods=['GET', 'POST'])
@csrf_exempt
def index():
    """Handle the main page with image upload and effect selection."""
    form = ImageProcessForm()
    effect_specific_form = None
    selected_effect_key = form.effect.data  # Get initial effect key if form is populated (e.g. from POST)

    if request.method == 'GET':
        # If a specific effect is pre-selected via query param (for dynamic loading on frontend)
        # Or if the form was submitted with GET for some reason, keep the selection
        query_effect_key = request.args.get('effect')
        if query_effect_key:
            selected_effect_key = query_effect_key
        
        if selected_effect_key and selected_effect_key in EFFECT_FORM_MAP:
            EffectFormClass = EFFECT_FORM_MAP[selected_effect_key]
            effect_specific_form = EffectFormClass(request.args) # Instantiate for GET request display, can fill from query args
            form.effect.data = selected_effect_key # Ensure main form selector reflects this
        elif selected_effect_key: # Effect key provided but not in map
            logger.warning(f"GET request with unknown effect key: {selected_effect_key}")
            # Optionally, clear it or set a default, or let the template handle it
            selected_effect_key = None # Reset to avoid issues

    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    logger.debug(f"Request method: {request.method}, AJAX: {is_ajax}, Selected Effect Key: {selected_effect_key}")
    
    if request.method == 'POST':
        logger.debug(f"POST Form data: {request.form}")
        logger.debug(f"POST Files: {request.files}")
        
        if form.validate_on_submit(): # Validates primary_image and effect selection
            logger.debug("Main ImageProcessForm validated successfully")
            selected_effect_key = form.effect.data # Confirmed selected effect
            EffectFormClass = EFFECT_FORM_MAP.get(selected_effect_key)

            if not EffectFormClass:
                logger.error(f"Unknown effect key selected: {selected_effect_key}")
                error_msg = f"Unknown effect: {selected_effect_key}"
                if is_ajax:
                    return jsonify({"success": False, "error": error_msg}), 400
                else:
                    # Re-render with main form, no specific form, and error
                    return render_template('index.html', form=form, effect_specific_form=None, selected_effect_key=selected_effect_key, error=error_msg)

            # Instantiate the specific form. Flask-WTF will populate from request.form and request.files.
            effect_specific_form = EffectFormClass()
            logger.debug(f"Instantiated specific form {EffectFormClass.__name__} for effect '{selected_effect_key}'")

            if effect_specific_form.validate():
                logger.debug(f"Specific form {EffectFormClass.__name__} validated successfully.")
                try:
                    primary_image_file = form.primary_image.data # From main form
                    filename = secure_filename(primary_image_file.filename)
                    primary_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    primary_image_file.save(primary_path)
                    logger.debug(f"Primary image saved to {primary_path}")
                    
                    original_image_pil = Image.open(primary_path)
                    original_size = original_image_pil.size
                    image = load_image(
                        primary_path,
                        max_width_config=app.config.get('MAX_IMAGE_WIDTH'),
                        max_height_config=app.config.get('MAX_IMAGE_HEIGHT')
                    )
                    
                    if image is None:
                        logger.error(f"Failed to load image from {primary_path}")
                        error_msg = "Error loading image"
                        # Pass back main form and the specific form that failed (or was attempted)
                        return jsonify({"success": False, "error": error_msg}) if is_ajax else render_template('index.html', form=form, effect_specific_form=effect_specific_form, selected_effect_key=selected_effect_key, error=error_msg), 400
                    
                    was_resized = original_size != image.size
                    if was_resized:
                        logger.info(f"Image resized from {original_size} to {image.size}")
                    
                    # Use selected_effect_key for dispatch
                    effect = selected_effect_key 
                    processed_image = None
                    settings = ""
                    
                    # --- EFFECT DISPATCH LOGIC ---
                    if effect == 'pixel_sort_chunk':
                        chunk_width = effect_specific_form.width.data
                        chunk_height = effect_specific_form.height.data
                        chunk_size_str = f"{chunk_width}x{chunk_height}"
                        sort_by = effect_specific_form.sort_by.data
                        sort_mode = effect_specific_form.sort_mode.data
                        reverse_sort = effect_specific_form.reverse_sort.data
                        sort_order_str = 'desc' if reverse_sort else 'asc'
                        
                        if sort_mode == 'diagonal':
                            starting_corner = effect_specific_form.starting_corner.data
                            logger.debug(f"Pixel Sort Chunk (Diagonal) params: chunk_size={chunk_size_str}, sort_by={sort_by}, starting_corner={starting_corner}, sort_order={sort_order_str}")
                            processed_image = pixel_sorting(image, sort_mode, chunk_size_str, sort_by, starting_corner=starting_corner, sort_order=sort_order_str)
                            settings = f"chunk_diag_{chunk_width}x{chunk_height}_{sort_by}_{starting_corner}_{sort_order_str}"
                        else:
                            logger.debug(f"Pixel Sort Chunk (Horiz/Vert) params: mode={sort_mode}, chunk_size={chunk_size_str}, sort_by={sort_by}, sort_order={sort_order_str}")
                            processed_image = pixel_sorting(image, sort_mode, chunk_size_str, sort_by, sort_order=sort_order_str)
                            settings = f"chunk_{sort_mode}_{chunk_width}x{chunk_height}_{sort_by}_{sort_order_str}"
                    
                    elif effect == 'color_channel':
                        manipulation_type = effect_specific_form.manipulation_type.data
                        logger.debug(f"Color channel manipulation type: {manipulation_type}")
                        params = [image, manipulation_type]
                        setting_suffix = ""
                        if manipulation_type == 'swap':
                            choice = effect_specific_form.swap_choice.data
                            params.append(choice)
                            setting_suffix = f"swap_{choice}"
                        elif manipulation_type == 'invert':
                            choice = effect_specific_form.invert_choice.data
                            params.append(choice)
                            setting_suffix = f"invert_{choice}"
                        elif manipulation_type == 'negative':
                            params.append(None) # No choice needed
                            setting_suffix = "negative"
                        else:  # adjust
                            choice = effect_specific_form.adjust_choice.data
                            factor = effect_specific_form.intensity_factor.data
                            params.extend([choice, factor])
                            setting_suffix = f"adjust_{choice}_{factor}"
                        processed_image = color_channel_manipulation(*params)
                        settings = setting_suffix

                    elif effect == 'double_expose':
                        secondary_image_file = effect_specific_form.secondary_image.data # From specific form
                        blend_mode = effect_specific_form.blend_mode.data
                        opacity = effect_specific_form.opacity.data
                        
                        if not secondary_image_file: # Should be caught by form validation if FileRequired
                            logger.error("Secondary image required but not provided (post-validation check)")
                            error_msg = "Secondary image is required for Double Expose."
                            return jsonify({"success": False, "error": error_msg}) if is_ajax else render_template('index.html', form=form, effect_specific_form=effect_specific_form, selected_effect_key=selected_effect_key, error=error_msg), 400

                        secondary_filename = secure_filename(secondary_image_file.filename)
                        secondary_path = os.path.join(app.config['UPLOAD_FOLDER'], f"secondary_{secondary_filename}")
                        secondary_image_file.save(secondary_path)
                        secondary_img_pil = load_image(
                            secondary_path,
                            max_width_config=app.config.get('MAX_IMAGE_WIDTH'),
                            max_height_config=app.config.get('MAX_IMAGE_HEIGHT')
                        )
                        
                        if secondary_img_pil is None:
                            logger.error(f"Failed to load secondary image from {secondary_path}")
                            error_msg = "Error loading secondary image for Double Expose."
                            return jsonify({"success": False, "error": error_msg}) if is_ajax else render_template('index.html', form=form, effect_specific_form=effect_specific_form, selected_effect_key=selected_effect_key, error=error_msg), 400
                            
                        logger.debug(f"Double expose params: blend_mode={blend_mode}, opacity={opacity}")
                        processed_image = double_expose(image, secondary_img_pil, blend_mode, opacity)
                        settings = f"doubleexpose_{blend_mode}_{opacity}"

                    elif effect == 'pixel_drift':
                        direction = effect_specific_form.direction.data
                        bands = effect_specific_form.bands.data
                        intensity = effect_specific_form.intensity.data
                        logger.debug(f"Pixel drift params: direction={direction}, bands={bands}, intensity={intensity}")
                        processed_image = pixel_drift(image, direction, bands, intensity)
                        settings = f"drift_{direction}_{bands}_{intensity}"

                    elif effect == 'bit_manipulation':
                        logger.debug("Applying bit manipulation")
                        chunk_size = effect_specific_form.chunk_size.data
                        offset_val = effect_specific_form.offset.data # Renamed from 'offset' to 'offset_val' to avoid conflict
                        xor_value = effect_specific_form.xor_value.data
                        skip_pattern = effect_specific_form.skip_pattern.data
                        manipulation_type = effect_specific_form.manipulation_type.data
                        shift_amount = effect_specific_form.shift_amount.data
                        randomize = effect_specific_form.randomize_effect.data
                        seed_val = effect_specific_form.seed.data # Renamed from 'seed'
                        
                        logger.debug(f"Bit manipulation params: chunk_size={chunk_size}, offset={offset_val}, " 
                                    f"xor_value={xor_value}, skip_pattern={skip_pattern}, "
                                    f"manipulation_type={manipulation_type}, bit_shift={shift_amount}, "
                                    f"randomize={randomize}, random_seed={seed_val}")
                        
                        processed_image = bit_manipulation(
                            image, chunk_size=chunk_size, offset=offset_val, xor_value=xor_value,
                            skip_pattern=skip_pattern, manipulation_type=manipulation_type,
                            bit_shift=shift_amount, randomize=randomize, random_seed=seed_val
                        )
                        settings = f"bitmanip_{chunk_size}_{offset_val}_{xor_value}_{skip_pattern}_{manipulation_type}"
                        if shift_amount != 1: settings += f"_{shift_amount}"
                        if randomize: settings += "_random"
                        if seed_val is not None: settings += f"_s{seed_val}"

                    elif effect == 'data_mosh_blocks':
                        operations = effect_specific_form.operations.data
                        block_size = effect_specific_form.block_size.data
                        movement = effect_specific_form.movement.data
                        color_swap = effect_specific_form.color_swap.data
                        invert_colors = effect_specific_form.invert_colors.data
                        shift_values = effect_specific_form.shift_values.data
                        flip_blocks = effect_specific_form.flip_blocks.data
                        seed_val = effect_specific_form.seed.data # Renamed from 'seed'
                        
                        logger.debug(f"Data mosh blocks params: operations={operations}, block_size={block_size}, "
                                    f"movement={movement}, color_swap={color_swap}, invert={invert_colors}, "
                                    f"shift={shift_values}, flip={flip_blocks}, seed={seed_val}")
                        processed_image = data_mosh_blocks(
                            image, operations, block_size, movement, color_swap, 
                            invert_colors, shift_values, flip_blocks, seed_val
                        )
                        settings = f"datamosh_{operations}_{block_size}_{movement}_{color_swap}_{invert_colors}_{shift_values}_{flip_blocks}"
                        if seed_val is not None: settings += f"_s{seed_val}"

                    elif effect == 'full_frame_sort':
                        direction = effect_specific_form.direction.data
                        sort_by = effect_specific_form.sort_by.data
                        reverse = effect_specific_form.reverse_sort.data
                        logger.debug(f"Full frame sort params: direction={direction}, sort_by={sort_by}, reverse={reverse}")
                        processed_image = full_frame_sort(image, direction, sort_by, reverse)
                        settings = f"fullframe_{direction}_{sort_by}_{'desc' if reverse else 'asc'}"

                    elif effect == 'polar_sort':
                        chunk_size = effect_specific_form.chunk_size.data
                        sort_by = effect_specific_form.sort_by.data
                        reverse = effect_specific_form.reverse_sort.data
                        logger.debug(f"Polar sort params: chunk_size={chunk_size}, sort_by={sort_by}, reverse={reverse}")
                        processed_image = polar_sorting(image, chunk_size, sort_by, reverse)
                        settings = f"polar_{chunk_size}_{sort_by}_{'desc' if reverse else 'asc'}"

                    elif effect == 'perlin_noise_sort':
                        chunk_width = effect_specific_form.chunk_width.data
                        chunk_height = effect_specific_form.chunk_height.data
                        chunk_size_str = f"{chunk_width}x{chunk_height}"
                        noise_scale = effect_specific_form.noise_scale.data
                        direction = effect_specific_form.direction.data
                        reverse = effect_specific_form.reverse_sort.data
                        seed_val = effect_specific_form.seed.data # Renamed from 'seed'
                        logger.debug(f"Perlin noise sort params: chunk_size={chunk_size_str}, noise_scale={noise_scale}, direction={direction}, reverse={reverse}, seed={seed_val}")
                        processed_image = perlin_noise_sorting(image, chunk_size_str, noise_scale, direction, reverse, seed_val)
                        settings = f"perlin_{chunk_width}x{chunk_height}_{noise_scale}_{direction}_{'desc' if reverse else 'asc'}"
                        if seed_val is not None: settings += f"_s{seed_val}"
                        
                    elif effect == 'perlin_full_frame':
                        noise_scale = effect_specific_form.noise_scale.data
                        sort_by = effect_specific_form.sort_by.data
                        reverse = effect_specific_form.reverse_sort.data
                        seed_val = effect_specific_form.seed.data # Renamed from 'seed'
                        pattern_width = effect_specific_form.pattern_width.data
                        logger.debug(f"Perlin full frame params: noise_scale={noise_scale}, sort_by={sort_by}, reverse={reverse}, seed={seed_val}, pattern_width={pattern_width}")
                        processed_image = perlin_full_frame_sort(image, noise_scale, sort_by, reverse, seed_val, pattern_width)
                        settings = f"perlinff_{noise_scale}_{sort_by}_{'desc' if reverse else 'asc'}_w{pattern_width}"
                        if seed_val is not None: settings += f"_s{seed_val}"

                    elif effect == 'spiral_sort_2':
                        chunk_size = effect_specific_form.chunk_size.data
                        sort_by = effect_specific_form.sort_by.data
                        reverse = effect_specific_form.reverse_sort.data
                        logger.debug(f"Spiral sort 2 params: chunk_size={chunk_size}, sort_by={sort_by}, reverse={reverse}")
                        processed_image = spiral_sort_2(image, chunk_size, sort_by, reverse)
                        settings = f"spiral2_{chunk_size}_{sort_by}_{'desc' if reverse else 'asc'}"

                    elif effect == 'pixelate':
                        pixel_width = effect_specific_form.width.data
                        pixel_height = effect_specific_form.height.data
                        attribute = effect_specific_form.attribute.data
                        num_bins = effect_specific_form.bins.data
                        logger.debug(f"Pixelate params: width={pixel_width}, height={pixel_height}, attribute={attribute}, bins={num_bins}")
                        processed_image = pixelate_by_attribute(image, pixel_width, pixel_height, attribute, num_bins)
                        settings = f"pixelate_{pixel_width}x{pixel_height}_{attribute}_{num_bins}"

                    elif effect == 'concentric_shapes':
                        num_points = effect_specific_form.num_points.data
                        shape_type = effect_specific_form.shape_type.data
                        thickness = effect_specific_form.thickness.data
                        spacing_val = effect_specific_form.spacing.data # Renamed
                        rotation_angle = effect_specific_form.rotation_angle.data
                        darken_step = effect_specific_form.darken_step.data
                        color_shift_amount = effect_specific_form.color_shift_amount.data
                        seed_val = effect_specific_form.seed.data # Renamed
                        logger.debug(f"Concentric shapes params: points={num_points}, shape_type={shape_type}, thickness={thickness}, spacing={spacing_val}, rotation={rotation_angle}, darken={darken_step}, color_shift={color_shift_amount}, seed={seed_val}")
                        # Assuming concentric_shapes function is updated to accept seed
                        processed_image = concentric_shapes(image, num_points, shape_type, thickness, spacing_val, rotation_angle, darken_step, color_shift=color_shift_amount, seed=seed_val)
                        settings = f"concentric_{shape_type}_{num_points}_{thickness}_{spacing_val}_{rotation_angle}_{darken_step}_{color_shift_amount}"
                        if seed_val is not None: settings += f"_s{seed_val}"

                    elif effect == 'color_shift_expansion':
                        num_points = effect_specific_form.num_points.data
                        shift_amount = effect_specific_form.shift_amount.data
                        expansion_type = effect_specific_form.expansion_type.data
                        pattern_type = effect_specific_form.pattern_type.data
                        color_theme = effect_specific_form.color_theme.data
                        saturation_boost = effect_specific_form.saturation_boost.data
                        value_boost = effect_specific_form.value_boost.data
                        decay_factor = effect_specific_form.decay_factor.data
                        seed_val = effect_specific_form.seed.data # Renamed
                        
                        logger.debug(f"Color shift expansion params: num_points={num_points}, shift_amount={shift_amount}, expansion_type={expansion_type}, pattern_type={pattern_type}, color_theme={color_theme}, saturation_boost={saturation_boost}, value_boost={value_boost}, decay_factor={decay_factor}, seed={seed_val}")
                        mode = 'xtreme' # Mode is hardcoded as per previous logic
                        processed_image = color_shift_expansion(
                            image=image, num_points=num_points, shift_amount=shift_amount, expansion_type=expansion_type,
                            mode=mode, saturation_boost=saturation_boost, value_boost=value_boost, pattern_type=pattern_type,
                            color_theme=color_theme, decay_factor=decay_factor, seed=seed_val
                        )
                        settings = f"colorshiftexp_{num_points}_{shift_amount}_{expansion_type}_{pattern_type}_{color_theme}"
                        if seed_val is not None: settings += f"_s{seed_val}"

                    elif effect == 'perlin_displacement':
                        scale = effect_specific_form.scale.data
                        intensity = effect_specific_form.intensity.data
                        octaves = effect_specific_form.octaves.data
                        seed_val = effect_specific_form.seed.data # Renamed
                        logger.debug(f"Perlin displacement params: scale={scale}, intensity={intensity}, octaves={octaves}, seed={seed_val}")
                        processed_image = perlin_noise_displacement(image, scale, intensity, octaves, seed=seed_val)
                        settings = f"perlindisp_{scale}_{intensity}_{octaves}"
                        if seed_val is not None: settings += f"_s{seed_val}"

                    elif effect == 'voronoi_sort':
                        num_cells = effect_specific_form.num_cells.data
                        size_variation = effect_specific_form.size_variation.data
                        sort_by = effect_specific_form.sort_by.data
                        sort_order = effect_specific_form.sort_order.data
                        orientation_val = effect_specific_form.orientation.data # Renamed
                        start_position = effect_specific_form.start_position.data
                        seed_val = effect_specific_form.seed.data # Renamed
                        
                        logger.debug(f"Voronoi sort params: num_cells={num_cells}, size_variation={size_variation}, sort_by={sort_by}, sort_order={sort_order}, orientation={orientation_val}, start_position={start_position}, seed={seed_val}")
                        processed_image = voronoi_pixel_sort(
                            image, num_cells, size_variation, sort_by, sort_order, seed_val, 
                            orientation_val, start_position
                        )
                        settings = f"voronoi_{num_cells}_{size_variation}_{sort_by}_{sort_order}_{orientation_val}_{start_position}"
                        if seed_val is not None: settings += f"_s{seed_val}"

                    elif effect == 'channel_shift': # Uses split_and_shift_channels
                        shift_amount = effect_specific_form.shift_amount.data
                        direction = effect_specific_form.direction.data
                        centered_channel = effect_specific_form.center_channel.data
                        mode = effect_specific_form.mode.data
                        
                        if mode == 'mirror':
                            logger.debug(f"Channel shift (mirror): centered_channel={centered_channel}")
                            processed_image = split_and_shift_channels(image, 0, 'horizontal', centered_channel, mode) # amount/direction ignored
                            settings = f"channelshift_{mode}_{centered_channel}"
                        else: # shift mode
                            logger.debug(f"Channel shift (shift): amount={shift_amount}, direction={direction}, centered_channel={centered_channel}")
                            processed_image = split_and_shift_channels(image, shift_amount, direction, centered_channel, mode)
                            settings = f"channelshift_{mode}_{shift_amount}_{direction}_{centered_channel}"
                            
                    elif effect == 'jpeg_artifacts':
                        intensity = effect_specific_form.intensity.data
                        logger.debug(f"JPEG artifacts params: intensity={intensity}")
                        processed_image = simulate_jpeg_artifacts(image, intensity)
                        settings = f"jpegart_{intensity}"

                    elif effect == 'pixel_scatter':
                        direction = effect_specific_form.direction.data
                        select_by = effect_specific_form.select_by.data
                        min_val = effect_specific_form.min_value.data
                        max_val = effect_specific_form.max_value.data
                        logger.debug(f"Pixel scatter params: direction={direction}, select_by={select_by}, min_val={min_val}, max_val={max_val}")
                        processed_image = pixel_scatter(image, direction, select_by, min_val, max_val)
                        settings = f"scatter_{direction}_{select_by}_{min_val}_{max_val}"

                    elif effect == 'databend':
                        intensity = effect_specific_form.intensity.data
                        preserve_header = effect_specific_form.preserve_header.data # Boolean now
                        seed_val = effect_specific_form.seed.data # Renamed
                        logger.debug(f"Databending params: intensity={intensity}, preserve_header={preserve_header}, seed={seed_val}")
                        processed_image = databend_image(image, intensity, preserve_header, seed_val)
                        settings = f"databend_{intensity}_{preserve_header}"
                        if seed_val is not None: settings += f"_s{seed_val}"
                        
                    elif effect == 'histogram_glitch':
                        r_mode = effect_specific_form.r_mode.data
                        g_mode = effect_specific_form.g_mode.data
                        b_mode = effect_specific_form.b_mode.data
                        r_freq = effect_specific_form.r_freq.data
                        r_phase = effect_specific_form.r_phase.data
                        g_freq = effect_specific_form.g_freq.data
                        g_phase = effect_specific_form.g_phase.data
                        b_freq = effect_specific_form.b_freq.data
                        b_phase = effect_specific_form.b_phase.data
                        gamma_val = effect_specific_form.gamma_value.data
                        
                        logger.debug(f"Hist. glitch params: modes=({r_mode},{g_mode},{b_mode}), freqs=({r_freq},{g_freq},{b_freq}), phases=({r_phase},{g_phase},{b_phase}), gamma={gamma_val}")
                        processed_image = histogram_glitch(
                            image, r_mode, g_mode, b_mode, r_freq, r_phase, g_freq, g_phase, b_freq, b_phase, gamma_val
                        )
                        settings = f"histglitch_{r_mode}_{g_mode}_{b_mode}" # Simplified settings string

                    elif effect == 'ripple':
                        num_droplets = effect_specific_form.num_droplets.data
                        amplitude = effect_specific_form.amplitude.data
                        frequency_val = effect_specific_form.frequency.data # Renamed
                        decay = effect_specific_form.decay.data
                        distortion_type = effect_specific_form.distortion_type.data
                        seed_val = effect_specific_form.seed.data # Renamed
                        
                        distortion_params = {}
                        if distortion_type == 'color_shift':
                            distortion_params = {
                                'factor_r': effect_specific_form.color_r_factor.data,
                                'factor_g': effect_specific_form.color_g_factor.data,
                                'factor_b': effect_specific_form.color_b_factor.data
                            }
                        elif distortion_type == 'pixelation':
                            distortion_params = {
                                'scale': effect_specific_form.pixelation_scale.data,
                                'max_mag': effect_specific_form.pixelation_magnitude.data
                            }
                        
                        logger.debug(f"Ripple params: droplets={num_droplets}, amp={amplitude}, freq={frequency_val}, decay={decay}, dist_type={distortion_type}, dist_params={distortion_params}, seed={seed_val}")
                        processed_image = ripple_effect(
                            image, num_droplets, amplitude, frequency_val, decay, distortion_type, distortion_params, seed=seed_val
                        )
                        settings = f"ripple_{num_droplets}_{amplitude}_{frequency_val}_{decay}_{distortion_type}"
                        if seed_val is not None: settings += f"_s{seed_val}"

                    elif effect == 'masked_merge':
                        secondary_image_file = effect_specific_form.secondary_image.data # From mixin
                        mask_type = effect_specific_form.mask_type.data
                        seed_val = effect_specific_form.seed.data # From mixin, Renamed
                        
                        if not secondary_image_file: # Should be caught by FileRequired
                            # Handle error
                            pass 

                        secondary_filename = secure_filename(secondary_image_file.filename)
                        secondary_path = os.path.join(app.config['UPLOAD_FOLDER'], f"secondary_mm_{secondary_filename}")
                        secondary_image_file.save(secondary_path)
                        secondary_img_pil = load_image(
                            secondary_path,
                            max_width_config=app.config.get('MAX_IMAGE_WIDTH'),
                            max_height_config=app.config.get('MAX_IMAGE_HEIGHT')
                        )

                        if secondary_img_pil is None:
                            # Handle error
                            pass

                        # Determine active width based on mask_type
                        active_width = effect_specific_form.mask_width.data
                        if mask_type == 'concentric_rectangles':
                            active_width = effect_specific_form.rectangle_band_width.data
                        elif mask_type == 'concentric_circles':
                            active_width = effect_specific_form.circle_band_width.data
                        
                        logger.debug(f"Masked Merge: type={mask_type}, active_width={active_width}, seed={seed_val}")
                        
                        processed_image = masked_merge(
                            image, secondary_img_pil, mask_type,
                            width=active_width, # Use determined width
                            height=effect_specific_form.mask_height.data,
                            random_seed=seed_val,
                            stripe_width=effect_specific_form.stripe_width.data,
                            stripe_angle=effect_specific_form.stripe_angle.data,
                            gradient_direction=effect_specific_form.gradient_direction.data,
                            perlin_noise_scale=effect_specific_form.perlin_noise_scale.data,
                            threshold=effect_specific_form.perlin_threshold.data,
                            perlin_octaves=effect_specific_form.perlin_octaves.data,
                            voronoi_cells=effect_specific_form.voronoi_num_cells.data,
                            circle_origin=effect_specific_form.circle_origin.data,
                            triangle_size=effect_specific_form.triangle_size.data
                        )
                        # Simplified settings string, can be expanded
                        settings = f"maskedmerge_{mask_type}_w{active_width}"
                        if seed_val is not None: settings += f"_s{seed_val}"

                    elif effect == 'offset':
                        raw_offset_x = effect_specific_form.x_value.data
                        unit_x = effect_specific_form.x_unit.data
                        raw_offset_y = effect_specific_form.y_value.data
                        unit_y = effect_specific_form.y_unit.data
                        
                        offset_x = raw_offset_x if raw_offset_x is not None else 0.0
                        offset_y = raw_offset_y if raw_offset_y is not None else 0.0
                        
                        logger.debug(f"Offset params: x={offset_x}{unit_x}, y={offset_y}{unit_y}")
                        processed_image = offset_effect(image, offset_x=offset_x, offset_y=offset_y, unit_x=unit_x, unit_y=unit_y)
                        settings = f"offset_{offset_x}{unit_x}_{offset_y}{unit_y}"

                    elif effect == 'slice_shuffle':
                        count = effect_specific_form.count.data
                        orientation_val = effect_specific_form.orientation.data # Renamed
                        seed_val = effect_specific_form.seed.data # Renamed
                        logger.debug(f"Slice Shuffle: count={count}, orientation={orientation_val}, seed={seed_val}")
                        processed_image = slice_shuffle(image, count, orientation_val, seed_val)
                        settings = f"sliceshuffle_{count}_{orientation_val}"
                        if seed_val is not None: settings += f"_s{seed_val}"

                    elif effect == 'slice_offset':
                        count = effect_specific_form.count.data
                        max_offset_val = effect_specific_form.max_offset.data # Renamed
                        orientation_val = effect_specific_form.orientation.data # Renamed
                        offset_mode = effect_specific_form.offset_mode.data
                        sine_frequency = effect_specific_form.frequency.data if offset_mode == 'sine' else None
                        seed_val = effect_specific_form.seed.data if offset_mode == 'random' else None # Renamed
                        
                        logger.debug(f"Slice Offset: count={count}, max_offset={max_offset_val}, orientation={orientation_val}, mode={offset_mode}, freq={sine_frequency}, seed={seed_val}")
                        processed_image = slice_offset(
                            image, count, max_offset_val, orientation_val, 
                            offset_mode=offset_mode, sine_frequency=sine_frequency, seed=seed_val
                        )
                        settings = f"sliceoffset_{count}_{max_offset_val}_{orientation_val}_{offset_mode}"
                        if offset_mode == 'sine' and sine_frequency: settings += f"_freq{sine_frequency}"
                        elif offset_mode == 'random' and seed_val is not None: settings += f"_s{seed_val}"

                    elif effect == 'slice_reduction':
                        count = effect_specific_form.count.data
                        reduction_value = effect_specific_form.reduction_value.data
                        orientation_val = effect_specific_form.orientation.data # Renamed
                        logger.debug(f"Slice Reduction: count={count}, reduction={reduction_value}, orientation={orientation_val}")
                        processed_image = slice_reduction(image, count, reduction_value, orientation_val)
                        settings = f"slicereduct_{count}_{reduction_value}_{orientation_val}"

                    elif effect == 'posterize':
                        levels = effect_specific_form.levels.data
                        logger.debug(f"Posterize: levels={levels}")
                        processed_image = posterize(image, levels)
                        settings = f"posterize_{levels}"

                    elif effect == 'curved_hue_shift':
                        curve_value = effect_specific_form.curve_value.data
                        shift_amount = effect_specific_form.shift_amount.data
                        logger.debug(f"Curved Hue Shift: curve={curve_value}, amount={shift_amount}")
                        processed_image = curved_hue_shift(image, curve_value, shift_amount)
                        settings = f"hueskift_C{curve_value}_A{shift_amount}"
                        
                    elif effect == 'contour':
                        num_levels = effect_specific_form.num_levels.data
                        noise_std = effect_specific_form.noise_std.data
                        smooth_sigma = effect_specific_form.smooth_sigma.data
                        line_thickness = effect_specific_form.line_thickness.data
                        grad_threshold = effect_specific_form.grad_threshold.data
                        min_distance = effect_specific_form.min_distance.data
                        max_line_length = effect_specific_form.max_line_length.data
                        blur_kernel_size = effect_specific_form.blur_kernel_size.data
                        sobel_kernel_size = effect_specific_form.sobel_kernel_size.data
                        seed_val = effect_specific_form.seed.data # Renamed
                        
                        logger.debug(f"Contour params: levels={num_levels}, noise={noise_std}, smooth={smooth_sigma}, thickness={line_thickness}, grad_thresh={grad_threshold}, min_dist={min_distance}, max_line={max_line_length}, blur_kernel={blur_kernel_size}, sobel_kernel={sobel_kernel_size}, seed={seed_val}")
                        processed_image = contour_effect(
                            image, num_levels=num_levels, noise_std=noise_std, smooth_sigma=smooth_sigma,
                            line_thickness=line_thickness, grad_threshold=grad_threshold,
                            min_distance=min_distance, max_line_length=max_line_length,
                            blur_kernel_size=blur_kernel_size, sobel_kernel_size=sobel_kernel_size, seed=seed_val
                        )
                        settings = f"contour_{num_levels}_{noise_std}_{smooth_sigma}_{line_thickness}_{grad_threshold}_{min_distance}_{max_line_length}_{blur_kernel_size}_{sobel_kernel_size}"
                        if seed_val is not None: settings += f"_s{seed_val}"

                    elif effect == 'block_shuffle':
                        block_width = effect_specific_form.block_width.data
                        block_height = effect_specific_form.block_height.data
                        seed_val = effect_specific_form.seed.data # Renamed
                        logger.debug(f"Block Shuffle params: width={block_width}, height={block_height}, seed={seed_val}")
                        processed_image = block_shuffle(image, block_width, block_height, seed_val)
                        settings = f"blockshuffle_{block_width}x{block_height}"
                        if seed_val is not None: settings += f"_s{seed_val}"
                        
                    else: # Should not be reached if EffectFormClass was found
                        logger.error(f"Effect '{effect}' has a form but no dispatch logic.")
                        error_msg = f"Processing logic for effect '{effect}' not implemented."
                        return jsonify({"success": False, "error": error_msg}) if is_ajax else render_template('index.html', form=form, effect_specific_form=effect_specific_form, selected_effect_key=selected_effect_key, error=error_msg), 500

                    # --- END OF EFFECT DISPATCH LOGIC ---

                    if processed_image is None:
                        logger.error(f"Effect function for '{effect}' did not return an image.")
                        error_msg = f"Error applying effect '{effect}'. Result was empty."
                        return jsonify({"success": False, "error": error_msg}) if is_ajax else render_template('index.html', form=form, effect_specific_form=effect_specific_form, selected_effect_key=selected_effect_key, error=error_msg), 500

                    output_filename = generate_output_filename(filename, effect, settings)
                    output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
                    processed_image.save(output_path)
                    logger.debug(f"Processed image saved to {output_path}")
                    
                    processed_url = url_for('processed_file', filename=output_filename)
                    
                    if is_ajax:
                        response_data = {"success": True, "processed_url": processed_url}
                        if was_resized:
                            response_data["was_resized"] = True
                            response_data["original_size"] = f"{original_size[0]}x{original_size[1]}"
                            response_data["new_size"] = f"{image.size[0]}x{image.size[1]}"
                        return jsonify(response_data)
                    else:
                        return redirect(processed_url)

                except Exception as e:
                    tb_str = traceback.format_exc()
                    logger.error(f"Error processing image for effect '{selected_effect_key}': {str(e)}")
                    logger.error(tb_str) # Log it as before
                    error_msg_user = f"An unexpected error occurred: {str(e)}"
                    
                    if is_ajax:
                        # For debugging, include traceback in AJAX response
                        return jsonify({"success": False, "error": error_msg_user, "traceback": tb_str}), 500
                    else:
                        # For non-AJAX, pass a simplified error
                        return render_template('index.html', form=form, effect_specific_form=effect_specific_form, selected_effect_key=selected_effect_key, error=error_msg_user)
            else: # Specific form validation failed
                logger.error(f"Specific form validation failed for '{selected_effect_key}': {effect_specific_form.errors}")
                if is_ajax:
                    # Consolidate errors from specific form
                    form_errors = {f: effect_specific_form.errors.get(f) for f in effect_specific_form.errors}
                    return jsonify({"success": False, "error": "Form validation failed", "form_errors": form_errors}), 400
                else:
                    # Re-render with main form, specific form (with errors), and selected effect
                    return render_template('index.html', form=form, effect_specific_form=effect_specific_form, selected_effect_key=selected_effect_key)
        else: # Main form validation failed
            logger.error(f"Main form validation failed: {form.errors}")
            if is_ajax:
                # Consolidate errors from main form
                form_errors = {f: form.errors.get(f) for f in form.errors}
                return jsonify({"success": False, "error": "Main form validation failed", "form_errors": form_errors}), 400
            else:
                # Re-render with main form (with errors), no specific form unless pre-selected from GET
                # If there was a selected_effect_key from GET, try to instantiate its form for display
                if selected_effect_key and selected_effect_key in EFFECT_FORM_MAP and not effect_specific_form:
                     EffectFormClass = EFFECT_FORM_MAP[selected_effect_key]
                     effect_specific_form = EffectFormClass() # For display
                return render_template('index.html', form=form, effect_specific_form=effect_specific_form, selected_effect_key=selected_effect_key)
    
    # For GET requests, or if POST failed and re-rendering
    return render_template('index.html', form=form, effect_specific_form=effect_specific_form, selected_effect_key=selected_effect_key)

@app.route('/get-effect-form/<string:effect_key>', methods=['GET'])
@csrf_exempt # Exempt CSRF for this GET request if needed, or handle appropriately
def get_effect_form_html(effect_key):
    """Return HTML for the fields of a specific effect form."""
    logger.debug(f"Request to get form HTML for effect: {effect_key}")
    EffectFormClass = EFFECT_FORM_MAP.get(effect_key)
    effect_specific_form = None
    form = ImageProcessForm() # Needed for the label in the partial
    form.effect.data = effect_key # Set the main form's effect to get the correct label

    if EffectFormClass:
        effect_specific_form = EffectFormClass() # Instantiate the specific form
        # Render a partial template containing only the fields for this form
        html = render_template('_effect_form_fields.html', 
                               effect_specific_form=effect_specific_form, 
                               selected_effect_key=effect_key,
                               form=form) # Pass main form for label context
        return jsonify({"success": True, "html": html})
    else:
        logger.warning(f"No form class found for effect key: {effect_key} in /get-effect-form")
        return jsonify({"success": False, "error": "Invalid effect selected"}), 404

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded original images."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    """Serve processed images."""
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/debug')
def debug_info():
    """Return debug information about the application."""
    debug_info = {
        'csrf_enabled': app.config['WTF_CSRF_ENABLED'],
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'processed_folder': app.config['PROCESSED_FOLDER'],
        'secret_key_set': bool(app.secret_key),
        'routes': [str(rule) for rule in app.url_map.iter_rules()]
    }
    return jsonify(debug_info)

@app.route('/check-form', methods=['POST'])
def check_form():
    """Check form submission details."""
    logger.debug("Check form endpoint called")
    logger.debug(f"Request method: {request.method}")
    logger.debug(f"Request headers: {request.headers}")
    logger.debug(f"Form data: {request.form}")
    logger.debug(f"Files: {request.files}")
    
    return jsonify({
        'success': True,
        'message': 'Form submission received',
        'headers': {k: v for k, v in request.headers.items()},
        'form_data': {k: v for k, v in request.form.items()},
        'files': [f for f in request.files.keys()]
    })

@app.route('/test-form')
def test_form():
    """A simple test form to verify that form submission is working correctly."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Form</title>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }
            form { background-color: #f5f5f5; padding: 20px; border-radius: 5px; }
            input[type="submit"] { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; }
            #debug { margin-top: 20px; padding: 10px; background-color: #f0f0f0; border-radius: 5px; display: none; }
        </style>
    </head>
    <body>
        <h1>Test Form</h1>
        <p>This is a simple test form to verify that form submission is working correctly.</p>
        <form id="test-form" method="POST" action="/check-form" enctype="multipart/form-data">
            <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
            <div>
                <label for="test_name">Name:</label>
                <input type="text" id="test_name" name="test_name" required>
            </div>
            <div style="margin-top: 10px;">
                <label for="test_file">File (optional):</label>
                <input type="file" id="test_file" name="test_file">
            </div>
            <div style="margin-top: 20px;">
                <input type="submit" value="Submit Test Form">
            </div>
        </form>
        <div id="debug"></div>
        <script>
            $(document).ready(function() {
                $('#test-form').on('submit', function(e) {
                    e.preventDefault();
                    console.log('Form submitted');
                    
                    var formData = new FormData(this);
                    
                    $.ajax({
                        url: '/check-form',
                        type: 'POST',
                        data: formData,
                        dataType: 'json',
                        headers: {
                            'X-Requested-With': 'XMLHttpRequest'
                        },
                        success: function(data) {
                            console.log('Success:', data);
                            $('#debug').show().html('<pre>' + JSON.stringify(data, null, 2) + '</pre>');
                        },
                        error: function(xhr, status, error) {
                            console.error('Error:', error);
                            $('#debug').show().html('<p>Error: ' + error + '</p><pre>' + xhr.responseText + '</pre>');
                        },
                        cache: false,
                        contentType: false,
                        processData: false
                    });
                });
            });
        </script>
    </body>
    </html>
    '''

if __name__ == '__main__':
    # Ensure upload and processed directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    
    # Log important configuration info
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"Processed folder: {app.config['PROCESSED_FOLDER']}")
    logger.info(f"Running on Heroku: {is_heroku}")
    
    # Get port from environment variable for Heroku compatibility
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)