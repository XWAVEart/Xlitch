from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import os
import traceback
import logging
import numpy as np
from forms import ImageProcessForm
from glitch_art import (
    load_image, pixel_sorting, color_channel_manipulation, double_expose, 
    pixel_drift, bit_manipulation, generate_output_filename, 
    full_frame_sort, spiral_sort_2, polar_sorting, perlin_noise_sorting, 
    perlin_full_frame_sort, pixelate_by_mode, concentric_shapes, 
    color_shift_expansion, perlin_noise_displacement, data_mosh_blocks, 
    voronoi_pixel_sort, split_and_shift_channels, simulate_jpeg_artifacts, 
    pixel_scatter, databend_image, histogram_glitch, resize_image_if_needed, 
    Ripple, masked_merge
)
from flask_wtf.csrf import CSRFProtect
import random
import secrets
from PIL import Image

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
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PROCESSED_FOLDER'] = 'processed/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['WTF_CSRF_ENABLED'] = True  # Enable CSRF protection
app.config['WTF_CSRF_TIME_LIMIT'] = 3600  # Extend CSRF token validity to 1 hour
app.config['WTF_CSRF_CHECK_DEFAULT'] = False  # We'll check it manually for AJAX
app.config['MAX_IMAGE_WIDTH'] = 1920  # Maximum image width before resizing
app.config['MAX_IMAGE_HEIGHT'] = 1920  # Maximum image height before resizing - set to same as width for longest dimension resize

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    
    # Check if this is an AJAX request
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    logger.debug(f"Request method: {request.method}, AJAX: {is_ajax}")
    logger.debug(f"Request headers: {request.headers}")
    
    if request.method == 'POST':
        logger.debug(f"Form data: {request.form}")
        logger.debug(f"Files: {request.files}")
        
        # Process the form submission
        if form.validate_on_submit():
            logger.debug("Form validated successfully")
            try:
                # Handle primary image upload
                primary_image = form.primary_image.data
                filename = secure_filename(primary_image.filename)
                primary_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                primary_image.save(primary_path)
                logger.debug(f"Primary image saved to {primary_path}")
                
                # Load the image and check if it was resized
                original_image = Image.open(primary_path)
                original_size = original_image.size
                image = load_image(primary_path)
                
                if image is None:
                    logger.error(f"Failed to load image from {primary_path}")
                    error_msg = "Error loading image"
                    return jsonify({"success": False, "error": error_msg}) if is_ajax else error_msg, 400
                
                # Check if the image was resized
                was_resized = original_size != image.size
                if was_resized:
                    logger.info(f"Image resized from {original_size} to {image.size} for better performance (longest dimension limited to 1920px)")
                
                # Determine the selected effect and process accordingly
                effect = form.effect.data
                logger.debug(f"Selected effect: {effect}")
                
                if effect == 'pixel_sort_chunk':
                    # Get common parameters
                    chunk_width = form.chunk_width.data
                    chunk_height = form.chunk_height.data
                    chunk_size = f"{chunk_width}x{chunk_height}"
                    sort_by = form.sort_by.data
                    sort_mode = form.sort_mode.data
                    sort_order = form.sort_order.data
                    
                    # Process based on sort mode
                    if sort_mode == 'diagonal':
                        # Diagonal pixel sorting
                        starting_corner = form.starting_corner.data
                        logger.debug(f"Diagonal pixel sort params: chunk_size={chunk_size}, sort_by={sort_by}, starting_corner={starting_corner}, sort_order={sort_order}")
                        processed_image = pixel_sorting(image, sort_mode, chunk_size, sort_by, starting_corner=starting_corner, sort_order=sort_order)
                        settings = f"chunk_diagonal_{chunk_size}_{sort_by}_{starting_corner}_{sort_order}"
                    else:
                        # Regular pixel sorting (horizontal or vertical)
                        logger.debug(f"Regular pixel sort params: mode={sort_mode}, chunk_size={chunk_size}, sort_by={sort_by}, sort_order={sort_order}")
                        processed_image = pixel_sorting(image, sort_mode, chunk_size, sort_by, sort_order=sort_order)
                        settings = f"chunk_{sort_mode}_{chunk_size}_{sort_by}_{sort_order}"
                elif effect == 'color_channel':
                    manipulation_type = form.manipulation_type.data
                    logger.debug(f"Color channel manipulation type: {manipulation_type}")
                    if manipulation_type == 'swap':
                        choice = form.swap_choice.data
                        logger.debug(f"Swap choice: {choice}")
                        processed_image = color_channel_manipulation(image, manipulation_type, choice)
                        settings = f"swap_{choice}"
                    elif manipulation_type == 'invert':
                        choice = form.invert_choice.data
                        logger.debug(f"Invert choice: {choice}")
                        processed_image = color_channel_manipulation(image, manipulation_type, choice)
                        settings = f"invert_{choice}"
                    elif manipulation_type == 'negative':
                        logger.debug("Creating negative image")
                        processed_image = color_channel_manipulation(image, manipulation_type, None)
                        settings = "negative"
                    else:  # adjust
                        choice = form.adjust_choice.data
                        factor = form.intensity_factor.data
                        logger.debug(f"Adjust params: choice={choice}, factor={factor}")
                        processed_image = color_channel_manipulation(image, manipulation_type, choice, factor)
                        settings = f"adjust_{choice}_{factor}"
                elif effect == 'double_expose':
                    secondary_image = form.secondary_image.data
                    blend_mode = form.blend_mode.data
                    opacity = form.opacity.data
                    if not secondary_image:
                        logger.error("Secondary image required for Double Expose but not provided")
                        error_msg = "Secondary image required for Double Expose"
                        return jsonify({"success": False, "error": error_msg}) if is_ajax else error_msg, 400
                    secondary_filename = secure_filename(secondary_image.filename)
                    secondary_path = os.path.join(app.config['UPLOAD_FOLDER'], secondary_filename)
                    secondary_image.save(secondary_path)
                    logger.debug(f"Secondary image saved to {secondary_path}")
                    logger.debug(f"Double expose params: blend_mode={blend_mode}, opacity={opacity}")
                    secondary_img = load_image(secondary_path)
                    if secondary_img is None:
                        logger.error(f"Failed to load secondary image from {secondary_path}")
                        error_msg = "Error loading secondary image"
                        return jsonify({"success": False, "error": error_msg}) if is_ajax else error_msg, 400
                    processed_image = double_expose(image, secondary_img, blend_mode, opacity)
                    settings = f"doubleexpose_{blend_mode}_{opacity}"
                elif effect == 'pixel_drift':
                    direction = form.drift_direction.data
                    drift_bands = form.drift_bands.data
                    drift_intensity = form.drift_intensity.data
                    logger.debug(f"Pixel drift params: direction={direction}, bands={drift_bands}, intensity={drift_intensity}")
                    processed_image = pixel_drift(image, direction, drift_bands, drift_intensity)
                    settings = f"drift_{direction}_{drift_bands}_{drift_intensity}"
                elif effect == 'spiral_sort':
                    # Redirect to spiral_sort_2 with equivalent parameters
                    chunk_size = form.spiral_chunk_size.data
                    order = form.spiral_order.data
                    # Convert order to equivalent parameters for spiral_sort_2
                    sort_by = 'brightness'  # Always brightness in original
                    reverse = order == 'darkest-to-lightest'  # Determine reverse based on order
                    logger.debug(f"Converting spiral sort to spiral sort 2: chunk_size={chunk_size}, sort_by={sort_by}, reverse={reverse}")
                    processed_image = spiral_sort_2(image, chunk_size, sort_by, reverse)
                    settings = f"spiral2_{chunk_size}_{sort_by}_{'desc' if reverse else 'asc'}"
                elif effect == 'bit_manipulation':
                    logger.debug("Applying bit manipulation")
                    chunk_size = form.bit_chunk_size.data
                    logger.debug(f"Bit manipulation chunk size: {chunk_size}")
                    processed_image = bit_manipulation(image, chunk_size)
                    settings = f"bitmanip_{chunk_size}"
                elif effect == 'data_mosh_blocks':
                    num_operations = form.data_mosh_operations.data
                    max_block_size = form.data_mosh_block_size.data
                    block_movement = form.data_mosh_movement.data
                    color_swap = form.data_mosh_color_swap.data
                    invert = form.data_mosh_invert.data
                    shift = form.data_mosh_shift.data
                    flip = form.data_mosh_flip.data
                    seed = form.data_mosh_seed.data
                    
                    logger.debug(f"Data mosh blocks params: operations={num_operations}, block_size={max_block_size}, "
                                f"movement={block_movement}, color_swap={color_swap}, invert={invert}, "
                                f"shift={shift}, flip={flip}, seed={seed}")
                    
                    processed_image = data_mosh_blocks(image, num_operations, max_block_size, block_movement,
                                                    color_swap, invert, shift, flip, seed)
                    
                    settings = f"datamosh_{num_operations}_{max_block_size}_{block_movement}_{color_swap}_{invert}_{shift}_{flip}"
                    if seed:
                        settings += f"_{seed}"
                elif effect == 'full_frame_sort':
                    direction = form.full_frame_direction.data
                    sort_by = form.full_frame_sort_by.data
                    reverse = form.full_frame_reverse.data == 'true'
                    logger.debug(f"Full frame sort params: direction={direction}, sort_by={sort_by}, reverse={reverse}")
                    processed_image = full_frame_sort(image, direction, sort_by, reverse)
                    settings = f"fullframe_{direction}_{sort_by}_{'desc' if reverse else 'asc'}"
                elif effect == 'polar_sort':
                    chunk_size = form.polar_chunk_size.data
                    sort_by = form.polar_sort_by.data
                    reverse = form.polar_reverse.data == 'true'
                    logger.debug(f"Polar sort params: chunk_size={chunk_size}, sort_by={sort_by}, reverse={reverse}")
                    processed_image = polar_sorting(image, chunk_size, sort_by, reverse)
                    settings = f"polar_{chunk_size}_{sort_by}_{'desc' if reverse else 'asc'}"
                elif effect == 'perlin_noise_sort':
                    chunk_width = form.perlin_chunk_width.data
                    chunk_height = form.perlin_chunk_height.data
                    chunk_size = f"{chunk_width}x{chunk_height}"
                    noise_scale = form.perlin_noise_scale.data
                    direction = form.perlin_direction.data
                    reverse = form.perlin_reverse.data == 'true'
                    seed = form.perlin_seed.data
                    logger.debug(f"Perlin noise sort params: chunk_size={chunk_size}, noise_scale={noise_scale}, direction={direction}, reverse={reverse}, seed={seed}")
                    processed_image = perlin_noise_sorting(image, chunk_size, noise_scale, direction, reverse, seed)
                    settings = f"perlin_{chunk_size}_{noise_scale}_{direction}_{'desc' if reverse else 'asc'}_{seed}"
                elif effect == 'perlin_full_frame':
                    noise_scale = form.perlin_full_frame_noise_scale.data
                    sort_by = form.perlin_full_frame_sort_by.data
                    reverse = form.perlin_full_frame_reverse.data == 'true'
                    seed = form.perlin_full_frame_seed.data
                    logger.debug(f"Perlin full frame params: noise_scale={noise_scale}, sort_by={sort_by}, reverse={reverse}, seed={seed}")
                    processed_image = perlin_full_frame_sort(image, noise_scale, sort_by, reverse, seed)
                    settings = f"perlinfullframe_{noise_scale}_{sort_by}_{'desc' if reverse else 'asc'}_{seed}"
                elif effect == 'spiral_sort_2':
                    chunk_size = form.spiral2_chunk_size.data
                    sort_by = form.spiral2_sort_by.data
                    reverse = form.spiral2_reverse.data == 'true'
                    logger.debug(f"Spiral sort 2 params: chunk_size={chunk_size}, sort_by={sort_by}, reverse={reverse}")
                    processed_image = spiral_sort_2(image, chunk_size, sort_by, reverse)
                    settings = f"spiral2_{chunk_size}_{sort_by}_{'desc' if reverse else 'asc'}"
                elif effect == 'pixelate':
                    pixel_width = form.pixelate_width.data
                    pixel_height = form.pixelate_height.data
                    attribute = form.pixelate_attribute.data
                    num_bins = form.pixelate_bins.data
                    logger.debug(f"Pixelate params: width={pixel_width}, height={pixel_height}, attribute={attribute}, bins={num_bins}")
                    processed_image = pixelate_by_mode(image, pixel_width, pixel_height, attribute, num_bins)
                    settings = f"pixelate_{pixel_width}x{pixel_height}_{attribute}_{num_bins}"
                elif effect == 'concentric_shapes':
                    num_points = form.concentric_num_points.data
                    shape_type = form.shape_type.data
                    thickness = form.concentric_thickness.data
                    spacing = form.spacing.data
                    rotation_angle = form.rotation_angle.data
                    darken_step = form.darken_step.data
                    color_shift = form.color_shift.data
                    logger.debug(f"Concentric shapes params: points={num_points}, shape_type={shape_type}, thickness={thickness}, spacing={spacing}, rotation={rotation_angle}, darken={darken_step}, color_shift={color_shift}")
                    processed_image = concentric_shapes(image, num_points, shape_type, thickness, spacing, rotation_angle, darken_step, color_shift)
                    settings = f"concentric_{shape_type}_{num_points}_{thickness}_{spacing}_{rotation_angle}_{darken_step}_{color_shift}"
                elif effect == 'color_shift_expansion':
                    num_points = form.color_shift_num_points.data
                    shift_amount = form.color_shift_amount.data
                    expansion_type = form.expansion_type.data
                    pattern_type = form.pattern_type.data
                    color_theme = form.color_theme.data
                    saturation_boost = form.saturation_boost.data
                    value_boost = form.value_boost.data
                    decay_factor = form.decay_factor.data
                    
                    logger.debug(f"Color shift expansion params: num_points={num_points}, shift_amount={shift_amount}, "
                                f"expansion_type={expansion_type}, pattern_type={pattern_type}, color_theme={color_theme}, "
                                f"saturation_boost={saturation_boost}, value_boost={value_boost}, decay_factor={decay_factor}")
                    
                    # Use a fixed 'xtreme' mode since we no longer have a mode selector in the form
                    mode = 'xtreme'
                    processed_image = color_shift_expansion(
                        image=image, 
                        num_points=num_points, 
                        shift_amount=shift_amount, 
                        expansion_type=expansion_type,
                        mode=mode,  # Use the hardcoded mode
                        saturation_boost=saturation_boost, 
                        value_boost=value_boost, 
                        pattern_type=pattern_type, 
                        color_theme=color_theme, 
                        decay_factor=decay_factor
                    )
                    
                    settings = f"colorshift_{num_points}_{shift_amount}_{expansion_type}_{pattern_type}_{color_theme}"
                elif effect == 'perlin_displacement':
                    scale = form.perlin_displacement_scale.data
                    intensity = form.perlin_displacement_intensity.data
                    octaves = form.perlin_displacement_octaves.data
                    persistence = form.perlin_displacement_persistence.data
                    lacunarity = form.perlin_displacement_lacunarity.data
                    logger.debug(f"Perlin displacement params: scale={scale}, intensity={intensity}, octaves={octaves}, persistence={persistence}, lacunarity={lacunarity}")
                    processed_image = perlin_noise_displacement(image, scale, intensity, octaves, persistence, lacunarity)
                    settings = f"perlindisplacement_{scale}_{intensity}_{octaves}_{persistence}_{lacunarity}"
                elif effect == 'voronoi_sort':
                    num_cells = form.voronoi_num_cells.data
                    size_variation = form.voronoi_size_variation.data
                    sort_by = form.voronoi_sort_by.data
                    sort_order = form.voronoi_sort_order.data
                    orientation = form.voronoi_orientation.data
                    start_position = form.voronoi_start_position.data
                    seed = form.voronoi_seed.data
                    
                    logger.debug(f"Voronoi sort params: num_cells={num_cells}, size_variation={size_variation}, "
                               f"sort_by={sort_by}, sort_order={sort_order}, orientation={orientation}, "
                               f"start_position={start_position}, seed={seed}")
                    
                    processed_image = voronoi_pixel_sort(image, num_cells, size_variation, sort_by, sort_order, seed, 
                                                       orientation, start_position)
                    
                    settings = f"voronoi_{num_cells}_{size_variation}_{sort_by}_{sort_order}_{orientation}_{start_position}"
                    if seed:
                        settings += f"_{seed}"
                elif effect == 'channel_shift':
                    shift_amount = form.channel_shift_amount.data
                    direction = form.channel_shift_direction.data
                    centered_channel = form.channel_shift_center.data
                    mode = form.channel_mode.data
                    
                    if mode == 'mirror':
                        logger.debug(f"Channel shift params (mirror mode): centered_channel={centered_channel}, mode={mode}")
                        # Direction is ignored in mirror mode, shift_amount is also ignored
                        processed_image = split_and_shift_channels(image, 0, 'horizontal', centered_channel, mode)
                        settings = f"channelshift_{mode}_{centered_channel}"
                    else:  # shift mode
                        logger.debug(f"Channel shift params (shift mode): amount={shift_amount}, direction={direction}, centered_channel={centered_channel}")
                        processed_image = split_and_shift_channels(image, shift_amount, direction, centered_channel, mode)
                        settings = f"channelshift_{shift_amount}_{direction}_{centered_channel}_{mode}"
                elif effect == 'jpeg_artifacts':
                    intensity = form.jpeg_intensity.data
                    logger.debug(f"JPEG artifacts params: intensity={intensity}")
                    
                    processed_image = simulate_jpeg_artifacts(image, intensity)
                    settings = f"jpeg_artifacts_{intensity}"
                elif effect == 'pixel_scatter':
                    direction = form.scatter_direction.data
                    select_by = form.scatter_select_by.data
                    min_val = form.scatter_min_value.data
                    max_val = form.scatter_max_value.data
                    logger.debug(f"Pixel scatter params: direction={direction}, select_by={select_by}, min_val={min_val}, max_val={max_val}")
                    processed_image = pixel_scatter(image, direction, select_by, min_val, max_val)
                    settings = f"scatter_{direction}_{select_by}_{min_val}_{max_val}"
                elif effect == 'databend':
                    try:
                        # Apply databending effect
                        logger.info("Applying databending effect with pixel manipulation")
                        intensity = form.databend_intensity.data
                        preserve_header = form.databend_preserve_header.data == 'true'
                        seed = form.databend_seed.data
                        logger.debug(f"Databending params: intensity={intensity}, preserve_header={preserve_header}, seed={seed}")
                        processed_image = databend_image(image, intensity, preserve_header, seed)
                        settings = f"databend_{intensity}_{preserve_header}_{seed}"
                    except Exception as e:
                        logger.error(f"Error during databending process: {e}")
                        error_msg = "Error applying databending effect. Try a different intensity or seed value."
                        return jsonify({"success": False, "error": error_msg}) if is_ajax else error_msg, 400
                elif effect == 'histogram_glitch':
                    # Extract parameters
                    r_mode = form.hist_r_mode.data
                    g_mode = form.hist_g_mode.data
                    b_mode = form.hist_b_mode.data
                    r_freq = form.hist_r_freq.data
                    r_phase = form.hist_r_phase.data
                    g_freq = form.hist_g_freq.data
                    g_phase = form.hist_g_phase.data
                    b_freq = form.hist_b_freq.data
                    b_phase = form.hist_b_phase.data
                    gamma_val = form.hist_gamma.data
                    
                    logger.debug(f"Histogram glitch params: r_mode={r_mode}, g_mode={g_mode}, b_mode={b_mode}, "
                               f"frequencies=({r_freq}, {g_freq}, {b_freq}), phases=({r_phase}, {g_phase}, {b_phase}), "
                               f"gamma={gamma_val}")
                    
                    processed_image = histogram_glitch(
                        image, r_mode, g_mode, b_mode, r_freq, r_phase, g_freq, g_phase, b_freq, b_phase, gamma_val
                    )
                    
                    settings = f"histogram_{r_mode}_{g_mode}_{b_mode}"
                elif effect == 'ripple':
                    # Extract parameters
                    num_droplets = form.ripple_num_droplets.data
                    amplitude = form.ripple_amplitude.data
                    frequency = form.ripple_frequency.data
                    decay = form.ripple_decay.data
                    distortion_type = form.ripple_distortion_type.data
                    
                    # Create distortion parameters based on the selected distortion type
                    distortion_params = {}
                    if distortion_type == 'color_shift':
                        distortion_params = {
                            'factor_r': form.ripple_color_r.data,
                            'factor_g': form.ripple_color_g.data,
                            'factor_b': form.ripple_color_b.data
                        }
                    elif distortion_type == 'pixelation':
                        distortion_params = {
                            'scale': form.ripple_pixelation_scale.data,
                            'max_mag': form.ripple_pixelation_max_mag.data
                        }
                    
                    logger.debug(f"Ripple effect params: num_droplets={num_droplets}, amplitude={amplitude}, "
                               f"frequency={frequency}, decay={decay}, distortion_type={distortion_type}, "
                               f"distortion_params={distortion_params}")
                    
                    processed_image = Ripple(
                        image, num_droplets, amplitude, frequency, decay, distortion_type, distortion_params
                    )
                    
                    settings = f"ripple_{num_droplets}_{amplitude}_{frequency}_{decay}_{distortion_type}"
                elif effect == 'masked_merge':
                    # Extract parameters
                    secondary_image = form.masked_merge_secondary.data
                    mask_type = form.mask_type.data
                    
                    # Determine the correct width parameter based on the mask type
                    if mask_type == 'concentric_rectangles':
                        current_mask_width = form.concentric_rectangle_width.data
                    else:
                        # Use the general mask_width for other types like checkerboard
                        current_mask_width = form.mask_width.data 
                        
                    # Keep mask_height for checkerboard types
                    mask_height = form.mask_height.data 
                    
                    random_seed = form.mask_random_seed.data if mask_type in ['random_checkerboard', 'perlin', 'voronoi'] else None
                    stripe_width = form.stripe_width.data
                    stripe_angle = form.stripe_angle.data
                    perlin_noise_scale = form.perlin_noise_scale.data
                    perlin_threshold = form.perlin_threshold.data
                    perlin_octaves = form.perlin_octaves.data
                    voronoi_cells = form.voronoi_cells.data
                    
                    if not secondary_image:
                        logger.error("Secondary image required for Masked Merge but not provided")
                        error_msg = "Secondary image required for Masked Merge"
                        return jsonify({"success": False, "error": error_msg}) if is_ajax else error_msg, 400
                    
                    secondary_filename = secure_filename(secondary_image.filename)
                    secondary_path = os.path.join(app.config['UPLOAD_FOLDER'], secondary_filename)
                    secondary_image.save(secondary_path)
                    logger.debug(f"Secondary image saved to {secondary_path}")
                    
                    # Log parameters based on mask type
                    if mask_type in ['checkerboard', 'random_checkerboard']:
                        logger.debug(f"Masked merge params: mask_type={mask_type}, mask_width={current_mask_width}, "
                                    f"mask_height={mask_height}, random_seed={random_seed}")
                    elif mask_type in ['striped', 'gradient_striped']:
                        logger.debug(f"Masked merge params: mask_type={mask_type}, stripe_width={stripe_width}, "
                                    f"stripe_angle={stripe_angle}")
                    elif mask_type == 'perlin':
                        logger.debug(f"Masked merge params: mask_type={mask_type}, noise_scale={perlin_noise_scale}, "
                                    f"threshold={perlin_threshold}, random_seed={random_seed}, octaves={perlin_octaves}")
                    elif mask_type == 'voronoi':
                        logger.debug(f"Masked merge params: mask_type={mask_type}, voronoi_cells={voronoi_cells}, "
                                    f"random_seed={random_seed}")
                    elif mask_type == 'concentric_rectangles':
                        # Use the specific width variable for logging
                        logger.debug(f"Masked merge params: mask_type={mask_type}, width={current_mask_width}")
                    
                    secondary_img = load_image(secondary_path)
                    if secondary_img is None:
                        logger.error(f"Failed to load secondary image from {secondary_path}")
                        error_msg = "Error loading secondary image"
                        return jsonify({"success": False, "error": error_msg}) if is_ajax else error_msg, 400
                    
                    # Call the function with appropriate parameters
                    # Pass the determined width (current_mask_width) as the 'width' parameter
                    processed_image = masked_merge(
                        image,
                        secondary_img,
                        mask_type,
                        width=current_mask_width, # Use the determined width
                        height=mask_height, # Still needed for other modes
                        random_seed=random_seed,
                        stripe_width=stripe_width,
                        stripe_angle=stripe_angle,
                        perlin_noise_scale=perlin_noise_scale,
                        threshold=perlin_threshold,
                        voronoi_cells=voronoi_cells,
                        perlin_octaves=perlin_octaves
                    )
                    
                    # Create settings string based on mask type
                    if mask_type in ['checkerboard', 'random_checkerboard']:
                        settings = f"maskedmerge_{mask_type}_{current_mask_width}x{mask_height}"
                        if mask_type == 'random_checkerboard' and random_seed:
                            settings += f"_{random_seed}"
                    elif mask_type in ['striped', 'gradient_striped']:
                        settings = f"maskedmerge_{mask_type}_{stripe_width}px_{stripe_angle}deg"
                    elif mask_type == 'perlin':
                        settings = f"maskedmerge_{mask_type}_{perlin_noise_scale}_{perlin_threshold}"
                        if random_seed:
                            settings += f"_{random_seed}"
                    elif mask_type == 'voronoi':
                        settings = f"maskedmerge_{mask_type}_{voronoi_cells}"
                        if random_seed:
                            settings += f"_{random_seed}"
                    elif mask_type == 'concentric_rectangles':
                        # Use the specific width variable for settings
                        settings = f"maskedmerge_{mask_type}_{current_mask_width}px"
                else:
                    logger.error(f"Unknown effect: {effect}")
                    error_msg = f"Unknown effect: {effect}"
                    return jsonify({"success": False, "error": error_msg}) if is_ajax else render_template('index.html', form=form, error=error_msg)
                
                # Save the processed image
                output_filename = generate_output_filename(filename, effect, settings)
                output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
                processed_image.save(output_path)
                logger.debug(f"Processed image saved to {output_path}")
                
                # Prepare the response
                processed_url = url_for('processed_file', filename=output_filename)
                
                # Return a success response
                if is_ajax:
                    response_data = {
                        "success": True,
                        "processed_url": processed_url
                    }
                    # Add information about resizing if it happened
                    if was_resized:
                        response_data["was_resized"] = True
                        response_data["original_size"] = f"{original_size[0]}x{original_size[1]}"
                        response_data["new_size"] = f"{image.size[0]}x{image.size[1]}"
                    
                    return jsonify(response_data)
                else:
                    # For non-AJAX requests, redirect to the processed image
                    return redirect(processed_url)
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                logger.error(traceback.format_exc())
                error_msg = f"Error: {str(e)}"
                return jsonify({"success": False, "error": error_msg}) if is_ajax else error_msg, 500
        else:
            # Form validation failed
            logger.error(f"Form validation failed: {form.errors}")
            if is_ajax:
                errors = {}
                for field_name, field_errors in form.errors.items():
                    errors[field_name] = field_errors
                return jsonify({"success": False, "error": "Form validation failed", "form_errors": errors}), 400
            # For non-AJAX requests, the form will be re-rendered with errors
    
    return render_template('index.html', form=form)

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
    
    # Get port from environment variable for Heroku compatibility
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)