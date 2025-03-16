from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import os
import traceback
import logging
import numpy as np
from forms import ImageProcessForm
from utils import load_image, pixel_sorting, color_channel_manipulation, data_moshing, pixel_drift, bit_manipulation, generate_output_filename, spiral_sort

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Replace with a secure key in production
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PROCESSED_FOLDER'] = 'processed/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['WTF_CSRF_ENABLED'] = True  # Enable CSRF protection

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Ensure upload and processed directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
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
                
                # Load the image
                image = load_image(primary_path)
                if image is None:
                    logger.error(f"Failed to load image from {primary_path}")
                    error_msg = "Error loading image"
                    return jsonify({"success": False, "error": error_msg}) if is_ajax else error_msg, 400
                
                # Determine the selected effect and process accordingly
                effect = form.effect.data
                logger.debug(f"Selected effect: {effect}")
                
                if effect == 'pixel_sort_original':
                    direction = form.direction.data
                    chunk_width = form.chunk_width.data
                    chunk_height = form.chunk_height.data
                    chunk_size = f"{chunk_width}x{chunk_height}"
                    sort_by = form.sort_by.data
                    logger.debug(f"Pixel sort params: direction={direction}, chunk_size={chunk_size}, sort_by={sort_by}")
                    processed_image = pixel_sorting(image, direction, chunk_size, sort_by)
                    settings = f"{direction}_{chunk_size}_{sort_by}"
                elif effect == 'pixel_sort_corner':
                    chunk_width = form.corner_chunk_width.data
                    chunk_height = form.corner_chunk_height.data
                    chunk_size = f"{chunk_width}x{chunk_height}"
                    sort_by = form.corner_sort_by.data
                    starting_corner = form.starting_corner.data
                    direction = form.corner_direction.data
                    logger.debug(f"Corner pixel sort params: chunk_size={chunk_size}, sort_by={sort_by}, starting_corner={starting_corner}, direction={direction}")
                    processed_image = pixel_sorting(image, direction, chunk_size, sort_by, starting_corner=starting_corner)
                    settings = f"corner_{chunk_size}_{sort_by}_{starting_corner}_{direction}"
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
                    else:  # adjust
                        choice = form.adjust_choice.data
                        factor = form.intensity_factor.data
                        logger.debug(f"Adjust params: choice={choice}, factor={factor}")
                        processed_image = color_channel_manipulation(image, manipulation_type, choice, factor)
                        settings = f"adjust_{choice}_{factor}"
                elif effect == 'data_moshing':
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
                    processed_image = data_moshing(image, secondary_img, blend_mode, opacity)
                    settings = f"doubleexpose_{blend_mode}_{opacity}"
                elif effect == 'pixel_drift':
                    direction = form.drift_direction.data
                    drift_bands = form.drift_bands.data
                    drift_intensity = form.drift_intensity.data
                    logger.debug(f"Pixel drift params: direction={direction}, bands={drift_bands}, intensity={drift_intensity}")
                    processed_image = pixel_drift(image, direction, drift_bands, drift_intensity)
                    settings = f"drift_{direction}_{drift_bands}_{drift_intensity}"
                elif effect == 'spiral_sort':
                    chunk_size = form.spiral_chunk_size.data
                    order = form.spiral_order.data
                    logger.debug(f"Spiral sort params: chunk_size={chunk_size}, order={order}")
                    processed_image = spiral_sort(image, chunk_size, order)
                    settings = f"spiral_{chunk_size}_{order}"
                elif effect == 'bit_manipulation':
                    logger.debug("Applying bit manipulation")
                    chunk_size = form.bit_chunk_size.data
                    logger.debug(f"Bit manipulation chunk size: {chunk_size}")
                    processed_image = bit_manipulation(image, chunk_size)
                    settings = f"bitmanip_{chunk_size}"
                
                # Save the processed image
                output_filename = generate_output_filename(filename, effect, settings)
                output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
                processed_image.save(output_path)
                logger.debug(f"Processed image saved to {output_path}")
                
                # Return appropriate response based on request type
                if is_ajax:
                    logger.debug("Returning JSON response for AJAX request")
                    return jsonify({
                        'success': True,
                        'original_url': url_for('uploaded_file', filename=filename),
                        'processed_url': url_for('processed_file', filename=output_filename)
                    })
                else:
                    logger.debug("Rendering result.html template for non-AJAX request")
                    return render_template('result.html', 
                                          original=filename, 
                                          processed=output_filename,
                                          original_url=url_for('uploaded_file', filename=filename),
                                          processed_url=url_for('processed_file', filename=output_filename))
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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)