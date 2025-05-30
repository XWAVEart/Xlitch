$(document).ready(function() {
    console.log("Document ready, initializing Glitch Art app...");
    
    // Set up CSRF token for AJAX requests
    var csrftoken = $('input[name=csrf_token]').val();
    console.log("CSRF token present:", csrftoken ? "Yes" : "No");
    
    // Add debug info to the page
    function addDebugInfo(message) {
        $('#debug-content').prepend('<div>' + new Date().toISOString() + ': ' + message + '</div>');
        $('#debug-info').show();
    }
    
    // Enable debug mode
    var debugMode = true;
    if (debugMode) {
        $('#debug-info').show();
        addDebugInfo("Debug mode enabled");
        addDebugInfo("CSRF token: " + (csrftoken ? "Present" : "Missing"));
    }
    
    $.ajaxSetup({
        beforeSend: function(xhr, settings) {
            if (!/^(GET|HEAD|OPTIONS|TRACE)$/i.test(settings.type) && !this.crossDomain) {
                // Get the token from the form
                var csrftoken = $('input[name=csrf_token]').val();
                xhr.setRequestHeader("X-CSRFToken", csrftoken);
            }
        }
    });
    
    // Handle effect selection
    $('#effect').change(function() {
        var effect = $(this).val();
        console.log("Effect changed to:", effect);
        addDebugInfo("Effect changed to: " + effect);
        
        // Hide all effect fields first
        $('div[id$="_fields"]').hide();
        
        if (effect) {
            // Show the fields for the selected effect
            $('#' + effect + '_fields').show();
            
            // Special handling for color_channel effect
            if (effect === 'color_channel') {
                $('#manipulation_type').trigger('change');
            }
            
            // Special handling for channel_shift effect
            if (effect === 'channel_shift') {
                $('#channel_mode').trigger('change');
            }
            
            // Special handling for histogram_glitch effect
            if (effect === 'histogram_glitch') {
                // Set default values for histogram glitch if needed
                if (!$('[name="hist_r_mode"]').val()) {
                    $('[name="hist_r_mode"]').val('solarize');
                }
                if (!$('[name="hist_g_mode"]').val()) {
                    $('[name="hist_g_mode"]').val('solarize');
                }
                if (!$('[name="hist_b_mode"]').val()) {
                    $('[name="hist_b_mode"]').val('solarize');
                }
                if (!$('[name="hist_r_freq"]').val()) {
                    $('[name="hist_r_freq"]').val('1.0');
                }
                if (!$('[name="hist_r_phase"]').val()) {
                    $('[name="hist_r_phase"]').val('0.0');
                }
                if (!$('[name="hist_g_freq"]').val()) {
                    $('[name="hist_g_freq"]').val('1.0');
                }
                if (!$('[name="hist_g_phase"]').val()) {
                    $('[name="hist_g_phase"]').val('0.2');
                }
                if (!$('[name="hist_b_freq"]').val()) {
                    $('[name="hist_b_freq"]').val('1.0');
                }
                if (!$('[name="hist_b_phase"]').val()) {
                    $('[name="hist_b_phase"]').val('0.4');
                }
                if (!$('[name="hist_gamma"]').val()) {
                    $('[name="hist_gamma"]').val('0.5');
                }
                
                // Update UI after setting defaults
                updateHistogramGlitchUI();
            }
            
            // Special handling for ripple effect
            if (effect === 'ripple') {
                $('#ripple_distortion_type').trigger('change');
            }
            
            // Special handling for masked_merge effect
            if (effect === 'masked_merge') {
                $('#mask_type').trigger('change');
            }
            
            // Special handling for bit_manipulation effect
            if (effect === 'bit_manipulation') {
                $('#bit_manipulation_type').trigger('change');
            }
        }
    });
    
    // Handle manipulation type selection for color channel effect
    $('#manipulation_type').change(function() {
        var type = $(this).val();
        console.log("Manipulation type changed to:", type);
        addDebugInfo("Manipulation type changed to: " + type);
        
        // Hide all manipulation type fields first
        $('#swap_fields, #invert_fields, #adjust_fields, #negative_fields').hide();
        
        if (type) {
            // Show the fields for the selected manipulation type
            $('#' + type + '_fields').show();
        }
    });
    
    // Handle distortion type selection for ripple effect
    $('#ripple_distortion_type').change(function() {
        var type = $(this).val();
        console.log("Ripple distortion type changed to:", type);
        addDebugInfo("Ripple distortion type changed to: " + type);
        
        // Hide all distortion type fields first
        $('#ripple_color_shift_fields, #ripple_pixelation_fields').hide();
        
        if (type === 'color_shift') {
            $('#ripple_color_shift_fields').show();
        } else if (type === 'pixelation') {
            $('#ripple_pixelation_fields').show();
        }
        // 'none' type doesn't show any additional fields
    });
    
    // Handle channel mode selection for RGB Channel Shift
    $('#channel_mode').change(function() {
        var mode = $(this).val();
        console.log("Channel mode changed to:", mode);
        addDebugInfo("Channel mode changed to: " + mode);
        
        if (mode === 'shift') {
            $('#shift_mode_fields').show();
            $('#mirror_mode_info').hide();
            $('#center_channel_help').text('Channel that stays centered (others will shift away)');
        } else if (mode === 'mirror') {
            $('#shift_mode_fields').hide();
            $('#mirror_mode_info').show();
            $('#center_channel_help').text('Channel that stays unchanged (others will be mirrored)');
        }
    });
    
    // Preview primary image when selected
    $('#primary-image-input').change(function() {
        console.log("Primary image selected");
        addDebugInfo("Primary image selected: " + (this.files && this.files[0] ? this.files[0].name : "none"));
        
        if (this.files && this.files[0]) {
            var reader = new FileReader();
            reader.onload = function(e) {
                $('#original-preview').attr('src', e.target.result);
                $('#image-preview').show();
                $('#processed-container').hide();
                console.log("Image preview displayed");
                addDebugInfo("Image preview displayed");
            };
            reader.readAsDataURL(this.files[0]);
        }
    });
    
    // Preview secondary image when selected (for data moshing)
    $('#secondary-image-input').change(function() {
        // No preview needed for secondary image, but we could add one if desired
        addDebugInfo("Secondary image selected: " + (this.files && this.files[0] ? this.files[0].name : "none"));
    });
    
    // Clear any existing error messages
    function clearErrors() {
        $('.error-message').remove();
        $('.form-group').removeClass('has-error');
        addDebugInfo("Cleared error messages");
    }
    
    // Display error messages
    function displayErrors(errors) {
        clearErrors();
        console.log("Displaying errors:", errors);
        addDebugInfo("Displaying errors: " + JSON.stringify(errors));
        
        if (errors.form_errors) {
            $.each(errors.form_errors, function(field, messages) {
                var fieldElement = $('[name="' + field + '"]');
                var errorMessage = '<div class="error-message">' + messages.join(', ') + '</div>';
                fieldElement.after(errorMessage);
                fieldElement.closest('.form-group').addClass('has-error');
                addDebugInfo("Field error: " + field + " - " + messages.join(', '));
            });
        } else if (errors.error) {
            $('#image-form').before('<div class="error-message global-error">' + errors.error + '</div>');
            addDebugInfo("Global error: " + errors.error);
        }
    }
    
    // Set default values for form fields if they're empty
    function setDefaultValues() {
        // Set default chunk width and height if empty
        if ($('#pixel_sort_original_fields').is(':visible')) {
            if (!$('[name="chunk_width"]').val()) {
                $('[name="chunk_width"]').val('32');
                addDebugInfo("Set default chunk_width to 32");
            }
            if (!$('[name="chunk_height"]').val()) {
                $('[name="chunk_height"]').val('32');
                addDebugInfo("Set default chunk_height to 32");
            }
        }
        
        if ($('#pixel_sort_corner_fields').is(':visible')) {
            if (!$('[name="corner_chunk_width"]').val()) {
                $('[name="corner_chunk_width"]').val('32');
                addDebugInfo("Set default corner_chunk_width to 32");
            }
            if (!$('[name="corner_chunk_height"]').val()) {
                $('[name="corner_chunk_height"]').val('32');
                addDebugInfo("Set default corner_chunk_height to 32");
            }
        }
        
        if ($('#polar_sort_fields').is(':visible')) {
            if (!$('[name="polar_chunk_size"]').val()) {
                $('[name="polar_chunk_size"]').val('32');
                addDebugInfo("Set default polar_chunk_size to 32");
            }
        }
        
        if ($('#perlin_noise_sort_fields').is(':visible')) {
            if (!$('[name="perlin_chunk_width"]').val()) {
                $('[name="perlin_chunk_width"]').val('32');
                addDebugInfo("Set default perlin_chunk_width to 32");
            }
            if (!$('[name="perlin_chunk_height"]').val()) {
                $('[name="perlin_chunk_height"]').val('32');
                addDebugInfo("Set default perlin_chunk_height to 32");
            }
            if (!$('[name="perlin_noise_scale"]').val()) {
                $('[name="perlin_noise_scale"]').val('0.01');
                addDebugInfo("Set default perlin_noise_scale to 0.01");
            }
            if (!$('[name="perlin_seed"]').val()) {
                $('[name="perlin_seed"]').val('42');
                addDebugInfo("Set default perlin_seed to 42");
            }
        }
        
        if ($('#perlin_full_frame_fields').is(':visible')) {
            if (!$('[name="perlin_full_frame_noise_scale"]').val()) {
                $('[name="perlin_full_frame_noise_scale"]').val('0.01');
                addDebugInfo("Set default perlin_full_frame_noise_scale to 0.01");
            }
            if (!$('[name="perlin_full_frame_seed"]').val()) {
                $('[name="perlin_full_frame_seed"]').val('42');
                addDebugInfo("Set default perlin_full_frame_seed to 42");
            }
        }
        
        if ($('#data_mosh_blocks_fields').is(':visible')) {
            if (!$('[name="data_mosh_operations"]').val()) {
                $('[name="data_mosh_operations"]').val('10');
                addDebugInfo("Set default data_mosh_operations to 10");
            }
            if (!$('[name="data_mosh_block_size"]').val()) {
                $('[name="data_mosh_block_size"]').val('50');
                addDebugInfo("Set default data_mosh_block_size to 50");
            }
        }
        
        if ($('#pixelate_fields').is(':visible')) {
            if (!$('[name="pixelate_width"]').val()) {
                $('[name="pixelate_width"]').val('8');
                addDebugInfo("Set default pixelate_width to 8");
            }
            if (!$('[name="pixelate_height"]').val()) {
                $('[name="pixelate_height"]').val('8');
                addDebugInfo("Set default pixelate_height to 8");
            }
            if (!$('[name="pixelate_bins"]').val()) {
                $('[name="pixelate_bins"]').val('100');
                addDebugInfo("Set default pixelate_bins to 100");
            }
        }
        
        if ($('#pixel_scatter_fields').is(':visible')) {
            if (!$('[name="scatter_min_value"]').val()) {
                $('[name="scatter_min_value"]').val('100');
                addDebugInfo("Set default scatter_min_value to 100");
            }
            if (!$('[name="scatter_max_value"]').val()) {
                $('[name="scatter_max_value"]').val('200');
                addDebugInfo("Set default scatter_max_value to 200");
            }
        }
        
        if ($('#channel_shift_fields').is(':visible')) {
            if (!$('[name="channel_shift_amount"]').val()) {
                $('[name="channel_shift_amount"]').val('10');
                addDebugInfo("Set default channel_shift_amount to 10");
            }
            if (!$('[name="channel_mode"]').val()) {
                $('[name="channel_mode"]').val('shift');
                addDebugInfo("Set default channel_mode to shift");
            }
        }
        
        // Set default intensity factor if empty
        if ($('#adjust_fields').is(':visible') && !$('[name="intensity_factor"]').val()) {
            $('[name="intensity_factor"]').val('1.5');
            addDebugInfo("Set default intensity_factor to 1.5");
        }
        
        if ($('#databend_fields').is(':visible')) {
            if (!$('[name="databend_intensity"]').val()) {
                $('[name="databend_intensity"]').val('0.1');
                addDebugInfo("Set default databend_intensity to 0.1");
            }
            if (!$('[name="databend_seed"]').val()) {
                $('[name="databend_seed"]').val('42');
                addDebugInfo("Set default databend_seed to 42");
            }
        }
        
        if ($('#histogram_glitch_fields').is(':visible')) {
            // Channel modes
            if (!$('[name="hist_r_mode"]').val()) {
                $('[name="hist_r_mode"]').val('solarize');
                addDebugInfo("Set default hist_r_mode to solarize");
            }
            if (!$('[name="hist_g_mode"]').val()) {
                $('[name="hist_g_mode"]').val('solarize');
                addDebugInfo("Set default hist_g_mode to solarize");
            }
            if (!$('[name="hist_b_mode"]').val()) {
                $('[name="hist_b_mode"]').val('solarize');
                addDebugInfo("Set default hist_b_mode to solarize");
            }
            
            // Solarize parameters
            if (!$('[name="hist_r_freq"]').val()) {
                $('[name="hist_r_freq"]').val('1.0');
                addDebugInfo("Set default hist_r_freq to 1.0");
            }
            if (!$('[name="hist_r_phase"]').val()) {
                $('[name="hist_r_phase"]').val('0.0');
                addDebugInfo("Set default hist_r_phase to 0.0");
            }
            if (!$('[name="hist_g_freq"]').val()) {
                $('[name="hist_g_freq"]').val('1.0');
                addDebugInfo("Set default hist_g_freq to 1.0");
            }
            if (!$('[name="hist_g_phase"]').val()) {
                $('[name="hist_g_phase"]').val('0.2');
                addDebugInfo("Set default hist_g_phase to 0.2");
            }
            if (!$('[name="hist_b_freq"]').val()) {
                $('[name="hist_b_freq"]').val('1.0');
                addDebugInfo("Set default hist_b_freq to 1.0");
            }
            if (!$('[name="hist_b_phase"]').val()) {
                $('[name="hist_b_phase"]').val('0.4');
                addDebugInfo("Set default hist_b_phase to 0.4");
            }
            
            // Gamma parameter
            if (!$('[name="hist_gamma"]').val()) {
                $('[name="hist_gamma"]').val('0.5');
                addDebugInfo("Set default hist_gamma to 0.5");
            }
        }
        
        if ($('#masked_merge_fields').is(':visible')) {
            // Get the current mask type
            var maskType = $('#mask_type').val();
            
            // Only set default mask_width if it's not concentric_circles or if the field is empty
            if (maskType !== 'concentric_circles' && !$('[name="mask_width"]').val()) {
                $('[name="mask_width"]').val('32');
                addDebugInfo("Set default mask_width to 32");
            } else if (maskType === 'concentric_circles') {
                // For concentric_circles, preserve the user's value and log it
                var userWidth = $('[name="mask_width"]').val();
                if (userWidth) {
                    addDebugInfo("Using concentric circles band thickness: " + userWidth);
                } else {
                    // Only set default if empty
                    $('[name="mask_width"]').val('32');
                    addDebugInfo("Set default concentric circles band thickness to 32");
                }
            }
            
            if (!$('[name="mask_height"]').val()) {
                $('[name="mask_height"]').val('32');
                addDebugInfo("Set default mask_height to 32");
            }
            if (!$('[name="stripe_width"]').val()) {
                $('[name="stripe_width"]').val('16');
                addDebugInfo("Set default stripe_width to 16");
            }
            if (!$('[name="stripe_angle"]').val()) {
                $('[name="stripe_angle"]').val('45');
                addDebugInfo("Set default stripe_angle to 45");
            }
            if (!$('[name="perlin_noise_scale"]').val()) {
                $('[name="perlin_noise_scale"]').val('0.01');
                addDebugInfo("Set default perlin_noise_scale to 0.01");
            }
            if (!$('[name="perlin_threshold"]').val()) {
                $('[name="perlin_threshold"]').val('0.5');
                addDebugInfo("Set default perlin_threshold to 0.5");
            }
            
            // Special handling for concentric_circles - preserve user input and don't override
            var maskType = $('#mask_type').val();
            if (maskType === 'concentric_circles') {
                var userWidth = $('[name="mask_width"]').val();
                if (userWidth) {
                    addDebugInfo("Preserving user-set concentric circles band thickness: " + userWidth);
                }
            }
        }
        
        if ($('#posterize_fields').is(':visible')) {
            if (!$('[name="posterize_levels"]').val()) {
                $('[name="posterize_levels"]').val('4');
                addDebugInfo("Set default posterize_levels to 4");
            }
        }
        
        if ($('#curved_hue_shift_fields').is(':visible')) {
            if (!$('[name="hue_curve"]').val()) {
                $('[name="hue_curve"]').val('180');
                addDebugInfo("Set default hue_curve to 180");
            }
            if (!$('[name="hue_shift_amount"]').val()) {
                $('[name="hue_shift_amount"]').val('60');
                addDebugInfo("Set default hue_shift_amount to 60");
            }
        }
        
        if ($('#bit_manipulation_fields').is(':visible')) {
            if (!$('[name="bit_chunk_size"]').val()) {
                $('[name="bit_chunk_size"]').val('24');
                addDebugInfo("Set default bit_chunk_size to 24");
            }
            if (!$('[name="bit_manipulation_type"]').val()) {
                $('[name="bit_manipulation_type"]').val('xor');
                addDebugInfo("Set default bit_manipulation_type to xor");
            }
            if (!$('[name="bit_skip_pattern"]').val()) {
                $('[name="bit_skip_pattern"]').val('alternate');
                addDebugInfo("Set default bit_skip_pattern to alternate");
            }
            if (!$('[name="bit_offset"]').val()) {
                $('[name="bit_offset"]').val('0');
                addDebugInfo("Set default bit_offset to 0");
            }
            if (!$('[name="bit_xor_value"]').val()) {
                $('[name="bit_xor_value"]').val('255');
                addDebugInfo("Set default bit_xor_value to 255");
            }
            if (!$('[name="bit_shift"]').val()) {
                $('[name="bit_shift"]').val('1');
                addDebugInfo("Set default bit_shift to 1");
            }
            if (!$('[name="bit_randomize"]').val()) {
                $('[name="bit_randomize"]').val('false');
                addDebugInfo("Set default bit_randomize to false");
            }
            if (!$('[name="bit_random_seed"]').val()) {
                $('[name="bit_random_seed"]').val('42');
                addDebugInfo("Set default bit_random_seed to 42");
            }
        }
    }
    
    // Handle form submission via AJAX
    $(document).on('submit', '#image-form', function(e) {
        console.log("Form submitted");
        addDebugInfo("Form submitted");
        e.preventDefault();
        clearErrors();
        
        console.log("Form ID:", this.id);
        console.log("Form action:", $(this).attr('action'));
        console.log("Form method:", $(this).attr('method'));
        console.log("Form enctype:", $(this).attr('enctype'));
        
        addDebugInfo("Form ID: " + this.id);
        addDebugInfo("Form action: " + $(this).attr('action'));
        addDebugInfo("Form method: " + $(this).attr('method'));
        addDebugInfo("Form enctype: " + $(this).attr('enctype'));
        
        // Check if an image is selected
        if (!$('#primary-image-input')[0].files.length) {
            $('#image-form').before('<div class="error-message global-error">Please select an image to process</div>');
            console.error("No image selected");
            addDebugInfo("Error: No image selected");
            return false;
        }
        
        // Set default values for required fields
        setDefaultValues();
        
        // Create FormData object
        var formData = new FormData(this);
        
        // Debug form data
        console.log("Form data:");
        addDebugInfo("Form data:");
        for (var pair of formData.entries()) {
            console.log(pair[0] + ': ' + pair[1]);
            addDebugInfo("  " + pair[0] + ': ' + pair[1]);
        }
        
        // Show loading indicator
        $('#processed-container').hide();
        $('#loading').remove();
        $('<div id="loading"><div class="spinner"></div><span>Processing image...</span></div>').insertAfter('#image-form');
        addDebugInfo("Added loading indicator");
        
        // Disable the submit button to prevent multiple submissions
        $('#process-button').prop('disabled', true).css('opacity', '0.5');
        addDebugInfo("Disabled submit button");
        
        console.log("Submitting form via AJAX...");
        addDebugInfo("Submitting form via AJAX...");
        
        // Send the AJAX request
        $.ajax({
            url: '/',
            type: 'POST',
            data: formData,
            dataType: 'json',
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            },
            success: function(data) {
                console.log("AJAX success:", data);
                addDebugInfo("AJAX success: " + JSON.stringify(data));
                
                // Remove loading indicator
                $('#loading').remove();
                
                // Re-enable the submit button
                $('#process-button').prop('disabled', false).css('opacity', '1');
                addDebugInfo("Re-enabled submit button");
                
                if (data.success) {
                    // Update the processed image preview with the URLs from the JSON response
                    if (data.processed_url) {
                        console.log("Setting processed image URL:", data.processed_url);
                        addDebugInfo("Setting processed image URL: " + data.processed_url);
                        
                        // Show notification if image was resized
                        if (data.was_resized) {
                            console.log("Image was automatically resized from", data.original_size, "to", data.new_size);
                            addDebugInfo("Image was automatically resized from " + data.original_size + " to " + data.new_size);
                            
                            // Add a notification about the resize
                            var resizeMsg = '<div class="alert alert-info resize-notification">Large image detected. Image was automatically resized from ' + 
                                            data.original_size + ' to ' + data.new_size + ' for better performance (longest dimension limited to 1920px).</div>';
                            $('#image-form').before(resizeMsg);
                            
                            // Set a timeout to remove the notification after 10 seconds
                            setTimeout(function() {
                                $('.resize-notification').fadeOut(500, function() { $(this).remove(); });
                            }, 10000);
                        }
                        
                        // Force browser to reload the image by adding a timestamp
                        var timestamp = new Date().getTime();
                        var processedUrl = data.processed_url + '?t=' + timestamp;
                        addDebugInfo("Timestamped URL: " + processedUrl);
                        
                        // Create a new image element to force reload
                        var img = new Image();
                        img.onload = function() {
                            console.log("Processed image loaded successfully");
                            addDebugInfo("Processed image loaded successfully");
                            $('#processed-preview').attr('src', processedUrl);
                            $('#download-link').attr('href', data.processed_url);
                            $('#processed-container').show();
                            
                            // Scroll to the image preview
                            $('html, body').animate({
                                scrollTop: $('#image-preview').offset().top
                            }, 500);
                        };
                        img.onerror = function() {
                            console.error("Failed to load processed image:", processedUrl);
                            addDebugInfo("Error: Failed to load processed image: " + processedUrl);
                            $('#image-form').before('<div class="error-message global-error">Error: Failed to load processed image</div>');
                        };
                        img.src = processedUrl;
                    } else {
                        console.error("No processed_url in response:", data);
                        addDebugInfo("Error: No processed_url in response");
                        $('#image-form').before('<div class="error-message global-error">Error: No processed image URL in response</div>');
                    }
                } else if (data.error) {
                    console.error("Error in response:", data.error);
                    addDebugInfo("Error in response: " + data.error);
                    displayErrors(data);
                }
            },
            error: function(xhr, status, error) {
                console.error("AJAX error:", status, error);
                console.error("Response:", xhr.responseText);
                addDebugInfo("AJAX error: " + status + " - " + error);
                addDebugInfo("Response: " + xhr.responseText);
                
                $('#loading').remove();
                
                // Re-enable the submit button
                $('#process-button').prop('disabled', false).css('opacity', '1');
                
                try {
                    // Try to parse the error response as JSON
                    const errorData = JSON.parse(xhr.responseText);
                    displayErrors(errorData);
                } catch (e) {
                    // If parsing fails, use the original error
                    $('#image-form').before('<div class="error-message global-error">Error processing image: ' + error + '</div>');
                    addDebugInfo("Failed to parse error response: " + e.message);
                }
            },
            cache: false,
            contentType: false,
            processData: false
        });
        
        return false; // Prevent default form submission
    });
    
    // Trigger initial change to set visibility based on default selection
    $('#effect').trigger('change');
    
    // Add a direct click handler to the process button as a fallback
    $('#process-button').click(function(e) {
        console.log("Process button clicked directly");
        addDebugInfo("Process button clicked directly");
        
        e.preventDefault();
        
        // Manually trigger the form submission logic
        var form = $('#image-form')[0];
        
        console.log("Form submitted via button click");
        addDebugInfo("Form submitted via button click");
        clearErrors();
        
        console.log("Form ID:", form.id);
        console.log("Form action:", $(form).attr('action'));
        console.log("Form method:", $(form).attr('method'));
        console.log("Form enctype:", $(form).attr('enctype'));
        
        addDebugInfo("Form ID: " + form.id);
        addDebugInfo("Form action: " + $(form).attr('action'));
        addDebugInfo("Form method: " + $(form).attr('method'));
        addDebugInfo("Form enctype: " + $(form).attr('enctype'));
        
        // Check if an image is selected
        if (!$('#primary-image-input')[0].files.length) {
            $('#image-form').before('<div class="error-message global-error">Please select an image to process</div>');
            console.error("No image selected");
            addDebugInfo("Error: No image selected");
            return false;
        }
        
        // Set default values for required fields
        setDefaultValues();
        
        // Create FormData object
        var formData = new FormData(form);
        
        // Ensure we have the latest CSRF token
        var csrftoken = $('input[name=csrf_token]').val();
        console.log("Using CSRF token:", csrftoken);
        addDebugInfo("Using CSRF token: " + csrftoken);
        
        // Debug form data
        console.log("Form data:");
        addDebugInfo("Form data:");
        for (var pair of formData.entries()) {
            console.log(pair[0] + ': ' + pair[1]);
            addDebugInfo("  " + pair[0] + ': ' + pair[1]);
        }
        
        // Show loading indicator
        $('#processed-container').hide();
        $('#loading').remove();
        $('<div id="loading"><div class="spinner"></div><span>Processing image...</span></div>').insertAfter('#image-form');
        addDebugInfo("Added loading indicator");
        
        // Disable the submit button to prevent multiple submissions
        $('#process-button').prop('disabled', true).css('opacity', '0.5');
        addDebugInfo("Disabled submit button");
        
        console.log("Submitting form via AJAX...");
        addDebugInfo("Submitting form via AJAX...");
        
        // Send the AJAX request
        $.ajax({
            url: '/',
            type: 'POST',
            data: formData,
            dataType: 'json',
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            },
            success: function(data) {
                console.log("AJAX success:", data);
                addDebugInfo("AJAX success: " + JSON.stringify(data));
                
                // Remove loading indicator
                $('#loading').remove();
                
                // Re-enable the submit button
                $('#process-button').prop('disabled', false).css('opacity', '1');
                addDebugInfo("Re-enabled submit button");
                
                if (data.success) {
                    // Update the processed image preview with the URLs from the JSON response
                    if (data.processed_url) {
                        console.log("Setting processed image URL:", data.processed_url);
                        addDebugInfo("Setting processed image URL: " + data.processed_url);
                        
                        // Show notification if image was resized
                        if (data.was_resized) {
                            console.log("Image was automatically resized from", data.original_size, "to", data.new_size);
                            addDebugInfo("Image was automatically resized from " + data.original_size + " to " + data.new_size);
                            
                            // Add a notification about the resize
                            var resizeMsg = '<div class="alert alert-info resize-notification">Large image detected. Image was automatically resized from ' + 
                                            data.original_size + ' to ' + data.new_size + ' for better performance (longest dimension limited to 1920px).</div>';
                            $('#image-form').before(resizeMsg);
                            
                            // Set a timeout to remove the notification after 10 seconds
                            setTimeout(function() {
                                $('.resize-notification').fadeOut(500, function() { $(this).remove(); });
                            }, 10000);
                        }
                        
                        // Force browser to reload the image by adding a timestamp
                        var timestamp = new Date().getTime();
                        var processedUrl = data.processed_url + '?t=' + timestamp;
                        addDebugInfo("Timestamped URL: " + processedUrl);
                        
                        // Create a new image element to force reload
                        var img = new Image();
                        img.onload = function() {
                            console.log("Processed image loaded successfully");
                            addDebugInfo("Processed image loaded successfully");
                            $('#processed-preview').attr('src', processedUrl);
                            $('#download-link').attr('href', data.processed_url);
                            $('#processed-container').show();
                            
                            // Scroll to the image preview
                            $('html, body').animate({
                                scrollTop: $('#image-preview').offset().top
                            }, 500);
                        };
                        img.onerror = function() {
                            console.error("Failed to load processed image:", processedUrl);
                            addDebugInfo("Error: Failed to load processed image: " + processedUrl);
                            $('#image-form').before('<div class="error-message global-error">Error: Failed to load processed image</div>');
                        };
                        img.src = processedUrl;
                    } else {
                        console.error("No processed_url in response:", data);
                        addDebugInfo("Error: No processed_url in response");
                        $('#image-form').before('<div class="error-message global-error">Error: No processed image URL in response</div>');
                    }
                } else if (data.error) {
                    console.error("Error in response:", data.error);
                    addDebugInfo("Error in response: " + data.error);
                    displayErrors(data);
                }
            },
            error: function(xhr, status, error) {
                console.error("AJAX error:", status, error);
                console.error("Response:", xhr.responseText);
                addDebugInfo("AJAX error: " + status + " - " + error);
                addDebugInfo("Response: " + xhr.responseText);
                
                $('#loading').remove();
                
                // Re-enable the submit button
                $('#process-button').prop('disabled', false).css('opacity', '1');
                
                try {
                    // Try to parse the error response as JSON
                    const errorData = JSON.parse(xhr.responseText);
                    displayErrors(errorData);
                } catch (e) {
                    // If parsing fails, use the original error
                    $('#image-form').before('<div class="error-message global-error">Error processing image: ' + error + '</div>');
                    addDebugInfo("Failed to parse error response: " + e.message);
                }
            },
            cache: false,
            contentType: false,
            processData: false
        });
    });
    
    // Add a handler for the "Use as Input Image" button
    $('#use-processed-button').click(function() {
        console.log("Use processed image as input button clicked");
        addDebugInfo("Use processed image as input button clicked");
        
        // Get the processed image URL
        var processedUrl = $('#processed-preview').attr('src');
        if (!processedUrl) {
            console.error("No processed image URL found");
            addDebugInfo("Error: No processed image URL found");
            return;
        }
        
        // Create a new Image to load the processed image
        var img = new Image();
        img.crossOrigin = "Anonymous"; // Enable CORS for the image
        
        img.onload = function() {
            console.log("Processed image loaded for conversion");
            addDebugInfo("Processed image loaded for conversion");
            
            // Create a canvas to convert the image to a Blob
            var canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            var ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);
            
            // Convert the canvas to a Blob
            canvas.toBlob(function(blob) {
                // Create a File object from the Blob
                var file = new File([blob], "processed_image.png", { type: "image/png" });
                
                // Create a FileList-like object
                var dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                
                // Set the file input's files property
                $('#primary-image-input')[0].files = dataTransfer.files;
                
                // Trigger the change event to update the preview
                $('#primary-image-input').trigger('change');
                
                console.log("Processed image set as input");
                addDebugInfo("Processed image set as input");
            }, 'image/png');
        };
        
        img.onerror = function() {
            console.error("Failed to load processed image for conversion");
            addDebugInfo("Error: Failed to load processed image for conversion");
            $('#image-form').before('<div class="error-message global-error">Error: Failed to load processed image</div>');
        };
        
        // Load the image
        img.src = processedUrl;
    });
    
    function updateHistogramGlitchUI() {
        // Get the current modes
        var rMode = $('[name="hist_r_mode"]').val();
        var gMode = $('[name="hist_g_mode"]').val();
        var bMode = $('[name="hist_b_mode"]').val();
        
        // Show/hide relevant fields based on modes
        if (rMode === 'solarize') {
            $('[name="hist_r_freq"], [name="hist_r_phase"]').closest('.form-group').show();
        } else {
            $('[name="hist_r_freq"], [name="hist_r_phase"]').closest('.form-group').hide();
        }
        
        if (gMode === 'solarize') {
            $('[name="hist_g_freq"], [name="hist_g_phase"]').closest('.form-group').show();
        } else {
            $('[name="hist_g_freq"], [name="hist_g_phase"]').closest('.form-group').hide();
        }
        
        if (bMode === 'solarize') {
            $('[name="hist_b_freq"], [name="hist_b_phase"]').closest('.form-group').show();
        } else {
            $('[name="hist_b_freq"], [name="hist_b_phase"]').closest('.form-group').hide();
        }
        
        // Show/hide gamma field
        if (rMode === 'gamma' || gMode === 'gamma' || bMode === 'gamma') {
            $('[name="hist_gamma"]').closest('.form-group').show();
        } else {
            $('[name="hist_gamma"]').closest('.form-group').hide();
        }
    }
    
    // Update histogram fields when modes change
    $('[name="hist_r_mode"], [name="hist_g_mode"], [name="hist_b_mode"]').change(function() {
        updateHistogramGlitchUI();
    });
    
    // Handle mask type selection for Masked Merge effect
    $('#mask_type').change(function() {
        var type = $(this).val();
        console.log("Mask type changed to:", type);
        addDebugInfo("Mask type changed to: " + type);
        
        // Hide all mask-specific fields first
        $('#checkerboard_fields, #mask_random_seed_field, #striped_fields, #gradient_description, #perlin_fields, #voronoi_fields, #concentric_rectangles_fields, #concentric_circles_fields, #gradient_direction_field, #random_triangles_fields').hide();
        
        // Show appropriate fields based on mask type
        if (type === 'checkerboard') {
            $('#checkerboard_fields').show();
        } else if (type === 'random_checkerboard') {
            $('#checkerboard_fields').show();
            $('#mask_random_seed_field').show();
        } else if (type === 'striped') {
            $('#striped_fields').show();
        } else if (type === 'gradient_striped') {
            $('#striped_fields').show();
            $('#gradient_description').show();
        } else if (type === 'linear_gradient_striped') {
            $('#striped_fields').show();
            $('#gradient_direction_field').show();
        } else if (type === 'perlin') {
            $('#perlin_fields').show();
            $('#mask_random_seed_field').show();
        } else if (type === 'voronoi') {
            $('#voronoi_fields').show();
        } else if (type === 'concentric_rectangles') {
            $('#concentric_rectangles_fields').show();
        } else if (type === 'concentric_circles') {
            $('#concentric_circles_fields').show();
        } else if (type === 'random_triangles') {
            $('#random_triangles_fields').show();
            $('#mask_random_seed_field').show();
        }
    });
    
    // Handle bit manipulation type selection to show/hide relevant fields
    $('#bit_manipulation_type').change(function() {
        var type = $(this).val();
        console.log("Bit manipulation type changed to:", type);
        addDebugInfo("Bit manipulation type changed to: " + type);
        
        // Show/hide fields based on the selected manipulation type
        if (type === 'xor') {
            $('#bit_xor_value').closest('.form-group').show();
            $('#bit_shift').closest('.form-group').hide();
        } else if (type === 'shift') {
            $('#bit_xor_value').closest('.form-group').hide();
            $('#bit_shift').closest('.form-group').show();
        } else if (type === 'swap' || type === 'invert') {
            $('#bit_xor_value').closest('.form-group').hide();
            $('#bit_shift').closest('.form-group').hide();
        }
    });
    
    console.log("Glitch Art app initialized");
    addDebugInfo("Glitch Art app initialized");
});