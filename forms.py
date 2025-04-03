from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SelectField, StringField, FloatField, IntegerField, SubmitField
from wtforms.validators import DataRequired, Optional, ValidationError, NumberRange
import re
import logging

# Configure logging
logger = logging.getLogger(__name__)

def validate_chunk_size(form, field):
    """Validate that chunk size is in the format NxN where N is a number."""
    if not field.data:
        return
    pattern = re.compile(r'^\d+x\d+$')
    if not pattern.match(field.data):
        raise ValidationError('Chunk size must be in the format NxN (e.g., 32x32)')

def validate_multiple_of_8(form, field):
    """Validate that the value is a multiple of 8."""
    if not field.data:
        return
    if field.data % 8 != 0:
        raise ValidationError('Value must be a multiple of 8 (e.g., 8, 16, 24, 32, etc.)')

class ImageProcessForm(FlaskForm):
    """Form for image upload and effect selection."""
    primary_image = FileField('Primary Image', validators=[
        FileRequired(),
        FileAllowed(['jpg', 'jpeg', 'png', 'gif'], 'Images only!')
    ])
    effect = SelectField('Effect', choices=[
        # Full Frame Sorting
        ('full_frame_sort', 'Full Frame Sorting'),
        ('perlin_full_frame', 'Perlin Full Frame'),
        # Chunked Sorting
        ('pixel_sort_chunk', 'Pixel Sort Chunk'),
        ('polar_sort', 'Polar Sorting'),
        ('spiral_sort_2', 'Spiral Sort 2'),
        ('voronoi_sort', 'Voronoi Pixel Sort'),
        ('perlin_noise_sort', 'Perlin Noise Sorting'),
        # Glitchy/Distortion Effects
        ('pixel_drift', 'Pixel Drift'),
        ('bit_manipulation', 'Bit Manipulation'),
        ('data_mosh_blocks', 'Data Mosh Blocks'),
        ('jpeg_artifacts', 'JPEG Artifacts'),
        ('pixel_scatter', 'Pixel Scatter'),
        ('databend', 'Databending'),
        ('perlin_displacement', 'Perlin Displacement'),
        ('ripple', 'Ripple Effect'),
        # Color/Pattern Effects
        ('color_channel', 'Color Channel Manipulation'),
        ('channel_shift', 'RGB Channel Shift'),
        ('histogram_glitch', 'Histogram Glitch'),
        ('color_shift_expansion', 'Color Shift Expansion'),
        ('pixelate', 'Pixelate'),
        ('concentric_shapes', 'Concentric Shapes'),
        # Merge Effects
        ('double_expose', 'Double Expose'),
        ('masked_merge', 'Masked Merge')
    ], validators=[DataRequired()])
    
    # Pixel Sort Chunk (Combined pixel_sort_original and pixel_sort_corner)
    chunk_width = IntegerField('Chunk Width', 
                              default=48,
                              validators=[Optional(), NumberRange(min=8, max=2048), validate_multiple_of_8])
    chunk_height = IntegerField('Chunk Height', 
                               default=48,
                               validators=[Optional(), NumberRange(min=8, max=2048), validate_multiple_of_8])
    sort_by = SelectField('Sort By', choices=[
        ('color', 'Color (R+G+B)'), 
        ('brightness', 'Brightness'), 
        ('hue', 'Hue'),
        ('red', 'Red Channel'),
        ('green', 'Green Channel'),
        ('blue', 'Blue Channel'),
        ('saturation', 'Saturation'),
        ('luminance', 'Luminance'),
        ('contrast', 'Contrast')
    ], default='brightness', validators=[Optional()])
    sort_order = SelectField('Sort Order', choices=[
        ('ascending', 'Ascending (Low to High)'), 
        ('descending', 'Descending (High to Low)')
    ], default='descending', validators=[Optional()])
    sort_mode = SelectField('Sort Mode', choices=[
        ('horizontal', 'Horizontal'), 
        ('vertical', 'Vertical'),
        ('diagonal', 'Diagonal')
    ], default='horizontal', validators=[Optional()])
    starting_corner = SelectField('Starting Corner (for Diagonal mode)', choices=[
        ('top-left', 'Top-Left'), 
        ('top-right', 'Top-Right'),
        ('bottom-left', 'Bottom-Left'), 
        ('bottom-right', 'Bottom-Right')
    ], default='top-left', validators=[Optional()])
    
    # Color Channel Manipulation
    manipulation_type = SelectField('Manipulation Type', choices=[
        ('swap', 'Swap Channels'), 
        ('invert', 'Invert Channel'), 
        ('adjust', 'Adjust Channel Intensity'),
        ('negative', 'Negative (Invert All Channels)')
    ], default='swap', validators=[Optional()])
    swap_choice = SelectField('Swap Channels', choices=[
        ('red-green', 'Red-Green'), 
        ('red-blue', 'Red-Blue'), 
        ('green-blue', 'Green-Blue')
    ], default='red-green', validators=[Optional()])
    invert_choice = SelectField('Invert Channel', choices=[
        ('red', 'Red'), 
        ('green', 'Green'), 
        ('blue', 'Blue')
    ], default='red', validators=[Optional()])
    adjust_choice = SelectField('Adjust Channel', choices=[
        ('red', 'Red'), 
        ('green', 'Green'), 
        ('blue', 'Blue')
    ], default='red', validators=[Optional()])
    intensity_factor = FloatField('Intensity Factor', 
                                 default=1.5,
                                 validators=[Optional(), NumberRange(min=0.1, max=10.0)])
    
    # Data Moshing
    secondary_image = FileField('Secondary Image', validators=[
        Optional(),
        FileAllowed(['jpg', 'jpeg', 'png', 'gif'], 'Images only!')
    ])
    blend_mode = SelectField('Blend Mode', choices=[
        ('classic', 'Classic Blend'),
        ('screen', 'Screen'),
        ('multiply', 'Multiply'),
        ('overlay', 'Overlay'),
        ('difference', 'Difference'),
        ('color_dodge', 'Color Dodge')
    ], default='classic', validators=[Optional()])
    opacity = FloatField('Opacity', 
                        default=0.5,
                        validators=[Optional(), NumberRange(min=0.1, max=1.0)])
    
    # Pixel Drift
    drift_direction = SelectField('Drift Direction', choices=[
        ('up', 'Up'), 
        ('down', 'Down'), 
        ('left', 'Left'), 
        ('right', 'Right')
    ], default='right', validators=[Optional()])
    drift_bands = IntegerField('Number of Bands', 
                              default=12,
                              validators=[Optional(), NumberRange(min=1, max=48)])
    drift_intensity = FloatField('Drift Intensity Multiplier', 
                               default=4.0,
                               validators=[Optional(), NumberRange(min=0.1, max=10.0)])
    
    # Spiral Sort 2 (Replaces original Spiral Sort)
    spiral2_chunk_size = IntegerField('Chunk Size', 
                                   default=64,
                                   validators=[Optional(), NumberRange(min=8, max=128), validate_multiple_of_8])
    spiral2_sort_by = SelectField('Sort By', choices=[
        ('color', 'Color (R+G+B)'), 
        ('brightness', 'Brightness'),
        ('hue', 'Hue'),
        ('red', 'Red Channel'),
        ('green', 'Green Channel'),
        ('blue', 'Blue Channel'),
        ('saturation', 'Saturation'),
        ('luminance', 'Luminance'),
        ('contrast', 'Contrast')
    ], default='brightness', validators=[Optional()])
    spiral2_reverse = SelectField('Sort Order', choices=[
        ('false', 'Ascending (Low to High)'), 
        ('true', 'Descending (High to Low)')
    ], default='false', validators=[Optional()])
    
    # Bit Manipulation
    bit_chunk_size = IntegerField('Chunk Size', 
                                default=24,
                                validators=[Optional(), NumberRange(min=8, max=128), validate_multiple_of_8])
    
    # Full Frame Sorting
    full_frame_direction = SelectField('Direction', choices=[
        ('vertical', 'Vertical'), 
        ('horizontal', 'Horizontal')
    ], default='vertical', validators=[Optional()])
    full_frame_sort_by = SelectField('Sort By', choices=[
        ('color', 'Color (R+G+B)'), 
        ('brightness', 'Brightness'),
        ('hue', 'Hue'),
        ('red', 'Red Channel'),
        ('green', 'Green Channel'),
        ('blue', 'Blue Channel'),
        ('saturation', 'Saturation'),
        ('luminance', 'Luminance'),
        ('contrast', 'Contrast')
    ], default='hue', validators=[Optional()])
    full_frame_reverse = SelectField('Sort Order', choices=[
        ('false', 'Ascending (Low to High)'), 
        ('true', 'Descending (High to Low)')
    ], default='true', validators=[Optional()])
    
    # Polar Sorting
    polar_chunk_size = IntegerField('Chunk Size', 
                                  default=64,
                                  validators=[Optional(), NumberRange(min=8, max=128), validate_multiple_of_8])
    polar_sort_by = SelectField('Sort By', choices=[
        ('angle', 'Angle (around center)'), 
        ('radius', 'Radius (distance from center)')
    ], default='radius', validators=[Optional()])
    polar_reverse = SelectField('Sort Order', choices=[
        ('false', 'Ascending (Low to High)'), 
        ('true', 'Descending (High to Low)')
    ], default='true', validators=[Optional()])
    
    # Perlin Noise Sorting
    perlin_chunk_width = IntegerField('Chunk Width', 
                                    default=128,
                                    validators=[Optional(), NumberRange(min=8, max=1024), validate_multiple_of_8])
    perlin_chunk_height = IntegerField('Chunk Height', 
                                     default=1024,
                                     validators=[Optional(), NumberRange(min=8, max=1024), validate_multiple_of_8])
    perlin_noise_scale = FloatField('Noise Scale', 
                                  default=0.008,
                                  validators=[Optional(), NumberRange(min=0.001, max=0.1)])
    perlin_seed = IntegerField('Noise Seed', 
                             default=420,
                             validators=[Optional(), NumberRange(min=1, max=9999)])
    perlin_direction = SelectField('Direction', choices=[
        ('horizontal', 'Horizontal'), 
        ('vertical', 'Vertical')
    ], default='horizontal', validators=[Optional()])
    perlin_reverse = SelectField('Sort Order', choices=[
        ('false', 'Ascending (Low to High)'), 
        ('true', 'Descending (High to Low)')
    ], default='true', validators=[Optional()])
    
    # Perlin Full Frame
    perlin_full_frame_noise_scale = FloatField('Noise Scale', 
                                            default=0.005,
                                            validators=[Optional(), NumberRange(min=0.001, max=0.1)])
    perlin_full_frame_sort_by = SelectField('Sort By', choices=[
        ('color', 'Color (R+G+B)'), 
        ('brightness', 'Brightness'),
        ('hue', 'Hue'),
        ('red', 'Red Channel'),
        ('green', 'Green Channel'),
        ('blue', 'Blue Channel'),
        ('saturation', 'Saturation'),
        ('luminance', 'Luminance'),
        ('contrast', 'Contrast')
    ], default='hue', validators=[Optional()])
    perlin_full_frame_reverse = SelectField('Sort Order', choices=[
        ('false', 'Ascending (Low to High)'), 
        ('true', 'Descending (High to Low)')
    ], default='false', validators=[Optional()])
    perlin_full_frame_seed = IntegerField('Noise Seed',
                                        default=69,
                                        validators=[Optional(), NumberRange(min=1, max=9999)])
    
    # Pixelate
    pixelate_width = IntegerField('Pixel Width',
                                default=16,
                                validators=[Optional(), NumberRange(min=2, max=64)])
    pixelate_height = IntegerField('Pixel Height',
                                 default=16,
                                 validators=[Optional(), NumberRange(min=2, max=64)])
    pixelate_attribute = SelectField('Attribute', choices=[
        ('color', 'Color (Most Common)'),
        ('brightness', 'Brightness'),
        ('hue', 'Hue'),
        ('saturation', 'Saturation'),
        ('luminance', 'Luminance')
    ], default='hue', validators=[Optional()])
    pixelate_bins = IntegerField('Number of Bins',
                               default=200,
                               validators=[Optional(), NumberRange(min=10, max=1000)])
    
    # Concentric Shapes
    concentric_num_points = IntegerField('Number of Points',
                                      default=7,
                                      validators=[Optional(), NumberRange(min=1, max=100)])
    shape_type = SelectField('Shape Type', choices=[
        ('square', 'Square'),
        ('circle', 'Circle'),
        ('hexagon', 'Hexagon'),
        ('triangle', 'Triangle')
    ], default='triangle', validators=[Optional()])
    concentric_thickness = IntegerField('Line Thickness',
                                     default=2,
                                     validators=[Optional(), NumberRange(min=1, max=10)])
    spacing = IntegerField('Spacing',
                         default=20,
                         validators=[Optional(), NumberRange(min=1, max=50)])
    rotation_angle = IntegerField('Rotation Angle',
                                default=9,
                                validators=[Optional(), NumberRange(min=0, max=360)])
    darken_step = IntegerField('Darken Step',
                             default=0,
                             validators=[Optional(), NumberRange(min=0, max=255)])
    color_shift = IntegerField('Color Shift',
                             default=15,
                             validators=[Optional(), NumberRange(min=0, max=360)])
    
    # Color Shift Expansion
    color_shift_num_points = IntegerField('Number of Seed Points',
                                 default=7,
                                 validators=[Optional(), NumberRange(min=1, max=100)])
    color_shift_amount = IntegerField('Shift Amount',
                                 default=7,
                                 validators=[Optional(), NumberRange(min=1, max=10)])
    expansion_type = SelectField('Expansion Type', choices=[
        ('square', 'Square (8 directions)'),
        ('cross', 'Cross (4 directions)'),
        ('circular', 'Circular')
    ], default='circular', validators=[Optional()])
    pattern_type = SelectField('Seed Point Pattern', choices=[
        ('random', 'Random'),
        ('grid', 'Grid'),
        ('radial', 'Radial'),
        ('spiral', 'Spiral')
    ], default='random', validators=[Optional()])
    color_theme = SelectField('Color Theme', choices=[
        ('full-spectrum', 'Full Spectrum'),
        ('warm', 'Warm Colors'),
        ('cool', 'Cool Colors'),
        ('complementary', 'Complementary'),
        ('analogous', 'Analogous')
    ], default='complementary', validators=[Optional()])
    saturation_boost = FloatField('Saturation Boost',
                              default=0.5,
                              validators=[Optional(), NumberRange(min=0.0, max=1.0)])
    value_boost = FloatField('Brightness Boost',
                         default=0.0,
                         validators=[Optional(), NumberRange(min=0.0, max=1.0)])
    decay_factor = FloatField('Decay Factor',
                          default=0.2,
                          validators=[Optional(), NumberRange(min=0.0, max=1.0)])
    
    # Perlin Displacement
    perlin_displacement_scale = IntegerField('Noise Scale',
                                          default=200,
                                          validators=[Optional(), NumberRange(min=10, max=500)])
    perlin_displacement_intensity = IntegerField('Displacement Intensity',
                                              default=40,
                                              validators=[Optional(), NumberRange(min=1, max=100)])
    perlin_displacement_octaves = IntegerField('Octaves',
                                            default=4,
                                            validators=[Optional(), NumberRange(min=1, max=10)])
    perlin_displacement_persistence = FloatField('Persistence',
                                              default=0.6,
                                              validators=[Optional(), NumberRange(min=0.1, max=1.0)])
    perlin_displacement_lacunarity = FloatField('Lacunarity',
                                             default=2.0,
                                             validators=[Optional(), NumberRange(min=1.0, max=4.0)])
    
    # Add form fields for Data Mosh Blocks effect
    data_mosh_operations = IntegerField('Number of Operations', 
                                      default=69,
                                      validators=[Optional(), NumberRange(min=1, max=256)])
    data_mosh_block_size = IntegerField('Max Block Size', 
                                      default=128,
                                      validators=[Optional(), NumberRange(min=1, max=500)])
    data_mosh_movement = SelectField('Block Movement', choices=[
        ('swap', 'Swap Blocks'), 
        ('in_place', 'In Place')
    ], default='swap', validators=[Optional()])
    data_mosh_color_swap = SelectField('Color Channel Swap', choices=[
        ('never', 'Never'), 
        ('always', 'Always'),
        ('random', 'Random')
    ], default='random', validators=[Optional()])
    data_mosh_invert = SelectField('Color Inversion', choices=[
        ('never', 'Never'), 
        ('always', 'Always'),
        ('random', 'Random')
    ], default='random', validators=[Optional()])
    data_mosh_shift = SelectField('Channel Value Shift', choices=[
        ('never', 'Never'), 
        ('always', 'Always'),
        ('random', 'Random')
    ], default='random', validators=[Optional()])
    data_mosh_flip = SelectField('Block Flipping', choices=[
        ('never', 'Never'), 
        ('vertical', 'Vertical'),
        ('horizontal', 'Horizontal'),
        ('random', 'Random')
    ], default='random', validators=[Optional()])
    data_mosh_seed = IntegerField('Random Seed', 
                                validators=[Optional(), NumberRange(min=1, max=9999)])
    
    # Voronoi Pixel Sort
    voronoi_num_cells = IntegerField('Number of Cells',
                                  default=69,
                                  validators=[Optional(), NumberRange(min=10, max=1000)])
    voronoi_size_variation = FloatField('Size Variation',
                                     default=0.8,
                                     validators=[Optional(), NumberRange(min=0.0, max=1.0)])
    voronoi_sort_by = SelectField('Sort By', choices=[
        ('color', 'Color (R+G+B)'), 
        ('brightness', 'Brightness'),
        ('hue', 'Hue'),
        ('red', 'Red Channel'),
        ('green', 'Green Channel'),
        ('blue', 'Blue Channel'),
        ('saturation', 'Saturation'),
        ('luminance', 'Luminance'),
        ('contrast', 'Contrast')
    ], default='hue', validators=[Optional()])
    voronoi_sort_order = SelectField('Sort Order', choices=[
        ('clockwise', 'Clockwise'), 
        ('counter-clockwise', 'Counter-Clockwise')
    ], default='clockwise', validators=[Optional()])
    voronoi_orientation = SelectField('Line Orientation', choices=[
        ('horizontal', 'Horizontal'),
        ('vertical', 'Vertical'),
        ('radial', 'Radial'),
        ('spiral', 'Spiral')
    ], default='spiral', validators=[Optional()])
    voronoi_start_position = SelectField('Start Position', choices=[
        ('left', 'Left'),
        ('right', 'Right'),
        ('top', 'Top'),
        ('bottom', 'Bottom'),
        ('center', 'Center')
    ], default='center', validators=[Optional()])
    voronoi_seed = IntegerField('Random Seed',
                             default=420,
                             validators=[Optional(), NumberRange(min=1, max=9999)])
    
    # RGB Channel Shift
    channel_shift_amount = IntegerField('Shift Amount', 
                                   default=42,
                                   validators=[Optional(), NumberRange(min=1, max=500)])
    channel_shift_direction = SelectField('Shift Direction', choices=[
        ('horizontal', 'Horizontal'), 
        ('vertical', 'Vertical')
    ], default='horizontal', validators=[Optional()])
    channel_shift_center = SelectField('Centered Channel', choices=[
        ('red', 'Red'), 
        ('green', 'Green'), 
        ('blue', 'Blue')
    ], default='green', validators=[Optional()])
    channel_mode = SelectField('Mode', choices=[
        ('shift', 'Shift Channels'), 
        ('mirror', 'Mirror Channels')
    ], default='shift', validators=[Optional()])
    
    # JPEG Artifacts
    jpeg_intensity = FloatField('Artifact Intensity', 
                               default=1.0,
                               validators=[Optional(), NumberRange(min=0.0, max=1.0)])
    
    # Pixel Scatter
    scatter_direction = SelectField('Scatter Direction', choices=[
        ('horizontal', 'Horizontal'),
        ('vertical', 'Vertical')
    ], default='horizontal', validators=[Optional()])
    
    scatter_select_by = SelectField('Select Pixels By', choices=[
        ('brightness', 'Brightness'),
        ('red', 'Red Channel'),
        ('green', 'Green Channel'),
        ('blue', 'Blue Channel'),
        ('hue', 'Hue'),
        ('saturation', 'Saturation'),
        ('luminance', 'Luminance'),
        ('contrast', 'Contrast')
    ], default='red', validators=[Optional()])
    
    scatter_min_value = FloatField('Minimum Value',
                              default=180,
                              validators=[Optional(), NumberRange(min=0, max=360)])
    
    scatter_max_value = FloatField('Maximum Value',
                              default=360,
                              validators=[Optional(), NumberRange(min=0, max=360)])
    
    # Databending
    databend_intensity = FloatField('Databend Intensity',
                                 default=0.1,
                                 validators=[Optional(), NumberRange(min=0.1, max=1.0)])
    
    databend_preserve_header = SelectField('Preserve Header', choices=[
        ('true', 'Yes (More Stable)'),
        ('false', 'No (More Glitchy)')
    ], default='true', validators=[Optional()])
    
    databend_seed = IntegerField('Random Seed',
                               default=42,
                               validators=[Optional(), NumberRange(min=1, max=9999)])
    
    # Histogram Glitch
    hist_r_mode = SelectField('Red Channel Treatment', choices=[
        ('solarize', 'Solarize (Sine Wave)'),
        ('log', 'Logarithmic Compression'),
        ('gamma', 'Gamma Adjustment'),
        ('normal', 'Normal (No Change)')
    ], default='solarize', validators=[Optional()])
    
    hist_g_mode = SelectField('Green Channel Treatment', choices=[
        ('solarize', 'Solarize (Sine Wave)'),
        ('log', 'Logarithmic Compression'),
        ('gamma', 'Gamma Adjustment'),
        ('normal', 'Normal (No Change)')
    ], default='log', validators=[Optional()])
    
    hist_b_mode = SelectField('Blue Channel Treatment', choices=[
        ('solarize', 'Solarize (Sine Wave)'),
        ('log', 'Logarithmic Compression'),
        ('gamma', 'Gamma Adjustment'),
        ('normal', 'Normal (No Change)')
    ], default='gamma', validators=[Optional()])
    
    hist_r_freq = FloatField('Red Solarize Frequency',
                           default=1.0,
                           validators=[Optional(), NumberRange(min=0.1, max=10.0)])
    
    hist_r_phase = FloatField('Red Solarize Phase',
                            default=0.0,
                            validators=[Optional(), NumberRange(min=0.0, max=6.28)])
    
    hist_g_freq = FloatField('Green Solarize Frequency',
                           default=1.0,
                           validators=[Optional(), NumberRange(min=0.1, max=10.0)])
    
    hist_g_phase = FloatField('Green Solarize Phase',
                            default=0.0,
                            validators=[Optional(), NumberRange(min=0.0, max=6.28)])
    
    hist_b_freq = FloatField('Blue Solarize Frequency',
                           default=1.0,
                           validators=[Optional(), NumberRange(min=0.1, max=10.0)])
    
    hist_b_phase = FloatField('Blue Solarize Phase',
                            default=0.0,
                            validators=[Optional(), NumberRange(min=0.0, max=6.28)])
    
    hist_gamma = FloatField('Gamma Value',
                          default=0.5,
                          validators=[Optional(), NumberRange(min=0.1, max=3.0)])
    
    # Ripple Effect
    ripple_num_droplets = IntegerField('Number of Droplets',
                                     default=9,
                                     validators=[Optional(), NumberRange(min=1, max=20)])
    
    ripple_amplitude = FloatField('Ripple Amplitude',
                                default=45.0,
                                validators=[Optional(), NumberRange(min=1.0, max=50.0)])
                                
    ripple_frequency = FloatField('Ripple Frequency',
                               default=0.4,
                               validators=[Optional(), NumberRange(min=0.01, max=1.0)])
                               
    ripple_decay = FloatField('Ripple Decay',
                           default=0.004,
                           validators=[Optional(), NumberRange(min=0.001, max=0.1)])
                           
    ripple_distortion_type = SelectField('Distortion Type', choices=[
        ('color_shift', 'Color Shift'),
        ('pixelation', 'Pixelation'),
        ('none', 'None (Plain Ripple)')
    ], default='color_shift', validators=[Optional()])
    
    ripple_color_r = FloatField('Red Channel Factor',
                             default=0.8,
                             validators=[Optional(), NumberRange(min=0.5, max=1.5)])
                             
    ripple_color_g = FloatField('Green Channel Factor',
                             default=1.0,
                             validators=[Optional(), NumberRange(min=0.5, max=1.5)])
                             
    ripple_color_b = FloatField('Blue Channel Factor',
                             default=1.2,
                             validators=[Optional(), NumberRange(min=0.5, max=1.5)])
                             
    ripple_pixelation_scale = IntegerField('Pixelation Scale',
                                        default=10,
                                        validators=[Optional(), NumberRange(min=2, max=20)])
                                        
    ripple_pixelation_max_mag = FloatField('Pixelation Magnitude',
                                        default=10.0,
                                        validators=[Optional(), NumberRange(min=1.0, max=20.0)])
    
    # Masked Merge
    masked_merge_secondary = FileField('Secondary Image', validators=[
        Optional(),
        FileAllowed(['jpg', 'jpeg', 'png', 'gif'], 'Images only!')
    ])
    mask_type = SelectField('Mask Type', choices=[
        ('checkerboard', 'Checkerboard'),
        ('random_checkerboard', 'Random Checkerboard'),
        ('striped', 'Striped'),
        ('gradient_striped', 'Gradient Striped'),
        ('perlin', 'Perlin Noise'),
        ('voronoi', 'Voronoi Cells'),
        ('concentric_rectangles', 'Concentric Rectangles')
    ], default='checkerboard', validators=[Optional()])
    mask_width = IntegerField('Width (pixels)',
                          default=32,
                          validators=[Optional(), NumberRange(min=8, max=512)])
    mask_height = IntegerField('Height (pixels)',
                           default=32,
                           validators=[Optional(), NumberRange(min=8, max=512)])
    mask_random_seed = IntegerField('Random Seed',
                               default=42,
                               validators=[Optional(), NumberRange(min=1, max=9999)])
    stripe_width = IntegerField('Stripe Width (pixels)',
                            default=16,
                            validators=[Optional(), NumberRange(min=1, max=200)])
    stripe_angle = IntegerField('Stripe Angle (degrees)',
                           default=45,
                           validators=[Optional(), NumberRange(min=0, max=180)])
    perlin_noise_scale = FloatField('Noise Scale',
                                default=0.01,
                                validators=[Optional(), NumberRange(min=0.001, max=0.1)])
    perlin_threshold = FloatField('Threshold',
                              default=0.5,
                              validators=[Optional(), NumberRange(min=0.0, max=1.0)])
    perlin_octaves = IntegerField('Octaves (1=smooth, 8=detailed)',
                              default=1,
                              validators=[Optional(), NumberRange(min=1, max=8)])
    voronoi_cells = IntegerField('Number of Cells',
                             default=50,
                             validators=[Optional(), NumberRange(min=10, max=500)])
    
    # Concentric Rectangles specific width
    concentric_rectangle_width = IntegerField('Band Width (pixels)',
                                           default=16, # Provide a distinct default
                                           validators=[Optional(), NumberRange(min=2, max=100)])
    
    # Submit Button (already present, no changes needed)
    submit = SubmitField('Process Image')
    
    def validate(self, extra_validators=None):
        """Custom validation based on the selected effect."""
        logger.debug(f"Validating form with effect: {self.effect.data}")
        
        # First run the standard validation
        if not super(ImageProcessForm, self).validate(extra_validators=extra_validators):
            logger.debug(f"Standard validation failed: {self.errors}")
            return False
            
        # Get the selected effect
        effect = self.effect.data
        logger.debug(f"Selected effect: {effect}")
        
        # Validate fields based on the selected effect
        if effect == 'pixel_sort_chunk':
            if not self.chunk_width.data:
                self.chunk_width.errors = ['Chunk width is required for Pixel Sort Chunk']
                logger.debug("Missing chunk_width for pixel_sort_chunk")
                return False
            if not self.chunk_height.data:
                self.chunk_height.errors = ['Chunk height is required for Pixel Sort Chunk']
                logger.debug("Missing chunk_height for pixel_sort_chunk")
                return False
            if not self.sort_by.data:
                self.sort_by.errors = ['Sort by is required for Pixel Sort Chunk']
                logger.debug("Missing sort_by for pixel_sort_chunk")
                return False
            if not self.sort_order.data:
                self.sort_order.errors = ['Sort order is required for Pixel Sort Chunk']
                logger.debug("Missing sort_order for pixel_sort_chunk")
                return False
            if not self.sort_mode.data:
                self.sort_mode.errors = ['Sort mode is required for Pixel Sort Chunk']
                logger.debug("Missing sort_mode for pixel_sort_chunk")
                return False
            
            # Validate starting_corner when sort_mode is diagonal
            if self.sort_mode.data == 'diagonal' and not self.starting_corner.data:
                self.starting_corner.errors = ['Starting corner is required for Diagonal sort mode']
                logger.debug("Missing starting_corner for diagonal sort mode")
                return False
                
        elif effect == 'full_frame_sort':
            if not self.full_frame_direction.data:
                self.full_frame_direction.errors = ['Direction is required for Full Frame Sorting']
                logger.debug("Missing full_frame_direction for full_frame_sort")
                return False
            if not self.full_frame_sort_by.data:
                self.full_frame_sort_by.errors = ['Sort by is required for Full Frame Sorting']
                logger.debug("Missing full_frame_sort_by for full_frame_sort")
                return False
            if not self.full_frame_reverse.data:
                self.full_frame_reverse.errors = ['Sort order is required for Full Frame Sorting']
                logger.debug("Missing full_frame_reverse for full_frame_sort")
                return False
                
        elif effect == 'polar_sort':
            if not self.polar_chunk_size.data:
                self.polar_chunk_size.errors = ['Chunk size is required for Polar Sorting']
                logger.debug("Missing polar_chunk_size for polar_sort")
                return False
            if not self.polar_sort_by.data:
                self.polar_sort_by.errors = ['Sort by is required for Polar Sorting']
                logger.debug("Missing polar_sort_by for polar_sort")
                return False
            if not self.polar_reverse.data:
                self.polar_reverse.errors = ['Sort order is required for Polar Sorting']
                logger.debug("Missing polar_reverse for polar_sort")
                return False
                
        elif effect == 'perlin_noise_sort':
            logger.debug("Validating perlin_noise_sort fields")
            if not self.perlin_chunk_width.data:
                self.perlin_chunk_width.errors = ['Chunk width is required for Perlin Noise Sorting']
                logger.debug("Missing perlin_chunk_width for perlin_noise_sort")
                return False
            if not self.perlin_chunk_height.data:
                self.perlin_chunk_height.errors = ['Chunk height is required for Perlin Noise Sorting']
                logger.debug("Missing perlin_chunk_height for perlin_noise_sort")
                return False
            if not self.perlin_noise_scale.data:
                self.perlin_noise_scale.errors = ['Noise scale is required for Perlin Noise Sorting']
                logger.debug("Missing perlin_noise_scale for perlin_noise_sort")
                return False
            if not self.perlin_direction.data:
                self.perlin_direction.errors = ['Direction is required for Perlin Noise Sorting']
                logger.debug("Missing perlin_direction for perlin_noise_sort")
                return False
            if not self.perlin_reverse.data:
                self.perlin_reverse.errors = ['Sort order is required for Perlin Noise Sorting']
                logger.debug("Missing perlin_reverse for perlin_noise_sort")
                return False
                
        elif effect == 'perlin_full_frame':
            logger.debug("Validating perlin_full_frame fields")
            if not self.perlin_full_frame_noise_scale.data:
                self.perlin_full_frame_noise_scale.errors = ['Noise scale is required for Perlin Full Frame']
                logger.debug("Missing perlin_full_frame_noise_scale for perlin_full_frame")
                return False
            if not self.perlin_full_frame_sort_by.data:
                self.perlin_full_frame_sort_by.errors = ['Sort by is required for Perlin Full Frame']
                logger.debug("Missing perlin_full_frame_sort_by for perlin_full_frame")
                return False
            if not self.perlin_full_frame_reverse.data:
                self.perlin_full_frame_reverse.errors = ['Sort order is required for Perlin Full Frame']
                logger.debug("Missing perlin_full_frame_reverse for perlin_full_frame")
                return False
                
        elif effect == 'pixelate':
            if not self.pixelate_width.data:
                self.pixelate_width.errors = ['Pixel width is required for Pixelate']
                logger.debug("Missing pixelate_width for pixelate")
                return False
            if not self.pixelate_height.data:
                self.pixelate_height.errors = ['Pixel height is required for Pixelate']
                logger.debug("Missing pixelate_height for pixelate")
                return False
            if not self.pixelate_attribute.data:
                self.pixelate_attribute.errors = ['Attribute is required for Pixelate']
                logger.debug("Missing pixelate_attribute for pixelate")
                return False
            if not self.pixelate_bins.data:
                self.pixelate_bins.errors = ['Number of bins is required for Pixelate']
                logger.debug("Missing pixelate_bins for pixelate")
                return False
                
        elif effect == 'concentric_shapes':
            if not self.concentric_num_points.data:
                self.concentric_num_points.errors = ['Number of points is required for Concentric Shapes']
                logger.debug("Missing concentric_num_points for concentric_shapes")
                return False
            if not self.shape_type.data:
                self.shape_type.errors = ['Shape type is required for Concentric Shapes']
                logger.debug("Missing shape_type for concentric_shapes")
                return False
            if not self.concentric_thickness.data:
                self.concentric_thickness.errors = ['Line thickness is required for Concentric Shapes']
                logger.debug("Missing concentric_thickness for concentric_shapes")
                return False
            if not self.spacing.data:
                self.spacing.errors = ['Spacing is required for Concentric Shapes']
                logger.debug("Missing spacing for concentric_shapes")
                return False
            # These parameters can be optional with defaults
            if self.rotation_angle.data is None:
                self.rotation_angle.data = 0
            if self.darken_step.data is None:
                self.darken_step.data = 0
            if self.color_shift.data is None:
                self.color_shift.data = 0
                
        elif effect == 'color_shift_expansion':
            if not self.color_shift_num_points.data:
                self.color_shift_num_points.errors = ['Number of seed points is required for Color Shift Expansion']
                logger.debug("Missing color_shift_num_points for color_shift_expansion")
                return False
            if not self.color_shift_amount.data:
                self.color_shift_amount.errors = ['Shift amount is required for Color Shift Expansion']
                logger.debug("Missing color_shift_amount for color_shift_expansion")
                return False
            if not self.expansion_type.data:
                self.expansion_type.errors = ['Expansion type is required for Color Shift Expansion']
                logger.debug("Missing expansion_type for color_shift_expansion")
                return False
            if not self.pattern_type.data:
                self.pattern_type.errors = ['Seed point pattern is required for Color Shift Expansion']
                logger.debug("Missing pattern_type for color_shift_expansion")
                return False
            if not self.color_theme.data:
                self.color_theme.errors = ['Color theme is required for Color Shift Expansion']
                logger.debug("Missing color_theme for color_shift_expansion")
                return False
            # These can have default values
            if self.saturation_boost.data is None:
                self.saturation_boost.data = 0.0
            if self.value_boost.data is None:
                self.value_boost.data = 0.0
            if self.decay_factor.data is None:
                self.decay_factor.data = 0.0
                
        elif effect == 'perlin_displacement':
            if not self.perlin_displacement_scale.data:
                self.perlin_displacement_scale.errors = ['Noise scale is required for Perlin Displacement']
                logger.debug("Missing perlin_displacement_scale for perlin_displacement")
                return False
            if not self.perlin_displacement_intensity.data:
                self.perlin_displacement_intensity.errors = ['Displacement intensity is required for Perlin Displacement']
                logger.debug("Missing perlin_displacement_intensity for perlin_displacement")
                return False
            if not self.perlin_displacement_octaves.data:
                self.perlin_displacement_octaves.errors = ['Octaves is required for Perlin Displacement']
                logger.debug("Missing perlin_displacement_octaves for perlin_displacement")
                return False
            if not self.perlin_displacement_persistence.data:
                self.perlin_displacement_persistence.errors = ['Persistence is required for Perlin Displacement']
                logger.debug("Missing perlin_displacement_persistence for perlin_displacement")
                return False
            if not self.perlin_displacement_lacunarity.data:
                self.perlin_displacement_lacunarity.errors = ['Lacunarity is required for Perlin Displacement']
                logger.debug("Missing perlin_displacement_lacunarity for perlin_displacement")
                return False
                
        elif effect == 'voronoi_sort':
            logger.debug("Validating voronoi_sort fields")
            if not self.voronoi_num_cells.data:
                self.voronoi_num_cells.errors = ['Number of cells is required for Voronoi Pixel Sort']
                logger.debug("Missing voronoi_num_cells for voronoi_sort")
                return False
            if not self.voronoi_size_variation.data and self.voronoi_size_variation.data != 0:
                self.voronoi_size_variation.errors = ['Size variation is required for Voronoi Pixel Sort']
                logger.debug("Missing voronoi_size_variation for voronoi_sort")
                return False
            if not self.voronoi_sort_by.data:
                self.voronoi_sort_by.errors = ['Sort by is required for Voronoi Pixel Sort']
                logger.debug("Missing voronoi_sort_by for voronoi_sort")
                return False
            if not self.voronoi_sort_order.data:
                self.voronoi_sort_order.errors = ['Sort order is required for Voronoi Pixel Sort']
                logger.debug("Missing voronoi_sort_order for voronoi_sort")
                return False
            if not self.voronoi_orientation.data:
                self.voronoi_orientation.errors = ['Line orientation is required for Voronoi Pixel Sort']
                logger.debug("Missing voronoi_orientation for voronoi_sort")
                return False
            if not self.voronoi_start_position.data:
                self.voronoi_start_position.errors = ['Start position is required for Voronoi Pixel Sort']
                logger.debug("Missing voronoi_start_position for voronoi_sort")
                return False
        
        # Check if channel_shift is valid        
        elif effect == 'channel_shift':
            logger.debug("Validating channel_shift fields")
            if not self.channel_shift_amount.data:
                self.channel_shift_amount.errors = ['Shift amount is required for RGB Channel Shift']
                logger.debug("Missing channel_shift_amount for channel_shift")
                return False
            if not self.channel_shift_direction.data:
                self.channel_shift_direction.errors = ['Shift direction is required for RGB Channel Shift']
                logger.debug("Missing channel_shift_direction for channel_shift")
                return False
            if not self.channel_shift_center.data:
                self.channel_shift_center.errors = ['Centered channel is required for RGB Channel Shift']
                logger.debug("Missing channel_shift_center for channel_shift")
                return False
            if not self.channel_mode.data:
                self.channel_mode.errors = ['Mode is required for RGB Channel Shift']
                logger.debug("Missing channel_mode for channel_shift")
                return False
                
        # Check if pixel_scatter is valid
        elif effect == 'pixel_scatter':
            logger.debug("Validating pixel_scatter fields")
            if not self.scatter_direction.data:
                self.scatter_direction.errors = ['Scatter direction is required for Pixel Scatter']
                logger.debug("Missing scatter_direction for pixel_scatter")
                return False
            if not self.scatter_select_by.data:
                self.scatter_select_by.errors = ['Select by is required for Pixel Scatter']
                logger.debug("Missing scatter_select_by for pixel_scatter")
                return False
            if self.scatter_min_value.data is None:
                self.scatter_min_value.errors = ['Minimum value is required for Pixel Scatter']
                logger.debug("Missing scatter_min_value for pixel_scatter")
                return False
            if self.scatter_max_value.data is None:
                self.scatter_max_value.errors = ['Maximum value is required for Pixel Scatter']
                logger.debug("Missing scatter_max_value for pixel_scatter")
                return False
            if self.scatter_min_value.data >= self.scatter_max_value.data:
                self.scatter_min_value.errors = ['Minimum value must be less than maximum value']
                logger.debug("Invalid value range for pixel_scatter")
                return False
                
        # Check if databend is valid
        elif effect == 'databend':
            logger.debug("Validating databend fields")
            if self.databend_intensity.data is None:
                self.databend_intensity.errors = ['Databend intensity is required']
                logger.debug("Missing databend_intensity for databend")
                return False
            if self.databend_intensity.data < 0.1 or self.databend_intensity.data > 1.0:
                self.databend_intensity.errors = ['Databend intensity must be between 0.1 and 1.0']
                logger.debug(f"Invalid databend_intensity: {self.databend_intensity.data}")
                return False
            if not self.databend_preserve_header.data:
                self.databend_preserve_header.errors = ['Preserve header option is required for Databending']
                logger.debug("Missing databend_preserve_header for databend")
                return False
        
        # Check if histogram_glitch is valid
        elif effect == 'histogram_glitch':
            logger.debug("Validating histogram_glitch fields")
            if not self.hist_r_mode.data:
                self.hist_r_mode.errors = ['Red channel treatment is required for Histogram Glitch']
                logger.debug("Missing hist_r_mode for histogram_glitch")
                return False
            if not self.hist_g_mode.data:
                self.hist_g_mode.errors = ['Green channel treatment is required for Histogram Glitch']
                logger.debug("Missing hist_g_mode for histogram_glitch")
                return False
            if not self.hist_b_mode.data:
                self.hist_b_mode.errors = ['Blue channel treatment is required for Histogram Glitch']
                logger.debug("Missing hist_b_mode for histogram_glitch")
                return False
            
            # Check solarize parameters if any channel is set to solarize mode
            if self.hist_r_mode.data == 'solarize':
                if not self.hist_r_freq.data or not self.hist_r_phase.data:
                    self.hist_r_freq.errors = ['Frequency required for solarize mode']
                    self.hist_r_phase.errors = ['Phase required for solarize mode']
                    logger.debug("Missing solarize parameters for red channel")
                    return False
            
            if self.hist_g_mode.data == 'solarize':
                if not self.hist_g_freq.data or not self.hist_g_phase.data:
                    self.hist_g_freq.errors = ['Frequency required for solarize mode']
                    self.hist_g_phase.errors = ['Phase required for solarize mode']
                    logger.debug("Missing solarize parameters for green channel")
                    return False
            
            if self.hist_b_mode.data == 'solarize':
                if not self.hist_b_freq.data or not self.hist_b_phase.data:
                    self.hist_b_freq.errors = ['Frequency required for solarize mode']
                    self.hist_b_phase.errors = ['Phase required for solarize mode']
                    logger.debug("Missing solarize parameters for blue channel")
                    return False
            
            # Check gamma if any channel is set to gamma mode
            if 'gamma' in [self.hist_r_mode.data, self.hist_g_mode.data, self.hist_b_mode.data]:
                if not self.hist_gamma.data:
                    self.hist_gamma.errors = ['Gamma value is required when a channel is set to gamma mode']
                    logger.debug("Missing hist_gamma for histogram_glitch")
                    return False
                if self.hist_gamma.data < 0.1 or self.hist_gamma.data > 3.0:
                    self.hist_gamma.errors = ['Gamma value must be between 0.1 and 3.0']
                    return False
        
        # If we get here, validation passed
        logger.debug("Validation passed")
        return True