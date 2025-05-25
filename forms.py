from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SelectField, StringField, FloatField, IntegerField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Optional, ValidationError, NumberRange
import re
import logging

# Configure logging
logger = logging.getLogger(__name__)

# --- Custom Validators ---
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

def validate_multiple_of_2(form, field):
    """Validate that the value is a multiple of 2."""
    if not field.data:
        return
    if field.data % 2 != 0:
        raise ValidationError('Value must be a multiple of 2 (e.g., 2, 4, 6, 8, etc.)')

def validate_is_odd(form, field):
    """Validate that the value is an odd number."""
    if field.data is None: # Allow optional fields to be empty
        return
    if field.data % 2 == 0:
        raise ValidationError('Value must be an odd number.')

def validate_hex_color(form, field):
    """Validate that the value is a valid hex color."""
    if not field.data:
        return
    pattern = re.compile(r'^#[0-9A-Fa-f]{6}$')
    if not pattern.match(field.data):
        raise ValidationError('Color must be in hex format (e.g., #FF0000)')

# --- Base Forms and Mixins ---

class BaseEffectForm(FlaskForm):
    """
    Conceptual base for specific effect forms.
    Can be used for common methods or configurations if needed later.
    Individual effect forms will inherit from FlaskForm directly or via mixins.
    """
    pass

class SortingFieldsMixin:
    """Mixin for common sorting parameters."""
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
    ], default='brightness', validators=[DataRequired()])

    reverse_sort = BooleanField('Descending Sort (High to Low)', default=True, validators=[Optional()])

class SeedMixin:
    """Mixin for a common random seed parameter."""
    seed = IntegerField('Random Seed', validators=[Optional(), NumberRange(min=1, max=99999)])

class SecondaryImageMixin:
    """Mixin for a secondary image upload field."""
    secondary_image = FileField('Secondary Image', validators=[
        FileRequired(message="A secondary image is required for this effect."),
        FileAllowed(['jpg', 'jpeg', 'png', 'gif'], 'Secondary image must be a JPG, PNG, or GIF.')
    ])

# --- Effect Specific Forms ---

class PixelSortChunkForm(FlaskForm, SortingFieldsMixin):
    """Form for Pixel Sort Chunk effect."""
    width = IntegerField('Chunk Width',
                         default=48,
                         validators=[DataRequired(), NumberRange(min=2, max=2048), validate_multiple_of_2])
    height = IntegerField('Chunk Height',
                           default=48,
                           validators=[DataRequired(), NumberRange(min=2, max=2048), validate_multiple_of_2])
    sort_mode = SelectField('Sort Mode', choices=[
        ('horizontal', 'Horizontal'),
        ('vertical', 'Vertical'),
        ('diagonal', 'Diagonal')
    ], default='horizontal', validators=[DataRequired()])
    starting_corner = SelectField('Starting Corner (for Diagonal mode)', choices=[
        ('top-left', 'Top-Left'),
        ('top-right', 'Top-Right'),
        ('bottom-left', 'Bottom-Left'),
        ('bottom-right', 'Bottom-Right')
    ], default='top-left', validators=[Optional()]) # Optional, only if sort_mode is diagonal

class ColorChannelForm(FlaskForm):
    """Form for Color Channel Manipulation effect."""
    manipulation_type = SelectField('Manipulation Type', choices=[
        ('swap', 'Swap Channels'),
        ('invert', 'Invert Channel'),
        ('adjust', 'Adjust Channel Intensity'),
        ('negative', 'Negative (Invert All Channels)')
    ], default='swap', validators=[DataRequired()])
    # These fields will be shown/hidden via JS based on manipulation_type
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
    intensity_factor = FloatField('Intensity Factor (for Adjust)',
                                 default=1.5,
                                 validators=[Optional(), NumberRange(min=0.1, max=10.0)])

class BitManipulationForm(FlaskForm, SeedMixin):
    """Form for Bit Manipulation effect."""
    chunk_size = IntegerField('Chunk Size', 
                                default=24,
                                validators=[DataRequired(), NumberRange(min=8, max=128), validate_multiple_of_8])
    offset = IntegerField('Byte Offset',
                            default=1,
                            validators=[DataRequired(), NumberRange(min=0, max=1000000)])
    xor_value = IntegerField('XOR Value',
                               default=255,
                               validators=[DataRequired(), NumberRange(min=0, max=255)])
    skip_pattern = SelectField('Skip Pattern', choices=[
        ('alternate', 'Every Other Chunk'),
        ('every_third', 'Every Third Chunk'),
        ('every_fourth', 'Every Fourth Chunk'),
        ('random', 'Random Chunks')
    ], default='alternate', validators=[DataRequired()])
    manipulation_type = SelectField('Manipulation Type', choices=[
        ('xor', 'XOR (Blend)'),
        ('invert', 'Invert (Negative)'),
        ('shift', 'Bit Shift'),
        ('swap', 'Swap Chunks')
    ], default='xor', validators=[DataRequired()])
    shift_amount = IntegerField('Bit Shift Amount',
                           default=1,
                           validators=[DataRequired(), NumberRange(min=-7, max=7)])
    randomize_effect = BooleanField('Add Randomness', default=False, validators=[Optional()])

class DoubleExposeForm(FlaskForm, SecondaryImageMixin):
    """Form for Double Expose effect."""
    # secondary_image is inherited from SecondaryImageMixin
    blend_mode = SelectField('Blend Mode', choices=[
        ('classic', 'Classic Blend'),
        ('screen', 'Screen'),
        ('multiply', 'Multiply'),
        ('overlay', 'Overlay'),
        ('hard_light', 'Hard Light'),
        ('difference', 'Difference'),
        ('exclusion', 'Exclusion'),
        ('add', 'Add (Linear Dodge)'),
        ('subtract', 'Subtract'),
        ('darken_only', 'Darken Only'),
        ('lighten_only', 'Lighten Only'),
        ('color_dodge', 'Color Dodge'),
        ('burn', 'Color Burn')
    ], default='classic', validators=[DataRequired()])
    opacity = FloatField('Opacity', 
                        default=0.5,
                        validators=[DataRequired(), NumberRange(min=0.0, max=1.0)])

class FullFrameSortForm(FlaskForm, SortingFieldsMixin):
    """Form for Full Frame Sorting effect."""
    direction = SelectField('Direction', choices=[
        ('vertical', 'Vertical'), 
        ('horizontal', 'Horizontal')
    ], default='vertical', validators=[DataRequired()])
    # sort_by and reverse_sort are inherited from SortingFieldsMixin
    # Note: original form had full_frame_reverse as a SelectField ('true'/'false').
    # Mixin uses BooleanField reverse_sort, which is preferred.

class PixelDriftForm(FlaskForm):
    """Form for Pixel Drift effect."""
    direction = SelectField('Drift Direction', choices=[
        ('up', 'Up'), 
        ('down', 'Down'), 
        ('left', 'Left'), 
        ('right', 'Right')
    ], default='right', validators=[DataRequired()])
    bands = IntegerField('Number of Bands', 
                              default=12,
                              validators=[DataRequired(), NumberRange(min=1, max=48)]) # Renamed from drift_bands
    intensity = FloatField('Drift Intensity Multiplier', 
                               default=4.0,
                               validators=[DataRequired(), NumberRange(min=0.1, max=10.0)]) # Renamed from drift_intensity

class DatabendingForm(FlaskForm, SeedMixin):
    """Form for Databending effect."""
    intensity = FloatField('Databend Intensity',
                                 default=0.1,
                                 validators=[DataRequired(), NumberRange(min=0.1, max=1.0)]) # Renamed from databend_intensity
    preserve_header = BooleanField('Preserve Header (More Stable)', default=True, validators=[Optional()]) # Renamed & changed from SelectField
    # seed is inherited from SeedMixin (renamed from databend_seed)

class PolarSortForm(FlaskForm):
    """Form for Polar Sorting effect."""
    chunk_size = IntegerField('Chunk Size', 
                                default=64,
                                validators=[DataRequired(), NumberRange(min=8, max=128), validate_multiple_of_8])
    sort_by = SelectField('Sort By', choices=[
        ('angle', 'Angle (around center)'), 
        ('radius', 'Radius (distance from center)')
    ], default='radius', validators=[DataRequired()])
    reverse_sort = BooleanField('Descending Sort', default=True, validators=[Optional()]) # Standardized from polar_reverse

class SpiralSort2Form(FlaskForm, SortingFieldsMixin):
    """Form for Spiral Sort 2 effect."""
    chunk_size = IntegerField('Chunk Size', 
                                default=64,
                                validators=[DataRequired(), NumberRange(min=8, max=128), validate_multiple_of_8])
    # sort_by and reverse_sort are inherited from SortingFieldsMixin
    # Original spiral2_sort_by had same choices as SortingFieldsMixin.default='brightness'
    # Original spiral2_reverse was SelectField 'false'/'true', now handled by mixin's BooleanField.

class JpegArtifactsForm(FlaskForm):
    """Form for Simulate JPEG Artifacts effect."""
    intensity = FloatField('Artifact Intensity', 
                               default=1.0,
                               validators=[DataRequired(), NumberRange(min=0.0, max=1.0)]) # Renamed from jpeg_intensity

class PerlinFullFrameForm(FlaskForm, SortingFieldsMixin, SeedMixin):
    """Form for Perlin Full Frame Sorting effect."""
    noise_scale = FloatField('Noise Scale', 
                                default=0.005,
                                validators=[DataRequired(), NumberRange(min=0.001, max=0.1)]) # Renamed
    pattern_width = IntegerField('Pattern Width',
                                default=1,
                                validators=[DataRequired(), NumberRange(min=1, max=8)]) # Renamed
    # sort_by and reverse_sort from SortingFieldsMixin. Original perlin_full_frame_sort_by used same choices.
    # seed from SeedMixin. Original perlin_full_frame_seed.

class VoronoiSortForm(FlaskForm, SeedMixin):
    """Form for Voronoi Pixel Sort effect."""
    num_cells = IntegerField('Number of Cells',
                                default=69,
                                validators=[DataRequired(), NumberRange(min=10, max=1000)]) # Renamed
    size_variation = FloatField('Size Variation',
                                default=0.8,
                                validators=[DataRequired(), NumberRange(min=0.0, max=1.0)]) # Renamed
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
    ], default='hue', validators=[DataRequired()]) # Defined specifically as sort_order is custom
    sort_order = SelectField('Sort Order', choices=[
        ('clockwise', 'Clockwise'), 
        ('counter-clockwise', 'Counter-Clockwise')
    ], default='clockwise', validators=[DataRequired()]) # Renamed
    orientation = SelectField('Line Orientation', choices=[
        ('horizontal', 'Horizontal'),
        ('vertical', 'Vertical'),
        ('radial', 'Radial'),
        ('spiral', 'Spiral')
    ], default='spiral', validators=[DataRequired()]) # Renamed
    start_position = SelectField('Start Position', choices=[
        ('left', 'Left'),
        ('right', 'Right'),
        ('top', 'Top'),
        ('bottom', 'Bottom'),
        ('center', 'Center')
    ], default='center', validators=[DataRequired()]) # Renamed
    # seed from SeedMixin. Original voronoi_seed.

class PerlinNoiseSortForm(FlaskForm, SeedMixin):
    """Form for Perlin Noise Sorting effect."""
    chunk_width = IntegerField('Chunk Width', 
                                default=120,
                                validators=[DataRequired(), NumberRange(min=8, max=1024), validate_multiple_of_8])
    chunk_height = IntegerField('Chunk Height', 
                                default=1024,
                                validators=[DataRequired(), NumberRange(min=8, max=1024), validate_multiple_of_8])
    noise_scale = FloatField('Noise Scale', 
                                default=0.008,
                                validators=[DataRequired(), NumberRange(min=0.001, max=0.1)])
    direction = SelectField('Direction', choices=[
        ('horizontal', 'Horizontal'), 
        ('vertical', 'Vertical')
    ], default='horizontal', validators=[DataRequired()]) # Renamed from perlin_direction
    reverse_sort = BooleanField('Descending Sort', default=True, validators=[Optional()]) # Standardized from perlin_reverse
    # seed from SeedMixin. Original perlin_seed.

class PixelScatterForm(FlaskForm):
    """Form for Pixel Scatter effect."""
    direction = SelectField('Scatter Direction', choices=[
        ('horizontal', 'Horizontal'),
        ('vertical', 'Vertical')
    ], default='horizontal', validators=[DataRequired()]) # Renamed from scatter_direction
    select_by = SelectField('Select Pixels By', choices=[
        ('brightness', 'Brightness'),
        ('red', 'Red Channel'),
        ('green', 'Green Channel'),
        ('blue', 'Blue Channel'),
        ('hue', 'Hue'),
        ('saturation', 'Saturation'),
        ('luminance', 'Luminance'),
        ('contrast', 'Contrast')
    ], default='red', validators=[DataRequired()]) # Renamed from scatter_select_by
    min_value = FloatField('Minimum Value',
                              default=180,
                              validators=[DataRequired(), NumberRange(min=0, max=360)]) # Renamed from scatter_min_value
    max_value = FloatField('Maximum Value',
                              default=360,
                              validators=[DataRequired(), NumberRange(min=0, max=360)]) # Renamed from scatter_max_value
    
    def validate_max_value(self, field):
        if self.min_value.data is not None and field.data is not None:
            if field.data < self.min_value.data:
                raise ValidationError('Maximum value must not be less than minimum value.')

class PerlinDisplacementForm(FlaskForm, SeedMixin): # Added SeedMixin assuming Perlin noise can be seeded
    """Form for Perlin Noise Displacement effect."""
    scale = IntegerField('Noise Scale',
                        default=44, # Changed from 25
                        validators=[DataRequired(), NumberRange(min=10, max=500)]) # Renamed
    intensity = IntegerField('Displacement Intensity',
                            default=25, # Changed from 40
                            validators=[DataRequired(), NumberRange(min=1, max=100)]) # Renamed
    octaves = IntegerField('Octaves',
                        default=3, # Changed from 4
                        validators=[DataRequired(), NumberRange(min=1, max=10)]) # Renamed
    # SeedMixin provides the seed field

class RippleEffectForm(FlaskForm, SeedMixin): # Added SeedMixin assuming droplet placement can be seeded
    """Form for Ripple Effect."""
    num_droplets = IntegerField('Number of Droplets',
                                default=9,
                                validators=[DataRequired(), NumberRange(min=1, max=20)]) # Renamed
    amplitude = FloatField('Ripple Amplitude',
                            default=45.0,
                            validators=[DataRequired(), NumberRange(min=1.0, max=50.0)]) # Renamed
    frequency = FloatField('Ripple Frequency',
                            default=0.4,
                            validators=[DataRequired(), NumberRange(min=0.01, max=1.0)]) # Renamed
    decay = FloatField('Ripple Decay',
                        default=0.004,
                        validators=[DataRequired(), NumberRange(min=0.001, max=0.1)]) # Renamed
    distortion_type = SelectField('Distortion Type', choices=[
        ('color_shift', 'Color Shift'),
        ('pixelation', 'Pixelation'),
        ('none', 'None (Plain Ripple)')
    ], default='color_shift', validators=[DataRequired()]) # Renamed
    # Conditional fields based on distortion_type (shown/hidden with JS)
    color_r_factor = FloatField('Red Channel Factor (for Color Shift)',
                                default=0.8,
                                validators=[Optional(), NumberRange(min=0.5, max=1.5)]) # Renamed
    color_g_factor = FloatField('Green Channel Factor (for Color Shift)',
                                default=1.0,
                                validators=[Optional(), NumberRange(min=0.5, max=1.5)]) # Renamed
    color_b_factor = FloatField('Blue Channel Factor (for Color Shift)',
                                default=1.2,
                                validators=[Optional(), NumberRange(min=0.5, max=1.5)]) # Renamed
    pixelation_scale = IntegerField('Pixelation Scale (for Pixelation)',
                                    default=10,
                                    validators=[Optional(), NumberRange(min=2, max=20)]) # Renamed
    pixelation_magnitude = FloatField('Pixelation Magnitude (for Pixelation)',
                                    default=10.0,
                                    validators=[Optional(), NumberRange(min=1.0, max=20.0)]) # Renamed
    # seed from SeedMixin

class RGBChannelShiftForm(FlaskForm):
    """Form for RGB Channel Shift effect."""
    shift_amount = IntegerField('Shift Amount',
                                default=42,
                                validators=[DataRequired(), NumberRange(min=1, max=500)]) # Renamed
    direction = SelectField('Shift Direction', choices=[
        ('horizontal', 'Horizontal'), 
        ('vertical', 'Vertical')
    ], default='horizontal', validators=[DataRequired()]) # Renamed
    center_channel = SelectField('Centered Channel', choices=[
        ('red', 'Red'), 
        ('green', 'Green'), 
        ('blue', 'Blue')
    ], default='green', validators=[DataRequired()]) # Renamed
    mode = SelectField('Mode', choices=[
        ('shift', 'Shift Channels'), 
        ('mirror', 'Mirror Channels')
    ], default='shift', validators=[DataRequired()]) # Renamed

class HistogramGlitchForm(FlaskForm):
    """Form for Histogram Glitch effect."""
    # Renamed fields by removing 'hist_' prefix and standardizing freq/phase
    r_mode = SelectField('Red Channel Treatment', choices=[
        ('solarize', 'Solarize (Sine Wave)'),
        ('log', 'Logarithmic Compression'),
        ('gamma', 'Gamma Adjustment'),
        ('normal', 'Normal (No Change)')
    ], default='solarize', validators=[DataRequired()])
    g_mode = SelectField('Green Channel Treatment', choices=[
        ('solarize', 'Solarize (Sine Wave)'),
        ('log', 'Logarithmic Compression'),
        ('gamma', 'Gamma Adjustment'),
        ('normal', 'Normal (No Change)')
    ], default='solarize', validators=[DataRequired()])
    b_mode = SelectField('Blue Channel Treatment', choices=[
        ('solarize', 'Solarize (Sine Wave)'),
        ('log', 'Logarithmic Compression'),
        ('gamma', 'Gamma Adjustment'),
        ('normal', 'Normal (No Change)')
    ], default='solarize', validators=[DataRequired()])
    
    # Conditional fields (shown/hidden with JS based on mode selections)
    r_freq = FloatField('Red Solarize Frequency',
                        default=1.0,
                        validators=[Optional(), NumberRange(min=0.1, max=10.0)])
    r_phase = FloatField('Red Solarize Phase',
                        default=0.4,
                        validators=[Optional(), NumberRange(min=0.0, max=6.28)])
    g_freq = FloatField('Green Solarize Frequency',
                        default=1.0,
                        validators=[Optional(), NumberRange(min=0.1, max=10.0)])
    g_phase = FloatField('Green Solarize Phase',
                        default=0.3,
                        validators=[Optional(), NumberRange(min=0.0, max=6.28)])
    b_freq = FloatField('Blue Solarize Frequency',
                        default=1.0,
                        validators=[Optional(), NumberRange(min=0.1, max=10.0)])
    b_phase = FloatField('Blue Solarize Phase',
                        default=0.2,
                        validators=[Optional(), NumberRange(min=0.0, max=6.28)])
    gamma_value = FloatField('Gamma Value (for Gamma Adjustment)',
                            default=0.5,
                            validators=[Optional(), NumberRange(min=0.1, max=3.0)]) # Renamed

class ColorShiftExpansionForm(FlaskForm, SeedMixin): # Added SeedMixin
    """Form for Color Shift Expansion effect."""
    # Renamed fields for clarity/consistency
    num_points = IntegerField('Number of Seed Points',
                                default=7,
                                validators=[DataRequired(), NumberRange(min=1, max=100)])
    shift_amount = IntegerField('Shift Amount',
                                default=20, # Default was previously updated in a commit
                                validators=[DataRequired(), NumberRange(min=1, max=50)])
    expansion_type = SelectField('Expansion Type', choices=[
        ('square', 'Square (8 directions)'),
        ('diamond', 'Diamond (4 directions)'),
        ('circle', 'Circle')
    ], default='circle', validators=[DataRequired()])
    pattern_type = SelectField('Seed Point Pattern', choices=[
        ('random', 'Random'),
        ('grid', 'Grid'),
        ('edges', 'Around Edges')
    ], default='random', validators=[DataRequired()])
    color_theme = SelectField('Color Theme', choices=[
        ('full-spectrum', 'Full Spectrum'),
        ('warm', 'Warm Colors'),
        ('cool', 'Cool Colors'),
        ('pastel', 'Pastel Colors')
    ], default='full-spectrum', validators=[DataRequired()])
    saturation_boost = FloatField('Saturation Boost',
                                default=0.5,
                                validators=[DataRequired(), NumberRange(min=0.0, max=1.0)])
    value_boost = FloatField('Brightness Boost',
                            default=0.0,
                            validators=[Optional(), NumberRange(min=0.0, max=1.0)])
    decay_factor = FloatField('Decay Factor',
                            default=0.2,
                            validators=[DataRequired(), NumberRange(min=0.0, max=1.0)])
    # seed is from SeedMixin

class PixelateForm(FlaskForm):
    """Form for Pixelate effect."""
    width = IntegerField('Pixel Width',
                        default=16,
                        validators=[DataRequired(), NumberRange(min=2, max=64)]) # Renamed
    height = IntegerField('Pixel Height',
                        default=16,
                        validators=[DataRequired(), NumberRange(min=2, max=64)]) # Renamed
    attribute = SelectField('Attribute', choices=[
        ('color', 'Color (Most Common)'),
        ('brightness', 'Brightness'),
        ('hue', 'Hue'),
        ('saturation', 'Saturation'),
        ('luminance', 'Luminance')
    ], default='hue', validators=[DataRequired()]) # Renamed
    bins = IntegerField('Number of Bins',
                        default=200,
                        validators=[DataRequired(), NumberRange(min=10, max=1000)]) # Renamed

class VoronoiPixelateForm(FlaskForm, SeedMixin):
    """Form for Voronoi Pixelate effect."""
    num_cells = IntegerField('Number of Voronoi Cells',
                            default=50,
                            validators=[DataRequired(), NumberRange(min=10, max=500)])
    attribute = SelectField('Cell Color Attribute', choices=[
        ('color', 'Color (Most Common)'),
        ('brightness', 'Brightness'),
        ('hue', 'Hue'),
        ('saturation', 'Saturation'),
        ('luminance', 'Luminance')
    ], default='hue', validators=[DataRequired()])
    # seed is from SeedMixin

class ConcentricShapesForm(FlaskForm, SeedMixin): # Added SeedMixin
    """Form for Concentric Shapes effect."""
    num_points = IntegerField('Number of Points',
                                default=7,
                                validators=[DataRequired(), NumberRange(min=1, max=100)]) # Renamed
    shape_type = SelectField('Shape Type', choices=[
        ('square', 'Square'),
        ('circle', 'Circle'),
        ('hexagon', 'Hexagon'),
        ('triangle', 'Triangle')
    ], default='triangle', validators=[DataRequired()])
    thickness = IntegerField('Line Thickness',
                                default=2,
                                validators=[DataRequired(), NumberRange(min=1, max=10)]) # Renamed
    spacing = IntegerField('Spacing',
                        default=20,
                        validators=[DataRequired(), NumberRange(min=1, max=50)])
    rotation_angle = IntegerField('Rotation Angle',
                                default=9,
                                validators=[DataRequired(), NumberRange(min=0, max=360)])
    darken_step = IntegerField('Darken Step',
                            default=0,
                            validators=[Optional(), NumberRange(min=0, max=255)])
    color_shift_amount = IntegerField('Color Shift',
                                default=15,
                                validators=[DataRequired(), NumberRange(min=0, max=360)]) # Renamed
    # seed is from SeedMixin

class PosterizeForm(FlaskForm):
    """Form for Posterize effect."""
    levels = IntegerField('Color Levels',
                        default=4,
                        validators=[DataRequired(), NumberRange(min=2, max=32, message="Levels must be between 2 and 32")]) # Renamed

class CurvedHueShiftForm(FlaskForm):
    """Form for Curved Hue Shift effect."""
    curve_value = FloatField('Curve Value',
                            default=180,
                            validators=[DataRequired(), NumberRange(min=1, max=360)]) # Renamed from hue_curve
    shift_amount = FloatField('Shift Amount (degrees)',
                                default=45.0, # Default was previously updated in a commit
                                validators=[DataRequired(), NumberRange(min=-360.0, max=360.0)]) # Renamed from hue_shift_amount

class ColorFilterForm(FlaskForm):
    """Form for Color Filter effect."""
    filter_type = SelectField('Filter Type', choices=[
        ('solid', 'Solid Color'),
        ('gradient', 'Gradient')
    ], default='solid', validators=[DataRequired()])
    
    blend_mode = SelectField('Blend Mode', choices=[
        ('overlay', 'Overlay'),
        ('soft_light', 'Soft Light')
    ], default='overlay', validators=[DataRequired()])
    
    opacity = FloatField('Filter Opacity',
                        default=0.5,
                        validators=[DataRequired(), NumberRange(min=0.0, max=1.0)])
    
    # Color picker fields - using StringField with hex validation
    color = StringField('Filter Color',
                       default='#FF0000',
                       validators=[DataRequired(), validate_hex_color])
    
    # Gradient-specific fields (conditional)
    gradient_color2 = StringField('Gradient End Color',
                                 default='#0000FF',
                                 validators=[Optional(), validate_hex_color])
    
    gradient_angle = IntegerField('Gradient Angle (degrees)',
                                 default=0,
                                 validators=[Optional(), NumberRange(min=0, max=360)])

class GaussianBlurForm(FlaskForm):
    """Form for Gaussian Blur effect."""
    radius = FloatField('Blur Radius',
                       default=5.0,
                       validators=[DataRequired(), NumberRange(min=0.1, max=50.0)])
    
    sigma = FloatField('Sigma (Standard Deviation)',
                      default=None,
                      validators=[Optional(), NumberRange(min=0.1, max=20.0)])

class NoiseEffectForm(FlaskForm, SeedMixin):
    """Form for Noise Effect."""
    noise_type = SelectField('Noise Type', choices=[
        ('film_grain', 'Film Grain'),
        ('digital', 'Digital Noise'),
        ('colored', 'Colored Noise'),
        ('salt_pepper', 'Salt & Pepper'),
        ('gaussian', 'Gaussian Noise')
    ], default='film_grain', validators=[DataRequired()])
    
    intensity = FloatField('Noise Intensity',
                          default=0.3,
                          validators=[DataRequired(), NumberRange(min=0.0, max=1.0)])
    
    grain_size = FloatField('Grain/Particle Size',
                           default=1.0,
                           validators=[DataRequired(), NumberRange(min=0.5, max=5.0)])
    
    color_variation = FloatField('Color Variation',
                                default=0.2,
                                validators=[DataRequired(), NumberRange(min=0.0, max=1.0)])
    
    noise_color = StringField('Noise Color (for Colored type)',
                             default='#FFFFFF',
                             validators=[Optional(), validate_hex_color])
    
    blend_mode = SelectField('Blend Mode', choices=[
        ('overlay', 'Overlay'),
        ('add', 'Add'),
        ('multiply', 'Multiply'),
        ('screen', 'Screen')
    ], default='overlay', validators=[DataRequired()])
    
    pattern = SelectField('Noise Pattern', choices=[
        ('random', 'Random'),
        ('perlin', 'Perlin-like'),
        ('cellular', 'Cellular')
    ], default='random', validators=[DataRequired()])
    
    # seed is inherited from SeedMixin

class ChromaticAberrationForm(FlaskForm, SeedMixin):
    """Form for Chromatic Aberration effect."""
    intensity = FloatField('Aberration Intensity',
                          default=5.0,
                          validators=[DataRequired(), NumberRange(min=0.0, max=50.0)])
    
    pattern = SelectField('Aberration Pattern', choices=[
        ('radial', 'Radial (Lens-like)'),
        ('linear', 'Linear (Prism-like)'),
        ('barrel', 'Barrel Distortion'),
        ('custom', 'Custom Manual')
    ], default='radial', validators=[DataRequired()])
    
    # Manual displacement controls (for custom pattern or fine-tuning)
    red_shift_x = FloatField('Red Channel X Shift',
                            default=0.0,
                            validators=[Optional(), NumberRange(min=-20.0, max=20.0)])
    
    red_shift_y = FloatField('Red Channel Y Shift',
                            default=0.0,
                            validators=[Optional(), NumberRange(min=-20.0, max=20.0)])
    
    blue_shift_x = FloatField('Blue Channel X Shift',
                             default=0.0,
                             validators=[Optional(), NumberRange(min=-20.0, max=20.0)])
    
    blue_shift_y = FloatField('Blue Channel Y Shift',
                             default=0.0,
                             validators=[Optional(), NumberRange(min=-20.0, max=20.0)])
    
    # Center point controls
    center_x = FloatField('Center X Position',
                         default=0.5,
                         validators=[Optional(), NumberRange(min=0.0, max=1.0)])
    
    center_y = FloatField('Center Y Position',
                         default=0.5,
                         validators=[Optional(), NumberRange(min=0.0, max=1.0)])
    
    # Advanced controls
    falloff = SelectField('Distance Falloff', choices=[
        ('linear', 'Linear'),
        ('quadratic', 'Quadratic'),
        ('cubic', 'Cubic')
    ], default='quadratic', validators=[DataRequired()])
    
    edge_enhancement = FloatField('Edge Enhancement',
                                 default=0.0,
                                 validators=[Optional(), NumberRange(min=0.0, max=1.0)])
    
    color_boost = FloatField('Color Saturation Boost',
                            default=1.0,
                            validators=[Optional(), NumberRange(min=0.5, max=2.0)])
    
    # seed is inherited from SeedMixin

class VHSEffectForm(FlaskForm, SeedMixin):
    """Form for VHS Effect."""
    quality_preset = SelectField('VHS Quality Preset', choices=[
        ('high', 'High Quality (Fresh Tape)'),
        ('medium', 'Medium Quality (Rental Store)'),
        ('low', 'Low Quality (Old Tape)'),
        ('damaged', 'Damaged (Garage Sale Find)')
    ], default='medium', validators=[DataRequired()])
    
    # Scan line controls
    scan_line_intensity = FloatField('Scan Line Intensity',
                                    default=0.3,
                                    validators=[Optional(), NumberRange(min=0.0, max=1.0)])
    
    scan_line_spacing = IntegerField('Scan Line Spacing',
                                    default=2,
                                    validators=[Optional(), NumberRange(min=1, max=5)])
    
    # Static noise controls
    static_intensity = FloatField('Static Noise Intensity',
                                 default=0.2,
                                 validators=[Optional(), NumberRange(min=0.0, max=1.0)])
    
    static_type = SelectField('Static Noise Type', choices=[
        ('white', 'White Static'),
        ('colored', 'Colored Static'),
        ('mixed', 'Mixed Static')
    ], default='white', validators=[DataRequired()])
    
    # Vertical hold controls
    vertical_hold_frequency = FloatField('Vertical Hold Issues',
                                        default=0.1,
                                        validators=[Optional(), NumberRange(min=0.0, max=1.0)])
    
    vertical_hold_intensity = FloatField('Vertical Hold Intensity',
                                        default=5.0,
                                        validators=[Optional(), NumberRange(min=0.0, max=20.0)])
    
    # Color bleeding and chroma shift
    color_bleeding = FloatField('Color Bleeding',
                               default=0.3,
                               validators=[Optional(), NumberRange(min=0.0, max=1.0)])
    
    chroma_shift = FloatField('Chroma/Luma Separation',
                             default=0.2,
                             validators=[Optional(), NumberRange(min=0.0, max=1.0)])
    
    # Tracking and tape wear
    tracking_errors = FloatField('Tracking Errors',
                                default=0.15,
                                validators=[Optional(), NumberRange(min=0.0, max=1.0)])
    
    tape_wear = FloatField('Tape Wear & Dropouts',
                          default=0.1,
                          validators=[Optional(), NumberRange(min=0.0, max=1.0)])
    
    head_switching_noise = FloatField('Head Switching Noise',
                                     default=0.1,
                                     validators=[Optional(), NumberRange(min=0.0, max=1.0)])
    
    # Color degradation
    color_desaturation = FloatField('Color Desaturation',
                                   default=0.3,
                                   validators=[Optional(), NumberRange(min=0.0, max=1.0)])
    
    brightness_variation = FloatField('Brightness Variation',
                                     default=0.2,
                                     validators=[Optional(), NumberRange(min=0.0, max=1.0)])
    
    # seed is inherited from SeedMixin

class SharpenEffectForm(FlaskForm):
    """Form for Sharpen Effect."""
    method = SelectField('Sharpening Method', choices=[
        ('unsharp_mask', 'Unsharp Mask (Classic)'),
        ('high_pass', 'High-Pass Filter'),
        ('edge_enhance', 'Edge Enhancement'),
        ('custom', 'Custom Kernel')
    ], default='unsharp_mask', validators=[DataRequired()])
    
    intensity = FloatField('Sharpening Intensity',
                          default=1.0,
                          validators=[DataRequired(), NumberRange(min=0.0, max=5.0)])
    
    # Unsharp mask specific controls
    radius = FloatField('Blur Radius (for Unsharp Mask)',
                       default=1.0,
                       validators=[Optional(), NumberRange(min=0.1, max=10.0)])
    
    threshold = IntegerField('Threshold (for Unsharp Mask)',
                            default=0,
                            validators=[Optional(), NumberRange(min=0, max=255)])
    
    # High-pass specific controls
    high_pass_radius = FloatField('High-Pass Radius',
                                 default=3.0,
                                 validators=[Optional(), NumberRange(min=1.0, max=10.0)])
    
    # Custom kernel controls
    custom_kernel = SelectField('Custom Kernel Type', choices=[
        ('default', 'Default Sharpen'),
        ('laplacian', 'Laplacian'),
        ('sobel', 'Sobel'),
        ('prewitt', 'Prewitt')
    ], default='default', validators=[Optional()])
    
    # Additional enhancement
    edge_enhancement = FloatField('Additional Edge Enhancement',
                                 default=0.0,
                                 validators=[Optional(), NumberRange(min=0.0, max=2.0)])

class WaveDistortionForm(FlaskForm):
    """Form for Wave Distortion effect."""
    wave_type = SelectField('Wave Type', choices=[
        ('horizontal', 'Horizontal Waves'),
        ('vertical', 'Vertical Waves'),
        ('both', 'Both Horizontal & Vertical'),
        ('diagonal', 'Diagonal Waves'),
        ('radial', 'Radial Waves (from center)')
    ], default='horizontal', validators=[DataRequired()])
    
    # Primary wave controls
    amplitude = FloatField('Wave Amplitude (pixels)',
                          default=20.0,
                          validators=[DataRequired(), NumberRange(min=0.0, max=100.0)])
    
    frequency = FloatField('Wave Frequency',
                          default=0.02,
                          validators=[DataRequired(), NumberRange(min=0.001, max=0.1)])
    
    phase = FloatField('Wave Phase (degrees)',
                      default=0.0,
                      validators=[Optional(), NumberRange(min=0.0, max=360.0)])
    
    # Secondary wave controls
    secondary_wave = BooleanField('Enable Secondary Wave', default=False, validators=[Optional()])
    
    secondary_amplitude = FloatField('Secondary Wave Amplitude',
                                    default=10.0,
                                    validators=[Optional(), NumberRange(min=0.0, max=100.0)])
    
    secondary_frequency = FloatField('Secondary Wave Frequency',
                                    default=0.05,
                                    validators=[Optional(), NumberRange(min=0.001, max=0.1)])
    
    secondary_phase = FloatField('Secondary Wave Phase (degrees)',
                                default=90.0,
                                validators=[Optional(), NumberRange(min=0.0, max=360.0)])
    
    # Wave combination controls
    blend_mode = SelectField('Wave Blend Mode', choices=[
        ('add', 'Add (Combine waves)'),
        ('multiply', 'Multiply (Modulate waves)'),
        ('max', 'Maximum (Take strongest)'),
        ('interference', 'Interference Pattern')
    ], default='add', validators=[DataRequired()])
    
    # Edge handling
    edge_behavior = SelectField('Edge Behavior', choices=[
        ('wrap', 'Wrap Around'),
        ('clamp', 'Clamp to Edges'),
        ('reflect', 'Reflect at Edges')
    ], default='wrap', validators=[DataRequired()])
    
    # Interpolation quality
    interpolation = SelectField('Interpolation Quality', choices=[
        ('nearest', 'Nearest Neighbor (Fast)'),
        ('bilinear', 'Bilinear (Good)'),
        ('bicubic', 'Bicubic (Best)')
    ], default='bilinear', validators=[DataRequired()])

class MaskedMergeForm(FlaskForm, SeedMixin, SecondaryImageMixin):
    """Form for Masked Merge effect."""
    # secondary_image from SecondaryImageMixin (replaces masked_merge_secondary)
    # seed from SeedMixin (replaces mask_random_seed)
    mask_type = SelectField('Mask Type', choices=[
        ('checkerboard', 'Checkerboard'),
        ('random_checkerboard', 'Random Checkerboard'), # Will use seed
        ('striped', 'Striped'),
        ('gradient_striped', 'Gradient Striped (Sawtooth)'),
        ('linear_gradient_striped', 'Linear Gradient Striped'),
        ('perlin', 'Perlin Noise'), # Will use seed
        ('voronoi', 'Voronoi Cells'), # Will use seed
        ('concentric_rectangles', 'Concentric Rectangles'),
        ('concentric_circles', 'Concentric Circles'),
        ('random_triangles', 'Random Triangles') # Will use seed
    ], default='checkerboard', validators=[DataRequired()])

    # --- Fields for specific mask types (conditional logic in template/JS) ---
    # For Checkerboard, Random Checkerboard (mask_width, mask_height)
    # For Striped (stripe_width, stripe_angle)
    # For Gradient Striped, Linear Gradient Striped (stripe_width, stripe_angle, gradient_direction for linear)
    # For Perlin Noise (perlin_noise_scale, perlin_threshold, perlin_octaves)
    # For Voronoi Cells (voronoi_cells)
    # For Concentric Rectangles (concentric_rectangle_width)
    # For Concentric Circles (concentric_circle_width, concentric_origin)
    # For Random Triangles (triangle_size)
    
    # General mask dimensions (used by some, like checkerboard)
    mask_width = IntegerField('Mask/Cell Width (pixels)',
                            default=32,
                            validators=[Optional(), NumberRange(min=2, max=512)]) 
    mask_height = IntegerField('Mask/Cell Height (pixels)',
                            default=32,
                            validators=[Optional(), NumberRange(min=2, max=512)])

    # Striped masks
    stripe_width = IntegerField('Stripe Width (pixels)',
                            default=16,
                            validators=[Optional(), NumberRange(min=1, max=200)])
    stripe_angle = IntegerField('Stripe Angle (degrees)',
                            default=45,
                            validators=[Optional(), NumberRange(min=0, max=180)])
    gradient_direction = SelectField('Gradient Direction (for Linear Gradient Stripe)', choices=[
        ('up', 'Up (0 -> 255)'),
        ('down', 'Down (255 -> 0)')
    ], default='up', validators=[Optional()])

    # Perlin mask
    perlin_noise_scale = FloatField('Perlin Noise Scale',
                                default=0.01,
                                validators=[Optional(), NumberRange(min=0.001, max=0.1)])
    perlin_threshold = FloatField('Perlin Threshold',
                                default=0.5,
                                validators=[Optional(), NumberRange(min=0.0, max=1.0)])
    perlin_octaves = IntegerField('Perlin Octaves (1=smooth, 8=detailed)',
                                default=1,
                                validators=[Optional(), NumberRange(min=1, max=8)])

    # Voronoi mask
    voronoi_num_cells = IntegerField('Number of Voronoi Cells',
                                default=50,
                                validators=[Optional(), NumberRange(min=10, max=500)]) # Renamed

    # Concentric Rectangles mask
    rectangle_band_width = IntegerField('Rectangle Band Width (pixels)',
                                        default=16,
                                        validators=[Optional(), NumberRange(min=2, max=100)]) # Renamed

    # Concentric Circles mask
    circle_band_width = IntegerField('Circle Band Thickness (pixels)',
                                    default=16, # Original: concentric_circle_width
                                    validators=[Optional(), NumberRange(min=2, max=100)])
    circle_origin = SelectField('Circle Origin', choices=[
        ('center', 'Center'),
        ('top-left', 'Top-Left'),
        ('top-right', 'Top-Right'),
        ('bottom-left', 'Bottom-Left'),
        ('bottom-right', 'Bottom-Right')
    ], default='center', validators=[Optional()]) # Renamed

    # Random Triangles mask
    triangle_size = IntegerField('Triangle Size (pixels)',
                                default=50, # Added a default
                                validators=[Optional(), NumberRange(min=4, max=256)])

class OffsetEffectForm(FlaskForm):
    """Form for Offset effect.""" 
    # Original form had offset_x_value, offset_y_value, etc.
    x_value = FloatField('Horizontal Offset Value', default=0.0, validators=[Optional()])
    x_unit = SelectField('Horizontal Offset Unit', choices=[('pixels', 'Pixels'), ('percentage', 'Percentage')], default='pixels', validators=[DataRequired()])
    y_value = FloatField('Vertical Offset Value', default=0.0, validators=[Optional()])
    y_unit = SelectField('Vertical Offset Unit', choices=[('pixels', 'Pixels'), ('percentage', 'Percentage')], default='pixels', validators=[DataRequired()])

class SliceShuffleForm(FlaskForm, SeedMixin):
    """Form for Slice Shuffle effect."""
    count = IntegerField('Slice Count',
                        default=16, # Added a default, original had no default in form but used in effect
                        validators=[DataRequired(), NumberRange(min=4, max=128)]) # Renamed from slice_count
    orientation = SelectField('Slice Orientation', choices=[
        ('rows', 'Rows'), 
        ('columns', 'Columns')
    ], default='rows', validators=[DataRequired()]) # Renamed from slice_orientation
    # seed from SeedMixin (replaces slice_seed)

class SliceOffsetForm(FlaskForm, SeedMixin):
    """Form for Slice Offset effect."""
    count = IntegerField('Number of Slices',
                        default=16, # Added a default
                        validators=[DataRequired(), NumberRange(min=4, max=128)]) # Renamed
    max_offset = IntegerField('Maximum Offset (pixels)',
                            default=50, # Added a default
                            validators=[DataRequired(), NumberRange(min=1, max=512)]) # Renamed
    orientation = SelectField('Slice Orientation', choices=[
        ('rows', 'Rows (horizontal slices, horizontal offset)'), 
        ('columns', 'Columns (vertical slices, vertical offset)')
    ], default='rows', validators=[DataRequired()]) # Renamed
    offset_mode = SelectField('Offset Pattern', choices=[
        ('random', 'Random Offsets'), 
        ('sine', 'Sine Wave Pattern')
    ], default='random', validators=[DataRequired()]) # Renamed
    # Conditional field, only if offset_mode is 'sine'
    frequency = FloatField('Sine Wave Frequency',
                            default=0.1,
                            validators=[Optional(), NumberRange(min=0.01, max=1.0)]) # Renamed
    # seed from SeedMixin (replaces slice_offset_seed)

class SliceReductionForm(FlaskForm):
    """Form for Slice Reduction effect."""
    count = IntegerField('Number of Slices',
                        default=32, # Added a default
                        validators=[DataRequired(), NumberRange(min=16, max=256)]) # Renamed
    reduction_value = IntegerField('Reduction Value',
                                default=2, 
                                validators=[DataRequired(), NumberRange(min=2, max=8)]) # Renamed
    orientation = SelectField('Slice Orientation', choices=[
        ('rows', 'Rows (horizontal slices)'), 
        ('columns', 'Columns (vertical slices)')
    ], default='rows', validators=[DataRequired()]) # Renamed

class ContourForm(FlaskForm, SeedMixin): # Added SeedMixin
    """Form for Contour effect."""
    # Standardized names by removing 'contour_' prefix
    num_levels = IntegerField('Number of Contour Levels',
                            default=10,
                            validators=[DataRequired(), NumberRange(min=5, max=30)])
    noise_std = IntegerField('Noise Amount',
                            default=5,
                            validators=[DataRequired(), NumberRange(min=0, max=10)])
    smooth_sigma = IntegerField('Contour Smoothness',
                                default=16,
                                validators=[DataRequired(), NumberRange(min=1, max=20)])
    line_thickness = IntegerField('Line Thickness',
                                default=1,
                                validators=[DataRequired(), NumberRange(min=1, max=5)])
    grad_threshold = IntegerField('Gradient Threshold',
                                default=28,
                                validators=[DataRequired(), NumberRange(min=1, max=100)])
    min_distance = IntegerField('Minimum Distance Between Points',
                                default=3,
                                validators=[DataRequired(), NumberRange(min=1, max=20)])
    max_line_length = IntegerField('Maximum Line Segment Length',
                                default=256,
                                validators=[DataRequired(), NumberRange(min=50, max=500)])
    blur_kernel_size = IntegerField('Blur Kernel Size (odd number)',
                                default=5,
                                validators=[DataRequired(), NumberRange(min=3, max=33), validate_is_odd]) # Added validate_is_odd
    sobel_kernel_size = IntegerField('Edge Detection Kernel Size (odd number)',
                                default=15,
                                validators=[DataRequired(), NumberRange(min=3, max=33), validate_is_odd]) # Added validate_is_odd
    # seed from SeedMixin

class BlockShuffleForm(FlaskForm, SeedMixin):
    """Form for Block Shuffle effect."""
    # Standardized names
    block_width = IntegerField('Block Width',
                                default=32, # Added a default
                                validators=[DataRequired(), NumberRange(min=2, max=512)])
    block_height = IntegerField('Block Height',
                                default=32, # Added a default
                                validators=[DataRequired(), NumberRange(min=2, max=512)])
    # seed from SeedMixin

class DataMoshBlocksForm(FlaskForm, SeedMixin):
    """Form for Data Mosh Blocks effect."""
    # Standardized names by removing 'data_mosh_' prefix
    operations = IntegerField('Number of Operations',
                            default=69,
                            validators=[DataRequired(), NumberRange(min=1, max=256)])
    block_size = IntegerField('Max Block Size',
                            default=128,
                            validators=[DataRequired(), NumberRange(min=1, max=500)])
    movement = SelectField('Block Movement', choices=[
        ('swap', 'Swap Blocks'), 
        ('in_place', 'In Place')
    ], default='swap', validators=[DataRequired()])
    color_swap = SelectField('Color Channel Swap', choices=[
        ('never', 'Never'), 
        ('always', 'Always'),
        ('random', 'Random') # If random, effect function can use seed
    ], default='random', validators=[DataRequired()])
    invert_colors = SelectField('Color Inversion', choices=[
        ('never', 'Never'), 
        ('always', 'Always'),
        ('random', 'Random') # If random, effect function can use seed
    ], default='random', validators=[DataRequired()]) # Renamed
    shift_values = SelectField('Channel Value Shift', choices=[
        ('never', 'Never'), 
        ('always', 'Always'),
        ('random', 'Random') # If random, effect function can use seed
    ], default='random', validators=[DataRequired()]) # Renamed
    flip_blocks = SelectField('Block Flipping', choices=[
        ('never', 'Never'), 
        ('vertical', 'Vertical'),
        ('horizontal', 'Horizontal'),
        ('random', 'Random') # If random, effect function can use seed
    ], default='random', validators=[DataRequired()]) # Renamed
    # seed from SeedMixin

# === CONSOLIDATED EFFECT FORMS ===

class SliceBlockManipulationForm(FlaskForm, SeedMixin):
    """Consolidated form for slice and block manipulation effects."""
    
    manipulation_type = SelectField('Manipulation Type', choices=[
        ('slice_shuffle', 'Slice Shuffle'),
        ('slice_offset', 'Slice Offset'),
        ('slice_reduction', 'Slice Reduction'),
        ('block_shuffle', 'Block Shuffle')
    ], default='slice_shuffle', validators=[DataRequired()])
    
    # Common parameters
    orientation = SelectField('Orientation', choices=[
        ('rows', 'Rows'),
        ('columns', 'Columns')
    ], default='rows', validators=[DataRequired()])
    
    # Slice parameters
    slice_count = IntegerField('Number of Slices', 
                              default=16, 
                              validators=[Optional(), NumberRange(min=4, max=256)])
    
    # Block parameters
    block_width = IntegerField('Block Width', 
                              default=32, 
                              validators=[Optional(), NumberRange(min=2, max=512)])
    block_height = IntegerField('Block Height', 
                               default=32, 
                               validators=[Optional(), NumberRange(min=2, max=512)])
    
    # Offset-specific parameters
    max_offset = IntegerField('Maximum Offset (pixels)', 
                             default=50, 
                             validators=[Optional(), NumberRange(min=1, max=512)])
    offset_mode = SelectField('Offset Pattern', choices=[
        ('random', 'Random Offsets'),
        ('sine', 'Sine Wave Pattern')
    ], default='random', validators=[Optional()])
    frequency = FloatField('Sine Wave Frequency', 
                          default=0.1, 
                          validators=[Optional(), NumberRange(min=0.01, max=1.0)])
    
    # Reduction-specific parameters
    reduction_value = IntegerField('Reduction Value', 
                                  default=2, 
                                  validators=[Optional(), NumberRange(min=2, max=8)])

class AdvancedPixelSortingForm(FlaskForm, SortingFieldsMixin, SeedMixin):
    """Consolidated form for all pixel sorting effects."""
    
    # Primary sorting method selector
    sorting_method = SelectField('Sorting Method', choices=[
        ('chunk', 'Chunk-Based Sorting'),
        ('full_frame', 'Full Frame Sorting'),
        ('polar', 'Polar Sorting'),
        ('spiral', 'Spiral Sorting'),
        ('voronoi', 'Voronoi-Based Sorting'),
        ('perlin_noise', 'Perlin Noise Sorting'),
        ('perlin_full_frame', 'Perlin Full Frame Sorting'),
        ('wrapped', 'Wrapped Sort')
    ], default='chunk', validators=[DataRequired()])
    
    # Chunk-specific parameters (conditional)
    chunk_width = IntegerField('Chunk Width', 
                              default=48, 
                              validators=[Optional(), NumberRange(min=2, max=2048), validate_multiple_of_2])
    chunk_height = IntegerField('Chunk Height', 
                               default=48, 
                               validators=[Optional(), NumberRange(min=2, max=2048), validate_multiple_of_2])
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
    
    # Voronoi-specific parameters
    num_cells = IntegerField('Number of Cells', 
                            default=69, 
                            validators=[Optional(), NumberRange(min=10, max=1000)])
    size_variation = FloatField('Size Variation', 
                               default=0.8, 
                               validators=[Optional(), NumberRange(min=0.0, max=1.0)])
    sort_order = SelectField('Sort Order', choices=[
        ('clockwise', 'Clockwise'),
        ('counter-clockwise', 'Counter-Clockwise')
    ], default='clockwise', validators=[Optional()])
    voronoi_orientation = SelectField('Line Orientation', choices=[
        ('horizontal', 'Horizontal'),
        ('vertical', 'Vertical'),
        ('radial', 'Radial'),
        ('spiral', 'Spiral')
    ], default='spiral', validators=[Optional()])
    start_position = SelectField('Start Position', choices=[
        ('center', 'Center'),
        ('top-left', 'Top-Left'),
        ('top-right', 'Top-Right'),
        ('bottom-left', 'Bottom-Left'),
        ('bottom-right', 'Bottom-Right')
    ], default='center', validators=[Optional()])
    
    # Perlin noise parameters
    noise_scale = FloatField('Noise Scale', 
                            default=0.008, 
                            validators=[Optional(), NumberRange(min=0.001, max=0.1)])
    pattern_width = IntegerField('Pattern Width', 
                                default=1, 
                                validators=[Optional(), NumberRange(min=1, max=100)])
    
    # Perlin noise specific chunk parameters (different from general chunk params)
    perlin_chunk_width = IntegerField('Perlin Chunk Width', 
                                     default=120, 
                                     validators=[Optional(), NumberRange(min=8, max=1024), validate_multiple_of_8])
    perlin_chunk_height = IntegerField('Perlin Chunk Height', 
                                      default=1024, 
                                      validators=[Optional(), NumberRange(min=8, max=1024), validate_multiple_of_8])
    
    # Direction for applicable methods
    direction = SelectField('Direction', choices=[
        ('horizontal', 'Horizontal'),
        ('vertical', 'Vertical')
    ], default='horizontal', validators=[Optional()])
    
    # Polar-specific parameters
    polar_sort_by = SelectField('Polar Sort By', choices=[
        ('angle', 'Angle (around center)'),
        ('radius', 'Radius (distance from center)')
    ], default='radius', validators=[Optional()])
    
    # Chunk size for methods that need it (polar, spiral)
    chunk_size = IntegerField('Chunk Size', 
                             default=64, 
                             validators=[Optional(), NumberRange(min=8, max=128), validate_multiple_of_8])
    
    # Wrapped sort specific parameters
    wrapped_chunk_width = IntegerField('Wrapped Chunk Width', 
                                      default=12, 
                                      validators=[Optional(), NumberRange(min=1, max=500)])
    wrapped_chunk_height = IntegerField('Wrapped Chunk Height', 
                                       default=123, 
                                       validators=[Optional(), NumberRange(min=1, max=500)])
    wrapped_starting_corner = SelectField('Wrapped Starting Corner', choices=[
        ('top-left', 'Top-Left'),
        ('top-right', 'Top-Right'),
        ('bottom-left', 'Bottom-Left'),
        ('bottom-right', 'Bottom-Right')
    ], default='top-left', validators=[Optional()])
    wrapped_flow_direction = SelectField('Chunk Flow Direction', choices=[
        ('primary', 'Primary (down/right from corner)'),
        ('secondary', 'Secondary (right/down from corner)')
    ], default='primary', validators=[Optional()])

# Add other specific effect forms here following the pattern:
# class AnotherEffectForm(FlaskForm, OptionalMixin1, OptionalMixin2):
#     param1 = ...
#     param2 = ...

# --- Main Form for Effect Selection ---
class ImageProcessForm(FlaskForm):
    """Main form for image upload and primary effect selection."""
    primary_image = FileField('Primary Image', validators=[
        FileRequired(),
        FileAllowed(['jpg', 'jpeg', 'png', 'gif'], 'Images only!')
    ])
    effect = SelectField('Effect', choices=[
        # === COLOR & TONE EFFECTS ===
        ('color_filter', 'Color Filter'),
        ('color_channel', 'Color Channel Manipulation'),
        ('channel_shift', 'RGB Channel Shift'),
        ('curved_hue_shift', 'Hue Skrift'),
        ('color_shift_expansion', 'Color Shift Expansion'),
        ('histogram_glitch', 'Histogram Glitch'),
        ('posterize', 'Posterfy'),
        
        # === PIXEL SORTING EFFECTS (CONSOLIDATED) ===
        ('advanced_pixel_sorting', 'Advanced Pixel Sorting'),
        
        # === PIXELATION & STYLIZATION ===
        ('pixelate', 'Pixelate'),
        ('voronoi_pixelate', 'Voronoi Pixelate'),
        ('gaussian_blur', 'Gaussian Blur'),
        ('sharpen_effect', 'Sharpen Effect'),
        ('vhs_effect', 'VHS Effect'),
        ('concentric_shapes', 'Concentric Shapes'),
        ('contour', 'Contour'),
        
        # === DISTORTION & DISPLACEMENT ===
        ('pixel_drift', 'Pixel Drift'),
        ('perlin_displacement', 'Perlin Displacement'),
        ('wave_distortion', 'Wave Distortion'),
        ('ripple', 'Ripple Effect'),
        ('pixel_scatter', 'Pixel Scatter'),
        ('offset', 'Offset'),
        
        # === SLICE & BLOCK EFFECTS (CONSOLIDATED) ===
        ('slice_block_manipulation', 'Slice & Block Manipulation'),
        
        # === GLITCH & CORRUPTION ===
        ('bit_manipulation', 'Bit Manipulation'),
        ('data_mosh_blocks', 'Data Mosh Blocks'),
        ('databend', 'Databending'),
        ('jpeg_artifacts', 'JPEG Artifacts'),
        ('noise_effect', 'Noise Effect'),
        ('chromatic_aberration', 'Chromatic Aberration'),
        
        # === BLEND & COMPOSITE ===
        ('double_expose', 'Double Expose'),
        ('masked_merge', 'Masked Merge'),
    ], validators=[DataRequired()])
    submit = SubmitField('Process Image')

# The monolithic ImageProcessForm class previously containing all effect parameters
# and its custom .validate() method has been removed.
# Effect-specific parameters are now defined in their own form classes.
# The custom validators (validate_chunk_size, etc.) are preserved at the top.

# Contour Effect Fields
contour_num_levels = IntegerField('Number of Contour Levels',
default=10,
validators=[Optional(), NumberRange(min=5, max=30, message="Number of contour levels must be between 5 and 30")])
contour_noise_std = IntegerField('Noise Amount',
default=5,
validators=[Optional(), NumberRange(min=0, max=10, message="Noise amount must be between 0 and 10")])
contour_smooth_sigma = IntegerField('Contour Smoothness',
default=16,
validators=[Optional(), NumberRange(min=1, max=20, message="Smoothness must be between 1 and 20")])
contour_line_thickness = IntegerField('Line Thickness',
default=1,
validators=[Optional(), NumberRange(min=1, max=5, message="Line thickness must be between 1 and 5")])
contour_grad_threshold = IntegerField('Gradient Threshold',
default=28,
validators=[Optional(), NumberRange(min=1, max=100, message="Gradient threshold must be between 1 and 100")])
contour_min_distance = IntegerField('Minimum Distance Between Points',
default=3,
validators=[Optional(), NumberRange(min=1, max=20, message="Minimum distance must be between 1 and 20")])
contour_max_line_length = IntegerField('Maximum Line Segment Length',
default=256,
validators=[Optional(), NumberRange(min=50, max=500, message="Maximum line length must be between 50 and 500")])
contour_blur_kernel_size = IntegerField('Blur Kernel Size',
default=5,
validators=[Optional(), NumberRange(min=3, max=33, message="Blur kernel size must be between 3 and 33")])
contour_sobel_kernel_size = IntegerField('Edge Detection Kernel Size',
default=15,
validators=[Optional(), NumberRange(min=3, max=33, message="Edge detection kernel size must be between 3 and 33")])

# Block Shuffle
block_shuffle_block_width = IntegerField('Block Width', validators=[Optional(), NumberRange(min=2, max=512)])
block_shuffle_block_height = IntegerField('Block Height', validators=[Optional(), NumberRange(min=2, max=512)])
block_shuffle_seed = IntegerField('Random Seed (optional)', validators=[Optional(), NumberRange(min=1, max=9999)])