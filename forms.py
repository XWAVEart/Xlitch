from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SelectField, StringField, FloatField, IntegerField
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
        ('pixel_sort_original', 'Pixel Sorting (Original)'),
        ('pixel_sort_corner', 'Pixel Sorting (Corner-to-Corner)'),
        ('full_frame_sort', 'Full Frame Sorting'),
        ('polar_sort', 'Polar Sorting'),
        ('perlin_noise_sort', 'Perlin Noise Sorting'),
        ('perlin_full_frame', 'Perlin Full Frame'),
        ('perlin_merge', 'Perlin Merge'),
        ('perlin_displacement', 'Perlin Displacement'),
        ('pixelate', 'Pixelate'),
        ('concentric_shapes', 'Concentric Shapes'),
        ('color_shift_expansion', 'Color Shift Expansion'),
        ('color_channel', 'Color Channel Manipulation'),
        ('double_expose', 'Double Expose'),
        ('data_mosh_blocks', 'Data Mosh Blocks'),
        ('pixel_drift', 'Pixel Drift'),
        ('spiral_sort', 'Spiral Sort'),
        ('spiral_sort_2', 'Spiral Sort 2'),
        ('bit_manipulation', 'Bit Manipulation')
    ], validators=[DataRequired()])
    
    # Pixel Sorting (Original)
    direction = SelectField('Direction', choices=[
        ('horizontal', 'Horizontal'), 
        ('vertical', 'Vertical')
    ], default='horizontal', validators=[Optional()])
    chunk_width = IntegerField('Chunk Width', 
                              default=32,
                              validators=[Optional(), NumberRange(min=8, max=2048), validate_multiple_of_8])
    chunk_height = IntegerField('Chunk Height', 
                               default=32,
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
    
    # Pixel Sorting (Corner-to-Corner)
    corner_chunk_width = IntegerField('Chunk Width', 
                                     default=32,
                                     validators=[Optional(), NumberRange(min=8, max=2048), validate_multiple_of_8])
    corner_chunk_height = IntegerField('Chunk Height', 
                                      default=32,
                                      validators=[Optional(), NumberRange(min=8, max=2048), validate_multiple_of_8])
    corner_sort_by = SelectField('Sort By', choices=[
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
    starting_corner = SelectField('Starting Corner', choices=[
        ('top-left', 'Top-Left'), 
        ('top-right', 'Top-Right'),
        ('bottom-left', 'Bottom-Left'), 
        ('bottom-right', 'Bottom-Right')
    ], default='top-left', validators=[Optional()])
    corner_direction = SelectField('Direction', choices=[
        ('horizontal', 'Horizontal'), 
        ('vertical', 'Vertical')
    ], default='horizontal', validators=[Optional()])
    
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
                              default=8,
                              validators=[Optional(), NumberRange(min=8, max=48), validate_multiple_of_8])
    drift_intensity = FloatField('Drift Intensity Multiplier', 
                               default=1.0,
                               validators=[Optional(), NumberRange(min=0.1, max=10.0)])
    
    # Spiral Sort
    spiral_chunk_size = IntegerField('Chunk Size', 
                                   default=32,
                                   validators=[Optional(), NumberRange(min=8, max=128), validate_multiple_of_8])
    spiral_order = SelectField('Sort Order', choices=[
        ('lightest-to-darkest', 'Lightest to Darkest'), 
        ('darkest-to-lightest', 'Darkest to Lightest')
    ], default='lightest-to-darkest', validators=[Optional()])
    
    # Spiral Sort 2
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
                                default=8,
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
    ], default='brightness', validators=[Optional()])
    full_frame_reverse = SelectField('Sort Order', choices=[
        ('false', 'Ascending (Low to High)'), 
        ('true', 'Descending (High to Low)')
    ], default='false', validators=[Optional()])
    
    # Polar Sorting
    polar_chunk_size = IntegerField('Chunk Size', 
                                  default=32,
                                  validators=[Optional(), NumberRange(min=8, max=128), validate_multiple_of_8])
    polar_sort_by = SelectField('Sort By', choices=[
        ('angle', 'Angle (around center)'), 
        ('radius', 'Radius (distance from center)')
    ], default='angle', validators=[Optional()])
    polar_reverse = SelectField('Sort Order', choices=[
        ('false', 'Ascending (Low to High)'), 
        ('true', 'Descending (High to Low)')
    ], default='false', validators=[Optional()])
    
    # Perlin Noise Sorting
    perlin_chunk_width = IntegerField('Chunk Width', 
                                   default=32,
                                   validators=[Optional(), NumberRange(min=8, max=1024), validate_multiple_of_8])
    perlin_chunk_height = IntegerField('Chunk Height', 
                                    default=32,
                                    validators=[Optional(), NumberRange(min=8, max=1024), validate_multiple_of_8])
    perlin_noise_scale = FloatField('Noise Scale', 
                                  default=0.01,
                                  validators=[Optional(), NumberRange(min=0.001, max=0.1)])
    perlin_direction = SelectField('Direction', choices=[
        ('horizontal', 'Horizontal'), 
        ('vertical', 'Vertical')
    ], default='horizontal', validators=[Optional()])
    perlin_reverse = SelectField('Sort Order', choices=[
        ('false', 'Ascending (Low to High)'), 
        ('true', 'Descending (High to Low)')
    ], default='false', validators=[Optional()])
    perlin_seed = IntegerField('Noise Seed',
                             default=42,
                             validators=[Optional(), NumberRange(min=1, max=9999)])
    
    # Perlin Full Frame
    perlin_full_frame_noise_scale = FloatField('Noise Scale', 
                                            default=0.01,
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
    ], default='brightness', validators=[Optional()])
    perlin_full_frame_reverse = SelectField('Sort Order', choices=[
        ('false', 'Ascending (Low to High)'), 
        ('true', 'Descending (High to Low)')
    ], default='false', validators=[Optional()])
    perlin_full_frame_seed = IntegerField('Noise Seed',
                                        default=42,
                                        validators=[Optional(), NumberRange(min=1, max=9999)])
    
    # Perlin Merge
    perlin_merge_secondary = FileField('Secondary Image', validators=[
        Optional(),
        FileAllowed(['jpg', 'jpeg', 'png', 'gif'], 'Images only!')
    ])
    perlin_merge_noise_scale = FloatField('Noise Scale', 
                                       default=0.01,
                                       validators=[Optional(), NumberRange(min=0.001, max=0.1)])
    perlin_merge_threshold = FloatField('Threshold', 
                                     default=0.5,
                                     validators=[Optional(), NumberRange(min=0.0, max=1.0)])
    perlin_merge_seed = IntegerField('Noise Seed',
                                   default=42,
                                   validators=[Optional(), NumberRange(min=1, max=9999)])
    
    # Pixelate
    pixelate_width = IntegerField('Pixel Width',
                                default=8,
                                validators=[Optional(), NumberRange(min=2, max=64)])
    pixelate_height = IntegerField('Pixel Height',
                                 default=8,
                                 validators=[Optional(), NumberRange(min=2, max=64)])
    pixelate_attribute = SelectField('Attribute', choices=[
        ('color', 'Color (Most Common)'),
        ('brightness', 'Brightness'),
        ('hue', 'Hue'),
        ('saturation', 'Saturation'),
        ('luminance', 'Luminance')
    ], default='color', validators=[Optional()])
    pixelate_bins = IntegerField('Number of Bins',
                               default=100,
                               validators=[Optional(), NumberRange(min=10, max=1000)])
    
    # Concentric Shapes
    concentric_num_points = IntegerField('Number of Points',
                                      default=10,
                                      validators=[Optional(), NumberRange(min=1, max=100)])
    shape_type = SelectField('Shape Type', choices=[
        ('square', 'Square'),
        ('circle', 'Circle'),
        ('hexagon', 'Hexagon'),
        ('triangle', 'Triangle')
    ], default='square', validators=[Optional()])
    concentric_thickness = IntegerField('Line Thickness',
                                     default=2,
                                     validators=[Optional(), NumberRange(min=1, max=10)])
    spacing = IntegerField('Spacing',
                         default=10,
                         validators=[Optional(), NumberRange(min=1, max=50)])
    rotation_angle = IntegerField('Rotation Angle',
                                default=0,
                                validators=[Optional(), NumberRange(min=0, max=360)])
    darken_step = IntegerField('Darken Step',
                             default=0,
                             validators=[Optional(), NumberRange(min=0, max=255)])
    color_shift = IntegerField('Color Shift',
                             default=0,
                             validators=[Optional(), NumberRange(min=0, max=360)])
    
    # Color Shift Expansion
    color_shift_num_points = IntegerField('Number of Seed Points',
                                       default=5,
                                       validators=[Optional(), NumberRange(min=1, max=100)])
    color_shift_amount = IntegerField('Shift Amount',
                                   default=5,
                                   validators=[Optional(), NumberRange(min=1, max=30)])
    expansion_type = SelectField('Expansion Type', choices=[
        ('square', 'Square (8-way)'),
        ('cross', 'Cross (4-way)'),
        ('circular', 'Circular')
    ], default='square', validators=[Optional()])
    color_shift_mode = SelectField('Mode', choices=[
        ('classic', 'Classic (Fixed Shift)'),
        ('xtreme', 'Xtreme (Distance-based Shift)')
    ], default='xtreme', validators=[Optional()])
    
    # Perlin Displacement
    perlin_displacement_scale = IntegerField('Noise Scale',
                                          default=100,
                                          validators=[Optional(), NumberRange(min=10, max=500)])
    perlin_displacement_intensity = IntegerField('Displacement Intensity',
                                              default=30,
                                              validators=[Optional(), NumberRange(min=1, max=100)])
    perlin_displacement_octaves = IntegerField('Octaves',
                                            default=6,
                                            validators=[Optional(), NumberRange(min=1, max=10)])
    perlin_displacement_persistence = FloatField('Persistence',
                                              default=0.5,
                                              validators=[Optional(), NumberRange(min=0.1, max=1.0)])
    perlin_displacement_lacunarity = FloatField('Lacunarity',
                                             default=2.0,
                                             validators=[Optional(), NumberRange(min=1.0, max=4.0)])
    
    # Add form fields for Data Mosh Blocks effect
    data_mosh_operations = IntegerField('Number of Operations', 
                                      default=64,
                                      validators=[Optional(), NumberRange(min=1, max=256)])
    data_mosh_block_size = IntegerField('Max Block Size', 
                                      default=50,
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
    ], default='never', validators=[Optional()])
    data_mosh_shift = SelectField('Channel Value Shift', choices=[
        ('never', 'Never'), 
        ('always', 'Always'),
        ('random', 'Random')
    ], default='never', validators=[Optional()])
    data_mosh_flip = SelectField('Block Flipping', choices=[
        ('never', 'Never'), 
        ('vertical', 'Vertical'),
        ('horizontal', 'Horizontal'),
        ('random', 'Random')
    ], default='never', validators=[Optional()])
    data_mosh_seed = IntegerField('Random Seed', 
                                validators=[Optional()])
    
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
        if effect == 'pixel_sort_original':
            if not self.direction.data:
                self.direction.errors = ['Direction is required for Pixel Sorting']
                logger.debug("Missing direction for pixel_sort_original")
                return False
            if not self.chunk_width.data:
                self.chunk_width.errors = ['Chunk width is required for Pixel Sorting']
                logger.debug("Missing chunk_width for pixel_sort_original")
                return False
            if not self.chunk_height.data:
                self.chunk_height.errors = ['Chunk height is required for Pixel Sorting']
                logger.debug("Missing chunk_height for pixel_sort_original")
                return False
            if not self.sort_by.data:
                self.sort_by.errors = ['Sort by is required for Pixel Sorting']
                logger.debug("Missing sort_by for pixel_sort_original")
                return False
                
        elif effect == 'pixel_sort_corner':
            if not self.corner_chunk_width.data:
                self.corner_chunk_width.errors = ['Chunk width is required for Corner Pixel Sorting']
                logger.debug("Missing corner_chunk_width for pixel_sort_corner")
                return False
            if not self.corner_chunk_height.data:
                self.corner_chunk_height.errors = ['Chunk height is required for Corner Pixel Sorting']
                logger.debug("Missing corner_chunk_height for pixel_sort_corner")
                return False
            if not self.corner_sort_by.data:
                self.corner_sort_by.errors = ['Sort by is required for Corner Pixel Sorting']
                logger.debug("Missing corner_sort_by for pixel_sort_corner")
                return False
            if not self.starting_corner.data:
                self.starting_corner.errors = ['Starting corner is required for Corner Pixel Sorting']
                logger.debug("Missing starting_corner for pixel_sort_corner")
                return False
            if not self.corner_direction.data:
                self.corner_direction.errors = ['Direction is required for Corner Pixel Sorting']
                logger.debug("Missing corner_direction for pixel_sort_corner")
                return False
                
        elif effect == 'color_channel':
            if not self.manipulation_type.data:
                self.manipulation_type.errors = ['Manipulation type is required for Color Channel Manipulation']
                logger.debug("Missing manipulation_type for color_channel")
                return False
                
            # Validate based on manipulation type
            manipulation_type = self.manipulation_type.data
            logger.debug(f"Manipulation type: {manipulation_type}")
            
            if manipulation_type == 'swap' and not self.swap_choice.data:
                self.swap_choice.errors = ['Swap choice is required for Color Channel Swap']
                logger.debug("Missing swap_choice for swap manipulation")
                return False
            elif manipulation_type == 'invert' and not self.invert_choice.data:
                self.invert_choice.errors = ['Invert choice is required for Color Channel Invert']
                logger.debug("Missing invert_choice for invert manipulation")
                return False
            elif manipulation_type == 'negative':
                # Negative doesn't require any additional parameters
                pass
            elif manipulation_type == 'adjust':
                if not self.adjust_choice.data:
                    self.adjust_choice.errors = ['Adjust choice is required for Color Channel Adjust']
                    logger.debug("Missing adjust_choice for adjust manipulation")
                    return False
                if not self.intensity_factor.data:
                    self.intensity_factor.errors = ['Intensity factor is required for Color Channel Adjust']
                    logger.debug("Missing intensity_factor for adjust manipulation")
                    return False
                    
        elif effect == 'double_expose':
            if not self.secondary_image.data:
                self.secondary_image.errors = ['Secondary image is required for Double Expose']
                logger.debug("Missing secondary_image for double_expose")
                return False
            if not self.blend_mode.data:
                self.blend_mode.errors = ['Blend mode is required for Double Expose']
                logger.debug("Missing blend_mode for double_expose")
                return False
            if not self.opacity.data:
                self.opacity.errors = ['Opacity is required for Double Expose']
                logger.debug("Missing opacity for double_expose")
                return False
                
        elif effect == 'data_mosh_blocks':
            if not self.data_mosh_operations.data:
                self.data_mosh_operations.errors = ['Number of operations is required for Data Mosh Blocks']
                logger.debug("Missing data_mosh_operations for data_mosh_blocks")
                return False
            if not self.data_mosh_block_size.data:
                self.data_mosh_block_size.errors = ['Max block size is required for Data Mosh Blocks']
                logger.debug("Missing data_mosh_block_size for data_mosh_blocks")
                return False
            if not self.data_mosh_movement.data:
                self.data_mosh_movement.errors = ['Block movement is required for Data Mosh Blocks']
                logger.debug("Missing data_mosh_movement for data_mosh_blocks")
                return False
            if not self.data_mosh_color_swap.data:
                self.data_mosh_color_swap.errors = ['Color channel swap is required for Data Mosh Blocks']
                logger.debug("Missing data_mosh_color_swap for data_mosh_blocks")
                return False
            if not self.data_mosh_invert.data:
                self.data_mosh_invert.errors = ['Color inversion is required for Data Mosh Blocks']
                logger.debug("Missing data_mosh_invert for data_mosh_blocks")
                return False
            if not self.data_mosh_shift.data:
                self.data_mosh_shift.errors = ['Channel value shift is required for Data Mosh Blocks']
                logger.debug("Missing data_mosh_shift for data_mosh_blocks")
                return False
            if not self.data_mosh_flip.data:
                self.data_mosh_flip.errors = ['Block flipping is required for Data Mosh Blocks']
                logger.debug("Missing data_mosh_flip for data_mosh_blocks")
                return False
                
        elif effect == 'pixel_drift':
            if not self.drift_direction.data:
                self.drift_direction.errors = ['Drift direction is required for Pixel Drift']
                logger.debug("Missing drift_direction for pixel_drift")
                return False
                
        elif effect == 'spiral_sort':
            if not self.spiral_chunk_size.data:
                self.spiral_chunk_size.errors = ['Chunk size is required for Spiral Sort']
                logger.debug("Missing spiral_chunk_size for spiral_sort")
                return False
            if not self.spiral_order.data:
                self.spiral_order.errors = ['Sort order is required for Spiral Sort']
                logger.debug("Missing spiral_order for spiral_sort")
                return False
                
        elif effect == 'spiral_sort_2':
            if not self.spiral2_chunk_size.data:
                self.spiral2_chunk_size.errors = ['Chunk size is required for Spiral Sort 2']
                logger.debug("Missing spiral2_chunk_size for spiral_sort_2")
                return False
            if not self.spiral2_sort_by.data:
                self.spiral2_sort_by.errors = ['Sort by is required for Spiral Sort 2']
                logger.debug("Missing spiral2_sort_by for spiral_sort_2")
                return False
            if not self.spiral2_reverse.data:
                self.spiral2_reverse.errors = ['Sort order is required for Spiral Sort 2']
                logger.debug("Missing spiral2_reverse for spiral_sort_2")
                return False
                
        elif effect == 'bit_manipulation':
            if not self.bit_chunk_size.data:
                self.bit_chunk_size.errors = ['Chunk size is required for Bit Manipulation']
                logger.debug("Missing bit_chunk_size for bit_manipulation")
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
                
        elif effect == 'perlin_merge':
            if not self.perlin_merge_secondary.data:
                self.perlin_merge_secondary.errors = ['Secondary image is required for Perlin Merge']
                logger.debug("Missing perlin_merge_secondary for perlin_merge")
                return False
            if not self.perlin_merge_noise_scale.data:
                self.perlin_merge_noise_scale.errors = ['Noise scale is required for Perlin Merge']
                logger.debug("Missing perlin_merge_noise_scale for perlin_merge")
                return False
            if not self.perlin_merge_threshold.data:
                self.perlin_merge_threshold.errors = ['Threshold is required for Perlin Merge']
                logger.debug("Missing perlin_merge_threshold for perlin_merge")
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
            if not self.color_shift_mode.data:
                self.color_shift_mode.errors = ['Mode is required for Color Shift Expansion']
                logger.debug("Missing color_shift_mode for color_shift_expansion")
                return False
                
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
                
        # If we get here, validation passed
        logger.debug("Validation passed")
        return True