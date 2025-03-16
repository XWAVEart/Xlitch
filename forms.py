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
        ('pixelate', 'Pixelate'),
        ('concentric_squares', 'Concentric Squares'),
        ('color_channel', 'Color Channel Manipulation'),
        ('data_moshing', 'Double Expose'),
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
        ('adjust', 'Adjust Channel Intensity')
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
    
    # Concentric Squares
    concentric_num_points = IntegerField('Number of Points',
                                      default=10,
                                      validators=[Optional(), NumberRange(min=1, max=100)])
    concentric_num_squares = IntegerField('Squares per Point',
                                       default=5,
                                       validators=[Optional(), NumberRange(min=1, max=20)])
    concentric_thickness = IntegerField('Line Thickness',
                                     default=2,
                                     validators=[Optional(), NumberRange(min=1, max=10)])
    
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
            elif manipulation_type == 'adjust':
                if not self.adjust_choice.data:
                    self.adjust_choice.errors = ['Adjust choice is required for Color Channel Adjust']
                    logger.debug("Missing adjust_choice for adjust manipulation")
                    return False
                if not self.intensity_factor.data:
                    self.intensity_factor.errors = ['Intensity factor is required for Color Channel Adjust']
                    logger.debug("Missing intensity_factor for adjust manipulation")
                    return False
                    
        elif effect == 'data_moshing':
            if not self.secondary_image.data:
                self.secondary_image.errors = ['Secondary image is required for Double Expose']
                logger.debug("Missing secondary_image for data_moshing")
                return False
            if not self.blend_mode.data:
                self.blend_mode.errors = ['Blend mode is required for Double Expose']
                logger.debug("Missing blend_mode for data_moshing")
                return False
            if not self.opacity.data:
                self.opacity.errors = ['Opacity is required for Double Expose']
                logger.debug("Missing opacity for data_moshing")
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
                
        # If we get here, validation passed
        logger.debug("Validation passed")
        return True