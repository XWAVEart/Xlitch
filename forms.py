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
        ('color_channel', 'Color Channel Manipulation'),
        ('data_moshing', 'Double Expose'),
        ('pixel_drift', 'Pixel Drift'),
        ('spiral_sort', 'Spiral Sort'),
        ('bit_manipulation', 'Bit Manipulation')
    ], validators=[DataRequired()])
    
    # Pixel Sorting (Original)
    direction = SelectField('Direction', choices=[
        ('horizontal', 'Horizontal'), 
        ('vertical', 'Vertical')
    ], default='horizontal', validators=[Optional()])
    chunk_width = IntegerField('Chunk Width', 
                              default=32,
                              validators=[Optional(), NumberRange(min=8, max=1000), validate_multiple_of_8])
    chunk_height = IntegerField('Chunk Height', 
                               default=32,
                               validators=[Optional(), NumberRange(min=8, max=1000), validate_multiple_of_8])
    sort_by = SelectField('Sort By', choices=[
        ('color', 'Color'), 
        ('brightness', 'Brightness'), 
        ('hue', 'Hue')
    ], default='brightness', validators=[Optional()])
    
    # Pixel Sorting (Corner-to-Corner)
    corner_chunk_width = IntegerField('Chunk Width', 
                                     default=32,
                                     validators=[Optional(), NumberRange(min=8, max=1000), validate_multiple_of_8])
    corner_chunk_height = IntegerField('Chunk Height', 
                                      default=32,
                                      validators=[Optional(), NumberRange(min=8, max=1000), validate_multiple_of_8])
    corner_sort_by = SelectField('Sort By', choices=[
        ('color', 'Color'), 
        ('brightness', 'Brightness'), 
        ('hue', 'Hue')
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
    
    # Bit Manipulation
    bit_chunk_size = IntegerField('Chunk Size', 
                                default=8,
                                validators=[Optional(), NumberRange(min=8, max=128), validate_multiple_of_8])
    
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
                
        elif effect == 'bit_manipulation':
            if not self.bit_chunk_size.data:
                self.bit_chunk_size.errors = ['Chunk size is required for Bit Manipulation']
                logger.debug("Missing bit_chunk_size for bit_manipulation")
                return False
                
        # If we get here, validation passed
        logger.debug("Validation passed")
        return True