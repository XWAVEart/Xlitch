import cv2
import numpy as np
from skimage import measure
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
import random

def do_lines_intersect(p1, p2, p3, p4):
    """Check if two line segments intersect"""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

class LineGrid:
    def __init__(self, cell_size=25):
        self.cell_size = cell_size
        self.grid = defaultdict(list)
    
    def get_cell(self, point):
        return (point[0] // self.cell_size, point[1] // self.cell_size)
    
    def add_line(self, p1, p2):
        # Get cells that this line passes through
        cells = set()
        x1, y1 = p1
        x2, y2 = p2
        
        # Add cells along the line using Bresenham's algorithm
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        n = 1 + dx + dy
        x_inc = 1 if x2 > x1 else -1
        y_inc = 1 if y2 > y1 else -1
        error = dx - dy
        dx *= 2
        dy *= 2
        
        for _ in range(n):
            cells.add(self.get_cell((x, y)))
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        
        # Add line to all relevant cells
        for cell in cells:
            self.grid[cell].append((p1, p2))
    
    def check_intersection(self, p1, p2):
        # Only check lines in the same or adjacent cells
        cell = self.get_cell(p1)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_cell = (cell[0] + dx, cell[1] + dy)
                for line in self.grid.get(check_cell, []):
                    if do_lines_intersect(p1, p2, line[0], line[1]):
                        return True
        return False

def contour_effect(image, num_levels=15, noise_std=2, smooth_sigma=8, 
                   line_thickness=1, grad_threshold=20, min_distance=5, 
                   max_line_length=256, blur_kernel_size=5, sobel_kernel_size=15, seed=None):
    """
    Create contour line art from an image.
    
    Args:
        image (PIL.Image): Input image to process.
        num_levels (int): Number of contour levels (5-30).
        noise_std (int): Standard deviation of noise perturbation (0-10).
        smooth_sigma (int): Smoothness of noise along contours (1-20).
        line_thickness (int): Thickness of contour lines (1-5).
        grad_threshold (int): Gradient threshold to reduce background clutter (1-100).
        min_distance (int): Minimum distance between contour points (1-20).
        max_line_length (int): Maximum length of a line segment (50-500).
        blur_kernel_size (int): Size of Gaussian blur kernel (3-33, must be odd).
        sobel_kernel_size (int): Size of Sobel edge detection kernel (3-33, must be odd).
        seed (int, optional): Seed for random number generation. Defaults to None.
    
    Returns:
        PIL.Image: Processed image with contour lines.
    """
    # Convert PIL Image to OpenCV format
    import numpy as np
    from PIL import Image

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    image_np = np.array(image)
    
    # Convert to grayscale if it's a color image
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np.copy()

    # Ensure blur_kernel_size is odd
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1
    
    # Apply Gaussian blur to smooth the image
    gray_blur = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)

    # Ensure sobel_kernel_size is odd
    if sobel_kernel_size % 2 == 0:
        sobel_kernel_size += 1
    
    # Compute the gradient magnitude to emphasize edges
    grad_x = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
    grad_y = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Create a blank output image (black background)
    height, width = gray.shape
    if len(image_np.shape) == 3:
        output = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        output = np.zeros((height, width), dtype=np.uint8)
    
    # Create a distance map to track drawn points
    distance_map = np.zeros((height, width), dtype=np.uint8)
    
    # Create a grid for efficient line intersection checking
    line_grid = LineGrid(cell_size=25)

    # Determine intensity range for contour levels
    min_val, max_val = np.min(gray_blur), np.max(gray_blur)
    levels = np.linspace(min_val, max_val, num_levels * 2)
    levels = min_val + (max_val - min_val) * np.sqrt((levels - min_val) / (max_val - min_val))
    levels = np.unique(levels)[:num_levels]

    # Generate and draw contours for each level
    for level in levels:
        # Find contours at the current intensity level using Marching Squares
        contours = measure.find_contours(gray_blur, level)
        
        # Process each contour
        for contour in contours:
            if len(contour) > 1:  # Skip single-point contours
                # Check if the contour is in a high-gradient area
                contour_points = contour.astype(np.int32)
                valid_indices = np.where(
                    (contour_points[:, 0] >= 0) & 
                    (contour_points[:, 0] < height) & 
                    (contour_points[:, 1] >= 0) & 
                    (contour_points[:, 1] < width)
                )[0]
                
                if len(valid_indices) < 2:
                    continue
                    
                contour_points = contour_points[valid_indices]
                grad_values = grad_mag[contour_points[:, 0], contour_points[:, 1]]
                avg_grad = np.mean(grad_values)
                
                if avg_grad < grad_threshold:
                    continue

                # Generate random noise for x and y coordinates
                offset_x = np.random.normal(0, noise_std, len(contour))
                offset_y = np.random.normal(0, noise_std, len(contour))
                
                # Smooth the noise to avoid jagged perturbations
                smooth_offset_x = gaussian_filter1d(offset_x, sigma=smooth_sigma)
                smooth_offset_y = gaussian_filter1d(offset_y, sigma=smooth_sigma)
                
                # Apply perturbation to contour points and swap coordinates
                perturbed = contour + np.column_stack((smooth_offset_y, smooth_offset_x))
                perturbed = np.fliplr(perturbed)
                
                # Convert to integer coordinates for drawing
                points = perturbed.astype(np.int32)
                
                # Filter points that are too close to already drawn points
                current_segment = []
                
                for i, point in enumerate(points):
                    x, y = point
                    if 0 <= x < width and 0 <= y < height:
                        if distance_map[y, x] == 0:
                            if i > 0:
                                prev_point = points[i-1]
                                # Check if line would be too long
                                if len(current_segment) > 0:
                                    dist = np.sqrt((x - current_segment[0][0])**2 + (y - current_segment[0][1])**2)
                                    if dist > max_line_length:
                                        # Draw current segment and start new one
                                        if len(current_segment) > 1:
                                            cv2.polylines(output, [np.array(current_segment)], isClosed=False,
                                                        color=(255, 255, 255), thickness=line_thickness)
                                            for j in range(len(current_segment) - 1):
                                                line_grid.add_line(current_segment[j], current_segment[j+1])
                                        current_segment = []
                                
                                # Check for intersections
                                if not line_grid.check_intersection(prev_point, point):
                                    current_segment.append(point)
                                    cv2.circle(distance_map, (x, y), min_distance, 255, -1)
                                else:
                                    # Draw current segment and start new one
                                    if len(current_segment) > 1:
                                        cv2.polylines(output, [np.array(current_segment)], isClosed=False,
                                                    color=(255, 255, 255), thickness=line_thickness)
                                        for j in range(len(current_segment) - 1):
                                            line_grid.add_line(current_segment[j], current_segment[j+1])
                                    current_segment = []
                            else:
                                current_segment.append(point)
                                cv2.circle(distance_map, (x, y), min_distance, 255, -1)
                
                # Draw any remaining segment
                if len(current_segment) > 1:
                    cv2.polylines(output, [np.array(current_segment)], isClosed=False,
                                color=(255, 255, 255), thickness=line_thickness)
                    for j in range(len(current_segment) - 1):
                        line_grid.add_line(current_segment[j], current_segment[j+1])

    # Convert back to PIL Image and return
    return Image.fromarray(output) 