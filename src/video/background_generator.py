
import numpy as np
import cv2
from moviepy.editor import VideoClip, ImageClip, ColorClip, CompositeVideoClip
import math
from typing import Tuple, List

class BackgroundGenerator:
    """
    Generates video backgrounds to ensure NO BLACK SCREENS.
    """
    
    @staticmethod
    def create_starfield(duration: float, width: int = 1080, height: int = 1920, speed: int = 2) -> VideoClip:
        """
        Creates a scrolling starfield background.
        """
        # Create a static star image larger than frame
        # For performance in MoviePy with Python, generating frames in numpy per second is slow.
        # Optimization: Generate one large image and scroll it?
        
        # Let's generate a simple particle effect using a make_frame function with modest particle count.
        
        num_stars = 150
        stars = np.random.randint(0, [width, height], size=(num_stars, 2))
        sizes = np.random.randint(1, 3, size=num_stars)
        colors = np.random.randint(200, 255, size=(num_stars,)) # Greyscale brightness
        
        def make_frame(t):
            # Dark Navy Background
            img = np.zeros((height, width, 3), dtype=np.uint8)
            img[:] = (39, 14, 10) # BGR for #0A0E27 (OpenCV uses BGR)
            
            # Draw stars
            for i, (x, y) in enumerate(stars):
                # Move stars
                y_pos = (y + t * 50 * speed) % height 
                x_pos = (x + math.sin(t + i) * 10) % width # Slight wobble
                
                c = int(colors[i])
                cv2.circle(img, (int(x_pos), int(y_pos)), sizes[i], (c, c, c), -1)
                
            return img # Returns BGR, MoviePy expects RGB? 
            # MoviePy uses RGB. OpenCV uses BGR.
            # We should construct RGB directly or convert.
            
        def make_frame_rgb(t):
            # RGB Background #0A0E27 -> (10, 14, 39)
            img = np.zeros((height, width, 3), dtype=np.uint8)
            img[:] = (10, 14, 39) 
            
            for i, (x, y) in enumerate(stars):
                y_pos = (y + t * 50 * speed) % height
                # x_pos = (x + math.sin(t + i) * 10) % width
                
                # Simple straight scroll for speed
                
                c = int(colors[i])
                # draw circle in RGB
                # manual circle or use cv2 and convert
                # Using cv2 is faster for drawing
                
            # Let's use CV2 for drawing then convert BGR to RGB
            bgr_img = np.zeros((height, width, 3), dtype=np.uint8)
            bgr_img[:] = (39, 14, 10) # BGR
            
            for i, (x, y) in enumerate(stars):
                y_pos = (y + t * 20 * speed) % height
                x_pos = x
                cv2.circle(bgr_img, (int(x_pos), int(y_pos)), sizes[i], (int(c), int(c), int(c)), -1)
                
            return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        return VideoClip(make_frame_rgb, duration=duration)

    @staticmethod
    def create_gradient(duration: float, color1: Tuple, color2: Tuple, width: int = 1080, height: int = 1920) -> VideoClip:
        """
        Creates a shifting gradient background.
        Not implemented efficiently in pure Python/Numpy for video.
        Fallback: Static Gradient Image.
        """
        # Create static gradient for now to save rendering time
        base = np.zeros((height, width, 3), dtype=np.uint8)
        # Vertical gradient
        for y in range(height):
            ratio = y / height
            r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
            g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
            b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
            base[y, :] = (r, g, b)
            
        return ImageClip(base).set_duration(duration)

    @staticmethod
    def get_standard_background(duration: float) -> VideoClip:
        """Returns the default 'Starfield' background."""
        return BackgroundGenerator.create_starfield(duration)
