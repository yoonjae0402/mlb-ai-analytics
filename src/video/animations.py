
from moviepy.editor import VideoClip, TextClip, CompositeVideoClip, vfx
import numpy as np
from typing import Tuple, Optional, Any
from PIL import Image, ImageDraw, ImageFont

def counter_animation(start: int, end: int, duration: float, font: str, fontsize: int = 150, color: str = 'white', center: Tuple[int, int] = (540, 960)) -> VideoClip:
    """
    Animate numbers counting up (for win probability, scores).
    Returns a VideoClip.
    """
    # Optimized PIL implementation directly
    return _pil_counter_viz(start, end, duration, font, fontsize, color, center)

def _pil_counter_viz(start, end, duration, font_name, fontsize, color, center):
    # Try to load font once
    try:
        font = ImageFont.truetype(font_name, fontsize)
    except:
        font = ImageFont.load_default()
        
    def make_frame(t):
        # Create transparent base
        img = Image.new('RGBA', (1080, 1920), (0,0,0,0))
        draw = ImageDraw.Draw(img)
        
        progress = min(1.0, t / duration)
        val = int(start + (end - start) * progress)
        text = f"{val}%"
        
        # Center text
        bbox = draw.textbbox((0,0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        draw.text((center[0] - w/2, center[1] - h/2), text, font=font, fill=color)
        
        return np.array(img)
        
    return VideoClip(make_frame, duration=duration, ismask=False)

def slide_in(clip: VideoClip, direction: str = 'left', duration: float = 0.5) -> VideoClip:
    """
    Slide element into frame.
    """
    w, h = clip.size
    screen_w = 1080
    
    def pos(t):
        if t > duration:
            # Stay at center
            return ('center', 'center')
            
        # Easing: Cubic Out for smoothness
        p = min(1.0, t / duration)
        ease = 1 - (1 - p) ** 3
        
        # Simplified: Slide from off-screen to 'center' horizontally.
        if direction == 'left':
            start_x = -w
            end_x = (screen_w - w) // 2 
            current_x = start_x + (end_x - start_x) * ease
            return (int(current_x), 'center')

        elif direction == 'right':
            start_x = screen_w + w
            end_x = (screen_w - w) // 2
            current_x = start_x + (end_x - start_x) * ease
            return (int(current_x), 'center')
            
        return ('center', 'center')

    return clip.set_position(pos)

def fade_in_out(clip: VideoClip, fade_duration: float = 0.5) -> VideoClip:
    """Smooth fade in and out."""
    # Check if duration allows
    d = clip.duration
    if d < 2 * fade_duration:
        fade_duration = d / 2
    return clip.fx(vfx.fadein, fade_duration).fx(vfx.fadeout, fade_duration)

def zoom_in(clip: VideoClip, duration: float = 4.0, zoom_ratio: float = 1.1) -> VideoClip:
    """Zoom in for emphasis (Ken Burns effect)."""
    # Note: duration arg matches standard signature, but clip.duration is source of truth usually.
    # We use arguments to define the rate.
    
    def resize(t):
        # Linear or Ease Out zoom over the duration of the clip
        # If t=0, scale=1.0. If t=duration, scale=zoom_ratio.
        progress = t / duration
        scale = 1.0 + (zoom_ratio - 1.0) * progress
        return scale
        
    return clip.resize(resize)

def pulse_effect(clip: VideoClip, duration: float, scale_factor: float = 1.05) -> VideoClip:
    """Pulse heartbeat effect."""
    def resize(t):
        # Period 1s approx
        cycle = np.sin(2 * np.pi * t) 
        # Scale between 1.0 and scale_factor
        scale = 1.0 + (scale_factor - 1.0) * abs(cycle)
        return scale
    return clip.resize(resize)
