
import logging
import os
import subprocess
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from moviepy.editor import VideoFileClip
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

class VideoValidationError(Exception):
    """Custom exception for video validation failures."""
    def __init__(self, message: str, discrepancies: List[str] = None):
        super().__init__(message)
        self.discrepancies = discrepancies or []

class VideoValidator:
    """
    Performs critical quality checks on generated viral videos.
    Ensures compliance with platform requirements (TikTok/Shorts).
    """

    MIN_DURATION = 45.0
    MAX_DURATION = 65.0 # buffer for 60s target
    TARGET_WIDTH = 1080
    TARGET_HEIGHT = 1920
    REQUIRED_FPS = 30
    
    def validate_video(self, video_path: str) -> bool:
        """
        Run all validation checks.
        Returns True if passed, raises VideoValidationError if failed.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        logger.info(f"Validating video: {path.name}")
        discrepancies = []
        
        try:
            clip = VideoFileClip(str(path))
            
            # 1. Duration Check
            if not (self.MIN_DURATION <= clip.duration <= self.MAX_DURATION):
                msg = f"Duration {clip.duration:.2f}s outside range [{self.MIN_DURATION}-{self.MAX_DURATION}s]"
                logger.error(msg)
                discrepancies.append(msg)
                
            # 2. Resolution Check
            if clip.w != self.TARGET_WIDTH or clip.h != self.TARGET_HEIGHT:
                 msg = f"Resolution {clip.w}x{clip.h} does not match target {self.TARGET_WIDTH}x{self.TARGET_HEIGHT}"
                 logger.error(msg)
                 discrepancies.append(msg)

            # 3. FPS Check (Approximate)
            if abs(clip.fps - self.REQUIRED_FPS) > 1.0:
                msg = f"FPS {clip.fps} mismatch (Target: {self.REQUIRED_FPS})"
                logger.error(msg)
                discrepancies.append(msg)
                
            # 4. Audio Check
            if not clip.audio:
                msg = "Video has no audio track."
                logger.error(msg)
                discrepancies.append(msg)
            else:
                # Check for silence? (Expensive, maybe skip for MVP or do sampling)
                pass

            # 5. Black Frame Detection (Sampling)
            if self._has_black_frames(clip):
                msg = "Significant black frames detected (Validation Failed)."
                logger.error(msg)
                discrepancies.append(msg)
                
            clip.close()

            if discrepancies:
                raise VideoValidationError("Video validation failed", discrepancies)
                
            logger.info("Video validation PASSED ✅")
            return True

        except Exception as e:
            if isinstance(e, VideoValidationError):
                raise
            logger.error(f"Validation process error: {e}")
            raise VideoValidationError(f"Validation crashed: {e}")

    def _has_black_frames(self, clip: VideoFileClip, threshold: int = 10, sample_rate: float = 1.0) -> bool:
        """
        Check for black frames by sampling.
        Returns True if a sequence of black frames is found.
        """
        # Sample every 'sample_rate' seconds
        times = np.arange(0, clip.duration, sample_rate)
        black_count = 0
        
        for t in times:
            try:
                frame = clip.get_frame(t)
                # Check mean brightness
                brightness = np.mean(frame)
                
                if brightness < threshold:
                    black_count += 1
                    logger.warning(f"Potential black frame at {t:.2f}s (Brightness: {brightness:.2f})")
                
                # If we see > 3 seconds of blackness purely by sampling, flag it
                if black_count > 3: 
                    return True
            except Exception:
                continue
                
        return False
