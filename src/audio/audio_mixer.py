
import logging
from pathlib import Path
from typing import List, Optional, Dict
from moviepy.editor import AudioFileClip, CompositeAudioClip, concatenate_audioclips

logger = logging.getLogger(__name__)

class AudioMixer:
    """
    Mixes TTS narration with background music (BGM) and sound effects (SFX).
    """

    def __init__(self, asset_dir: str = "assets/audio"):
        self.asset_dir = Path(asset_dir)
        self.sfx_dir = self.asset_dir / "sfx"
        self.bgm_dir = self.asset_dir / "bgm"
        
        # Ensure dirs exist
        self.sfx_dir.mkdir(parents=True, exist_ok=True)
        self.bgm_dir.mkdir(parents=True, exist_ok=True)

    def mix_full_track(
        self,
        scene_audio_paths: Dict[int, str],
        scene_durations: Dict[int, float],
        scenes: List[Dict],
        bgm_name: str = "energetic",
        bgm_volume: float = 0.15
    ) -> Optional[AudioFileClip]:
        """
        Create the full audio track for the video:
        1. Concatenate scene narrations (with pauses?)
        2. Layer SFX based on scene triggers
        3. Layer BGM with ducting (optional) or constant low volume
        """
        try:
            # 1. Sequence Narrations
            # We need to construct the timeline.
            # However, ViralVideoEngine concats VIDEO clips which already have their audio attached.
            # So mixing a master audio track separately and trying to sync it is hard if we change video lengths.
            
            # BETTER APPROACH for Viral Engine:
            # Mix audio PER SCENE before creating the video clip?
            # Or attach BGM at the very end to the final video.
            
            # Let's support "Final Mix" approach where we take the final video duration 
            # and create a BGM track for it.
            pass
            
        except Exception as e:
            logger.error(f"Audio mixing failed: {e}")
            return None

    def get_bgm_clip(self, bgm_name: str, duration: float, volume: float = 0.15) -> Optional[AudioFileClip]:
        """
        Get a looped BGM clip for the specific duration.
        """
        # Try finding file
        # Check mp3 and wav
        bgm_path = None
        for ext in [".mp3", ".wav"]:
            f = self.bgm_dir / f"{bgm_name}{ext}"
            if f.exists():
                bgm_path = f
                break
        
        if not bgm_path:
            logger.warning(f"BGM {bgm_name} not found in {self.bgm_dir}")
            return None
            
        try:
            bgm = AudioFileClip(str(bgm_path))
            
            # Loop if needed
            if bgm.duration < duration:
                bgm = bgm.loop(duration=duration)
            else:
                bgm = bgm.subclip(0, duration)
                
            # Set volume
            bgm = bgm.volumex(volume)
            
            # Fade in/out
            bgm = bgm.audio_fadein(1.0).audio_fadeout(2.0)
            
            return bgm
        except Exception as e:
            logger.error(f"Failed to load BGM: {e}")
            return None

    def get_sfx_clip(self, sfx_name: str) -> Optional[AudioFileClip]:
        """Get a sound effect clip."""
        # Check mp3 and wav
        sfx_path = None
        for ext in [".mp3", ".wav"]:
            f = self.sfx_dir / f"{sfx_name}{ext}"
            if f.exists():
                sfx_path = f
                break
                
        if not sfx_path:
            # Optional: Don't log warning for 'none'
            if sfx_name.lower() != "none":
                logger.debug(f"SFX {sfx_name} not found")
            return None
            
        try:
            return AudioFileClip(str(sfx_path))
        except Exception as e:
            logger.error(f"Failed to load SFX {sfx_name}: {e}")
            return None
