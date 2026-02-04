
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
from moviepy.editor import (
    VideoClip, ImageClip, AudioFileClip, 
    CompositeVideoClip, concatenate_videoclips, CompositeAudioClip
)

from src.video.stat_renderer import StatRenderer
from src.video.asset_manager import AssetManager
from src.audio.audio_mixer import AudioMixer
from src.content.image_generator import ImageGenerator
from src.video.animations import slide_in, pulse_effect, counter_animation, zoom_in, fade_in_out
from src.video.transition_manager import TransitionManager

logger = logging.getLogger(__name__)

class ViralVideoEngine:
    """
    Assembles the viral video from script scenes + stat graphics.
    Orchestrates:
    - Background Generation (ImageGenerator)
    - Stat/Overlay Rendering (StatRenderer)
    - Audio Mixing (AudioMixer)
    - Video Assembly (MoviePy)
    """
    
    FPS = 30
    WIDTH = 1080
    HEIGHT = 1920
    
    def __init__(self, output_dir: str = "outputs/videos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.asset_manager = AssetManager()
        self.renderer = StatRenderer(self.asset_manager)
        self.audio_mixer = AudioMixer()
        self.image_generator = ImageGenerator() # Uses Nano Banana / Imagen
        
    def render_video(
        self,
        script: Dict[str, Any],
        game_data: Dict[str, Any],
        prediction: Dict[str, Any],
        scene_audio_paths: Dict[int, str],
        output_filename: str
    ) -> Optional[str]:
        """
        Main assembly method.
        """
        try:
            scenes = script.get("scenes", [])
            
            # 1. Pre-fetch AI Backgrounds for all scenes (Parallel)
            # Optimization: Skip generation for specific text-heavy/graphic-heavy scenes
            no_gen_types = ["hook", "score", "transition", "cta"]
            scenes_to_generate = [
                s for s in scenes 
                if s.get("scene_type") not in no_gen_types
            ]
            
            logger.info(f"Generating AI backgrounds for {len(scenes_to_generate)} scenes (Skipping {len(scenes) - len(scenes_to_generate)})...")
            scene_backgrounds = self.image_generator.generate_scene_images(scenes_to_generate)
            
            clips = []
            
            for scene in scenes:
                scene_id = scene.get("scene_id")
                audio_path = scene_audio_paths.get(scene_id)
                bg_path = scene_backgrounds.get(scene_id)
                
                if not audio_path:
                    logger.warning(f"Scene {scene_id} missing audio, skipping.")
                    continue
                    
                clip = self._render_scene_clip(scene, game_data, prediction, audio_path, bg_path)
                if clip:
                    # TransitionManager handles fades now. Don't double apply.
                    clips.append(clip)
                    
            if not clips:
                logger.error("No clips generated.")
                return None
                
            # Use TransitionManager for smooth flows
            # Now returns a flat CompositeVideoClip (Much faster)
            final_clip = TransitionManager.apply_transitions(clips, transition_type='crossfade')
            
            # --- Audio Mixing ---
            # 1. Get current audio (Narration + SFX from scenes)
            main_audio = final_clip.audio
            
            # 2. Add Background Music (BGM)
            # Use 'energetic' for recap, maybe 'dramatic' if valid, but stick to one track for MVP
            bgm = self.audio_mixer.get_bgm_clip("energetic", final_clip.duration, volume=0.12)
            
            if bgm:
                # Composite BGM + Main Audio
                final_audio = CompositeAudioClip([main_audio, bgm])
                final_clip = final_clip.set_audio(final_audio)
                
            # --- Rendering ---
            logger.info(f"Writing viral video to {output_filename}...")
            
            # Performance Optimization:
            # - threads=4: Use multi-threading for writing
            # - preset='ultrafast': Much faster encoding for testing (use 'medium' for final if needed)
            # - audio_codec='aac': Standard
            final_clip.write_videofile(
                output_filename, 
                fps=self.FPS, 
                codec="libx264", 
                audio_codec="aac",
                threads=4,
                preset="ultrafast"
            )
            
            return output_filename
            
        except Exception as e:
            logger.error(f"Error rendering video: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _render_scene_clip(
        self, 
        scene: Dict, 
        game_data: Dict, 
        prediction: Dict, 
        audio_path: str,
        bg_path: Optional[str]
    ) -> Optional[VideoClip]:
        """
        Create a clip for a single scene with compositing and effects.
        """
        try:
            # 1. Render Base Frame (Pillow) with AI Background
            pil_image = self.renderer.render_scene_frame(
                scene, 
                game_data, 
                prediction, 
                background_path=bg_path
            )
            img_array = np.array(pil_image)
            
            # 2. Load Audio & Determine Duration
            narration_clip = AudioFileClip(audio_path)
            duration = narration_clip.duration
            # Min duration for visual stability
            if duration < 2.0:
                 duration = 2.0
            
            # 3. Create Base Image Clip
            base_clip = ImageClip(img_array).set_duration(duration)
            base_clip.fps = self.FPS
            
            # 4. Apply Animation Effects based on Scene Type
            scene_type = scene.get("scene_type", "")
            
            # Refined Animations
            if scene_type in ["hook", "key_moment", "cta"]:
                # High Energy: Pulse + Zoom
                base_clip = pulse_effect(base_clip, duration, scale_factor=1.05)
                base_clip = zoom_in(base_clip, duration, zoom_ratio=1.1)
                
            elif scene_type in ["top_hitter", "top_pitcher", "player_watch_1", "player_watch_2", "player_spotlight", "standings"]:
                # Information Reveal: Slide In + Zoom
                # Fix: duration is 3rd arg in slide_in(clip, direction, duration)
                base_clip = slide_in(base_clip, direction='left', duration=duration)
                base_clip = zoom_in(base_clip, duration, zoom_ratio=1.05)
                
            elif scene_type == "prediction_meter":
                # Focus: Subtle Zoom
                base_clip = zoom_in(base_clip, duration, zoom_ratio=1.1)
                
            else:
                # Default: Zoom In
                base_clip = zoom_in(base_clip, duration, zoom_ratio=1.1)
            
            # 5. Add Sound Effects (SFX)
            sfx_name = scene.get("sound_effect", "none")
            sfx_clip = self.audio_mixer.get_sfx_clip(sfx_name)
            
            final_audio = narration_clip
            
            if sfx_clip:
                # Composite Audio: Narration + SFX
                sfx_clip = sfx_clip.volumex(0.6) 
                
                # Truncate sfx if longer than scene, or let it stick out?
                # Sticking out requires scene overlap management. We'll truncate/fade for now.
                if sfx_clip.duration > duration:
                    sfx_clip = sfx_clip.subclip(0, duration).audio_fadeout(0.5)
                
                final_audio = CompositeAudioClip([narration_clip, sfx_clip])
                
            # Set audio and ensure duration matches exactly
            base_clip = base_clip.set_audio(final_audio)
            base_clip = base_clip.set_duration(duration)
            
            return base_clip
            
        except Exception as e:
            logger.error(f"Error rendering scene {scene.get('scene_id')}: {e}")
            return None
