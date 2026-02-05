
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
from src.video.image_budget_manager import ImageBudgetManager
from src.video.subtitle_generator import SubtitleGenerator

logger = logging.getLogger(__name__)

class ViralVideoEngine:
    """
    Assembles the viral video from script scenes + stat graphics.
    Orchestrates:
    - Background Generation (ImageGenerator)
    - Image Budget Management (10-image limit)
    - Stat/Overlay Rendering (StatRenderer)
    - Burnt-in Subtitles (SubtitleGenerator)
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
        self.image_generator = ImageGenerator()
        self.subtitle_generator = SubtitleGenerator()

    def render_video(
        self,
        script: Dict[str, Any],
        game_data: Dict[str, Any],
        prediction: Dict[str, Any],
        scene_audio_paths: Dict[int, str],
        output_filename: str
    ) -> Optional[str]:
        """Main assembly method."""
        try:
            scenes = script.get("scenes", [])

            # 1. Initialize image budget manager
            image_manager = ImageBudgetManager()

            # 2. Pre-fetch AI Backgrounds for eligible scenes
            no_gen_types = ["score", "transition", "cta"]
            scenes_to_generate = [
                s for s in scenes
                if s.get("scene_type") not in no_gen_types
            ]

            logger.info(f"Generating AI backgrounds for {len(scenes_to_generate)} scenes (Skipping {len(scenes) - len(scenes_to_generate)})...")
            scene_backgrounds = self.image_generator.generate_scene_images(scenes_to_generate)

            # 3. Allocate backgrounds to image budget slots
            image_manager.allocate_from_backgrounds(scene_backgrounds, scenes)
            logger.info(image_manager.get_allocation_summary())

            # 4. Render each scene clip
            clips = []
            rendered_scenes = []

            for scene in scenes:
                scene_id = scene.get("scene_id")
                audio_path = scene_audio_paths.get(scene_id)
                bg_path = scene_backgrounds.get(scene_id)

                if not audio_path:
                    logger.warning(f"Scene {scene_id} missing audio, skipping.")
                    continue

                clip = self._render_scene_clip(scene, game_data, prediction, audio_path, bg_path)
                if clip:
                    clips.append(clip)
                    rendered_scenes.append(scene)

            if not clips:
                logger.error("No clips generated.")
                return None

            logger.info(f"Generated {len(clips)} scene clips. Durations: {[c.duration for c in clips]}")

            # 4b. Enforce duration budget - drop low-priority scenes if over target
            clips, rendered_scenes = self._enforce_duration_budget(clips, rendered_scenes, target_max=58.0)
            scenes = rendered_scenes

            # 5. Apply transitions between scenes
            final_clip = TransitionManager.apply_transitions(clips, transition_type='crossfade')
            logger.info(f"Final clip after transitions: {final_clip.duration:.2f}s")

            # 6. Add burnt-in subtitles
            logger.info("Adding burnt-in subtitles...")
            full_script, total_audio_dur = self.subtitle_generator.generate_scene_subtitles(
                scenes, scene_audio_paths
            )
            if full_script:
                final_clip = self.subtitle_generator.add_subtitles_to_video(
                    final_clip, full_script, final_clip.duration
                )

            # 7. Audio Mixing - add BGM
            main_audio = final_clip.audio
            bgm = self.audio_mixer.get_bgm_clip("energetic", final_clip.duration, volume=0.12)

            if bgm:
                final_audio = CompositeAudioClip([main_audio, bgm])
                final_clip = final_clip.set_audio(final_audio)

            # 8. Render to file
            output_path = str(self.output_dir / output_filename)
            logger.info(f"Writing viral video to {output_path}...")

            final_clip.write_videofile(
                output_path,
                fps=self.FPS,
                codec="libx264",
                audio_codec="aac",
                threads=4,
                preset="fast"
            )

            return output_path

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
        """Create a clip for a single scene with compositing and effects."""
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
            # Minimum scene duration: 3.5s ensures adequate viewing time
            MIN_SCENE_DURATION = 2.0
            duration = max(narration_clip.duration, MIN_SCENE_DURATION)

            # Pad short audio with silence to match visual duration
            if narration_clip.duration < duration:
                from moviepy.editor import AudioClip
                silence = AudioClip(lambda t: [0], duration=duration, fps=44100)
                narration_clip = CompositeAudioClip([narration_clip, silence]).set_duration(duration)

            # 3. Create Base Image Clip
            base_clip = ImageClip(img_array).set_duration(duration)
            base_clip.fps = self.FPS

            # 4. Apply Animation Effects based on Scene Type
            scene_type = scene.get("scene_type", "")

            if scene_type in ["hook", "key_moment", "cta", "recap"]:
                base_clip = pulse_effect(base_clip, duration, scale_factor=1.05)
                base_clip = zoom_in(base_clip, duration, zoom_ratio=1.1)

            elif scene_type in ["top_hitter", "top_pitcher", "standings", "score"]:
                base_clip = fade_in_out(base_clip, fade_duration=0.4)
                base_clip = zoom_in(base_clip, duration, zoom_ratio=1.05)

            else:
                base_clip = zoom_in(base_clip, duration, zoom_ratio=1.1)

            # 5. Add Sound Effects (SFX)
            sfx_name = scene.get("sound_effect", "none")
            sfx_clip = self.audio_mixer.get_sfx_clip(sfx_name)

            final_audio = narration_clip

            if sfx_clip:
                sfx_clip = sfx_clip.volumex(0.6)
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

    @staticmethod
    def _enforce_duration_budget(
        clips: List[VideoClip],
        scenes: List[Dict],
        target_max: float = 58.0,
        transition_overlap: float = 0.5
    ) -> tuple:
        """
        Drop lowest-priority scenes if estimated total duration exceeds target.
        Returns trimmed (clips, scenes) lists.
        """
        def _estimate_total(clip_list):
            if not clip_list:
                return 0
            raw = sum(c.duration for c in clip_list)
            overlaps = (len(clip_list) - 1) * transition_overlap
            return raw - overlaps

        # Scene types ordered from most droppable to least (for 8-scene structure)
        droppable_priority = [
            'recap',  # Can drop recap if needed
        ]

        estimated = _estimate_total(clips)
        if estimated <= target_max:
            return clips, scenes

        logger.info(f"Duration budget: {estimated:.1f}s > {target_max}s, trimming scenes...")

        for drop_type in droppable_priority:
            if _estimate_total(clips) <= target_max:
                break
            # Find and remove this scene type (from the end to preserve order)
            for i in range(len(scenes) - 1, -1, -1):
                if scenes[i].get('scene_type') == drop_type:
                    removed = scenes[i]
                    logger.info(f"Dropping scene {removed.get('scene_id')} ({drop_type}, {clips[i].duration:.1f}s) to fit duration budget")
                    clips.pop(i)
                    scenes.pop(i)
                    break

        final_est = _estimate_total(clips)
        logger.info(f"Duration after trimming: {final_est:.1f}s ({len(clips)} clips)")
        return clips, scenes
