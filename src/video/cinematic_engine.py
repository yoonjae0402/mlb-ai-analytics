import logging
import platform
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from moviepy.editor import (
    ImageClip, AudioFileClip, CompositeVideoClip,
    concatenate_videoclips, ColorClip,
)

# Monkey patch for Pillow 10+ compatibility with MoviePy 1.x
if hasattr(Image, 'Resampling'):
    if not hasattr(Image, 'ANTIALIAS'):
        Image.ANTIALIAS = Image.Resampling.LANCZOS

logger = logging.getLogger(__name__)


class CinematicEngine:
    """
    Renders cinematic videos from AI-generated images with Ken Burns motion effects.
    Video format: 1080x1920 vertical (YouTube Shorts).
    """

    VIDEO_WIDTH = 1080
    VIDEO_HEIGHT = 1920
    FPS = 30

    # Motion parameters
    ZOOM_IN_START = 1.0
    ZOOM_IN_END = 1.12
    ZOOM_OUT_START = 1.12
    ZOOM_OUT_END = 1.0

    # Pan overshoot factor (image is loaded wider/taller than frame)
    PAN_OVERSHOOT = 1.15

    def __init__(self, output_dir: str = "outputs/videos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._hw_encoder = self._detect_hw_encoder()

    def render_video(
        self,
        script: Dict[str, Any],
        scene_images: Dict[int, str],
        scene_audio_paths: Dict[int, str],
        output_filename: str = "cinematic_video.mp4",
    ) -> Optional[str]:
        """
        Render all scenes into a single cinematic video.

        Args:
            script: Cinematic JSON script with 'scenes' list.
            scene_images: {scene_id: image_path} mapping.
            scene_audio_paths: {scene_id: audio_path} mapping.
            output_filename: Output video file name.

        Returns:
            Path to output video file, or None on failure.
        """
        try:
            scenes = script.get("scenes", [])
            if not scenes:
                logger.error("No scenes in script")
                return None

            scene_clips = []
            for scene in scenes:
                scene_id = scene.get("scene_id", 0)
                image_path = scene_images.get(scene_id)
                audio_path = scene_audio_paths.get(scene_id)

                if not image_path or not Path(image_path).exists():
                    logger.warning(f"Scene {scene_id}: missing image, using dark background")
                    image_path = None

                if not audio_path or not Path(audio_path).exists():
                    logger.warning(f"Scene {scene_id}: missing audio, skipping")
                    continue

                clip = self.render_scene(scene, image_path, audio_path)
                if clip:
                    scene_clips.append(clip)

            if not scene_clips:
                logger.error("No valid scene clips to render")
                return None

            # Concatenate all scenes
            final = concatenate_videoclips(scene_clips, method="compose")

            output_path = self.output_dir / output_filename

            # Encode
            codec = self._hw_encoder or "libx264"
            ffmpeg_params = ["-threads", "0"]
            if codec == "libx264":
                ffmpeg_params += ["-preset", "ultrafast", "-crf", "23"]
            elif codec == "h264_videotoolbox":
                ffmpeg_params += ["-b:v", "4M"]

            logger.info(f"Encoding cinematic video with codec={codec} -> {output_path}")
            final.write_videofile(
                str(output_path),
                fps=self.FPS,
                codec=codec,
                audio_codec="aac",
                threads=4,
                ffmpeg_params=ffmpeg_params,
                logger=None,
            )

            # Clean up clips
            for clip in scene_clips:
                clip.close()
            final.close()

            logger.info(f"Cinematic video rendered: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error rendering cinematic video: {e}")
            import traceback
            traceback.print_exc()
            return None

    def render_scene(
        self,
        scene: Dict[str, Any],
        image_path: Optional[str],
        audio_path: str,
    ) -> Optional[CompositeVideoClip]:
        """
        Render a single scene with motion effects, text overlay, and audio sync.

        Returns:
            CompositeVideoClip for this scene, or None on failure.
        """
        try:
            # Get audio duration (this is the source of truth for scene timing)
            audio_clip = AudioFileClip(audio_path)
            duration = audio_clip.duration

            motion_type = scene.get("motion_type", "zoom_in")
            text_overlay = scene.get("text_overlay", "")

            # Build background clip with motion
            if image_path:
                bg_clip = self._create_motion_clip(image_path, motion_type, duration)
            else:
                bg_clip = ColorClip(
                    size=(self.VIDEO_WIDTH, self.VIDEO_HEIGHT),
                    color=(20, 20, 30),
                    duration=duration,
                )

            layers = [bg_clip]

            # Add semi-transparent gradient strip at bottom for text readability
            if text_overlay:
                gradient = self._create_bottom_gradient(duration)
                layers.append(gradient)

                # Add text overlay
                text_clip = self._create_text_overlay(
                    text_overlay, duration,
                    position=("center", self.VIDEO_HEIGHT - 250),
                    font_size=56,
                )
                if text_clip is not None:
                    layers.append(text_clip)

            # Composite scene
            composite = CompositeVideoClip(layers, size=(self.VIDEO_WIDTH, self.VIDEO_HEIGHT))
            composite = composite.set_audio(audio_clip)
            composite = composite.set_duration(duration)

            return composite

        except Exception as e:
            logger.error(f"Error rendering scene {scene.get('scene_id', '?')}: {e}")
            return None

    def _create_motion_clip(
        self,
        image_path: str,
        motion_type: str,
        duration: float,
    ) -> ImageClip:
        """Create an ImageClip with the specified motion effect applied."""
        if motion_type in ("pan_left", "pan_right"):
            return self._pan_effect(image_path, motion_type.split("_")[1], duration)
        elif motion_type == "zoom_out":
            return self._zoom_effect(
                image_path, self.ZOOM_OUT_START, self.ZOOM_OUT_END, duration
            )
        else:
            # Default: zoom_in
            return self._zoom_effect(
                image_path, self.ZOOM_IN_START, self.ZOOM_IN_END, duration
            )

    def _zoom_effect(
        self,
        image_path: str,
        start_scale: float,
        end_scale: float,
        duration: float,
    ) -> ImageClip:
        """
        Apply zoom effect by dynamically resizing and center-cropping each frame.
        """
        # Load image at max needed scale
        max_scale = max(start_scale, end_scale)
        base_img = self._prepare_image(image_path, overshoot=max_scale)
        base_h, base_w = base_img.shape[:2]

        vw, vh = self.VIDEO_WIDTH, self.VIDEO_HEIGHT

        def make_frame(t):
            progress = t / duration if duration > 0 else 0
            scale = start_scale + (end_scale - start_scale) * progress

            # Calculate crop dimensions at current scale
            crop_w = int(vw / scale)
            crop_h = int(vh / scale)

            # Center crop
            cx, cy = base_w // 2, base_h // 2
            x1 = max(0, cx - crop_w // 2)
            y1 = max(0, cy - crop_h // 2)
            x2 = min(base_w, x1 + crop_w)
            y2 = min(base_h, y1 + crop_h)

            cropped = base_img[y1:y2, x1:x2]

            # Resize back to output dimensions
            pil_img = Image.fromarray(cropped)
            pil_img = pil_img.resize((vw, vh), Image.LANCZOS)
            return np.array(pil_img)

        clip = ImageClip(make_frame, ismask=False, duration=duration)
        clip.fps = self.FPS
        return clip

    def _pan_effect(
        self,
        image_path: str,
        direction: str,
        duration: float,
    ) -> ImageClip:
        """
        Apply horizontal pan by shifting the crop window across an oversized image.
        """
        base_img = self._prepare_image(image_path, overshoot=self.PAN_OVERSHOOT)
        base_h, base_w = base_img.shape[:2]

        vw, vh = self.VIDEO_WIDTH, self.VIDEO_HEIGHT
        overshoot_px = base_w - vw

        def make_frame(t):
            progress = t / duration if duration > 0 else 0

            if direction == "left":
                # Pan from right to left
                x_offset = int(overshoot_px * (1 - progress))
            else:
                # Pan from left to right
                x_offset = int(overshoot_px * progress)

            x_offset = max(0, min(x_offset, base_w - vw))

            # Vertical center crop
            cy = base_h // 2
            y1 = max(0, cy - vh // 2)
            y2 = min(base_h, y1 + vh)

            cropped = base_img[y1:y2, x_offset:x_offset + vw]

            # Ensure exact output size
            if cropped.shape[1] != vw or cropped.shape[0] != vh:
                pil_img = Image.fromarray(cropped)
                pil_img = pil_img.resize((vw, vh), Image.LANCZOS)
                return np.array(pil_img)

            return cropped

        clip = ImageClip(make_frame, ismask=False, duration=duration)
        clip.fps = self.FPS
        return clip

    def _create_text_overlay(
        self,
        text: str,
        duration: float,
        position: tuple = ("center", 1600),
        font_size: int = 56,
    ) -> Optional[ImageClip]:
        """Create a text overlay clip using Pillow."""
        if not text:
            return None

        width = self.VIDEO_WIDTH
        lines = text.split('\n')
        line_height = font_size + 10
        img_height = len(lines) * line_height + 40

        img = Image.new("RGBA", (width, img_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("Arial", font_size)
        except OSError:
            font = ImageFont.load_default()

        y_text = 20
        for line in lines:
            draw.text(
                (width / 2, y_text), line,
                font=font, fill="white", anchor="mt",
            )
            y_text += line_height

        text_array = np.array(img)
        clip = (
            ImageClip(text_array, ismask=False, transparent=True)
            .set_duration(duration)
            .set_position(position)
        )
        return clip

    def _create_bottom_gradient(self, duration: float) -> ImageClip:
        """Create a semi-transparent dark gradient at the bottom of the frame."""
        width = self.VIDEO_WIDTH
        grad_height = 400

        img = Image.new("RGBA", (width, grad_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        for y in range(grad_height):
            alpha = int(180 * (y / grad_height))
            draw.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))

        grad_array = np.array(img)
        clip = (
            ImageClip(grad_array, ismask=False, transparent=True)
            .set_duration(duration)
            .set_position(("center", self.VIDEO_HEIGHT - grad_height))
        )
        return clip

    def _prepare_image(self, image_path: str, overshoot: float = 1.15) -> np.ndarray:
        """
        Load and prepare image for motion effects.
        Resizes so both dimensions are at least overshoot * target,
        to allow room for zoom/pan without showing edges.
        """
        img = Image.open(image_path).convert("RGB")

        target_w = int(self.VIDEO_WIDTH * overshoot)
        target_h = int(self.VIDEO_HEIGHT * overshoot)

        # Scale to cover target dimensions
        scale_w = target_w / img.width
        scale_h = target_h / img.height
        scale = max(scale_w, scale_h)

        new_w = int(img.width * scale)
        new_h = int(img.height * scale)

        img = img.resize((new_w, new_h), Image.LANCZOS)
        return np.array(img)

    @staticmethod
    def _detect_hw_encoder() -> Optional[str]:
        """Detect if a hardware H.264 encoder is available."""
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return None
        if platform.system() == "Darwin":
            return "h264_videotoolbox"
        return None
