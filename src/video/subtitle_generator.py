
import logging
import re
import numpy as np
from typing import List, Dict, Optional
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import CompositeVideoClip, VideoClip, AudioFileClip, ImageClip

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SubtitleGenerator:
    """
    Creates burnt-in subtitles for viral videos using Pillow.

    Renders subtitle frames as images and composites them onto the video
    without requiring ImageMagick. Each subtitle is a 3-5 word chunk
    timed to the narration.
    """

    def __init__(
        self,
        fontsize: int = 48,
        color: tuple = (255, 255, 255),
        stroke_color: tuple = (0, 0, 0),
        stroke_width: int = 3,
        max_width: int = 900,
        y_position: int = 1650,
        video_width: int = 1080,
        video_height: int = 1920,
    ):
        self.fontsize = fontsize
        self.color = color
        self.stroke_color = stroke_color
        self.stroke_width = stroke_width
        self.max_width = max_width
        self.y_position = y_position
        self.video_width = video_width
        self.video_height = video_height

        # Load font
        self.font = self._load_font(fontsize)

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Load a bold font for subtitles."""
        for name in ["Arial Bold", "Arial Bold.ttf", "Arial Black.ttf", "Helvetica Bold.ttf", "Arial.ttf"]:
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        logger.warning("No system font found, using default")
        return ImageFont.load_default()

    def generate_word_timings(self, script: str, audio_duration: float, chunk_size: int = 4) -> List[Dict]:
        """Break script into natural word chunks at sentence boundaries."""
        if not script or not script.strip():
            return []

        # Split into sentences at punctuation boundaries
        sentences = re.split(r'(?<=[.!?,;])\s+', script.strip())

        # Break long sentences into sub-chunks
        all_chunks = []
        for sentence in sentences:
            words = sentence.split()
            if not words:
                continue
            if len(words) <= chunk_size + 1:
                all_chunks.append(sentence)
            else:
                for i in range(0, len(words), chunk_size):
                    sub = words[i:i + chunk_size]
                    all_chunks.append(' '.join(sub))

        if not all_chunks:
            return []

        # Assign timings proportional to word count
        total_words = sum(len(c.split()) for c in all_chunks)
        if total_words == 0:
            return []

        time_per_word = audio_duration / total_words
        current_time = 0.0

        timings = []
        for chunk_text in all_chunks:
            word_count = len(chunk_text.split())
            chunk_duration = word_count * time_per_word
            timings.append({
                'text': chunk_text,
                'start': current_time,
                'end': min(current_time + chunk_duration, audio_duration),
            })
            current_time += chunk_duration

        return timings

    def _render_subtitle_frame(self, text: str) -> np.ndarray:
        """Render subtitle as white text on dark bar (RGB)."""
        bar_height = 140
        img = Image.new('RGB', (self.video_width, bar_height), (10, 10, 20))
        draw = ImageDraw.Draw(img)

        lines = self._wrap_text(text, draw, self.font, self.max_width)
        line_height = self.fontsize + 8
        total_height = len(lines) * line_height
        y_offset = (bar_height - total_height) // 2

        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=self.font)
            text_w = bbox[2] - bbox[0]
            x = (self.video_width - text_w) // 2

            # Stroke
            for dx in range(-self.stroke_width, self.stroke_width + 1):
                for dy in range(-self.stroke_width, self.stroke_width + 1):
                    if dx * dx + dy * dy <= self.stroke_width * self.stroke_width:
                        draw.text((x + dx, y_offset + dy), line, font=self.font,
                                  fill=self.stroke_color)

            # Main text
            draw.text((x, y_offset), line, font=self.font, fill=self.color)
            y_offset += line_height

        return np.array(img)

    def _wrap_text(self, text: str, draw: ImageDraw.Draw, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
        """Word-wrap text to fit within max_width."""
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if (bbox[2] - bbox[0]) <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        return lines

    def create_subtitle_clips(self, word_timings: List[Dict]) -> List[VideoClip]:
        """Create subtitle clips with semi-transparent dark bar."""
        subtitle_clips = []

        for timing in word_timings:
            text = timing['text']
            start = timing['start']
            duration = timing['end'] - timing['start']

            if duration <= 0 or not text.strip():
                continue

            try:
                frame_array = self._render_subtitle_frame(text)

                clip = (
                    ImageClip(frame_array, ismask=False)
                    .set_position(('center', self.y_position))
                    .set_start(start)
                    .set_duration(duration)
                    .set_opacity(0.75)
                )

                subtitle_clips.append(clip)

            except Exception as e:
                logger.warning(f"Failed to create subtitle clip for '{text[:30]}...': {e}")
                continue

        logger.info(f"Created {len(subtitle_clips)} subtitle clips")
        return subtitle_clips

    def add_subtitles_to_video(
        self,
        video_clip: VideoClip,
        full_script: str,
        audio_duration: float,
    ) -> VideoClip:
        """
        Burn subtitles directly into the video.
        """
        if not full_script or not full_script.strip():
            logger.warning("No script text provided for subtitles")
            return video_clip

        word_timings = self.generate_word_timings(full_script, audio_duration)

        if not word_timings:
            logger.warning("No word timings generated")
            return video_clip

        subtitle_clips = self.create_subtitle_clips(word_timings)

        if not subtitle_clips:
            logger.warning("No subtitle clips created")
            return video_clip

        # Composite subtitles on top of video
        final_video = CompositeVideoClip(
            [video_clip] + subtitle_clips,
            size=video_clip.size,
        )

        final_video = final_video.set_duration(video_clip.duration)

        logger.info(f"Added {len(subtitle_clips)} subtitle chunks to video")
        return final_video

    def generate_scene_subtitles(
        self,
        scenes: list,
        scene_audio_paths: dict,
    ) -> tuple:
        """
        Collect full script text and total audio duration from scenes.
        """
        full_script_parts = []
        total_duration = 0.0

        for scene in scenes:
            narration = scene.get('narration', '')
            if narration:
                full_script_parts.append(narration)

            scene_id = scene.get('scene_id')
            audio_path = scene_audio_paths.get(scene_id)
            if audio_path:
                try:
                    audio = AudioFileClip(audio_path)
                    total_duration += audio.duration
                    audio.close()
                except Exception as e:
                    logger.warning(f"Could not read audio duration for scene {scene_id}: {e}")
                    total_duration += scene.get('duration', 4.0)
            else:
                total_duration += scene.get('duration', 4.0)

        full_script = ' '.join(full_script_parts)
        return full_script, total_duration
