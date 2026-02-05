
import logging
import uuid
from pathlib import Path
from typing import Optional, Dict
from PIL import Image, ImageDraw, ImageFont

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ImageBudgetManager:
    """
    Manages the 10-image limit for viral videos.

    Allocates images to named slots and provides intelligent
    reuse across scenes. Creates text overlay variants to
    maximize visual variety without consuming additional slots.
    """

    MAX_IMAGES = 10

    # Maps scene types to image slots for intelligent reuse
    SCENE_TO_SLOT = {
        'hook': 'hook',
        'score': 'score',
        'top_hitter': 'player_1',
        'top_pitcher': 'player_2',
        'key_moment': 'key_moment',
        'standings': 'standings',
        'transition': 'hook',            # Reuse hook image
        'prediction_meter': 'prediction',
        'factor_1': 'prediction',         # Reuse prediction image
        'factor_2': 'standings',          # Reuse standings image
        'factor_3': 'key_moment',         # Reuse key_moment image
        'player_watch_1': 'player_1',     # Reuse player_1
        'player_watch_2': 'player_2',     # Reuse player_2
        'cta': 'cta',
    }

    def __init__(self, temp_dir: str = "/tmp/mlb_variants"):
        self.image_slots: Dict[str, Optional[str]] = {
            'hook': None,
            'score': None,
            'celebration': None,
            'next_game': None,
            'cta': None,
            'player_1': None,
            'player_2': None,
            'key_moment': None,
            'standings': None,
            'prediction': None,
        }
        self.used_count = 0
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def allocate_image(self, slot_name: str, image_path: str) -> bool:
        """
        Allocate an image to a named slot.
        Returns False if budget exceeded or slot name invalid.
        """
        if slot_name not in self.image_slots:
            logger.warning(f"Unknown image slot: {slot_name}")
            return False

        if self.image_slots[slot_name] is not None:
            # Slot already filled, replace it (no budget change)
            logger.debug(f"Replacing image in slot '{slot_name}'")
            self.image_slots[slot_name] = image_path
            return True

        if self.used_count >= self.MAX_IMAGES:
            logger.warning(f"Image budget exhausted ({self.MAX_IMAGES}/{self.MAX_IMAGES}), cannot allocate '{slot_name}'")
            return False

        self.image_slots[slot_name] = image_path
        self.used_count += 1
        logger.info(f"Allocated image {self.used_count}/{self.MAX_IMAGES}: {slot_name}")
        return True

    def get_scene_image(self, scene_type: str) -> Optional[str]:
        """
        Get the image path for a given scene type.
        Maps scene types to slots, with fallback chain.
        """
        slot = self.SCENE_TO_SLOT.get(scene_type, 'hook')
        image = self.image_slots.get(slot)

        if image and Path(image).exists():
            return image

        # Fallback chain: celebration -> hook -> None
        for fallback_slot in ['celebration', 'hook', 'score']:
            fallback = self.image_slots.get(fallback_slot)
            if fallback and Path(fallback).exists():
                return fallback

        return None

    def allocate_from_backgrounds(self, scene_backgrounds: Dict[int, str], scenes: list):
        """
        Allocate AI-generated backgrounds from the image generator results.
        Maps scene_id -> background_path into named slots.
        Prioritizes hook and key_moment scenes for allocation.
        """
        # Prioritize critical scenes so they get images before budget runs out
        priority_types = ['hook', 'key_moment', 'standings', 'prediction_meter']
        sorted_scenes = sorted(scenes, key=lambda s: (
            0 if s.get('scene_type') in priority_types else 1
        ))

        for scene in sorted_scenes:
            scene_id = scene.get('scene_id')
            scene_type = scene.get('scene_type', '')
            bg_path = scene_backgrounds.get(scene_id)

            if not bg_path or not Path(bg_path).exists():
                continue

            slot = self.SCENE_TO_SLOT.get(scene_type)
            if slot and self.image_slots.get(slot) is None:
                self.allocate_image(slot, bg_path)

    def create_text_overlay_variant(self, base_image_path: str, text: str, position: str = 'center') -> str:
        """
        Create a visual variant of a base image by adding text overlay.
        This maximizes visual variety without using additional image slots.

        Args:
            base_image_path: Path to the base image
            text: Text to overlay
            position: 'top', 'center', or 'bottom'

        Returns:
            Path to the variant image
        """
        try:
            img = Image.open(base_image_path).convert("RGBA")

            # Add dark gradient overlay for text readability
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw_overlay = ImageDraw.Draw(overlay)

            if position == 'top':
                # Dark gradient at top
                for y in range(400):
                    alpha = int(180 * (1 - y / 400))
                    draw_overlay.line([(0, y), (img.width, y)], fill=(0, 0, 0, alpha))
            elif position == 'bottom':
                # Dark gradient at bottom
                for y in range(img.height - 400, img.height):
                    alpha = int(180 * ((y - (img.height - 400)) / 400))
                    draw_overlay.line([(0, y), (img.width, y)], fill=(0, 0, 0, alpha))
            else:
                # Center: full semi-transparent overlay
                draw_overlay.rectangle([(0, 0), img.size], fill=(0, 0, 0, 120))

            img = Image.alpha_composite(img, overlay)

            # Add text
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("Arial Black", 72)
            except OSError:
                try:
                    font = ImageFont.truetype("Arial Black.ttf", 72)
                except OSError:
                    font = ImageFont.load_default()

            # Calculate text position
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            x = (img.width - text_w) // 2

            if position == 'top':
                y = 150
            elif position == 'bottom':
                y = img.height - 300
            else:
                y = (img.height - text_h) // 2

            # Draw text with outline for readability
            for dx, dy in [(-3, -3), (-3, 3), (3, -3), (3, 3), (-3, 0), (3, 0), (0, -3), (0, 3)]:
                draw.text((x + dx, y + dy), text, font=font, fill='black')
            draw.text((x, y), text, font=font, fill='white')

            # Save variant
            variant_path = str(self.temp_dir / f"variant_{uuid.uuid4().hex[:8]}.png")
            img.save(variant_path)
            return variant_path

        except Exception as e:
            logger.error(f"Failed to create text overlay variant: {e}")
            return base_image_path

    @property
    def remaining_budget(self) -> int:
        return self.MAX_IMAGES - self.used_count

    def get_allocation_summary(self) -> str:
        """Return a summary of image allocations for logging."""
        filled = [k for k, v in self.image_slots.items() if v is not None]
        empty = [k for k, v in self.image_slots.items() if v is None]
        return f"Images: {self.used_count}/{self.MAX_IMAGES} | Filled: {filled} | Empty: {empty}"
