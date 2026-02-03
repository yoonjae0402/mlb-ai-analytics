import os
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import requests
from PIL import Image, ImageDraw
import numpy as np
from dotenv import load_dotenv

from src.utils.exceptions import DataFetchError

load_dotenv()
logger = logging.getLogger(__name__)


class ImageGenerator:
    """
    Generates AI images via the Nano Banana API.
    Implements prompt-based caching to prevent regenerating images for the same prompts.
    """

    DEFAULT_BASE_URL = "https://api.nanobanana.com/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        output_dir: str = "outputs/images",
        cache_enabled: bool = True,
        max_workers: int = 5,
    ):
        self.api_key = api_key or os.getenv("NANO_BANANA_API_KEY")
        self.base_url = base_url or os.getenv("NANO_BANANA_BASE_URL", self.DEFAULT_BASE_URL)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = cache_enabled
        self.max_workers = max_workers

        if not self.api_key:
            logger.warning("NANO_BANANA_API_KEY not found in environment variables.")

    def generate_image(
        self,
        prompt: str,
        width: int = 1080,
        height: int = 1920,
        style: str = "cinematic",
    ) -> Optional[str]:
        """
        Generate a single image from a prompt.

        Returns:
            Path to saved image file, or None on failure.
        """
        # Check cache first
        cached = self._check_cache(prompt)
        if cached:
            logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
            return cached

        try:
            image_bytes = self._call_nano_banana_api(prompt, width, height, style)
            cache_path = self._get_cache_path(prompt)
            return self._save_image(image_bytes, cache_path)
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return None

    def generate_scene_images(
        self,
        scenes: List[Dict],
        max_workers: Optional[int] = None,
    ) -> Dict[int, str]:
        """
        Generate images for all scenes in parallel.

        Args:
            scenes: List of scene dicts, each with 'scene_id' and 'image_prompt'.
            max_workers: Override default max parallel workers.

        Returns:
            Mapping of {scene_id: image_path}.
        """
        workers = max_workers or self.max_workers
        results: Dict[int, str] = {}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for scene in scenes:
                scene_id = scene.get("scene_id", 0)
                prompt = scene.get("image_prompt", "")
                if not prompt:
                    logger.warning(f"Scene {scene_id} has no image_prompt, using fallback")
                    results[scene_id] = self._create_fallback_image(scene_id)
                    continue
                future = pool.submit(self.generate_image, prompt)
                futures[future] = scene_id

            for future in as_completed(futures):
                scene_id = futures[future]
                try:
                    image_path = future.result()
                    if image_path:
                        results[scene_id] = image_path
                    else:
                        results[scene_id] = self._create_fallback_image(scene_id)
                except Exception as e:
                    logger.error(f"Image generation failed for scene {scene_id}: {e}")
                    results[scene_id] = self._create_fallback_image(scene_id)

        logger.info(f"Generated {len(results)} scene images ({len(scenes)} requested)")
        return results

    def _get_cache_path(self, prompt: str) -> Path:
        """Generate deterministic cache path from prompt hash."""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        return self.output_dir / f"{prompt_hash}.png"

    def _check_cache(self, prompt: str) -> Optional[str]:
        """Return cached image path if it exists and cache is enabled."""
        if not self.cache_enabled:
            return None
        cache_path = self._get_cache_path(prompt)
        if cache_path.exists():
            return str(cache_path)
        return None

    def _call_nano_banana_api(
        self,
        prompt: str,
        width: int,
        height: int,
        style: str = "cinematic",
    ) -> bytes:
        """
        Make HTTP request to Nano Banana API.

        Returns:
            Raw image bytes.

        Raises:
            DataFetchError: On API failure.
        """
        url = f"{self.base_url}/generate"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "style": style,
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)

            if response.status_code == 429:
                raise DataFetchError(
                    "Nano Banana API rate limit exceeded",
                    source="NanoBanana",
                    status_code=429,
                )

            response.raise_for_status()

            # Check if response is direct image bytes or JSON with URL
            content_type = response.headers.get("Content-Type", "")
            if "image" in content_type:
                return response.content

            # JSON response with image URL
            data = response.json()
            image_url = data.get("image_url") or data.get("url") or data.get("output")
            if not image_url:
                raise DataFetchError(
                    "Nano Banana API response missing image URL",
                    source="NanoBanana",
                )

            img_response = requests.get(image_url, timeout=120)
            img_response.raise_for_status()
            return img_response.content

        except DataFetchError:
            raise
        except requests.RequestException as e:
            raise DataFetchError(
                f"Nano Banana API request failed: {e}",
                source="NanoBanana",
            ) from e

    def _save_image(self, image_bytes: bytes, output_path: Path) -> str:
        """Save image bytes to disk. Returns path string."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_bytes)
        logger.info(f"Image saved: {output_path}")
        return str(output_path)

    def _create_fallback_image(self, scene_id: int) -> str:
        """Create a dark gradient fallback image when API fails."""
        width, height = 1080, 1920
        img = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(img)

        # Dark gradient from top to bottom
        for y in range(height):
            r = int(15 + (25 - 15) * y / height)
            g = int(15 + (20 - 15) * y / height)
            b = int(30 + (50 - 30) * y / height)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        fallback_path = self.output_dir / f"fallback_scene_{scene_id}.png"
        img.save(str(fallback_path))
        logger.warning(f"Created fallback image for scene {scene_id}: {fallback_path}")
        return str(fallback_path)
