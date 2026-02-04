
import os
import hashlib
import logging
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional
import io
import base64

from PIL import Image, ImageDraw
from dotenv import load_dotenv

# Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

from src.utils.exceptions import DataFetchError

load_dotenv()
logger = logging.getLogger(__name__)

class ImageGenerator:
    """
    Generates AI images via Google Gemini Imagen 3 or Nano Banana.
    For Viral Mode, we prefer Nano Banana for stylized backgrounds.
    """

    IMAGEN_MODEL_ID = "imagen-4.0-generate-001"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        nano_api_key: Optional[str] = None,
        output_dir: str = "outputs/images",
        cache_enabled: bool = True,
        max_workers: int = 4, # Parallel generation
        provider: str = "imagen"  # Default to Google Imagen
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.nano_api_key = nano_api_key or os.getenv("NANO_BANANA_API_KEY") # Kept for backward compat
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = cache_enabled
        self.max_workers = max_workers
        self.provider = provider
        
        # Init clients
        if _GENAI_AVAILABLE and self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None
            if self.provider == "imagen":
                logger.warning("Gemini Client not available. Falling back to simple generation.")

    def generate_image(
        self,
        prompt: str,
        style: str = "cinematic",
        width: int = 1024,
        height: int = 1792
    ) -> Optional[str]:
        """
        Generate a single image. Dispatches to configured provider.
        """
        # Check cache first
        cached = self._check_cache(prompt)
        if cached:
            logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
            return cached

        try:
            # Default to Imagen if configured
            if self.client and self.provider == "imagen":
                return self._generate_imagen(prompt, style)
            elif self.provider == "nano_banana" and self.nano_api_key:
                return self._generate_nano_banana(prompt, style, width, height)
            else:
                logger.warning(f"No valid provider configured (Provider: {self.provider}). Using fallback.")
                return self._create_fallback_image(prompt) # Hash prompt for unique ID
                
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return self._create_fallback_image(prompt)

    def _generate_imagen(self, prompt: str, style: str) -> Optional[str]:
        """Generate using Gemini Imagen 3."""
        try:
            full_prompt = f"{style} style. {prompt}. High quality, detailed, ultra-realistic baseball scene."
            logger.info(f"Generating via Imagen: {prompt[:40]}...")
            
            response = self.client.models.generate_images(
                model=self.IMAGEN_MODEL_ID,
                prompt=full_prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio="9:16",
                    output_mime_type="image/png",
                )
            )

            if not response.generated_images:
                raise DataFetchError("No images returned from Imagen 3")

            image_bytes = response.generated_images[0].image.image_bytes
            cache_path = self._get_cache_path(prompt)
            return self._save_image(image_bytes, cache_path)
            
        except Exception as e:
            logger.error(f"Imagen 3 failure: {e}")
            return None

    def _generate_nano_banana(self, prompt: str, style: str, width: int, height: int) -> Optional[str]:
        """
        Generate using Nano Banana API.
        We assume the model expects a payload like {"prompt": ..., "negative_prompt": ...}
        and returns base64 image data.
        """
        logger.info(f"Generating via Nano Banana: {prompt[:40]}...")
        
        if not self.nano_api_key:
             logger.warning("Nano Banana API Key missing.")
             return None

        try:
            # Construct Generic Payload for SDXL/Flux likely deployed on Banana
            payload = {
                "apiKey": self.nano_api_key,
                "modelKey": "YOUR_MODEL_KEY", # Placeholder if using a specific model endpoint
                "modelInputs": {
                    "prompt": f"{style} style. {prompt}",
                    "negative_prompt": "text, watermark, low quality, blurry",
                    "width": width,
                    "height": height,
                    "num_inference_steps": 30
                }
            }
            
            # Since we don't have the EXACT endpoint without user input, we stub the actual POST.
            # In a real run, this would be:
            # response = requests.post(self.nano_url, json=payload)
            # result = response.json()
            # image_b64 = result['modelOutputs'][0]['image_base64']
            
            # Simulate failure to trigger fallback for now until User provides URL
            # or simulate success if we wanted to mock it.
            logger.warning("Nano Banana endpoint call mocked. User must provide NANO_BANANA_URL.")
            return None 
            
            # If we had real data:
            # image_bytes = base64.b64decode(image_b64)
            # cache_path = self._get_cache_path(prompt)
            # return self._save_image(image_bytes, cache_path)
            
        except Exception as e:
            logger.error(f"Nano Banana failure: {e}")
            return None

    def generate_scene_images(
        self,
        scenes: List[Dict],
        max_workers: Optional[int] = None,
    ) -> Dict[int, str]:
        """
        Generate images for all scenes in parallel.
        """
        workers = max_workers or self.max_workers
        results: Dict[int, str] = {}
        
        # Performance: Limit total generations (User Request: 10 max)
        limit_count = 10
        generated_count = 0

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for scene in scenes:
                scene_id = scene.get("scene_id", 0)
                
                # If limit reached, skip generation logic (will fallback later or we can assign fallback here)
                if generated_count >= limit_count:
                     logger.info(f"Image limit ({limit_count}) reached. Skipping generation for Scene {scene_id}.")
                     results[scene_id] = self._create_fallback_image(str(scene_id))
                     continue

                # Look for the new visual image_prompt field, fallback to root
                prompt = scene.get("visual", {}).get("image_prompt", "") 
                if not prompt:
                    prompt = scene.get("image_prompt", "")
                
                if not prompt:
                    logger.warning(f"Scene {scene_id} has no image_prompt, using fallback")
                    results[scene_id] = self._create_fallback_image(str(scene_id))
                    continue
                    
                future = pool.submit(self.generate_image, prompt)
                futures[future] = scene_id
                generated_count += 1

            for future in as_completed(futures):
                scene_id = futures[future]
                try:
                    image_path = future.result()
                    if image_path:
                        results[scene_id] = image_path
                    else:
                        logger.warning(f"Failed to generate for scene {scene_id}, using fallback.")
                        results[scene_id] = self._create_fallback_image(str(scene_id))
                except Exception as e:
                    logger.error(f"Image generation failed for scene {scene_id}: {e}")
                    results[scene_id] = self._create_fallback_image(str(scene_id))

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

    def _save_image(self, image_bytes: bytes, output_path: Path) -> str:
        """Save image bytes to disk. Returns path string."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_bytes)
        logger.info(f"Image saved: {output_path}")
        return str(output_path)
    
    def _create_fallback_image(self, seed: str) -> str:
        """Create a generative-looking dark gradient fallback image."""
        width, height = 1080, 1920
        img = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(img)
        
        # Deterministic color based on seed
        h_val = int(hashlib.md5(seed.encode()).hexdigest(), 16) % 360
        
        # Dark gradient from top to bottom
        for y in range(height):
            # Deep space gradient
            r = int(10 + (30 - 10) * y / height)
            g = int(14 + (30 - 14) * y / height)
            b = int(39 + (80 - 39) * y / height)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
            
        # Add 'stars' or noise
        import random
        random.seed(seed)
        for _ in range(100):
            x = random.randint(0, width)
            y = random.randint(0, height)
            draw.point((x, y), fill=(255, 255, 255, random.randint(50, 200)))

        fallback_path = self.output_dir / f"fallback_{seed}.png"
        img.save(str(fallback_path))
        return str(fallback_path)
