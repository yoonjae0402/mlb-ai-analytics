import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional
import shutil

from config.settings import settings

logger = logging.getLogger(__name__)


class TTSCache:
    """
    Caching system for Text-to-Speech audio files.
    
    Uses MD5 hashing of text + voice parameters to create unique keys.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or settings.cache_dir / "tts"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.jsonl"

    def get_cache_key(self, text: str, voice: str, model: str) -> str:
        """Generate unique hash for cache key."""
        # Normalize inputs
        text = text.strip()
        voice = voice.lower()
        model = model.lower()
        
        # Create hash
        payload = f"{text}|{voice}|{model}".encode("utf-8")
        return hashlib.md5(payload).hexdigest()

    def get(self, text: str, voice: str, model: str) -> Optional[Path]:
        """
        Retrieve file from cache if it exists.
        
        Returns:
            Path to cached file or None if not found.
        """
        if not settings.enable_tts_cache:
            return None
            
        key = self.get_cache_key(text, voice, model)
        # Look for expected filename pattern
        # We search for any file starting with this hash to handle extensions
        for file_path in self.cache_dir.glob(f"{key}.*"):
            if file_path.exists():
                logger.debug(f"TTS Cache HIT: {key} ({file_path.name})")
                
                # Touch file to update modification time (for LRU cleanup)
                try:
                    file_path.touch()
                except OSError:
                    pass
                    
                return file_path
                
        logger.debug(f"TTS Cache MISS: {key}")
        return None

    def save(self, audio_data: bytes, text: str, voice: str, model: str, extension: str = "mp3") -> Path:
        """
        Save audio data to cache.
        
        Args:
            audio_data: Binary audio content
            text: Original text
            voice: Voice ID
            model: Model ID
            extension: File extension (default: mp3)
            
        Returns:
            Path to saved file
        """
        key = self.get_cache_key(text, voice, model)
        filename = f"{key}.{extension.lstrip('.')}"
        file_path = self.cache_dir / filename
        
        try:
            with open(file_path, "wb") as f:
                f.write(audio_data)
            
            # Log metadata
            self._log_metadata(key, text, voice, model, file_path.name)
            logger.debug(f"Saved to TTS cache: {file_path}")
            
            return file_path
        except Exception as e:
            logger.error(f"Failed to save to TTS cache: {e}")
            # If save fails, we should still return something usable if we wrote it, 
            # or maybe just raise. But for cache, we can just return the path 
            # even if writing failed (it will just fail later), or better, return user temp path.
            # But here we assume it worked or raised.
            raise

    def cleanup(self, max_days: int = 90) -> int:
        """
        Remove files older than max_days.
        
        Returns:
            Number of bytes freed.
        """
        now = time.time()
        cutoff = now - (max_days * 86400)
        bytes_freed = 0
        count = 0
        
        for file_path in self.cache_dir.glob("*"):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff:
                try:
                    size = file_path.stat().st_size
                    file_path.unlink()
                    bytes_freed += size
                    count += 1
                except OSError as e:
                    logger.warning(f"Failed to delete cached file {file_path}: {e}")
                    
        if count > 0:
            logger.info(f"TTS Cache cleanup: Removed {count} files, freed {bytes_freed / 1024 / 1024:.2f} MB")
            
        return bytes_freed

    def _log_metadata(self, key: str, text: str, voice: str, model: str, filename: str) -> None:
        """Append metadata to JSONL file."""
        entry = {
            "timestamp": time.time(),
            "key": key,
            "voice": voice,
            "model": model,
            "filename": filename,
            "text_preview": text[:50]
        }
        try:
            with open(self.metadata_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log cache metadata: {e}")
