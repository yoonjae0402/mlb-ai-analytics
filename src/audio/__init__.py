"""
MLB Video Pipeline - Audio Package

Text-to-speech narration using local Qwen3-TTS (free, no API key):
- tts_engine: Generate natural voice narration

Usage:
    from src.audio import TTSEngine

    tts = TTSEngine()
    audio_path = tts.generate_narration(script_text)
"""

from src.audio.tts_engine import TTSEngine

__all__ = ["TTSEngine"]
