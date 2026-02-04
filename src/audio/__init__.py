"""
MLB Video Pipeline - Audio Package

Text-to-speech narration and audio mixing:
- tts_engine: Generate natural voice narration
- audio_mixer: Mix TTS with SFX and background music

Usage:
    from src.audio import TTSEngine, AudioMixer

    tts = TTSEngine()
    audio_path = tts.generate_narration(script_text)
"""

from src.audio.tts_engine import TTSEngine
from src.audio.audio_mixer import AudioMixer

__all__ = ["TTSEngine", "AudioMixer"]
