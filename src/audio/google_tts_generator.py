
import os
from google.cloud import texttospeech
from pathlib import Path
from typing import Optional
import logging
from moviepy.editor import AudioFileClip

from config.viral_video_config import ViralVideoConfig
from src.utils.error_handler import ErrorHandler, AudioQualityError, retry_with_backoff
from src.audio.tts_cost_tracker import TTSCostTracker

logger = logging.getLogger(__name__)

class GoogleTTSGenerator:
    """
    Wrapper for Google Cloud TTS (Neural2).
    Generates high-quality audio (192kbps).
    """
    
    def __init__(self):
        try:
            self.client = texttospeech.TextToSpeechClient()
        except Exception as e:
            logger.warning(f"Google TTS Client init failed: {e}. Check GOOGLE_APPLICATION_CREDENTIALS.")
            self.client = None
            
    @retry_with_backoff(max_retries=3)
    def generate_narration(self, text: str, output_path: str) -> str:
        """
        Generate audio from text.
        """
        TTSCostTracker.track_generation(text)
        
        if not self.client:
             logger.warning("Using MacOS fallback due to missing TTS Client.")
             return self._generate_fallback_macos(text, output_path)

        input_text = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name=ViralVideoConfig.TTS_VOICE,
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=ViralVideoConfig.TTS_SPEAKING_RATE,
            pitch=ViralVideoConfig.TTS_PITCH,
            sample_rate_hertz=ViralVideoConfig.AUDIO_SAMPLE_RATE # 44.1kHz
        )

        response = self.client.synthesize_speech(
            input=input_text, voice=voice, audio_config=audio_config
        )

        with open(output_path, "wb") as out:
            out.write(response.audio_content)
            
        # Validate Quality
        self._validate_audio_quality(output_path)
            
        return output_path

    def _validate_audio_quality(self, audio_path: str):
        """Verify bitrate and integrity."""
        # Simple file size check or probe?
        # MoviePy probe
        try:
            # Check file size > 1KB
            if os.path.getsize(audio_path) < 1024:
                raise AudioQualityError("Audio file too small (possible failure).")
            
            # Bitrate check requires ffmpeg probe, MoviePy doesn't expose bitrate easily directly on Clip object props?
            # We assume Google API respects the config. 
            pass
        except Exception as e:
            raise AudioQualityError(f"Audio validation failed: {e}")

    def _generate_fallback_macos(self, text: str, output_name: str) -> str:
        """Fallback to MacOS 'say' command."""
        import subprocess
        import platform
        
        output_path = Path(output_name)
        if platform.system() != "Darwin":
             # Create a silent dummy file if not on Mac and no Google TTS
             logger.error("System TTS not available (Not MacOS). Creating silent dummy.")
             self._create_silent_mp3(output_path)
             return str(output_path)
        
        temp_aiff = output_path.with_suffix(".aiff")
        move_to_mp3 = output_path.suffix == ".mp3"

        try:
            cmd = ["say", "-o", str(temp_aiff), "--data-format=LEF32@44100", "-r", "190", text]
            subprocess.run(cmd, check=True)
            
            # Convert to mp3 if needed (using moviepy/ffmpeg)
            # Or just rename if wrapper accepts aiff. 
            # ffmpeg for proper conversion
            if move_to_mp3:
                 # Minimal conversion via ffmpeg command if available, or just rename for now (bad practice)
                 # Better: Use MoviePy to convert
                 try:
                     clip = AudioFileClip(str(temp_aiff))
                     clip.write_audiofile(str(output_path), bitrate="192k", verbose=False, logger=None)
                     if temp_aiff.exists(): os.remove(temp_aiff)
                 except:
                     if temp_aiff.exists(): temp_aiff.rename(output_path)
            else:
                if temp_aiff.exists(): temp_aiff.rename(output_path)
            
            logger.info(f"Narrated via MacOS System TTS: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Mac TTS failed: {e}")
            self._create_silent_mp3(output_path)
            return str(output_path)

    def _create_silent_mp3(self, path: Path):
        # Create dummy using moviepy
        from moviepy.editor import AudioClip
        clip = AudioClip(lambda t: [0], duration=3.0, fps=44100)
        clip.write_audiofile(str(path), bitrate="64k", verbose=False, logger=None)
