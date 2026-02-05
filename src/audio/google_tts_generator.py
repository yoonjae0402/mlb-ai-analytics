
import os
import re
from google.cloud import texttospeech
from pathlib import Path
from typing import Optional, List
import logging
from moviepy.editor import AudioFileClip

from config.viral_video_config import ViralVideoConfig
from src.utils.error_handler import ErrorHandler, AudioQualityError, retry_with_backoff
from src.audio.tts_cost_tracker import TTSCostTracker

logger = logging.getLogger(__name__)

class GoogleTTSGenerator:
    """
    Wrapper for Google Cloud TTS (Neural2).
    Generates high-quality audio (192kbps) with SSML support
    for natural-sounding speech with emphasis and pauses.
    """

    # Words that should be emphasized in MLB context
    DEFAULT_EMPHASIS_WORDS = [
        'win', 'wins', 'won', 'victory', 'defeat', 'defeated',
        'shock', 'shocked', 'upset', 'stunning', 'dominant', 'domination',
        'home run', 'homer', 'grand slam', 'strikeout', 'strikeouts',
        'shutout', 'no-hitter', 'walk-off', 'clutch', 'incredible',
    ]

    def __init__(self):
        try:
            self.client = texttospeech.TextToSpeechClient()
        except Exception as e:
            logger.warning(f"Google TTS Client init failed: {e}. Check GOOGLE_APPLICATION_CREDENTIALS.")
            self.client = None

    @retry_with_backoff(max_retries=3)
    def generate_narration(self, text: str, output_path: str, emphasis_words: Optional[List[str]] = None) -> str:
        """
        Generate audio from text using SSML for natural speech.

        Args:
            text: Plain text to synthesize
            output_path: Path to save the audio file
            emphasis_words: Additional words to emphasize (team/player names)
        """
        TTSCostTracker.track_generation(text)

        if not self.client:
             logger.warning("Using MacOS fallback due to missing TTS Client.")
             return self._generate_fallback_macos(text, output_path)

        # Convert plain text to SSML for natural speech
        ssml = self._text_to_ssml(text, emphasis_words)

        synthesis_input = texttospeech.SynthesisInput(ssml=ssml)

        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name=ViralVideoConfig.TTS_VOICE,
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=ViralVideoConfig.TTS_SPEAKING_RATE,
            pitch=ViralVideoConfig.TTS_PITCH,
            sample_rate_hertz=getattr(ViralVideoConfig, 'TTS_SAMPLE_RATE', 44100),
            effects_profile_id=getattr(ViralVideoConfig, 'TTS_EFFECTS_PROFILE', []),
        )

        response = self.client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        with open(output_path, "wb") as out:
            out.write(response.audio_content)

        # Validate Quality
        self._validate_audio_quality(output_path)

        return output_path

    def _text_to_ssml(self, text: str, extra_emphasis_words: Optional[List[str]] = None) -> str:
        """
        Convert plain text script to SSML markup for natural speech.

        Adds:
        - Pauses between sentences
        - Emphasis on key words (team names, player names, action words)
        - Prosody variation for excitement
        """
        emphasis_set = set(w.lower() for w in self.DEFAULT_EMPHASIS_WORDS)
        if extra_emphasis_words:
            for w in extra_emphasis_words:
                emphasis_set.add(w.lower())

        ssml_parts = ['<speak>']

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            # Add pause between sentences
            if i > 0:
                ssml_parts.append('<break time="250ms"/>')

            # Check if this is an exciting sentence (ends with !)
            is_excited = sentence.strip().endswith('!')

            if is_excited:
                ssml_parts.append('<prosody rate="105%" pitch="+1st">')

            # Process words for emphasis
            words = sentence.split()
            processed_words = []
            skip_next = False

            for j, word in enumerate(words):
                if skip_next:
                    skip_next = False
                    continue

                clean_word = re.sub(r'[.,!?;:]', '', word).lower()

                # Check two-word phrases (e.g., "home run", "grand slam")
                if j + 1 < len(words):
                    two_word = clean_word + ' ' + re.sub(r'[.,!?;:]', '', words[j + 1]).lower()
                    if two_word in emphasis_set:
                        processed_words.append(f'<emphasis level="strong">{word} {words[j + 1]}</emphasis>')
                        skip_next = True
                        continue

                # Check single word emphasis
                if clean_word in emphasis_set:
                    processed_words.append(f'<emphasis level="moderate">{word}</emphasis>')
                else:
                    processed_words.append(word)

            ssml_parts.append(' '.join(processed_words))

            if is_excited:
                ssml_parts.append('</prosody>')

        ssml_parts.append('</speak>')
        return ' '.join(ssml_parts)

    def _validate_audio_quality(self, audio_path: str):
        """Verify bitrate and integrity."""
        try:
            if os.path.getsize(audio_path) < 1024:
                raise AudioQualityError("Audio file too small (possible failure).")
        except AudioQualityError:
            raise
        except Exception as e:
            raise AudioQualityError(f"Audio validation failed: {e}")

    def _generate_fallback_macos(self, text: str, output_name: str) -> str:
        """Fallback to MacOS 'say' command."""
        import subprocess
        import platform

        output_path = Path(output_name)
        if platform.system() != "Darwin":
             logger.error("System TTS not available (Not MacOS). Creating silent dummy.")
             self._create_silent_mp3(output_path)
             return str(output_path)

        temp_aiff = output_path.with_suffix(".aiff")
        move_to_mp3 = output_path.suffix == ".mp3"

        try:
            cmd = ["say", "-o", str(temp_aiff), "--data-format=LEF32@44100", "-r", "190", text]
            subprocess.run(cmd, check=True)

            if move_to_mp3:
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
        from moviepy.editor import AudioClip
        clip = AudioClip(lambda t: [0], duration=3.0, fps=44100)
        clip.write_audiofile(str(path), bitrate="64k", verbose=False, logger=None)
