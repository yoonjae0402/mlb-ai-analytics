
import unittest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import json

from src.utils.cost_tracker import CostTracker
from src.audio.tts_cache import TTSCache
from src.audio.google_tts import GoogleTTSGenerator
from config.settings import settings

class TestCostTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = CostTracker(log_file="test_costs.jsonl")

    def tearDown(self):
        if Path("test_costs.jsonl").exists():
            Path("test_costs.jsonl").unlink()

    def test_calculate_tts_cost(self):
        # Neural2 price: 0.000016 per char
        cost = self.tracker.calculate_tts_cost("Hello", "neural2")
        self.assertAlmostEqual(cost, 5 * 0.000016)

        # Standard price: 0.000004 per char
        cost = self.tracker.calculate_tts_cost("Hello", "standard")
        self.assertAlmostEqual(cost, 5 * 0.000004)

    def test_track_cost(self):
        self.tracker.track_cost("test_provider", 0.50)
        self.assertEqual(self.tracker.get_session_cost("test_provider"), 0.50)
        self.assertEqual(self.tracker.get_total_cost(), 0.50)


class TestTTSCache(unittest.TestCase):
    def setUp(self):
        self.cache_dir = Path("test_cache")
        self.cache = TTSCache(cache_dir=self.cache_dir)

    def tearDown(self):
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)

    def test_cache_key(self):
        key1 = self.cache.get_cache_key("hello", "en-US-Neural2-J", "google-tts")
        key2 = self.cache.get_cache_key("hello", "en-us-neural2-j", "google-tts") # Case insensitive voice
        self.assertEqual(key1, key2)

    def test_save_and_get(self):
        data = b"fake_audio_data"
        text = "test text"
        voice = "voice1"
        model = "model1"
        
        path = self.cache.save(data, text, voice, model)
        self.assertTrue(path.exists())
        
        cached_path = self.cache.get(text, voice, model)
        self.assertEqual(path, cached_path)


class TestGoogleTTS(unittest.TestCase):
    @patch("src.audio.google_tts.texttospeech.TextToSpeechClient")
    def test_generate_success(self, MockClient):
        # Setup mock
        mock_instance = MockClient.return_value
        mock_response = MagicMock()
        mock_response.audio_content = b"audio_content"
        mock_instance.synthesize_speech.return_value = mock_response

        generator = GoogleTTSGenerator()
        
        # Patch cache to avoid FS writes
        with patch.object(generator.cache, 'save', return_value=Path("mock_audio.mp3")) as mock_save:
            with patch.object(generator.cache, 'get', return_value=None): # Force miss
                path = generator.generate("Hello world")
                
                self.assertEqual(path, Path("mock_audio.mp3"))
                mock_instance.synthesize_speech.assert_called_once()


if __name__ == "__main__":
    unittest.main()
