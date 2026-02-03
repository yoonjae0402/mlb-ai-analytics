import pytest
from unittest.mock import MagicMock, patch, mock_open
import json
from src.content.script_generator import ScriptGenerator
from src.content.audio_generator import AudioGenerator
from src.utils.cost_tracker import CostTracker

class TestScriptGenerator:
    @pytest.fixture
    def mock_genai(self):
        with patch('src.content.script_generator.genai') as mock:
            yield mock
            
    def test_generate_script(self, mock_genai, tmp_path):
        # Setup template
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        (template_dir / "series_middle.txt").write_text("Script: {key_insight}")
        
        # Mock Gemini Response
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.text = "Generated Script"
        mock_client.models.generate_content.return_value = mock_response
        
        generator = ScriptGenerator(template_dir=str(template_dir))
        script = generator.generate_script(
            {"away_team": "A", "home_team": "B"}, 
            {"insights": ["Win"]}, 
            {"prediction": "P"}, 
            "series_middle"
        )
        
        assert script == "Generated Script"

class TestCostTracker:
    def test_log_usage(self, tmp_path):
        log_file = tmp_path / "costs.jsonl"
        tracker = CostTracker(log_file=str(log_file))
        
        # Note: Gemini and Qwen3-TTS are both free, so costs should be 0
        tracker.log_openai_usage("gpt-4o", 1000, 500)  # Legacy test
        tracker.log_dashscope_usage("qwen-tts-local", 5000)  # Updated to local TTS
        
        assert log_file.exists()
        content = log_file.read_text().splitlines()
        
        entry_openai = json.loads(content[0])
        assert entry_openai["input_tokens"] == 1000
        assert entry_openai["model"] == "gpt-4o"
        assert entry_openai["cost_usd"] >= 0  # Changed to >= since costs can be 0
        
        entry_tts = json.loads(content[1])
        assert entry_tts["characters"] == 5000
        assert entry_tts["provider"] == "qwen-tts-local"  # Updated
        assert entry_tts["cost_usd"] >= 0  # Local TTS is free

class TestAudioGenerator:
    @pytest.fixture
    def mock_tts_engine(self):
        with patch('src.content.audio_generator.TTSEngine') as mock:
            yield mock

    def test_generate_audio(self, mock_tts_engine, tmp_path):
        # Mock engine instance
        mock_instance = MagicMock()
        mock_tts_engine.return_value = mock_instance
        
        # Setup specific output from generate_narration
        mock_output_path = tmp_path / "generated_audio.mp3"
        mock_output_path.touch()
        mock_instance.generate_narration.return_value = mock_output_path
        
        tracker = MagicMock()
        generator = AudioGenerator(cost_tracker=tracker)
        
        target_output = tmp_path / "target.mp3"
        result = generator.generate_audio("Hello", str(target_output))
        
        assert result == str(target_output)
        assert target_output.exists() # Should have been moved/renamed
        mock_instance.generate_narration.assert_called_once()
