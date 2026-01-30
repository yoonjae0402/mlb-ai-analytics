import pytest
from unittest.mock import MagicMock, patch, mock_open
import json
from src.content.script_generator import ScriptGenerator
from src.content.audio_generator import AudioGenerator
from src.utils.cost_tracker import CostTracker

class TestScriptGenerator:
    @pytest.fixture
    def mock_openai(self):
        with patch('src.content.script_generator.OpenAI') as mock:
            yield mock
            
    def test_generate_script(self, mock_openai, tmp_path):
        # Setup template
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        (template_dir / "series_middle.txt").write_text("Script: {key_insight}")
        
        # Mock Response
        # We need to structure the mock to survive property access chain
        mock_message = MagicMock()
        mock_message.content = "Generated Script" # Direct attribute
        
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
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
        
        tracker.log_openai_usage("gpt-4o", 1000, 500)
        
        assert log_file.exists()
        content = log_file.read_text()
        entry = json.loads(content)
        assert entry["input_tokens"] == 1000
        assert entry["model"] == "gpt-4o"
        assert entry["cost_usd"] > 0

class TestAudioGenerator:
    @pytest.fixture
    def mock_eleven(self):
        with patch('src.content.audio_generator.ElevenLabs') as mock:
            yield mock

    def test_generate_audio(self, mock_eleven, tmp_path):
        # Mock client
        mock_client = MagicMock()
        mock_eleven.return_value = mock_client
        mock_client.generate.return_value = [b"chunk1", b"chunk2"] # Generator
        
        tracker = MagicMock()
        generator = AudioGenerator(cost_tracker=tracker)
        generator.api_key = "fake_key" # Force init
        generator.client = mock_client # Force client injection
        
        output = tmp_path / "test.mp3"
        result = generator.generate_audio("Hello", str(output))
        
        assert result == str(output)
        assert output.exists()
        assert output.read_bytes() == b"chunk1chunk2"
        tracker.log_elevenlabs_usage.assert_called_with(generator.voice_id, 5)
