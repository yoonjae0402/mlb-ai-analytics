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
