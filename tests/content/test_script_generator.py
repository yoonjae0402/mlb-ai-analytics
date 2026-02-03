import pytest
from unittest.mock import MagicMock, patch
from src.content.script_generator import ScriptGenerator

class TestScriptGenerator:
    
    @pytest.fixture
    def mock_genai(self):
        with patch('src.content.script_generator.genai') as mock:
            yield mock
            
    def test_generate_script(self, mock_genai, tmp_path):
        # Create dummy template
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        (template_dir / "series_middle.txt").write_text("Test Script: {key_insight}, Prediction: {prediction_class}")
        
        # Mock Gemini API response
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.text = "Generated Script Content"
        mock_client.models.generate_content.return_value = mock_response
        
        generator = ScriptGenerator(template_dir=str(template_dir))
        
        # Test Data
        game_data = {"away_team": "A", "home_team": "B", "date": "2024-01-01"}
        analysis = {"insights": ["Big Win"], "top_performers": [{"player": "P1", "highlight": "HR"}]}
        prediction = {"prediction": "Above Average", "confidence": "High", "reasons": ["R1"]}
        
        script = generator.generate_script(game_data, analysis, prediction, "series_middle")
        
        assert script == "Generated Script Content"
        mock_client.models.generate_content.assert_called_once()
        
    def test_missing_template_fallback(self, mock_genai, tmp_path):
        # Only create series_middle
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        (template_dir / "series_middle.txt").write_text("Fallback Template")

        # Mock Gemini API
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.text = "Script"
        mock_client.models.generate_content.return_value = mock_response

        generator = ScriptGenerator(template_dir=str(template_dir))

        # Request non-existent template
        script = generator.generate_script({}, {}, {}, "non_existent")
        
        # Should have called API (meaning it successfully loaded fallback)
        assert script == "Script"
