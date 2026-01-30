import os
import sys
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.content import ScriptGenerator, AudioGenerator
from src.utils import CostTracker

def verify_content_generation():
    print("🎬 Verifying MLB Video Pipeline - Content Generation 🎬")
    
    # Mock External APIs to avoid costs during verification
    with patch('src.content.script_generator.OpenAI') as mock_openai, \
         patch('src.content.audio_generator.ElevenLabs') as mock_eleven:
        
        # Setup Mock OpenAI
        mock_choice = MagicMock()
        mock_choice.message.content = "Start with a bang! The Yankees win! Prediction: Red Sox next."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        # Setup Mock ElevenLabs
        mock_eleven.return_value.generate.return_value = [b"audio_chunk"]
        
        # 1. Script Generation
        print("\n1. Generating Script...")
        script_gen = ScriptGenerator()
        
        game_data = {
            "away_team": "Red Sox", 
            "home_team": "Yankees", 
            "date": "2024-07-04",
            "away_score": 3,
            "home_score": 5
        }
        analysis = {
            "insights": ["Comeback win"], 
            "top_performers": [{"player": "Judge", "highlight": "2 HR"}],
            "game_flow": {"summary": "Yankees win 5-3"}
        }
        prediction = {
            "prediction": "Above Average", 
            "confidence": "High", 
            "reasons": ["Hot streak"]
        }
        
        script = script_gen.generate_script(game_data, analysis, prediction)
        print(f"   Script: {script}")
        
        # 2. Audio Generation
        print("\n2. Generating Audio...")
        cost_tracker = CostTracker(log_file="data/test_costs.jsonl")
        audio_gen = AudioGenerator(cost_tracker=cost_tracker)
        # Inject fake key to bypass check
        audio_gen.api_key = "test_key"
        audio_gen.client = mock_eleven.return_value
        
        output_path = "data/test_audio.mp3"
        result = audio_gen.generate_audio(script, output_path)
        print(f"   Audio saved to: {result}")
        
        # 3. Cost Tracking
        print("\n3. Verifying Cost Tracking...")
        total = cost_tracker.get_total_cost()
        print(f"   Total Session Cost: ${total:.5f}")
        
        if os.path.exists(output_path):
            os.remove(output_path)
            
    print("\n✅ Content Generation Verification Successful!")

if __name__ == "__main__":
    verify_content_generation()
