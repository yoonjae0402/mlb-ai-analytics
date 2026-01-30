import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class ScriptGenerator:
    """
    Generates video scripts using GPT-4 based on game analysis and predictions.
    """
    
    def __init__(self, template_dir: str = "./src/content/templates"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.template_dir = Path(template_dir)
        
    def generate_script(
        self,
        game_data: Dict[str, Any],
        analysis: Dict[str, Any],
        prediction: Dict[str, Any],
        video_type: str = "series_middle"
    ) -> str:
        """
        Generate a video script.
        
        Args:
            game_data: Basic game info (teams, score, date)
            analysis: Output from MLBStatsAnalyzer
            prediction: Output from PredictionExplainer
            video_type: 'series_middle' or 'series_end'
            
        Returns:
            Generated script text.
        """
        try:
            prompt = self._create_prompt(game_data, analysis, prediction, video_type)
            
            logger.info(f"Generating script for {video_type} video...")
            response = self.client.chat.completions.create(
                model="gpt-4o", # Use GPT-4o for best quality/cost balance
                messages=[
                    {"role": "system", "content": "You are a professional MLB content creator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            script = response.choices[0].message.content.strip()
            return script
            
        except Exception as e:
            logger.error(f"Error generating script: {e}")
            return "Error generating script. Please check logs."

    def _create_prompt(
        self,
        game_data: Dict,
        analysis: Dict,
        prediction: Dict,
        video_type: str
    ) -> str:
        """Hydrate the appropriate template with data."""
        template_file = self.template_dir / f"{video_type}.txt"
        
        if not template_file.exists():
            # Fallback to middle if specific template missing
            logger.warning(f"Template {video_type} not found, using series_middle")
            template_file = self.template_dir / "series_middle.txt"
            
        with open(template_file, "r") as f:
            template = f.read()
            
        # Extract analysis fields safely
        insights = analysis.get("insights", ["Great game"])
        key_insight = insights[0] if insights else "Intense matchup"
        
        performers = analysis.get("top_performers", [])
        top_perf = performers[0] if performers else {"player": "The Team", "highlight": "solid play"}
        
        # Format variables
        context = {
            "away_team": game_data.get("away_team", "Away Team"),
            "home_team": game_data.get("home_team", "Home Team"),
            "game_date": game_data.get("date", "Today"),
            "result_text": analysis.get("game_flow", {}).get("summary", "Complete"),
            "away_score": game_data.get("away_score", 0),
            "home_score": game_data.get("home_score", 0),
            "key_insight": key_insight,
            "top_performer": top_perf.get("player"),
            "performer_impact": top_perf.get("highlight"),
            "game_number": "2", # Placeholder, would come from SeriesTracker
            "series_status": "Series tied 1-1", # Placeholder
            "prediction_class": prediction.get("prediction", "Unknown"),
            "confidence": prediction.get("confidence", "Low"),
            "prediction_reasons": ", ".join(prediction.get("reasons", [])),
            
            # Extra fields for series_end
            "series_summary": "Yankees take 2 of 3",
            "next_series_opponent": "Red Sox"
        }
        
        return template.format(**context)
