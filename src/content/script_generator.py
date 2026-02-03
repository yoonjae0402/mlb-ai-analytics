import os
import json
import logging
import re
from typing import Dict, Any, Optional, List
from pathlib import Path
from google import genai
from dotenv import load_dotenv

from src.utils.exceptions import ScriptGenerationError

load_dotenv()
logger = logging.getLogger(__name__)

CINEMATIC_SYSTEM_PROMPT = """You are a Real-Time MLB Video Producer. You create structured JSON scripts for cinematic YouTube Shorts recap videos.

You MUST output ONLY valid JSON matching this exact schema — no extra text, no markdown fences:

{
  "video_metadata": {
    "title": "short YouTube title (under 100 chars)",
    "description": "2-3 sentence YouTube description",
    "tags": ["tag1", "tag2", "tag3"],
    "duration_target": 45
  },
  "scenes": [
    {
      "scene_id": 1,
      "narration": "spoken narration text for this scene (15-20 words)",
      "image_prompt": "detailed cinematic image description for AI generation (stadium, players, action, lighting)",
      "motion_type": "zoom_in",
      "text_overlay": "SHORT ON-SCREEN TEXT",
      "duration": 8.0
    }
  ]
}

Rules:
- Generate exactly 4-5 scenes
- Each scene narration should be 15-20 words (concise, hype commentary)
- Total narration across all scenes should be under 80 words
- motion_type MUST be one of: "zoom_in", "zoom_out", "pan_left", "pan_right"
- Vary motion_type across scenes for visual variety
- image_prompt should describe a cinematic baseball scene (8K quality, dramatic lighting, stadium atmosphere)
- text_overlay should be short (3-6 words) key stat or moment
- duration_target should be 30-45 seconds total
- Tags should include team names, player names, and "MLB Shorts"
"""


class ScriptGenerator:
    """
    Generates video scripts using Google Gemini based on game analysis and predictions.
    """

    def __init__(self, template_dir: str = "./src/content/templates"):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables.")

        self.client = genai.Client(api_key=self.api_key)
        self.template_dir = Path(template_dir)

    # =========================================================================
    # Original plain-text script generation (backward compatible)
    # =========================================================================

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

            logger.info(f"Generating script for {video_type} video using Gemini...")

            # Generate content with Gemini
            response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=150,
                )
            )

            script = response.text.strip()
            return script

        except Exception as e:
            logger.error(f"Error generating script with Gemini: {e}")
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
            "game_number": "1",
            "series_status": game_data.get("series_status", ""),
            "prediction_class": prediction.get("prediction", "Unknown"),
            "confidence": prediction.get("confidence", "Low"),
            "prediction_reasons": ", ".join(prediction.get("reasons", [])),

            # Extra fields for series_end
            "series_summary": game_data.get("series_status", ""),
            "next_series_opponent": game_data.get("home_team", "TBD")
        }

        return template.format(**context)

    # =========================================================================
    # Cinematic JSON script generation (Phase 2A)
    # =========================================================================

    def generate_cinematic_script(
        self,
        game_data: Dict[str, Any],
        analysis: Dict[str, Any],
        prediction: Dict[str, Any],
        video_type: str = "series_middle"
    ) -> Dict[str, Any]:
        """
        Generate a structured JSON script for cinematic video production.

        Args:
            game_data: Basic game info (teams, score, date)
            analysis: Output from MLBStatsAnalyzer
            prediction: Output from PredictionExplainer
            video_type: 'series_middle' or 'series_end'

        Returns:
            Parsed dict with 'video_metadata' and 'scenes' keys.

        Raises:
            ScriptGenerationError: If JSON parsing or validation fails.
        """
        prompt = self._create_cinematic_prompt(game_data, analysis, prediction, video_type)

        logger.info(f"Generating cinematic script for {video_type} video using Gemini...")

        try:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=CINEMATIC_SYSTEM_PROMPT,
                    temperature=0.7,
                    max_output_tokens=1024,
                )
            )

            raw_text = response.text.strip()
            script = self._parse_script_json(raw_text)
            self._validate_script_structure(script)

            logger.info(
                f"Cinematic script generated: {len(script['scenes'])} scenes, "
                f"title='{script['video_metadata']['title']}'"
            )
            return script

        except ScriptGenerationError:
            raise
        except Exception as e:
            raise ScriptGenerationError(
                f"Cinematic script generation failed: {e}",
                model="gemini-2.0-flash",
                reason="api_error",
            ) from e

    def _create_cinematic_prompt(
        self,
        game_data: Dict,
        analysis: Dict,
        prediction: Dict,
        video_type: str,
    ) -> str:
        """Build the user prompt for cinematic JSON script generation."""
        away = game_data.get("away_team", "Away Team")
        home = game_data.get("home_team", "Home Team")
        away_score = game_data.get("away_score", 0)
        home_score = game_data.get("home_score", 0)
        date = game_data.get("date", "Today")
        series_status = game_data.get("series_status", "")

        insights = analysis.get("key_insights", analysis.get("insights", []))
        storylines = analysis.get("storylines", [])
        performers = analysis.get("top_performances", analysis.get("top_performers", []))
        game_flow = analysis.get("game_flow", {})

        pred_text = prediction.get("prediction", "Unknown")
        pred_confidence = prediction.get("confidence", "Low")
        pred_reasons = prediction.get("reasons", [])

        prompt = f"""Create a cinematic MLB Shorts recap video script for this game:

Game: {away} ({away_score}) at {home} ({home_score}) on {date}
Video type: {video_type}
Series status: {series_status}

Key insights: {json.dumps(insights[:3]) if insights else 'Competitive game'}
Storylines: {json.dumps(storylines[:3]) if storylines else 'Close matchup'}
Top performers: {json.dumps(performers[:3]) if performers else 'Solid team effort'}
Game flow: {json.dumps(game_flow) if game_flow else 'Standard game'}

Prediction: {pred_text} (Confidence: {pred_confidence})
Reasons: {', '.join(pred_reasons) if pred_reasons else 'Multiple factors'}

Generate the JSON script now."""

        return prompt

    def _parse_script_json(self, raw_text: str) -> Dict[str, Any]:
        """
        Parse Gemini's response into validated JSON.
        Strips markdown code fences if present.
        """
        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        cleaned = re.sub(r'^```(?:json)?\s*', '', raw_text, flags=re.MULTILINE)
        cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ScriptGenerationError(
                f"Failed to parse cinematic script JSON: {e}",
                model="gemini-2.0-flash",
                reason="json_parse_error",
            ) from e

    def _validate_script_structure(self, script: Dict[str, Any]) -> None:
        """Validate the cinematic script has all required fields."""
        if "video_metadata" not in script:
            raise ScriptGenerationError(
                "Cinematic script missing 'video_metadata'",
                model="gemini-2.0-flash",
                reason="validation_error",
            )

        if "scenes" not in script or not isinstance(script["scenes"], list):
            raise ScriptGenerationError(
                "Cinematic script missing 'scenes' array",
                model="gemini-2.0-flash",
                reason="validation_error",
            )

        if len(script["scenes"]) == 0:
            raise ScriptGenerationError(
                "Cinematic script has no scenes",
                model="gemini-2.0-flash",
                reason="validation_error",
            )

        allowed_motions = {"zoom_in", "zoom_out", "pan_left", "pan_right"}

        for i, scene in enumerate(script["scenes"]):
            # Ensure scene_id exists
            if "scene_id" not in scene:
                scene["scene_id"] = i + 1

            for field in ("narration", "image_prompt", "motion_type"):
                if field not in scene:
                    raise ScriptGenerationError(
                        f"Scene {scene.get('scene_id', i+1)} missing '{field}'",
                        model="gemini-2.0-flash",
                        reason="validation_error",
                    )

            if scene["motion_type"] not in allowed_motions:
                logger.warning(
                    f"Scene {scene['scene_id']} has invalid motion_type "
                    f"'{scene['motion_type']}', defaulting to 'zoom_in'"
                )
                scene["motion_type"] = "zoom_in"

            # Ensure optional fields have defaults
            scene.setdefault("text_overlay", "")
            scene.setdefault("duration", 8.0)
