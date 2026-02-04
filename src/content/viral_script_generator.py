
import logging
import json
import re
from typing import Dict, Any, List, Optional
from src.utils.logger import get_logger
from config.settings import settings
from src.video.timing_coordinator import TimingCoordinator

# Google GenAI
try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

logger = get_logger(__name__)

class ViralScriptGenerator:
    """
    Generates high-energy, fast-paced viral video scripts using Gemini.
    Output is strictly structured JSON for the ViralVideoEngine.
    Enforces 14-section structure for 60s target.
    """
    
    VIRAL_SYSTEM_PROMPT = """
You are a viral sports content creator. Create a STRICT 58-60 second MLB recap script for a vertical video (TikTok/Shorts).
Output STRICT JSON.

TIMING & STRUCTURE REQUIREMENTS:
You MUST generate exactly these 14 scenes with these IDs and types:

1.  HOOK (3s): Shocking stat or result. High energy.
2.  SCORE (4s): Final score reveal.
3.  TOP_HITTER (5s): Specific stats (e.g. "Judge: 2 HRs").
4.  TOP_PITCHER (5s): Specific stats (e.g. "Cole: 10 Ks").
5.  KEY_MOMENT (5s): The play that changed the game.
6.  STANDINGS (4s): Impact on playoff race.
7.  TRANSITION (2s): "But the real story is tomorrow..."
8.  PREDICTION_METER (5s): Visual win probability gauge.
9.  FACTOR_1 (4s): Reasoning #1 (e.g. "Home Field Advantage").
10. FACTOR_2 (4s): Reasoning #2 (e.g. "Hot Streak").
11. FACTOR_3 (4s): Reasoning #3 (e.g. "Head-to-Head").
12. PLAYER_WATCH_1 (5s): Prediction for Player A.
13. PLAYER_WATCH_2 (5s): Prediction for Player B.
14. CTA (4s): "Follow for daily picks."

Total Duration: ~59s.
Total Word Count: 140-160 Words (Narration needs to fill the time fast).

Output JSON Format:
{
  "video_metadata": {
    "title": "Clickbaity title (max 50 chars)",
    "description": "Short description with hashtags",
    "tags": ["mlb", "baseball", "team_name"]
  },
  "scenes": [
    {
      "scene_id": 1,
      "scene_type": "hook", 
      "duration": 3.0,
      "narration": "The Yankees just sent a message to the entire league!",
      "tts_pace": "fast",
      "visual": {
        "main_text": "YANKEES STATEMENT WIN",
        "sub_text": "10-2 DOMINATION",
        "image_prompt": "cinematic baseball stadium, explosion of lights, 8k render",
        "player_to_show": null,
        "stat_to_show": null
      },
      "sound_effect": "boom"
    }
    // ... all 14 scenes
  ]
}

Rules:
- Narrations must be CONCISE but punchy.
- Use the provided DATA explicitly (don't hallucinate stats).
- Visual 'main_text' should be 2-4 words max.
- 'player_to_show' must match the name in data EXACTLY for photo lookup.
- **IMAGE LIMIT**: You have a budget of 10 unique AI images. For Scenes 11-14, set "image_prompt" to null.
- **NO IMAGES FOR**: `hook`, `score`, `transition`, `cta`. Set "image_prompt" to null for these types. They will use a tech background.
    """

    def __init__(self):
        self.api_key = settings.gemini_api_key
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found. Script generation will fail.")
        
        if _GENAI_AVAILABLE and self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def generate_viral_script(
        self,
        game_data: Dict[str, Any],
        prediction: Dict[str, Any],
        next_game_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate the viral script JSON.
        """
        if not self.client:
            logger.error("Client not initialized")
            return self._fallback_script(game_data)

        prompt = self._create_prompt(game_data, prediction, next_game_data)
        
        try:
            logger.info("Calling Gemini for viral script...")
            response = self.client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.VIRAL_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    temperature=0.7,
                )
            )
            
            if not response.text:
                raise ValueError("Empty response from Gemini")

            # Parse JSON
            script = json.loads(response.text)
            self._validate_and_fix_script(script)
            logger.info(f"Generated viral script with {len(script.get('scenes', []))} scenes")
            return script

        except Exception as e:
            logger.error(f"Viral script generation failed: {e}")
            return self._fallback_script(game_data)

    def _create_prompt(self, game: Dict, prediction: Dict, next_game: Optional[Dict]) -> str:
        """Construct the user prompt with complete context."""
        
        # 1. Game Result Context
        winner = game.get("winner", "The Winner")
        score_str = f"{game.get('away_team')} {game.get('away_score')} - {game.get('home_team')} {game.get('home_score')}"
        
        # 2. Top Performers Context
        top_hitter = game.get('top_hitter', {})
        top_pitcher = game.get('top_pitcher', {})
        
        hitter_str = f"Top Hitter: {top_hitter.get('name')} ({top_hitter.get('stats')})"
        pitcher_str = f"Winning Pitcher: {top_pitcher.get('name')} ({top_pitcher.get('stats')})"
        
        # 3. Prediction Context
        factors = prediction.get('factors', [])
        players_watch = prediction.get('players_to_watch', [])
        
        factors_str = "\n".join([f"- Factor {i+1}: {f['factor']} ({f['impact']}) - {f['detail']}" for i, f in enumerate(factors)])
        players_str = "\n".join([f"- Player {i+1}: {p['name']} - Prediction: {p.get('predicted_stat', '')}" for i, p in enumerate(players_watch)])
        
        next_game_str = "Next game details unavailable"
        if next_game:
            next_game_str = f"{next_game.get('opponent')} on {next_game.get('date')}"

        prompt = f"""
        GENERATE SCRIPT FOR:
        
        GAME RECAP:
        - Matchup: {game.get('away_team')} vs {game.get('home_team')}
        - Score: {score_str}
        - Winner: {winner}
        - Key Moment: {game.get('key_moment', {}).get('description')}
        - Standings: {game.get('standings_impact', 'Big win')}
        
        STARS:
        - {hitter_str}
        - {pitcher_str}
        
        PREDICTION ({next_game_str}):
        - Win Prob: {int(prediction.get('win_probability', 0.5)*100)}% ({prediction.get('confidence')} Confidence)
        
        REASONING:
        {factors_str}
        
        PLAYERS TO WATCH:
        {players_str}
        """
        return prompt

    def _validate_and_fix_script(self, script: Dict):
        """Ensure script matches 14-scene structure and timings."""
        
        scenes = script.get("scenes", [])
        
        # Check scene count (warn but maybe allow minor deviance if valid?)
        # Strict mode: 14 scenes.
        if len(scenes) != 14:
            logger.warning(f"Generated {len(scenes)} scenes, expected 14. Timings might be off.")
            
        # Force timings from TimingCoordinator
        from src.video.timing_coordinator import TimingCoordinator
        adjusted_durations = TimingCoordinator.get_all_durations()
        
        # If types match 1-to-1 great, otherwise just iterate?
        # Because LLM output might order them correctly but name `scene_type` variably.
        # We rely on the PROMPT ID order mostly.
        
        ordered_keys = [
            'hook', 'score', 'top_hitter', 'top_pitcher', 'key_moment', 'standings', 
            'transition', 'prediction_meter', 'factor_1', 'factor_2', 'factor_3', 
            'player_watch_1', 'player_watch_2', 'cta'
        ]
        
        for i, scene in enumerate(scenes):
            if i < len(ordered_keys):
                expected_type = ordered_keys[i]
                # Force strictly correct timing and type to ensure pipeline stability
                scene['duration'] = adjusted_durations.get(expected_type, 4.0)
                # Correction of type if vaguely similar? 
                # e.g. "factor1" -> "factor_1"
                if scene['scene_type'] != expected_type:
                    logger.debug(f"Correcting scene {i+1} type {scene['scene_type']} -> {expected_type}")
                    scene['scene_type'] = expected_type

    def _fallback_script(self, game_data) -> Dict:
        """Return a simple hardcoded script if generation fails."""
        logger.warning("Using fallback viral script")
        # Generate dummy 14 scenes?
        # For brevity, we just do a shortened safe version, 
        # BUT pipeline expects strict structure now? 
        # If pipeline expects 14 scenes, fallback must provide 14 or handle it.
        # We attempt to provide the 14 empty slots.
        
        scenes = []
        ordered_keys = [
            'hook', 'score', 'top_hitter', 'top_pitcher', 'key_moment', 'standings', 
            'transition', 'prediction_meter', 'factor_1', 'factor_2', 'factor_3', 
            'player_watch_1', 'player_watch_2', 'cta'
        ]
        
        from src.video.timing_coordinator import TimingCoordinator
        durs = TimingCoordinator.get_all_durations()
        
        for i, key in enumerate(ordered_keys):
             scenes.append({
                 "scene_id": i+1,
                 "scene_type": key,
                 "duration": durs.get(key, 4.0),
                 "narration": f"Fallback content for {key}.",
                 "visual": {"main_text": key.upper().replace("_", " ")}
             })
             
        return {
            "video_metadata": {"title": "Game Recap", "description": "MLB Recap", "tags": ["MLB"]},
            "scenes": scenes
        }
