
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
    Generates game-neutral recap scripts for MLB games.
    Output is strictly structured JSON for the ViralVideoEngine.
    Focuses on game recap without team-specific predictions.
    """

    VIRAL_SYSTEM_PROMPT = """
You are a professional MLB sports broadcaster creating a 45-55 second viral video script.
This is a GAME RECAP - show BOTH teams fairly, not biased toward one team.
Output STRICT JSON.

CRITICAL RULES:
1. Use ONLY the team names, player names, scores, and stats provided in the data.
2. NEVER use placeholder text like "away team", "home team", "road team", "winning team", "key player".
3. Every sentence must reference ACTUAL team names and player names from the data.
4. State the EXACT final score with both team names.
5. Name SPECIFIC players with their EXACT stats from the data.
6. Include which team each player plays for when mentioning them.

STRUCTURE (8 scenes):
1.  HOOK (3s): Dramatic game result. Example: "The Yankees take down the Red Sox in a thriller!"
2.  SCORE (4s): State BOTH team names and the EXACT score. Example: "Final score: Yankees 5, Red Sox 3."
3.  TOP_HITTER (5s): Name the player, THEIR TEAM, and stats. Example: "Aaron Judge of the Yankees went 2 for 4 with 2 home runs."
4.  TOP_PITCHER (5s): Name the pitcher, THEIR TEAM, and stats. Example: "Gerrit Cole of the Yankees threw 7 innings with 10 strikeouts."
5.  KEY_MOMENT (5s): Describe the specific play that changed the game.
6.  STANDINGS (5s): Impact for BOTH teams. Example: "The Yankees climb to first in the AL East while the Red Sox drop to third."
7.  RECAP (4s): Quick summary of what this game means.
8.  CTA (3s): "Follow for daily MLB recaps and analysis."

Total Duration: ~34-40s (will be padded by audio).
Total Word Count: 180-220 Words.

Output JSON Format:
{
  "video_metadata": {
    "title": "Descriptive title with BOTH team names (max 60 chars)",
    "description": "Short description with hashtags",
    "tags": ["mlb", "baseball", "team1", "team2"]
  },
  "scenes": [
    {
      "scene_id": 1,
      "scene_type": "hook",
      "duration": 3.0,
      "narration": "The Yankees edge the Red Sox in a tight one!",
      "tts_pace": "fast",
      "visual": {
        "main_text": "YANKEES 5",
        "sub_text": "RED SOX 3",
        "image_prompt": null,
        "player_to_show": null,
        "stat_to_show": null
      },
      "sound_effect": "boom"
    }
  ]
}

Visual Rules:
- 'main_text' should show the score or key info (2-4 words max).
- 'player_to_show' must EXACTLY match the player name from the data.
- 'stat_to_show' must be the real stat line from the data.
- For player scenes, include image_prompt for AI generation.
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
        prediction: Dict[str, Any] = None,
        next_game_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate the game-neutral recap script JSON."""
        if not self.client:
            logger.error("Client not initialized")
            return self._fallback_script(game_data)

        prompt = self._create_prompt(game_data)

        try:
            logger.info("Calling Gemini for game recap script...")
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

            script = json.loads(response.text)
            self._validate_and_fix_script(script)
            self._validate_script_accuracy(script, game_data)
            logger.info(f"Generated script with {len(script.get('scenes', []))} scenes")
            return script

        except Exception as e:
            logger.error(f"Script generation failed: {e}")
            return self._fallback_script(game_data)

    def _create_prompt(self, game: Dict) -> str:
        """Construct the user prompt with ALL real data explicitly labeled."""

        home_team = game.get("home_team", "Unknown")
        away_team = game.get("away_team", "Unknown")
        home_score = game.get("home_score", 0)
        away_score = game.get("away_score", 0)
        winner = game.get("winner", home_team if home_score > away_score else away_team)
        loser = game.get("loser", away_team if winner == home_team else home_team)

        top_hitter = game.get('top_hitter', {})
        top_pitcher = game.get('top_pitcher', {})
        key_moment = game.get('key_moment', {})
        standings_impact = game.get('standings_impact', 'Important game for the standings')

        prompt = f"""
GENERATE A GAME-NEUTRAL RECAP SCRIPT USING ONLY THE DATA BELOW.
USE THESE EXACT NAMES AND NUMBERS. DO NOT CHANGE THEM.
SHOW BOTH TEAMS FAIRLY - this is not a highlight for one team.

=== GAME RESULT ===
Home Team: {home_team}
Away Team: {away_team}
Final Score: {away_team} {away_score}, {home_team} {home_score}
Winner: {winner}
Loser: {loser}
Standings Impact: {standings_impact}

=== TOP HITTER ===
Name: {top_hitter.get('name', 'Unknown')}
Team: {top_hitter.get('team', 'Unknown')}
Stats: {top_hitter.get('stats', 'N/A')}
Impact: {top_hitter.get('impact', 'Key performance')}

=== TOP PITCHER ===
Name: {top_pitcher.get('name', 'Unknown')}
Team: {top_pitcher.get('team', 'Unknown')}
Stats: {top_pitcher.get('stats', 'N/A')}
Impact: {top_pitcher.get('impact', 'Strong outing')}

=== KEY MOMENT ===
Description: {key_moment.get('description', 'Decisive play')}
Inning: {key_moment.get('inning', 'Late innings')}

REMEMBER:
- Use ONLY the names and numbers above
- Mention which TEAM each player is on
- Show impact for BOTH teams in standings
"""
        return prompt

    def _validate_and_fix_script(self, script: Dict):
        """Ensure script matches 8-scene structure and timings."""
        scenes = script.get("scenes", [])

        if len(scenes) != 8:
            logger.warning(f"Generated {len(scenes)} scenes, expected 8.")

        ordered_keys = [
            'hook', 'score', 'top_hitter', 'top_pitcher',
            'key_moment', 'standings', 'recap', 'cta'
        ]

        durations = {
            'hook': 3.0, 'score': 4.0, 'top_hitter': 5.0, 'top_pitcher': 5.0,
            'key_moment': 5.0, 'standings': 5.0, 'recap': 4.0, 'cta': 3.0
        }

        for i, scene in enumerate(scenes):
            if i < len(ordered_keys):
                expected_type = ordered_keys[i]
                scene['duration'] = durations.get(expected_type, 4.0)
                if scene.get('scene_type') != expected_type:
                    logger.debug(f"Correcting scene {i+1} type {scene.get('scene_type')} -> {expected_type}")
                    scene['scene_type'] = expected_type

    def _validate_script_accuracy(self, script: Dict, game_data: Dict):
        """Verify the generated script uses actual team and player names."""
        scenes = script.get("scenes", [])
        all_narration = " ".join(s.get("narration", "") for s in scenes).lower()

        # Check for forbidden placeholder text
        forbidden = [
            'away team', 'home team', 'road team',
            'winning team', 'losing team',
            'key player', 'top player',
        ]

        for phrase in forbidden:
            if phrase in all_narration:
                logger.warning(f"Script contains placeholder '{phrase}' - should use actual names")

        # Check team names are present
        home_team = game_data.get('home_team', '')
        away_team = game_data.get('away_team', '')

        if home_team and home_team.lower() not in all_narration:
            short_name = home_team.split()[-1].lower()
            if short_name not in all_narration:
                logger.warning(f"Script missing home team name: {home_team}")

        if away_team and away_team.lower() not in all_narration:
            short_name = away_team.split()[-1].lower()
            if short_name not in all_narration:
                logger.warning(f"Script missing away team name: {away_team}")

        logger.info("Script accuracy validation complete")

    def _fallback_script(self, game_data: Dict) -> Dict:
        """Return a game-neutral script when Gemini fails."""
        logger.warning("Using fallback script with real game data")

        home_team = game_data.get('home_team', 'Home Team')
        away_team = game_data.get('away_team', 'Away Team')
        home_score = game_data.get('home_score', 0)
        away_score = game_data.get('away_score', 0)
        winner = game_data.get('winner', home_team if home_score > away_score else away_team)
        loser = game_data.get('loser', away_team if winner == home_team else home_team)

        top_hitter = game_data.get('top_hitter', {})
        top_pitcher = game_data.get('top_pitcher', {})
        key_moment = game_data.get('key_moment', {})
        standings_impact = game_data.get('standings_impact', 'Big implications for the standings')

        hitter_name = top_hitter.get('name', 'The top hitter')
        hitter_team = top_hitter.get('team', winner)
        hitter_stats = top_hitter.get('stats', 'had a great game')

        pitcher_name = top_pitcher.get('name', 'The starter')
        pitcher_team = top_pitcher.get('team', winner)
        pitcher_stats = top_pitcher.get('stats', 'pitched well')

        # Get short team names for display
        winner_short = winner.split()[-1]
        loser_short = loser.split()[-1]

        scenes = [
            {
                "scene_id": 1, "scene_type": "hook", "duration": 3.0,
                "narration": f"The {winner} take down the {loser}!",
                "tts_pace": "fast",
                "visual": {
                    "main_text": f"{winner_short.upper()} {home_score if winner == home_team else away_score}",
                    "sub_text": f"{loser_short.upper()} {away_score if winner == home_team else home_score}",
                    "image_prompt": f"{winner} celebrating victory, baseball stadium, dramatic lighting",
                    "player_to_show": None, "stat_to_show": None
                },
                "sound_effect": "boom"
            },
            {
                "scene_id": 2, "scene_type": "score", "duration": 4.0,
                "narration": f"Final score: {away_team} {away_score}, {home_team} {home_score}.",
                "tts_pace": "normal",
                "visual": {
                    "main_text": "FINAL SCORE",
                    "sub_text": f"{away_score}-{home_score}",
                    "image_prompt": None, "player_to_show": None, "stat_to_show": None
                },
                "sound_effect": "none"
            },
            {
                "scene_id": 3, "scene_type": "top_hitter", "duration": 5.0,
                "narration": f"{hitter_name} of the {hitter_team} led the offense. {hitter_stats}.",
                "tts_pace": "normal",
                "visual": {
                    "main_text": hitter_name.split()[-1].upper(),
                    "sub_text": hitter_stats,
                    "image_prompt": f"{hitter_name} hitting, {hitter_team} uniform, dramatic stadium lighting",
                    "player_to_show": hitter_name, "stat_to_show": hitter_stats
                },
                "sound_effect": "none"
            },
            {
                "scene_id": 4, "scene_type": "top_pitcher", "duration": 5.0,
                "narration": f"{pitcher_name} of the {pitcher_team} was dominant on the mound. {pitcher_stats}.",
                "tts_pace": "normal",
                "visual": {
                    "main_text": pitcher_name.split()[-1].upper(),
                    "sub_text": pitcher_stats,
                    "image_prompt": f"{pitcher_name} pitching, {pitcher_team} uniform, intense expression",
                    "player_to_show": pitcher_name, "stat_to_show": pitcher_stats
                },
                "sound_effect": "none"
            },
            {
                "scene_id": 5, "scene_type": "key_moment", "duration": 5.0,
                "narration": f"The turning point: {key_moment.get('description', f'{winner} sealed it with a clutch play.')}",
                "tts_pace": "fast",
                "visual": {
                    "main_text": "TURNING POINT",
                    "sub_text": key_moment.get('inning', ''),
                    "image_prompt": "dramatic baseball moment, crowd cheering, stadium lights",
                    "player_to_show": None, "stat_to_show": None
                },
                "sound_effect": "boom"
            },
            {
                "scene_id": 6, "scene_type": "standings", "duration": 5.0,
                "narration": f"This win is huge for the {winner}. {standings_impact}. The {loser} will look to bounce back.",
                "tts_pace": "normal",
                "visual": {
                    "main_text": "STANDINGS IMPACT",
                    "sub_text": f"{winner_short} W / {loser_short} L",
                    "image_prompt": f"{winner} team celebration, playoff implications",
                    "player_to_show": None, "stat_to_show": None
                },
                "sound_effect": "none"
            },
            {
                "scene_id": 7, "scene_type": "recap", "duration": 4.0,
                "narration": f"A statement game for the {winner} as they continue their push.",
                "tts_pace": "normal",
                "visual": {
                    "main_text": f"{winner_short.upper()} WIN",
                    "sub_text": f"{max(home_score, away_score)}-{min(home_score, away_score)} FINAL",
                    "image_prompt": None, "player_to_show": None, "stat_to_show": None
                },
                "sound_effect": "none"
            },
            {
                "scene_id": 8, "scene_type": "cta", "duration": 3.0,
                "narration": "Follow for daily MLB recaps and analysis.",
                "tts_pace": "normal",
                "visual": {
                    "main_text": "FOLLOW",
                    "sub_text": "FOR DAILY MLB RECAPS",
                    "image_prompt": None, "player_to_show": None, "stat_to_show": None
                },
                "sound_effect": "none"
            },
        ]

        return {
            "video_metadata": {
                "title": f"{away_team} vs {home_team}: {away_score}-{home_score} Final",
                "description": f"MLB Recap: {away_team} at {home_team} #MLB #baseball #{winner_short} #{loser_short}",
                "tags": ["MLB", "baseball", winner_short.lower(), loser_short.lower(), "recap"]
            },
            "scenes": scenes
        }
