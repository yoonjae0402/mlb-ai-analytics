
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class FallbackHandler:
    """
    Provides graceful degradation when primary data sources fail.
    """
    
    @staticmethod
    def get_generic_player_data(role: str = "Player") -> Dict[str, Any]:
        """Returns safe default player data."""
        return {
            "name": f"Key {role}",
            "stats": "N/A",
            "impact": "Key contribution to the game",
            "photo_url": None,
            "id": None
        }

    @staticmethod
    def get_fallback_prediction_data(home_team: str, away_team: str) -> Dict[str, Any]:
        """Returns safe default prediction structure."""
        return {
            "win_probability": 0.50,
            "confidence": "Low",
            "factors": [
                {"factor": "Market Size", "detail": f"{home_team} has strong home support", "impact": "Neutral"},
                {"factor": "Recent Form", "detail": "Both teams competitive", "impact": "Neutral"},
                {"factor": "History", "detail": "Historic rivalry match", "impact": "Neutral"}
            ],
            "players_to_watch": [
                {"name": f"{home_team} Star", "prediction": "Key hit expected", "reasoning": "Team leader"},
                {"name": f"{away_team} Star", "prediction": "Solid performance", "reasoning": "Consistent form"}
            ]
        }
