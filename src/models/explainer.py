import torch
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Feature indices (must match predictor_data.py)
FEATURE_MAP = {
    0: "Batting Average",
    1: "OBP",
    2: "Slugging",
    3: "Home Runs",
    4: "RBIs",
    5: "Hits",
    6: "At Bats",
    7: "Walks",
    8: "Strikeouts",
    9: "Home Game",
    10: "Season Break"
}

class PredictionExplainer:
    """
    Generates human-readable explanations for model predictions.
    
    Current MVP Strategy: Heuristic Feature Analysis
    - Analyzes the input sequence to find 'standout' stats that align with the prediction.
    - Future: Integrated Gradients (Captum)
    """
    
    def __init__(self):
        pass
        
    def explain_prediction(
        self, 
        input_sequence: np.ndarray, 
        prediction_class: int, 
        confidence: float
    ) -> Dict[str, Any]:
        """
        Generate explanation for a single prediction.
        
        Args:
            input_sequence: (seq_len, 11) feature matrix
            prediction_class: 0 (Below), 1 (Average), 2 (Above)
            confidence: Probability of the predicted class
            
        Returns:
            {
                "prediction": "Above Average",
                "confidence": "72%",
                "reasons": ["Reason 1", "Reason 2", "Reason 3"]
            }
        """
        class_names = {0: "Below Average", 1: "Average", 2: "Above Average"}
        readable_class = class_names.get(prediction_class, "Unknown")
        
        # Calculate recent averages (last 5 games of sequence for more relevance)
        recent_seq = input_sequence[-5:] if len(input_sequence) >= 5 else input_sequence
        avg_stats = np.mean(recent_seq, axis=0)
        
        reasons = []
        
        if prediction_class == 2: # Above Average
            reasons = self._explain_above_average(avg_stats)
        elif prediction_class == 0: # Below Average
            reasons = self._explain_below_average(avg_stats)
        else: # Average
            reasons = self._explain_average(avg_stats)
            
        # Fallback if no specific reasons found
        if not reasons:
            reasons = ["Consistent recent play", "Standard performance metrics", "Neutral matchup factors"]
            
        return {
            "prediction": readable_class,
            "confidence": f"{confidence:.1%}",
            "reasons": reasons[:3]
        }

    def _explain_above_average(self, stats: np.ndarray) -> List[str]:
        """Generate reasons for high performance prediction."""
        reasons = []
        
        # Check standard metrics (thresholds are illustrative heuristics)
        if stats[0] > 0.280:
            reasons.append(f"Strong recent form: {stats[0]:.3f} Batting Average")
        if stats[2] > 0.450:
            reasons.append(f"High power output: {stats[2]:.3f} Slugging Pct")
        if stats[1] > 0.350:
            reasons.append(f"Getting on base frequently: {stats[1]:.3f} OBP")
        if stats[3] > 0.2: # Averaging > 0.2 HR per game recently
            reasons.append("Recent power surge (multiple HRs)")
        if stats[9] > 0.5: # Mostly home games
            reasons.append("Home field advantage")
        if stats[8] < 0.8: # Low strikeouts
            reasons.append("Excellent contact rate (low strikeouts)")
            
        return reasons

    def _explain_below_average(self, stats: np.ndarray) -> List[str]:
        """Generate reasons for low performance prediction."""
        reasons = []
        
        if stats[0] < 0.220:
            reasons.append(f"Cold streak: {stats[0]:.3f} recent Batting Average")
        if stats[8] > 1.2:
            reasons.append("High strikeout rate recently")
        if stats[1] < 0.280:
            reasons.append(f"Struggling to reach base: {stats[1]:.3f} OBP")
        if stats[9] < 0.5:
             reasons.append("Playing away from home")
             
        return reasons

    def _explain_average(self, stats: np.ndarray) -> List[str]:
        """Generate reasons for average performance."""
        reasons = []
        reasons.append(f"Steady performance ({stats[0]:.3f} BA)")
        if 0.4 < stats[9] < 0.6:
            reasons.append("Balanced schedule")
        return reasons
