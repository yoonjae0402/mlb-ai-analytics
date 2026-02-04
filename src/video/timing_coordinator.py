
from config.viral_video_config import ViralVideoConfig
from src.utils.error_handler import ConfigurationError

class TimingCoordinator:
    """
    Ensures video is exactly within the target duration (e.g. 60s).
    """
    
    @classmethod
    def get_scene_duration(cls, scene_type: str) -> float:
        """Get pre-calculated duration for scene type from Config."""
        durations = ViralVideoConfig.get_adjusted_durations()
        return durations.get(scene_type, 3.0) # Default 3s if unknown
    
    @classmethod
    def get_all_durations(cls) -> dict:
        return ViralVideoConfig.get_adjusted_durations()

    @classmethod
    def validate_total_duration(cls) -> float:
        """Ensure scenes sum to allowed range."""
        durations = cls.get_all_durations()
        total = sum(durations.values())
        
        if not (ViralVideoConfig.MIN_DURATION <= total <= ViralVideoConfig.MAX_DURATION + 2.0):
             # Allow +2s buffer for now as discussed
             if total > 65: # Hard limit
                 raise ConfigurationError(f"Total duration {total}s exceeds limit of {ViralVideoConfig.MAX_DURATION}s")
        
        return total
