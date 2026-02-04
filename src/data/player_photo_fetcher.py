
import os
import requests
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class PlayerPhotoFetcher:
    """
    Fetches official MLB player headshots and caches them locally.
    """
    
    CACHE_DIR = Path("assets/images/players")
    # Modern MLB Headshot URL Template
    # {id} is player_id
    URL_TEMPLATE = "https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_426,q_auto:best/v1/people/{player_id}/headshot/67/current"
    
    def __init__(self):
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.placeholder_path = Path("assets/images/defaults/player_placeholder.png")
        # Enhance: Ensure placeholder exists or create a generated one?
        
    def fetch_player_photo(self, player_id: int, player_name: str = "") -> Optional[str]:
        """
        Download and cache player headshot.
        Returns absolute path to the image.
        """
        if not player_id:
            return str(self.placeholder_path) if self.placeholder_path.exists() else None
            
        file_path = self.CACHE_DIR / f"{player_id}.png"
        
        # Return cached if exists
        if file_path.exists():
            return str(file_path.absolute())
            
        # Fetch
        url = self.URL_TEMPLATE.format(player_id=player_id)
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"Cached photo for {player_name} ({player_id})")
                return str(file_path.absolute())
            else:
                logger.warning(f"Failed to fetch photo for {player_id}: Status {response.status_code}")
                return str(self.placeholder_path) if self.placeholder_path.exists() else None
        except Exception as e:
            logger.error(f"Error fetching player photo: {e}")
            return str(self.placeholder_path) if self.placeholder_path.exists() else None

    def ensure_placeholder(self):
        """Create a basic placeholder if missing."""
        if not self.placeholder_path.exists():
            self.placeholder_path.parent.mkdir(parents=True, exist_ok=True)
            # Create a simple grey square using PIL
            from PIL import Image
            img = Image.new('RGB', (426, 426), color='grey')
            img.save(self.placeholder_path)
