import os
import requests
import logging
from pathlib import Path
from typing import Optional

import sys
import ctypes.util

# Ensure cairo shared library is discoverable across platforms
if not ctypes.util.find_library("cairo"):
    _lib_candidates = (
        ["/opt/homebrew/lib", "/usr/local/lib"] if sys.platform == "darwin"
        else ["/usr/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu"]
    )
    for _path in _lib_candidates:
        if os.path.isdir(_path):
            os.environ["DYLD_LIBRARY_PATH"] = _path
            os.environ["LD_LIBRARY_PATH"] = _path
            break

import cairosvg

logger = logging.getLogger(__name__)

class AssetManager:
    """
    Manages fetching and caching of visual assets (logos, headshots).
    """
    
    def __init__(self, cache_dir: str = "data/assets"):
        self.cache_dir = Path(cache_dir)
        self.logos_dir = self.cache_dir / "logos"
        self.headshots_dir = self.cache_dir / "headshots"
        self.backgrounds_dir = self.cache_dir / "backgrounds"
        
        for d in [self.logos_dir, self.headshots_dir, self.backgrounds_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
    def fetch_team_logo(self, team_id: int) -> Optional[str]:
        """
        Fetch team logo from MLB static CDN and convert SVG to PNG.
        """
        png_path = self.logos_dir / f"{team_id}.png"

        if png_path.exists():
            return str(png_path)

        url = f"https://www.mlbstatic.com/team-logos/team-cap-on-light/{team_id}.svg"
        svg_path = self.logos_dir / f"{team_id}.svg"
        result = self._download_file(url, svg_path)
        if not result:
            return None

        try:
            cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=200, output_height=200)
            svg_path.unlink(missing_ok=True)
            return str(png_path)
        except Exception as e:
            logger.error(f"Failed to convert SVG to PNG: {e}")
            return None

    def fetch_player_headshot(self, player_id: int) -> Optional[str]:
        """
        Fetch player headshot.
        """
        filename = f"{player_id}.png"
        output_path = self.headshots_dir / filename
        
        if output_path.exists():
            return str(output_path)
            
        # Standard MLB headshot URL pattern
        url = f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_213,q_auto:best/v1/people/{player_id}/headshot/67/current"
        return self._download_file(url, output_path)

    def get_background_video(self, video_type: str = "generic") -> Optional[str]:
        """
        Get background video (assumes manual placement for now).
        """
        # In a real app, this might download from an S3 bucket or similar
        # For now, we check if it exists locally
        path = self.backgrounds_dir / f"{video_type}.mp4"
        if path.exists():
            return str(path)
        
        logger.warning(f"Background video {video_type}.mp4 not found in {self.backgrounds_dir}")
        return None

    def _download_file(self, url: str, output_path: Path) -> Optional[str]:
        try:
            logger.info(f"Downloading asset from {url}")
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                return str(output_path)
            else:
                logger.warning(f"Failed to download asset: {url} (Status: {response.status_code})")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading asset: {e}")
            return None
