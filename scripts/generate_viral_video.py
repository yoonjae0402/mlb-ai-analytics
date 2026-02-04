#!/usr/bin/env python3
"""
Viral Video Generation Script
-----------------------------
Usage:
    python scripts/generate_viral_video.py --team Yankees --date 2024-05-15
"""
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import PIL.Image

# Monkey-patch Pillow 10+ to fix MoviePy 'ANTIALIAS' error
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.pipeline import PipelineOrchestrator
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate Viral MLB Video")
    parser.add_argument('--team', type=str, required=True, help="Team name")
    parser.add_argument('--date', type=str, help="Date (YYYY-MM-DD), default yesterday")
    parser.add_argument('--upload', action='store_true', help="Upload to YouTube")
    
    args = parser.parse_args()
    
    # Default to yesterday if date not provided
    if not args.date:
        args.date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
    logger.info(f"Starting Viral Video Generation: {args.team} on {args.date}")
    
    try:
        orchestrator = PipelineOrchestrator()
        video_path = orchestrator.run_viral_for_date(args.date, args.team)
        
        if video_path:
            print(f"OUTPUT_VIDEO_PATH={video_path}") # stdout for bash scripts
            logger.info("Viral generation successful.")
            
            if args.upload:
                # Import here to avoid heavy deps if not uploading
                from src.upload import YouTubeUploader
                uploader = YouTubeUploader()
                # Basic upload logic
                logger.info("Uploading feature not fully configured in this script yet.")
                
            sys.exit(0)
        else:
            logger.error("Viral generation failed.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
