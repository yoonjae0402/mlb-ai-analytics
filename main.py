#!/usr/bin/env python3
"""
MLB Video Pipeline - Main Entry Point

Usage:
    python main.py --date 2024-09-15                    # Process all final games for date
    python main.py --date 2024-09-15 --game-id 746545  # Process specific game
    python main.py --watch                              # Watch for live game completions
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import PIL.Image

# Monkey-patch Pillow 10+ to fix MoviePy 'ANTIALIAS' error
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import PipelineOrchestrator
from src.upload import YouTubeUploader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/pipeline.log')
    ]
)

logger = logging.getLogger(__name__)


def watch(args):
    """Run the game watcher for real-time game detection."""
    from src.watcher import GameWatcher

    orchestrator = PipelineOrchestrator()

    def on_game_final(game):
        game_id = game.get("game_id")
        away = game.get("away_name", "Away")
        home = game.get("home_name", "Home")

        logger.info(f"Game final: {away} @ {home} (ID: {game_id})")
        video_path = orchestrator.run_for_game(game_id)

        if video_path and args.upload:
            _upload_video(video_path, orchestrator)

    watcher = GameWatcher(on_game_final=on_game_final)

    logger.info("Starting game watcher... Press Ctrl+C to stop.")
    try:
        watcher.start()
    except KeyboardInterrupt:
        logger.info("Stopping game watcher...")
        watcher.stop()


def _upload_video(video_path, orchestrator):
    """Upload a generated video to YouTube."""
    try:
        uploader = YouTubeUploader()

        script = getattr(orchestrator, "_last_script", None)
        if script:
            metadata = script.get("video_metadata", {})
            title = metadata.get("title", "MLB Game Recap")
            description = metadata.get("description", "")
            tags = metadata.get("tags", ["MLB", "Baseball", "Recap"])
        else:
            title = "MLB Game Recap"
            description = "AI-generated MLB game recap"
            tags = ["MLB", "Baseball", "Recap"]

        video_id = uploader.upload_video(
            file_path=video_path,
            title=title,
            description=description,
            tags=tags,
            privacy_status="private",
        )

        if video_id:
            logger.info(f"Upload successful! Video ID: {video_id}")
        else:
            logger.warning("Upload failed")

    except Exception as e:
        logger.error(f"Upload error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='MLB Video Pipeline - Game recap video generation'
    )

    parser.add_argument(
        '--date',
        type=str,
        help='Date to process (YYYY-MM-DD). Defaults to yesterday.',
        default=None
    )

    parser.add_argument(
        '--game-id',
        type=int,
        help='Specific game ID to process',
        default=None
    )

    parser.add_argument(
        '--upload',
        action='store_true',
        help='Upload video to YouTube after generation'
    )

    parser.add_argument(
        '--watch',
        action='store_true',
        help='Watch for live game completions and auto-generate videos'
    )

    args = parser.parse_args()

    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("outputs/audio").mkdir(parents=True, exist_ok=True)
    Path("outputs/videos").mkdir(parents=True, exist_ok=True)
    Path("outputs/images").mkdir(parents=True, exist_ok=True)

    # Watch mode
    if args.watch:
        watch(args)
        return 0

    # Determine date
    if args.date:
        date = args.date
    else:
        yesterday = datetime.now() - timedelta(days=1)
        date = yesterday.strftime('%Y-%m-%d')

    logger.info("=" * 60)
    logger.info("MLB VIDEO PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Date: {date}")
    if args.game_id:
        logger.info(f"Game ID: {args.game_id}")
    logger.info(f"Upload: {args.upload}")
    logger.info("=" * 60)

    try:
        orchestrator = PipelineOrchestrator()

        video_paths = orchestrator.run_for_date(date, game_id=args.game_id)

        if video_paths:
            logger.info(f"SUCCESS! Generated {len(video_paths)} video(s):")
            for path in video_paths:
                logger.info(f"  - {path}")

            if args.upload:
                for path in video_paths:
                    logger.info(f"Uploading {path}...")
                    _upload_video(path, orchestrator)

            return 0
        else:
            logger.error("FAILED: No videos generated")
            return 1

    except Exception as e:
        logger.error(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
