"""
Alert service for win probability swings.

Sends notifications via Slack or Discord webhooks when
win probability changes exceed a configurable threshold.
"""

import json
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def check_and_alert(
    game: dict,
    threshold: float = 0.15,
    slack_webhook_url: Optional[str] = None,
    discord_webhook_url: Optional[str] = None,
) -> bool:
    """
    Check if the latest WP swing exceeds threshold and send alerts.

    Returns True if an alert was sent.
    """
    wp_history = game.get("wp_history", [])
    if len(wp_history) < 2:
        return False

    swing = abs(wp_history[-1] - wp_history[-2])
    if swing < threshold:
        return False

    matchup = f"{game['away_abbrev']} @ {game['home_abbrev']}"
    direction = "toward" if wp_history[-1] > wp_history[-2] else "away from"
    message = (
        f"Win Probability Alert: {matchup}\n"
        f"WP swung {swing:.0%} {direction} {game['home_abbrev']}\n"
        f"Score: {game['away_score']}-{game['home_score']} | "
        f"{game['half']} {game['inning']}\n"
        f"Current Home WP: {wp_history[-1]:.0%}"
    )

    sent = False

    if slack_webhook_url:
        sent |= _send_slack(slack_webhook_url, message)

    if discord_webhook_url:
        sent |= _send_discord(discord_webhook_url, message)

    if not slack_webhook_url and not discord_webhook_url:
        logger.info(f"Alert (no webhook configured): {message}")

    return sent


def _send_slack(webhook_url: str, message: str) -> bool:
    """Send a Slack webhook message."""
    try:
        resp = requests.post(
            webhook_url,
            json={"text": message},
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.warning(f"Slack alert failed: {e}")
        return False


def _send_discord(webhook_url: str, message: str) -> bool:
    """Send a Discord webhook message."""
    try:
        resp = requests.post(
            webhook_url,
            json={"content": message},
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.warning(f"Discord alert failed: {e}")
        return False
