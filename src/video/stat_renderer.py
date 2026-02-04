
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import math

from src.video.asset_manager import AssetManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

class StatRenderer:
    """
    Renders ESPN-style static graphics frames using Pillow.
    THEME: Modern Sports Analytics (Dark Navy + Neon Green).
    """
    
    WIDTH = 1080
    HEIGHT = 1920
    
    # Modern Sports Analytics Palette
    COLOR_BG = (10, 14, 39, 255)       # #0A0E27 Dark Navy
    COLOR_FG = (20, 24, 50, 200)       # Semi-transparent overlay
    COLOR_ACCENT_1 = (0, 255, 136, 255) # #00FF88 Neon Green (Positive)
    COLOR_ACCENT_2 = (255, 0, 85, 255)  # #FF0055 Neon Red/Pink (Alert)
    COLOR_TEXT_MAIN = (255, 255, 255, 255)
    COLOR_TEXT_SUB = (200, 200, 200, 255)
    
    def __init__(self, asset_manager: AssetManager):
        self.asset_manager = asset_manager
        # Fonts - Attempt to load impacting fonts, fallback to default
        self.font_header = self._load_font("Arial Black", 90)
        self.font_sub = self._load_font("Arial Bold", 45)
        self.font_body = self._load_font("Arial", 35)
        self.font_stat_huge = self._load_font("Arial Black", 150)
        
    def _load_font(self, name: str, size: int) -> ImageFont.FreeTypeFont:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            try:
                # Common Mac fonts
                if "Black" in name: return ImageFont.truetype("Arial Black.ttf", size)
                if "Bold" in name: return ImageFont.truetype("Arial Bold.ttf", size)
                return ImageFont.truetype("Arial.ttf", size)
            except:
                logger.warning(f"Could not load font {name}, using default.")
                return ImageFont.load_default()

    def render_scene_frame(
        self, 
        scene: Dict[str, Any], 
        game_data: Dict[str, Any],
        prediction_data: Optional[Dict[str, Any]] = None,
        background_path: Optional[str] = None
    ) -> Image.Image:
        """
        Render a single high-quality frame.
        Strategy: Background (NanoBanana) -> Overlays -> Text/Stats (Pillow)
        """
        scene_type = scene.get("scene_type", "hook")
        
        # 1. Base Layer: AI Background or Gradient Fallback
        if background_path and Path(background_path).exists():
            img = Image.open(background_path).convert("RGBA").resize((self.WIDTH, self.HEIGHT))
            # Darken AI background slightly for text readability
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 100))
            img = Image.alpha_composite(img, overlay)
        else:
            img = self._create_tech_bg()

        # 2. Render Layout
        # 2. Render Layout
        if scene_type == "hook":
            return self.render_hook(img, scene, game_data)
        elif scene_type == "scoreboard":
            return self.render_scoreboard(img, game_data)
        elif scene_type in ["top_hitter", "top_pitcher", "player_watch_1", "player_watch_2", "player_spotlight"]:
            return self.render_player_spotlight(img, scene, game_data)
        elif scene_type == "key_moment":
            return self.render_key_moment(img, scene, game_data)
        elif scene_type == "standings":
            return self.render_standings(img, scene, game_data)
        elif scene_type == "prediction_meter":
            return self.render_prediction_meter(img, prediction_data)
        elif scene_type in ["factor_1", "factor_2", "factor_3"]:
            # Maybe a simplified factor card? For now reuse prediction meter or generic
            # Let's make a specific "factor_card" view if we want, or just generic text.
            # Using generic text for factors is okay, or reuse prediction meter's factor panel style.
            return self.render_prediction_meter(img, prediction_data) 
        elif scene_type == "cta":
            return self.render_cta(img, scene)
        else:
            return self._render_generic_text(img, scene)

    def render_hook(self, img: Image.Image, scene: Dict, game_data: Dict) -> Image.Image:
        draw = ImageDraw.Draw(img)
        visual = scene.get("visual", {})
        
        # Giant Impact Text
        main_text = visual.get("main_text", "HOOK TEXT").upper()
        self._draw_centered_text(img, main_text, self.font_stat_huge, y_offset=600, color=self.COLOR_ACCENT_1, stroke_width=4, stroke_fill="black")
        
        # Subtext
        sub_text = visual.get("sub_text", "")
        if sub_text:
             self._draw_centered_text(img, sub_text, self.font_header, y_offset=900, color="white", stroke_width=2, stroke_fill="black")

        return img

    def render_scoreboard(self, img: Image.Image, game_data: Dict) -> Image.Image:
        draw = ImageDraw.Draw(img)
        
        # Glassmorphism Container
        box_rect = (100, 500, 980, 1100)
        self._draw_glass_panel(img, box_rect)
        
        # Data
        away_score = game_data.get('away_score', 0)
        home_score = game_data.get('home_score', 0)
        away_id = game_data.get('away_team_id')
        home_id = game_data.get('home_team_id')
        
        # Logos & Text
        y_away = 600
        y_home = 850
        
        self._draw_team_row(img, away_id, game_data.get('away_team'), away_score, y_away)
        self._draw_team_row(img, home_id, game_data.get('home_team'), home_score, y_home)
        
        # Final Label
        self._draw_centered_text(img, "FINAL SCORE", self.font_sub, y_offset=420, color=self.COLOR_ACCENT_1)
        
        return img

    def render_player_spotlight(self, img: Image.Image, scene: Dict, game_data: Dict) -> Image.Image:
        """
        Layout: Player Image (Cutout ideally) -> Stat Block overlaid.
        """
        draw = ImageDraw.Draw(img)
        visual = scene.get("visual", {})
        player_name = visual.get("player_to_show", "") or "Player Spotlight"
        
        # 1. Try to fetch player headshot/action shot (Placeholder logic for real fetch)
        # Assuming asset_manager has basic logic, we heavily rely on valid player_id logic which might be missing in MVP scripts.
        # Fallback: Just text or generic silhouette if name not found.
        
        # Draw Tech Card at bottom
        card_rect = (50, 1000, 1030, 1600)
        self._draw_glass_panel(img, card_rect)
        
        # Name
        draw.text((100, 1050), str(player_name).upper(), font=self.font_header, fill=self.COLOR_ACCENT_1)
        
        # Stats
        stats = str(visual.get("stat_to_show", "")).split(",")
        y_stat = 1200
        for stat in stats:
            draw.text((100, y_stat), stat.strip(), font=self.font_sub, fill="white")
            y_stat += 80
            
        return img

    def render_key_moment(self, img: Image.Image, scene: Dict, game_data: Dict) -> Image.Image:
        """
        Visual for the "Key Moment" scene.
        """
        draw = ImageDraw.Draw(img)
        visual = scene.get("visual", {})
        
        # Central Box
        self._draw_glass_panel(img, (100, 700, 980, 1200))
        
        # Title
        self._draw_centered_text(img, "TURNING POINT", self.font_header, y_offset=600, color=self.COLOR_ACCENT_2)
        
        # Description
        desc = game_data.get('key_moment', {}).get('description', 'Huge Play')
        self._draw_text_with_fit(img, desc, self.font_sub, (150, 750, 930, 1150))
        
        return img

    def render_standings(self, img: Image.Image, scene: Dict, game_data: Dict) -> Image.Image:
        """
        Visual for Standings/Impact scene.
        """
        draw = ImageDraw.Draw(img)
        
        self._draw_centered_text(img, "PLAYOFF IMPACT", self.font_header, y_offset=400, color=self.COLOR_ACCENT_1)
        
        # Mock Standings Table
        self._draw_glass_panel(img, (100, 600, 980, 1400))
        
        y_start = 700
        draw.text((150, y_start), f"1. {game_data.get('home_team')}", font=self.font_sub, fill="white")
        draw.text((800, y_start), "W", font=self.font_sub, fill=self.COLOR_ACCENT_1)
        
        draw.text((150, y_start + 100), f"2. {game_data.get('away_team')}", font=self.font_sub, fill="gray")
        draw.text((800, y_start + 100), "--", font=self.font_sub, fill="gray")
        
        return img

    def render_prediction_meter(self, img: Image.Image, prediction: Dict) -> Image.Image:
        draw = ImageDraw.Draw(img)
        prob = float(prediction.get("win_probability", 0.5))
        
        # Draw Circular Meter
        center_x, center_y = 540, 800
        radius = 350
        bbox = (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
        
        # Background Arc
        draw.arc(bbox, 135, 405, fill=(50, 50, 50), width=60)
        
        # Fill Arc (Map 0.0-1.0 to 135-405 deg)
        end_angle = 135 + (prob * 270)
        color = self.COLOR_ACCENT_1 if prob > 0.5 else self.COLOR_ACCENT_2
        draw.arc(bbox, 135, end_angle, fill=color, width=60)
        
        # Text
        pct_text = f"{int(prob*100)}%"
        self._draw_centered_text(img, pct_text, self.font_stat_huge, y_offset=center_y - 100, color="white")
        self._draw_centered_text(img, "WIN PROBABILITY", self.font_sub, y_offset=center_y + 100, color=self.COLOR_TEXT_SUB)
        
        # Factors Panel
        self._draw_glass_panel(img, (100, 1300, 980, 1700))
        factors = prediction.get("factors", [])
        y_fac = 1350
        draw.text((150, y_fac-50), "KEY FACTORS", font=self.font_sub, fill=self.COLOR_ACCENT_1)
        
        for f in factors[:3]:
            txt = f"• {f.get('factor', '')}: {f.get('impact', '')}"
            draw.text((150, y_fac), txt, font=self.font_body, fill="white")
            y_fac += 70

        return img

    def render_faceoff(self, img: Image.Image, scene: Dict, game_data: Dict, prediction: Dict) -> Image.Image:
        # Split screen effect for matchups
        draw = ImageDraw.Draw(img)
        
        # Divider
        draw.line([(540, 300), (540, 1600)], fill="white", width=5)
        
        # Left Side (Home/Favorite) - Right Side (Away/Underdog)
        # Just drawing text for now
        next_game = scene.get("visual", {})
        
        self._draw_centered_text(img, "NEXT MATCHUP", self.font_header, y_offset=200, color=self.COLOR_ACCENT_1)
        
        return img

    def render_cta(self, img: Image.Image, scene: Dict) -> Image.Image:
        draw = ImageDraw.Draw(img)
        self._draw_centered_text(img, "FOLLOW FOR DAILY", self.font_header, y_offset=700, color="white")
        self._draw_centered_text(img, "AI PREDICTIONS", self.font_stat_huge, y_offset=850, color=self.COLOR_ACCENT_1)
        return img

    def _render_generic_text(self, img: Image.Image, scene: Dict) -> Image.Image:
        visual = scene.get("visual", {})
        txt = visual.get("main_text", "")
        self._draw_centered_text(img, txt, self.font_header, y_offset=800)
        return img

    # --- Helpers ---
    
    def _create_tech_bg(self) -> Image.Image:
        # Fallback gradient
        img = Image.new("RGBA", (self.WIDTH, self.HEIGHT), self.COLOR_BG)
        draw = ImageDraw.Draw(img)
        # Grid lines
        for y in range(0, self.HEIGHT, 100):
            draw.line([(0, y), (self.WIDTH, y)], fill=(255, 255, 255, 10), width=1)
        return img


    def _draw_glass_panel(self, img: Image.Image, rect: Tuple[int, int, int, int]):
        """Draws a semi-transparent 'glass' panel with border."""
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Fill
        draw.rounded_rectangle(rect, radius=30, fill=self.COLOR_FG)
        # Border
        draw.rounded_rectangle(rect, radius=30, outline=self.COLOR_ACCENT_1, width=3)
        
        img.alpha_composite(overlay)

    def _draw_centered_text(self, img, text: str, font, y_offset=None, color="white", stroke_width=0, stroke_fill=None):
        """Draw text centered horizontally."""
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (self.WIDTH - w) / 2
        y = y_offset if y_offset else (self.HEIGHT - h) / 2
        draw.text((x, y), text, font=font, fill=color, stroke_width=stroke_width, stroke_fill=stroke_fill)


    def _draw_text_with_fit(
        self, 
        img: Image.Image, 
        text: str, 
        font: ImageFont.FreeTypeFont, 
        box_rect: Tuple[int, int, int, int], 
        color: Any = "white",
        align: str = "center",
        stroke_width: int = 0,
        stroke_fill: Any = None
    ):
        """
        Draws text within a bounding box, scaling down if necessary.
        """
        draw = ImageDraw.Draw(img)
        x1, y1, x2, y2 = box_rect
        max_w = x2 - x1
        max_h = y2 - y1
        
        current_font = font
        
        # Binary search or iterative shrink to find best fit
        # Simple iterative shrink for MVP
        for scale in range(100, 10, -5):
            # We can't easily resize a font object without reloading, 
            # so we assume 'font' was loaded with a size and we might need to reload or just guess
            # Pillow doesn't support dynamic resizing of loaded font object easily. 
            # We will just try to measure and if too big, we split lines or return error?
            
            # Better approach for MVP: Use textwrap if long, or just shrink font size?
            # Since loading fonts excessively is slow, let's try fitting width first.
            
            bbox = draw.textbbox((0, 0), text, font=current_font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            
            if w <= max_w and h <= max_h:
                break
            
            # Shrink font (naive approach: Requires reloading font or using a different size)
            # Since we can't reload easily here without path, we assume 'font' is fixed for now
            # and we just wrap text if possible.
            
            # TODO: Robust font resizing requires font path availability.
            # For now, we will just wrap text.
            pass
            
        # Wrap text
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = draw.textbbox((0,0), test_line, font=current_font)
            if (bbox[2] - bbox[0]) <= max_w:
                current_line.append(word)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
        lines.append(" ".join(current_line))
        
        # Draw lines centered vertically in box
        total_h = sum([draw.textbbox((0,0), line, font=current_font)[3] - draw.textbbox((0,0), line, font=current_font)[1] for line in lines])
        # Add simpler line height calc
        line_height = draw.textbbox((0,0), "Ag", font=current_font)[3] - draw.textbbox((0,0), "Ag", font=current_font)[1]
        total_h = len(lines) * line_height * 1.2
        
        start_y = y1 + (max_h - total_h) / 2
        
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0,0), line, font=current_font)
            lw = bbox[2] - bbox[0]
            lx = x1 + (max_w - lw) / 2 if align == "center" else x1
            ly = start_y + (i * line_height * 1.2)
            
            draw.text((lx, ly), line, font=current_font, fill=color, stroke_width=stroke_width, stroke_fill=stroke_fill)


    def _draw_team_row(self, img, team_id, name, score, y_pos):
        draw = ImageDraw.Draw(img)
        # Logo
        logo_path = self.asset_manager.fetch_team_logo(team_id)
        if logo_path:
            logo = Image.open(logo_path).convert("RGBA").resize((150, 150))
            img.paste(logo, (150, y_pos), logo)
            
        # Name (Use fit helper to avoid truncation)
        # Box for name: x=320, width=500
        self._draw_text_with_fit(
            img, 
            str(name).upper(), 
            self.font_header, 
            (320, y_pos, 820, y_pos + 150), 
            align="left"
        )

        # Score
        score_str = str(score)
        draw.text((850, y_pos + 40), score_str, font=self.font_header, fill=self.COLOR_ACCENT_1)

