
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
        if scene_type == "hook":
            return self.render_hook(img, scene, game_data)
        elif scene_type in ("scoreboard", "score"):
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
            return self.render_factor_card(img, scene, prediction_data)
        elif scene_type == "transition":
            return self.render_transition(img, scene)
        elif scene_type == "recap":
            return self.render_recap(img, scene, game_data)
        elif scene_type == "cta":
            return self.render_cta(img, scene)
        else:
            return self._render_generic_text(img, scene)

    def render_hook(self, img: Image.Image, scene: Dict, game_data: Dict) -> Image.Image:
        draw = ImageDraw.Draw(img)
        visual = scene.get("visual", {})

        # Use real team name from game_data if visual text is generic
        main_text = visual.get("main_text", "").upper()
        if not main_text or main_text in ("HOOK TEXT", "HOOK"):
            winner = game_data.get("winner", "")
            if winner:
                main_text = f"{winner.split()[-1].upper()} WIN!"
            else:
                main_text = "BIG WIN!"

        # Choose font size based on text length to prevent clipping
        if len(main_text) <= 8:
            hook_font = self.font_stat_huge  # 150pt fits short text
        else:
            hook_font = self.font_header     # 90pt for longer text

        self._draw_centered_text(img, main_text, hook_font, y_offset=600, color=self.COLOR_ACCENT_1, stroke_width=4, stroke_fill="black")

        # Subtext - show score if available
        sub_text = visual.get("sub_text", "")
        if not sub_text:
            away_score = game_data.get('away_score', '')
            home_score = game_data.get('home_score', '')
            if away_score != '' and home_score != '':
                sub_text = f"{away_score}-{home_score}"
        if sub_text:
             self._draw_centered_text(img, sub_text, self.font_header, y_offset=900, color="white", stroke_width=2, stroke_fill="black")

        return img

    def render_scoreboard(self, img: Image.Image, game_data: Dict) -> Image.Image:
        draw = ImageDraw.Draw(img)

        box_rect = (80, 470, 1000, 1100)
        self._draw_glass_panel(img, box_rect)

        away_score = game_data.get('away_score', 0)
        home_score = game_data.get('home_score', 0)
        away_id = game_data.get('away_team_id')
        home_id = game_data.get('home_team_id')

        y_away = 560
        y_home = 830

        self._draw_team_row(img, away_id, game_data.get('away_team'), away_score, y_away)

        # Divider line
        draw.line([(150, 760), (950, 760)], fill=(255, 255, 255, 40), width=2)

        self._draw_team_row(img, home_id, game_data.get('home_team'), home_score, y_home)

        self._draw_centered_text(img, "FINAL SCORE", self.font_sub, y_offset=420, color=self.COLOR_ACCENT_1)

        return img

    def render_player_spotlight(self, img: Image.Image, scene: Dict, game_data: Dict) -> Image.Image:
        """
        Layout: Player headshot (top) + name + stats card (bottom).
        Falls back to visual data from the script if game_data fields missing.
        """
        draw = ImageDraw.Draw(img)
        visual = scene.get("visual", {})
        scene_type = scene.get("scene_type", "")

        # Pull real player data from game_data based on scene type
        player_name = visual.get("player_to_show", "")
        player_stats = visual.get("stat_to_show", "")
        player_impact = ""
        player_id = None

        if scene_type in ("top_hitter", "player_watch_1"):
            hitter = game_data.get("top_hitter", {})
            if hitter.get("name"):
                player_name = player_name or hitter["name"]
                player_stats = player_stats or hitter.get("stats", "")
                player_impact = hitter.get("impact", "")
                player_id = hitter.get("id")
        elif scene_type in ("top_pitcher", "player_watch_2"):
            pitcher = game_data.get("top_pitcher", {})
            if pitcher.get("name"):
                player_name = player_name or pitcher["name"]
                player_stats = player_stats or pitcher.get("stats", "")
                player_impact = pitcher.get("impact", "")
                player_id = pitcher.get("id")

        if not player_name:
            player_name = "Player Spotlight"

        # Fix 2: Fetch and paste player headshot on top half
        if player_id:
            try:
                headshot_path = self.asset_manager.fetch_player_headshot(player_id)
                if headshot_path:
                    headshot = Image.open(headshot_path).convert("RGBA")
                    h_w, h_h = headshot.size
                    scale = min(500 / h_h, 600 / h_w)
                    new_size = (int(h_w * scale), int(h_h * scale))
                    headshot = headshot.resize(new_size, Image.LANCZOS)
                    x = (self.WIDTH - new_size[0]) // 2
                    img.paste(headshot, (x, 350), headshot)
            except Exception as e:
                logger.warning(f"Could not load player headshot: {e}")

        # Get player's team for display
        player_team = ""
        if scene_type == "top_hitter":
            player_team = game_data.get("top_hitter", {}).get("team", "")
        elif scene_type == "top_pitcher":
            player_team = game_data.get("top_pitcher", {}).get("team", "")

        # Header shows player's team
        if player_team:
            header_label = player_team.upper()
        else:
            header_label = "GAME LEADER"

        # Draw Tech Card at bottom
        card_rect = (50, 1000, 1030, 1650)
        self._draw_glass_panel(img, card_rect)

        # Header label above player name
        draw.text((100, 960), header_label, font=self.font_body, fill=self.COLOR_TEXT_SUB)

        # Fix 4: Player name - use text fit to prevent overflow
        self._draw_text_with_fit(
            img, str(player_name).upper(), self.font_header,
            (80, 1030, 1010, 1160),
            color=self.COLOR_ACCENT_1, align="left"
        )

        # Stats line (skip for player_watch to avoid duplicate content)
        if player_stats and scene_type not in ("player_watch_1", "player_watch_2"):
            stats_parts = str(player_stats).split(",")
            y_stat = 1200
            for stat in stats_parts:
                draw.text((100, y_stat), stat.strip(), font=self.font_sub, fill="white")
                y_stat += 70
        elif scene_type in ("player_watch_1", "player_watch_2"):
            # Show narration context instead of repeating stats
            narration = scene.get("narration", "")
            if narration:
                preview = narration[:80]
                self._draw_text_with_fit(img, preview, self.font_body, (100, 1200, 1000, 1400), color="white")

        # Impact line
        if player_impact:
            self._draw_text_with_fit(img, player_impact, self.font_body, (100, 1400, 1000, 1600), color=self.COLOR_TEXT_SUB)

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
        Visual for Standings/Impact scene with real team names.
        """
        draw = ImageDraw.Draw(img)

        self._draw_centered_text(img, "PLAYOFF IMPACT", self.font_header, y_offset=400, color=self.COLOR_ACCENT_1)

        self._draw_glass_panel(img, (100, 600, 980, 1400))

        # Show winner and loser with real names
        winner = game_data.get('winner', game_data.get('home_team', 'Team'))
        home_team = game_data.get('home_team', 'Home')
        away_team = game_data.get('away_team', 'Away')
        loser = away_team if winner == home_team else home_team

        y_start = 700
        draw.text((150, y_start), winner.upper(), font=self.font_sub, fill="white")
        draw.text((800, y_start), "W", font=self.font_sub, fill=self.COLOR_ACCENT_1)

        draw.text((150, y_start + 100), loser.upper(), font=self.font_sub, fill="gray")
        draw.text((800, y_start + 100), "L", font=self.font_sub, fill=self.COLOR_ACCENT_2)

        # Standings impact text
        impact_text = game_data.get('standings_impact', '')
        if impact_text:
            self._draw_text_with_fit(img, impact_text, self.font_body, (150, y_start + 250, 930, y_start + 550), color=self.COLOR_TEXT_SUB)

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

    def render_factor_card(self, img: Image.Image, scene: Dict, prediction: Optional[Dict] = None) -> Image.Image:
        """Render prediction factor with visual indicators."""
        draw = ImageDraw.Draw(img)
        visual = scene.get("visual", {})
        scene_type = scene.get("scene_type", "factor_1")

        factor_idx = int(scene_type.split("_")[-1]) - 1 if "_" in scene_type else 0

        factor_name = visual.get("main_text", "FACTOR")
        factor_detail = ""
        factor_impact = ""

        if prediction:
            factors = prediction.get("factors", [])
            if factor_idx < len(factors):
                f = factors[factor_idx]
                factor_name = f.get("factor", factor_name).upper()
                factor_detail = f.get("detail", "")
                factor_impact = f.get("impact", "")

        icon_colors = [
            (0, 200, 255, 255),    # Blue
            (255, 200, 0, 255),    # Gold
            (0, 255, 136, 255),    # Green
        ]
        icon_color = icon_colors[factor_idx] if factor_idx < 3 else self.COLOR_ACCENT_1

        # Badge
        badge_text = f"FACTOR {factor_idx + 1} OF 3"
        self._draw_centered_text(img, badge_text, self.font_body, y_offset=450, color=self.COLOR_TEXT_SUB)

        # Title
        self._draw_centered_text(img, factor_name, self.font_header, y_offset=600, color="white", stroke_width=2, stroke_fill="black")

        # Detail card
        if factor_detail:
            self._draw_glass_panel(img, (100, 800, 980, 1200))
            self._draw_text_with_fit(img, factor_detail, self.font_sub, (150, 850, 930, 1150))

        # Impact bar
        if factor_impact:
            impact_str = str(factor_impact).lower()
            if any(w in impact_str for w in ["high", "strong", "major", "dominant"]):
                bar_fill = 0.85
            elif any(w in impact_str for w in ["moderate", "medium", "solid"]):
                bar_fill = 0.6
            else:
                bar_fill = 0.4

            bar_y = 1280
            bar_x1, bar_x2 = 200, 880
            bar_width = bar_x2 - bar_x1

            draw.rounded_rectangle((bar_x1, bar_y, bar_x2, bar_y + 30), radius=15, fill=(50, 50, 50, 200))
            fill_x2 = bar_x1 + int(bar_width * bar_fill)
            draw.rounded_rectangle((bar_x1, bar_y, fill_x2, bar_y + 30), radius=15, fill=icon_color)

            self._draw_centered_text(img, f"IMPACT: {factor_impact.upper()}", self.font_body, y_offset=1330, color=self.COLOR_TEXT_SUB)

        return img

    def render_recap(self, img: Image.Image, scene: Dict, game_data: Dict) -> Image.Image:
        """Render game recap summary scene."""
        draw = ImageDraw.Draw(img)
        visual = scene.get("visual", {})

        winner = game_data.get('winner', 'Winner')
        winner_short = winner.split()[-1].upper()
        home_score = game_data.get('home_score', 0)
        away_score = game_data.get('away_score', 0)

        # Main result text
        main_text = visual.get("main_text", f"{winner_short} WIN")
        sub_text = visual.get("sub_text", f"{max(home_score, away_score)}-{min(home_score, away_score)} FINAL")

        self._draw_centered_text(img, main_text, self.font_stat_huge, y_offset=700, color=self.COLOR_ACCENT_1, stroke_width=4, stroke_fill="black")
        self._draw_centered_text(img, sub_text, self.font_header, y_offset=950, color="white", stroke_width=2, stroke_fill="black")

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
        self._draw_centered_text(img, "FOLLOW FOR DAILY", self.font_sub, y_offset=700, color="white")
        self._draw_centered_text(img, "AI PREDICTIONS", self.font_header, y_offset=850, color=self.COLOR_ACCENT_1, stroke_width=3, stroke_fill="black")
        self._draw_centered_text(img, "LIKE + FOLLOW", self.font_body, y_offset=1050, color=self.COLOR_TEXT_SUB)
        return img

    def render_transition(self, img: Image.Image, scene: Dict) -> Image.Image:
        """Render transition scene with UP NEXT visual."""
        draw = ImageDraw.Draw(img)

        # Dark gradient overlay band
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw_ov = ImageDraw.Draw(overlay)
        for y in range(800, 1120):
            alpha = int(160 * (1 - abs(y - 960) / 160))
            alpha = max(0, alpha)
            draw_ov.line([(0, y), (self.WIDTH, y)], fill=(0, 0, 0, alpha))
        img = Image.alpha_composite(img, overlay)

        draw = ImageDraw.Draw(img)
        draw.line([(200, 880), (880, 880)], fill=self.COLOR_ACCENT_1, width=4)

        self._draw_centered_text(img, "UP NEXT", self.font_header, y_offset=910, color=self.COLOR_ACCENT_1, stroke_width=2, stroke_fill="black")

        narration = scene.get("narration", "")
        if narration:
            preview = " ".join(narration.split()[:6]).upper()
            if len(narration.split()) > 6:
                preview += "..."
            self._draw_centered_text(img, preview, self.font_sub, y_offset=1030, color=self.COLOR_TEXT_SUB)

        return img

    def _render_generic_text(self, img: Image.Image, scene: Dict) -> Image.Image:
        visual = scene.get("visual", {})
        txt = visual.get("main_text", "")
        self._draw_centered_text(img, txt, self.font_header, y_offset=800)
        return img

    # --- Helpers ---
    
    def _create_tech_bg(self) -> Image.Image:
        """Smooth radial gradient background, no scan lines."""
        img = Image.new("RGBA", (self.WIDTH, self.HEIGHT), self.COLOR_BG)

        center_x, center_y = self.WIDTH // 2, self.HEIGHT // 2
        max_radius = int(math.sqrt(center_x**2 + center_y**2))

        overlay = Image.new("RGBA", (self.WIDTH, self.HEIGHT), (0, 0, 0, 0))
        draw_ov = ImageDraw.Draw(overlay)

        steps = 60
        for i in range(steps):
            t = i / steps
            alpha = int(35 * t)
            rx = int(max_radius * (1 - t))
            ry = int(max_radius * (1 - t))
            if rx > 0 and ry > 0:
                bbox = (center_x - rx, center_y - ry, center_x + rx, center_y + ry)
                draw_ov.ellipse(bbox, fill=(255, 255, 255, alpha))

        img = Image.alpha_composite(img, overlay)
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
        # Logo - reduced to 120px
        logo_path = self.asset_manager.fetch_team_logo(team_id)
        if logo_path:
            logo = Image.open(logo_path).convert("RGBA").resize((120, 120))
            img.paste(logo, (150, y_pos + 15), logo)

        # Name - use font_sub (45pt) to prevent overflow
        self._draw_text_with_fit(
            img,
            str(name).upper(),
            self.font_sub,
            (290, y_pos, 800, y_pos + 150),
            align="left"
        )

        # Score - positioned further right
        score_str = str(score)
        draw.text((880, y_pos + 30), score_str, font=self.font_header, fill=self.COLOR_ACCENT_1)

