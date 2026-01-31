"""
MLB Video Pipeline - Post-game Analysis Module

Analyzes completed games for video content generation.
Extracts storylines, key performances, and contextual insights
WITHOUT making predictions.

Usage:
    from src.data.analyzer import MLBStatsAnalyzer

    analyzer = MLBStatsAnalyzer()
    insights = analyzer.analyze_game(game_data)
"""

from typing import Any, Optional
from datetime import datetime

from src.utils.logger import get_logger
from src.data.fetcher import MLBDataFetcher


logger = get_logger(__name__)


class MLBStatsAnalyzer:
    """
    Generate insights from completed MLB games.

    Responsibilities:
    - Extract key storylines (streaks, milestones, comebacks)
    - Find interesting statistics and anomalies
    - Analyze player performance highlights
    - Provide season context and standings impact

    Note: This class does NOT make predictions. It only analyzes
    completed games for content generation.
    """

    def __init__(self, fetcher: Optional[MLBDataFetcher] = None):
        """
        Initialize the analyzer.

        Args:
            fetcher: Optional MLBDataFetcher instance for additional data
        """
        self.fetcher = fetcher or MLBDataFetcher()

    def analyze_game(self, game_data: dict) -> dict:
        """
        Analyze a completed game for content generation.

        Args:
            game_data: Game data dictionary from fetcher

        Returns:
            {
                "key_insights": [str, ...],  # Main talking points
                "top_performances": [dict, ...],  # Standout players
                "storylines": [str, ...],  # Narrative angles
                "season_context": {
                    "standings_impact": str,
                    "playoff_implications": str,
                    "streak_info": str
                },
                "game_flow": {
                    "turning_point": str,
                    "decisive_moment": str,
                    "momentum_shifts": int
                }
            }
        """
        logger.info(f"Analyzing game: {game_data.get('game_id')}")

        return {
            "key_insights": self._extract_key_insights(game_data),
            "top_performances": self._find_top_performances(game_data),
            "storylines": self.find_storylines(game_data),
            "season_context": self._get_season_context(game_data),
            "game_flow": self._analyze_game_flow(game_data),
        }

    def find_storylines(self, game_data: dict) -> list[str]:
        """
        Find interesting narratives from the game.

        Looks for:
        - Hitting/winning/losing streaks
        - Milestone achievements (100 RBIs, 20 wins, etc.)
        - Comeback victories
        - Pitcher duels / blowouts
        - Rookie performances
        - Player vs former team
        - Historical comparisons

        Args:
            game_data: Game data dictionary

        Returns:
            List of storyline descriptions
        """
        storylines = []

        # Check for comeback
        if self._is_comeback(game_data):
            storylines.append(self._describe_comeback(game_data))

        # Check for milestone performances
        milestones = self._find_milestones(game_data)
        storylines.extend(milestones)

        # Check for streaks
        streaks = self._find_streaks(game_data)
        storylines.extend(streaks)

        # Check for notable matchups
        matchups = self._find_notable_matchups(game_data)
        storylines.extend(matchups)

        return storylines[:5]  # Limit to top 5 storylines

    def _extract_key_insights(self, game_data: dict) -> list[str]:
        """Extract the most important talking points from the game."""
        insights = []

        # Score differential analysis
        home_score = game_data.get("home_score", 0)
        away_score = game_data.get("away_score", 0)
        diff = abs(home_score - away_score)

        if diff >= 7:
            winner = game_data.get("home_team" if home_score > away_score else "away_team")
            insights.append(f"Dominant victory by {winner} with a {diff}-run margin")
        elif diff <= 1:
            insights.append("Nail-biting finish decided by a single run")

        # High-scoring game
        total_runs = home_score + away_score
        if total_runs >= 15:
            insights.append(f"Offensive explosion with {total_runs} combined runs")

        # Pitching duel
        if total_runs <= 4 and home_score <= 2 and away_score <= 2:
            insights.append("Classic pitching duel with dominant performances on the mound")

        return insights

    def _find_top_performances(self, game_data: dict) -> list[dict]:
        """
        Find standout individual performances.

        Returns list of player performance dictionaries.
        """
        performances = []

        # Analyze box score if available
        box_score = game_data.get("box_score", {})

        # Find batting stars
        for team in ["home", "away"]:
            batters = box_score.get(f"{team}_batters", [])
            for batter in batters:
                # Multi-hit game
                if batter.get("hits", 0) >= 3:
                    performances.append({
                        "player": batter.get("name"),
                        "team": game_data.get(f"{team}_team"),
                        "type": "batting",
                        "highlight": f"{batter.get('hits')} hits",
                        "stats": batter,
                    })
                # Multi-homer game
                if batter.get("home_runs", 0) >= 2:
                    performances.append({
                        "player": batter.get("name"),
                        "team": game_data.get(f"{team}_team"),
                        "type": "power",
                        "highlight": f"{batter.get('home_runs')} home runs",
                        "stats": batter,
                    })

        # Sort by impact and return top performers
        return sorted(performances,
                     key=lambda x: self._calculate_impact_score(x),
                     reverse=True)[:3]

    def _get_season_context(self, game_data: dict) -> dict:
        """Get broader season context for the game."""
        return {
            "standings_impact": self._calculate_standings_impact(game_data),
            "playoff_implications": self._check_playoff_implications(game_data),
            "streak_info": self._get_streak_info(game_data),
        }

    def _analyze_game_flow(self, game_data: dict) -> dict:
        """Analyze how the game unfolded."""
        return {
            "turning_point": self._find_turning_point(game_data),
            "decisive_moment": self._find_decisive_moment(game_data),
            "momentum_shifts": self._count_momentum_shifts(game_data),
        }

    # Helper methods
    def _is_comeback(self, game_data: dict) -> bool:
        """Check if the game featured a comeback."""
        scoring = game_data.get("scoring_plays", [])
        if not scoring:
            return False

        max_deficit = 0
        current_diff = 0

        for play in scoring:
            runs = play.get("runs", 1)
            is_home = play.get("is_home_team", False)
            current_diff += runs if is_home else -runs
            max_deficit = max(max_deficit, abs(current_diff))

        final_winner_is_home = game_data.get("home_score", 0) > game_data.get("away_score", 0)
        return max_deficit >= 3

    def _describe_comeback(self, game_data: dict) -> str:
        """Describe a comeback victory."""
        winner = game_data.get("home_team" if
                               game_data.get("home_score", 0) > game_data.get("away_score", 0)
                               else "away_team")
        return f"{winner} stages an incredible comeback victory"

    def _find_milestones(self, game_data: dict) -> list[str]:
        """Find milestone achievements in the game."""
        milestones = []
        box_score = game_data.get("boxscore", {})

        # Check batting milestones
        for team in ["home", "away"]:
            batters = box_score.get(f"{team}_batters", [])
            for batter in batters:
                player_name = batter.get("name", "Unknown")
                season_stats = batter.get("season_stats", {})

                # Check for milestone home runs (10, 20, 30, 40, 50)
                hrs = season_stats.get("homeRuns", 0)
                if hrs > 0 and hrs % 10 == 0:
                    milestones.append(f"{player_name} hits their {hrs}th home run of the season")

                # Check for milestone RBIs (50, 100, 150)
                rbis = season_stats.get("rbi", 0)
                if rbis in [50, 100, 150]:
                    milestones.append(f"{player_name} reaches {rbis} RBIs for the season")

                # Check for milestone hits (100, 150, 200)
                hits = season_stats.get("hits", 0)
                if hits in [100, 150, 200]:
                    milestones.append(f"{player_name} collects their {hits}th hit of the season")

        # Check pitching milestones
        for team in ["home", "away"]:
            pitchers = box_score.get(f"{team}_pitchers", [])
            for pitcher in pitchers:
                player_name = pitcher.get("name", "Unknown")
                season_stats = pitcher.get("season_stats", {})

                # Check for milestone wins (10, 15, 20)
                wins = season_stats.get("wins", 0)
                if wins in [10, 15, 20]:
                    milestones.append(f"{player_name} earns their {wins}th win of the season")

                # Check for milestone strikeouts (100, 150, 200, 250, 300)
                strikeouts = season_stats.get("strikeOuts", 0)
                if strikeouts > 0 and strikeouts % 50 == 0:
                    milestones.append(f"{player_name} reaches {strikeouts} strikeouts for the season")

        return milestones

    def _find_streaks(self, game_data: dict) -> list[str]:
        """Find notable streaks extended or broken."""
        streaks = []

        # Team winning/losing streaks (would need historical data from fetcher)
        home_team = game_data.get("home_team", "")
        away_team = game_data.get("away_team", "")
        home_won = game_data.get("home_score", 0) > game_data.get("away_score", 0)

        # For now, use simplified logic - would ideally query recent games via fetcher
        box_score = game_data.get("boxscore", {})

        # Check for hitting streaks (multi-game hits)
        for team in ["home", "away"]:
            batters = box_score.get(f"{team}_batters", [])
            for batter in batters:
                hits_in_game = batter.get("hits", 0)
                player_name = batter.get("name", "Unknown")

                # If they got hits, mention notable streak potential
                if hits_in_game >= 2:
                    streaks.append(f"{player_name} extends hitting success with {hits_in_game} hits")

        # Check for pitcher performance streaks
        winning_pitcher = box_score.get("winning_pitcher", {})
        if winning_pitcher:
            pitcher_name = winning_pitcher.get("name", "Unknown")
            season_stats = winning_pitcher.get("season_stats", {})
            wins = season_stats.get("wins", 0)
            if wins >= 3:  # Winning pitcher with 3+ wins suggests a hot streak
                streaks.append(f"{pitcher_name} continues strong season performance")

        return streaks[:3]  # Limit to top 3

    def _find_notable_matchups(self, game_data: dict) -> list[str]:
        """Find notable individual matchups."""
        matchups = []

        home_team = game_data.get("home_team", "")
        away_team = game_data.get("away_team", "")

        # Check for division rivalry games (same division matchups)
        division_rivals = {
            "Yankees": ["Red Sox", "Blue Jays", "Rays", "Orioles"],
            "Red Sox": ["Yankees", "Blue Jays", "Rays", "Orioles"],
            "Dodgers": ["Giants", "Padres", "Diamondbacks", "Rockies"],
            "Giants": ["Dodgers", "Padres", "Diamondbacks", "Rockies"],
            "Cubs": ["Cardinals", "Brewers", "Pirates", "Reds"],
            "Cardinals": ["Cubs", "Brewers", "Pirates", "Reds"],
        }

        for team, rivals in division_rivals.items():
            if team in home_team and away_team in " ".join(rivals):
                matchups.append(f"Division rivalry: {home_team} vs {away_team}")
                break
            elif team in away_team and home_team in " ".join(rivals):
                matchups.append(f"Division rivalry: {away_team} vs {home_team}")
                break

        # Check for pitcher-batter matchups (would need detailed play-by-play)
        box_score = game_data.get("boxscore", {})
        starting_pitcher = box_score.get("away_starting_pitcher", {})
        if starting_pitcher:
            pitcher_name = starting_pitcher.get("name", "")
            pitcher_strikeouts = starting_pitcher.get("strikeouts", 0)
            if pitcher_strikeouts >= 10:
                matchups.append(f"{pitcher_name} dominates with {pitcher_strikeouts} strikeouts")

        return matchups

    def _calculate_impact_score(self, performance: dict) -> float:
        """Calculate impact score for ranking performances."""
        score = 0.0
        stats = performance.get("stats", {})

        score += stats.get("hits", 0) * 1.0
        score += stats.get("home_runs", 0) * 3.0
        score += stats.get("rbi", 0) * 1.5
        score += stats.get("runs", 0) * 1.0

        return score

    def _calculate_standings_impact(self, game_data: dict) -> str:
        """Calculate impact on division/wild card standings."""
        home_team = game_data.get("home_team", "")
        away_team = game_data.get("away_team", "")
        home_won = game_data.get("home_score", 0) > game_data.get("away_score", 0)
        winner = home_team if home_won else away_team

        # Get current standings if available via fetcher
        try:
            game_date = game_data.get("game_date", "")
            if game_date:
                year = int(game_date.split("-")[0]) if "-" in game_date else datetime.now().year
            else:
                year = datetime.now().year

            standings = self.fetcher.get_standings(season=year)

            # Find winner's position in standings
            for division in standings:
                for team_record in division.get("teams", []):
                    team_name = team_record.get("name", "")
                    if winner in team_name:
                        gb = team_record.get("gamesBack", "0")
                        if gb == "0" or gb == "-":
                            return f"{winner} maintains division lead"
                        elif float(gb) <= 3.0:
                            return f"{winner} closes gap in tight division race"
                        else:
                            return f"{winner} gains ground in the standings"

        except Exception as e:
            logger.debug(f"Could not fetch standings: {e}")

        return f"{winner} picks up an important victory"

    def _check_playoff_implications(self, game_data: dict) -> str:
        """Check playoff race implications."""
        try:
            game_date = game_data.get("game_date", "")
            if game_date:
                month = int(game_date.split("-")[1]) if "-" in game_date else datetime.now().month
            else:
                month = datetime.now().month

            # Playoff implications are more significant in late season
            if month >= 9:  # September or later
                home_team = game_data.get("home_team", "")
                away_team = game_data.get("away_team", "")
                home_won = game_data.get("home_score", 0) > game_data.get("away_score", 0)
                winner = home_team if home_won else away_team

                return f"Critical late-season victory for {winner} in the playoff race"
            elif month >= 7:  # July-August
                return "Important mid-season game as playoff picture takes shape"
            else:
                return "Early season matchup"

        except Exception as e:
            logger.debug(f"Could not determine playoff implications: {e}")
            return "Regular season game"

    def _get_streak_info(self, game_data: dict) -> str:
        """Get current streak information for both teams."""
        home_team = game_data.get("home_team", "")
        away_team = game_data.get("away_team", "")
        home_won = game_data.get("home_score", 0) > game_data.get("away_score", 0)

        # Try to get team records from box score
        box_score = game_data.get("boxscore", {})
        home_record = box_score.get("home_team_record", "")
        away_record = box_score.get("away_team_record", "")

        if home_won and home_record:
            return f"{home_team} improves to {home_record}"
        elif not home_won and away_record:
            return f"{away_team} improves to {away_record}"

        # Fallback to simple description
        winner = home_team if home_won else away_team
        return f"{winner} picks up the win"

    def _find_turning_point(self, game_data: dict) -> str:
        """Find the game's turning point."""
        scoring_plays = game_data.get("scoring_plays", [])
        if not scoring_plays:
            return "No significant turning point"

        # Find the play that resulted in the largest lead change
        max_swing = 0
        turning_point_play = None
        current_diff = 0

        for i, play in enumerate(scoring_plays):
            runs = play.get("runs", 1)
            is_home = play.get("is_home_team", False)

            # Calculate score before and after this play
            prev_diff = current_diff
            current_diff += runs if is_home else -runs

            # Calculate swing (change in differential)
            swing = abs(current_diff - prev_diff)

            if swing > max_swing or (swing == max_swing and abs(current_diff) < abs(prev_diff)):
                max_swing = swing
                turning_point_play = play

        if turning_point_play:
            inning = turning_point_play.get("inning", "unknown")
            description = turning_point_play.get("description", "key play")
            return f"Turning point in inning {inning}: {description}"

        return "Close game throughout"

    def _find_decisive_moment(self, game_data: dict) -> str:
        """Find the game's decisive moment."""
        scoring_plays = game_data.get("scoring_plays", [])
        if not scoring_plays:
            return "No decisive moment recorded"

        home_score = game_data.get("home_score", 0)
        away_score = game_data.get("away_score", 0)
        final_diff = abs(home_score - away_score)

        # The decisive moment is often the go-ahead run or insurance run
        # Look for the last play that gave the winning team the lead

        winning_team_is_home = home_score > away_score
        current_diff = 0
        decisive_play = None

        for play in scoring_plays:
            runs = play.get("runs", 1)
            is_home = play.get("is_home_team", False)

            prev_diff = current_diff
            current_diff += runs if is_home else -runs

            # Check if this play gave the eventual winner the lead
            if winning_team_is_home and is_home:
                if prev_diff <= 0 and current_diff > 0:  # Home team takes lead
                    decisive_play = play
            elif not winning_team_is_home and not is_home:
                if prev_diff >= 0 and current_diff < 0:  # Away team takes lead
                    decisive_play = play

        if decisive_play:
            inning = decisive_play.get("inning", "unknown")
            description = decisive_play.get("description", "go-ahead score")
            return f"Decisive moment in inning {inning}: {description}"

        # Fallback: if we didn't find a lead change, return the final scoring play
        if scoring_plays:
            last_play = scoring_plays[-1]
            inning = last_play.get("inning", "unknown")
            return f"Final runs scored in inning {inning}"

        return "No clear decisive moment"

    def _count_momentum_shifts(self, game_data: dict) -> int:
        """Count significant momentum shifts."""
        scoring_plays = game_data.get("scoring_plays", [])
        if not scoring_plays:
            return 0

        momentum_shifts = 0
        current_diff = 0
        last_scoring_team_is_home = None

        for play in scoring_plays:
            runs = play.get("runs", 1)
            is_home = play.get("is_home_team", False)

            # Count as momentum shift when:
            # 1. Lead changes hands
            # 2. Big inning (3+ runs) changes the game dynamic

            prev_diff = current_diff
            current_diff += runs if is_home else -runs

            # Lead change
            if (prev_diff > 0 and current_diff < 0) or (prev_diff < 0 and current_diff > 0):
                momentum_shifts += 1

            # Tie broken
            elif prev_diff == 0 and current_diff != 0:
                momentum_shifts += 1

            # Big scoring play (3+ runs)
            elif runs >= 3:
                momentum_shifts += 1

            # Answer-back momentum (opponent scores immediately after)
            elif last_scoring_team_is_home is not None and last_scoring_team_is_home != is_home:
                # Check if this is a quick response (within same inning or next)
                # For now, just count scoring by different teams in succession as potential shift
                pass

            last_scoring_team_is_home = is_home

        return momentum_shifts
