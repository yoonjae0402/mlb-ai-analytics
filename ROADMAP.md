# MLB Video Pipeline - Architecture & Strategy Roadmap

This document outlines the strategic approach and technical architecture for the automated MLB video pipeline, covering Phases 1-4.

## Core Philosophy
**"Data-Driven Storytelling"**: The system doesn't just recap games; it finds the *story* (e.g., a comeback, a pitching duel, a streak) and uses machine learning to look *forward* (predictions), creating actionable value for viewers.

---

## Phase 1: Foundation (Data Infrastructure)
**Goal**: Build a robust, fault-tolerant engine to fetch and normalize baseball data.

### Strategy
- **Source**: Use `MLB-StatsAPI` for official reliable data and `pybaseball` for advanced Statcast metrics.
- **Caching**: Implement a local file-based cache (`data/cache/`) to minimize API rate limits and speed up development.
- **Context Awareness**: It's not enough to know *who* won. We need to know if it's the *start* or *end* of a series to frame the video correctly.

### Key Components
- **`src/data/fetcher.py`**: The central data access layer. Handles caching and data normalization.
- **`src/data/series_tracker.py`**: Determines "Video Type" (Series Middle vs. Series End).
- **`src/analysis/analyzer.py`**: Heuristic engine that extracts "Key Insights" (e.g., "Dominant Victory", "Nail-biter", streaks, milestones).

---

## Phase 2: The Brain (Machine Learning)
**Goal**: Move beyond reporting stats to *predicting* outcomes, adding a unique value proposition.

### Strategy
- **Model Choice**: **LSTM (Long Short-Term Memory)** neural network. Baseball performance is sequential; a player's last 10 games matter more than their season average.
- **Problem Formulation**: Classification (3-Class). Instead of predicting exact stats (hard), predict impact: **Below Average**, **Average**, **Above Average**.
- **Explainability**: A "Black Box" model is useless for content. We implemented a heuristic **Explainer** that credits specific inputs (e.g., "High Slugging %") for the prediction.

### Key Components
- **`src/models/classifier.py`**: PyTorch LSTM architecture.
- **`src/models/explainer.py`**: Generates human-readable reasons for predictions.
- **`src/models/trainer.py`**: Automated training loop with early stopping.

---

## Phase 3: The Voice (Content Generation)
**Goal**: Convert raw data and predictions into engaging, viral-style short-form content.

### Strategy
- **LLM**: Use **GPT-4o** for scriptwriting. It understands nuance and can adopt a "Hype Sports Commentator" persona.
- **Templates**: Don't let the LLM guess the structure. Use rigid templates (`Hook` -> `Recap` -> `Prediction` -> `CTA`) to ensure viral pacing.
- **Voice**: **Alibaba DashScope TTS** for high-quality text-to-speech with Chinese language support.

### Key Components
- **`src/content/script_generator.py`**: Hydrates prompts with game data and ML predictions.
- **`src/content/audio_generator.py`**: Synthesizes speech and tracks character costs.
- **`src/utils/cost_tracker.py`**: Essential for monitoring OpenAI/DashScope burn rate.

---

## Phase 4: The Look (Video Production)
**Goal**: Assemble professional-grade vertical videos (9:16) without opening a video editor.

### Strategy
- **Programmatic Editing**: Use `MoviePy` to treat video editing like code.
- **Dynamic Assets**:
    - **AssetManager**: Fetches official logos/headshots from MLB's generic CDN.
    - **Visuals**: Dark-mode aesthetic with team colors.
- **Visual Proof**: Don't just *say* a player is hot, *show* it with a Trend Chart.

### Key Components
- **`src/video/video_assembler.py`**: Stitches Audio + Background + Overlays.
- **`src/video/chart_generator.py`**: Matplotlib/Seaborn for generating PNG overlays.
- **`src/video/asset_manager.py`**: Handles downloading/caching of visual assets.

---

## Phase 5: Automation (Upcoming)
**Goal**: Turn the codebase into a "Set and Forget" daily worker.

### Strategy
- **Orchestration**: Create a single `pipeline.py` that connects all previous modules.
- **Scheduling**: Use GitHub Actions or a simple cron job to run the pipeline every morning at 6 AM ET.
- **Deployment**: Automatic YouTube upload via `youtube_uploader.py`.

---

## Phase 6: Monitoring & Optimization (Upcoming)
**Goal**: Ensure quality and profitability.

### Strategy
- **Dashboard**: A Streamlit app to view recent predictions, API costs, and video accuracy.
- **Cost Alerting**: Slack/Email alerts if daily API usage exceeds $1.00.
- **A/B Testing**: Randomly vary video templates (e.g., "Hype" vs. "Analytical" tone) to see which performs better on YouTube.

---

## Future Expansion: KBO Adaptation
**Goal**: Port the pipeline to the Korean Baseball Organization.

### Adaptations Required
- **Data Source**: Switch from MLB API to scraping **Statiz** or using **KBReport**.
- **Language**: Translate GPT-4 prompts to Korean.
- **Voice**: Use Alibaba DashScope Korean TTS or Naver Clova Dubbing.
- **Assets**: Map KBO team IDs and scrape logos from official team sites.
