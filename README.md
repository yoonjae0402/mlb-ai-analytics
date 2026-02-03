# MLB Automated Video Pipeline

Fully automated system for generating baseball analysis videos from MLB game data. Uses Gemini AI for script generation, Google Cloud TTS for voiceover, and optional AI-generated cinematic images - no copyrighted footage, pure statistics, AI narration, and data visualization.

## Features

- **Data Collection**: Fetch game data, player stats, and standings from MLB Stats API
- **ML Predictions**: PyTorch LSTM models for player performance predictions
- **AI Scripts**: Gemini 2.0 Flash powered script generation
- **Natural TTS**: Google Cloud TTS (Neural2) with local Qwen3-TTS fallback
- **Cinematic Mode**: AI-generated images with Ken Burns motion effects via Nano Banana API
- **Video Generation**: Automated video creation with stats, charts, and graphics
- **Live Game Watcher**: Auto-detect game completions and trigger video generation
- **YouTube Upload**: Direct upload to YouTube with metadata
- **Monitoring**: Streamlit dashboard with email alerts
- **Cost Effective**: Smart caching + local fallback integration

## Quick Start

### 1. Clone and Setup

```bash
git clone <repo-url>
cd mlb-video-pipeline

# Create virtual environment
python3 -m venv mlb-env
source mlb-env/bin/activate  # On Windows: mlb-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required API keys:
- **Gemini**: Get from [Google AI Studio](https://aistudio.google.com/app/apikey)
- **Google Cloud TTS** (optional): Enable Text-to-Speech API and download service account JSON
- **Nano Banana** (optional): For cinematic AI image generation
- **YouTube** (optional): From [Google Cloud Console](https://console.cloud.google.com/)

Note:
- TTS uses Google Cloud (Neural2) by default, falling back to local Qwen3-TTS (free) if unavailable.
- Gemini API may incur costs depending on usage (check [pricing](https://ai.google.dev/pricing))

### 3. Run Tests

```bash
pytest tests/ -v
```

### 4. Generate Your First Video

```bash
# Generate for yesterday's game (defaults to Yankees)
python main.py --team Yankees

# Generate for a specific date
python main.py --date 2024-07-04 --team Yankees

# Cinematic mode with AI images
python main.py --date 2024-07-04 --team Yankees --cinematic

# Dry run (no upload)
python main.py --date 2024-07-04 --team Yankees --dry-run
```

## Project Structure

```
mlb-video-pipeline/
├── config/                  # Configuration
│   ├── settings.py          # API keys, paths, constants
│   └── league_config.py     # MLB teams, stats categories
├── data/                    # Data storage
│   ├── raw/                 # Raw MLB API responses
│   ├── processed/           # Cleaned data
│   └── cache/               # API response cache
├── models/                  # ML models
│   ├── player_predictor.pth # Trained model
│   └── training_data/       # Historical data
├── src/                     # Source code
│   ├── pipeline.py          # Pipeline orchestrator
│   ├── watcher.py           # Live game watcher
│   ├── data/                # Data fetching & processing
│   ├── analysis/            # Game analysis
│   ├── models/              # PyTorch model & trainer
│   ├── content/             # Gemini script generation
│   ├── audio/               # Google Cloud TTS + Qwen3-TTS fallback
│   ├── video/               # Video generation & cinematic engine
│   ├── upload/              # YouTube upload
│   └── utils/               # Logging, validation, cost tracking
├── scripts/                 # CLI tools
│   ├── migrate_to_google_tts.py # Verification script for Google TTS
│   └── train_model.py       # Train prediction model
├── outputs/                 # Generated content
│   ├── scripts/             # Text scripts
│   ├── audio/               # Audio narration
│   ├── images/              # AI-generated images
│   ├── videos/              # Final videos
│   └── thumbnails/          # Video thumbnails
├── tests/                   # Unit tests
├── dashboard/               # Streamlit dashboard
└── logs/                    # Runtime logs
```

## Usage Examples

### Fetch Game Data

```python
from src.data import MLBDataFetcher

fetcher = MLBDataFetcher()
games = fetcher.get_schedule(start_date="2024-07-04", end_date="2024-07-04")

for game in games:
    print(f"{game['away_name']} @ {game['home_name']}: {game['status']}")
```

### Generate Script

```python
from src.content import ScriptGenerator

generator = ScriptGenerator()
script = generator.generate_script(
    game_data={
        "away_team": "Red Sox",
        "home_team": "Yankees",
        "away_score": 3,
        "home_score": 5,
        "date": "2024-07-04",
    },
    analysis={"insights": ["Strong pitching performance"]},
    prediction={"prediction": "Above Average", "confidence": "High", "reasons": ["Hot streak"]},
    video_type="series_middle",
)
```

### Create Video

```python
from src.video.generator import VideoGenerator

generator = VideoGenerator(template="modern_dark")

video_path = generator.create_video(
    scenes=[
        {"type": "intro", "title": "GAME RECAP", "duration": 3},
        {"type": "stats", "data": stats_list, "duration": 5},
    ],
    audio_path=narration_path,
    output_name="yankees_recap",
)
```

### Train Prediction Model

```python
from src.models import PlayerPerformanceLSTM, ModelTrainer

model = PlayerPerformanceLSTM(input_size=11, hidden_size=64)
trainer = ModelTrainer(model, learning_rate=0.001)

history = trainer.train(train_loader, val_loader, epochs=50)
```

## CLI Commands

```bash
# Generate video for yesterday's game
python main.py --team Yankees

# Generate for a specific date
python main.py --date 2024-07-04 --team Yankees

# Cinematic mode with AI images
python main.py --date 2024-07-04 --team Yankees --cinematic

# Generate and upload to YouTube
python main.py --date 2024-07-04 --team Yankees --upload

# Watch for live game completions
python main.py --watch --team Yankees

# Dry run (no upload)
python main.py --date 2024-07-04 --team Yankees --dry-run

# Train model
python scripts/train_model.py --epochs 100 --learning-rate 0.001

# Run tests
pytest tests/ -v

# Verify Google TTS
python scripts/migrate_to_google_tts.py
```

## Dashboard

Launch the monitoring dashboard:

```bash
streamlit run dashboard/streamlit_app.py
```

Features:
- Pipeline overview and status
- Cost tracking and budget monitoring
- Video management
- Data file browser

## Video Templates

Available templates:
- `modern_dark`: Sleek dark theme with vibrant accents
- `classic_baseball`: Traditional baseball aesthetic
- `electric_neon`: High-energy style for highlights
- `clean_minimal`: Simple, professional look
- `team_branded`: Customizable with team colors

## Cost Management

The pipeline tracks all API costs:

```python
from src.utils import CostTracker

tracker = CostTracker()
print(f"Today's spend: ${tracker.get_daily_cost():.2f}")
print(f"Under budget: {tracker.check_budget(10.0)}")
```

Default limits:
- Daily spend limit: $10
- Google Cloud TTS: Neural2 voice ($0.000016/char), cached to minimize cost
- Local Qwen3-TTS fallback: Free
- Gemini: Check [pricing](https://ai.google.dev/pricing)

## Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific module
pytest tests/video/ -v
```

### Code Style

```bash
# Format code
black src/ scripts/ tests/

# Sort imports
isort src/ scripts/ tests/

# Type checking
mypy src/
```

## Troubleshooting

### API Key Issues

```bash
# Verify keys are loaded
python -c "from config.settings import settings; print(settings.validate_api_keys())"
```

### MoviePy/FFmpeg Issues

```bash
# Install FFmpeg
brew install ffmpeg  # macOS
apt install ffmpeg   # Ubuntu
```

### Cache Issues

```bash
# Clear API cache
rm -rf data/cache/*
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

---

Built with Python, PyTorch, Gemini, Google Cloud TTS, Qwen3-TTS, and MoviePy.
