# MLB Video Pipeline - Complete Usage Guide

This guide walks you through using the automated MLB video pipeline from setup to production.

## Table of Contents
1. [Initial Setup](#initial-setup)
2. [Basic Usage](#basic-usage)
3. [Advanced Features](#advanced-features)
4. [Monitoring](#monitoring)
5. [Troubleshooting](#troubleshooting)

---

## Initial Setup

### 1. Environment Setup

```bash
# Clone the repository
cd ~/Desktop/mlb-video-pipeline

# Activate virtual environment
source mlb-env/bin/activate

# Verify installation
python -c "import src; print('✅ Installation OK')"
```

### 2. Configure API Keys

Edit your `.env` file with required credentials:

```bash
# Required for script generation
OPENAI_API_KEY=sk-...

# Required for voice synthesis
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=...

# Optional: For alerts
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
ALERT_EMAIL_FROM=your-email@gmail.com
ALERT_EMAIL_TO=alerts@example.com
ALERT_EMAIL_PASSWORD=your-app-password
```

### 3. Verify Setup

```bash
# Run all tests
./mlb-env/bin/pytest tests/ -v

# Should see all tests passing
```

---

## Basic Usage

### Generate a Video (Simple)

The easiest way to generate a video:

```bash
# Generate video for yesterday's Yankees game
python main.py --team Yankees

# Generate for a specific date
python main.py --date 2024-07-04 --team "Red Sox"
```

**What happens:**
1. Fetches game data from MLB API
2. Analyzes the game for key insights
3. Generates ML prediction for next game
4. Creates AI-powered script
5. Synthesizes voiceover audio
6. Generates trend charts
7. Assembles final video

**Output:**
- Video: `outputs/videos/2024-07-04_Yankees.mp4`
- Audio: `outputs/audio/2024-07-04_Yankees.mp3`
- Charts: `data/charts/2024-07-04_trend.png`
- Logs: `logs/pipeline.log`

### Generate and Upload to YouTube

```bash
# Generate and upload (private by default)
python main.py --date 2024-07-04 --team Yankees --upload
```

**First-time YouTube setup:**
1. Download `client_secrets.json` from Google Cloud Console
2. Place it in project root
3. Run with `--upload` flag
4. Browser will open for OAuth authentication
5. Credentials saved to `token.pickle` for future use

---

## Advanced Features

### Custom Team Selection

```bash
# Any MLB team works
python main.py --team "Dodgers"
python main.py --team "Cubs"
python main.py --team "Astros"
```

### Date Range Processing

```bash
# Process multiple days (useful for catching up)
for date in 2024-07-{01..07}; do
    python main.py --date $date --team Yankees
done
```

### Dry Run (Testing)

```bash
# Test without uploading
python main.py --date 2024-07-04 --team Yankees --dry-run
```

---

## Monitoring

### Launch Dashboard

```bash
# Install dashboard dependencies (first time only)
pip install streamlit plotly

# Launch dashboard
streamlit run dashboard/app.py
```

The dashboard opens at `http://localhost:8501`

### Dashboard Features

**📊 Overview Page:**
- Total pipeline runs
- Success rate percentage
- Average processing time
- Total API costs
- Recent runs table

**💰 Cost Analysis:**
- Daily cost breakdown (OpenAI vs ElevenLabs)
- Stacked bar charts
- Cost per video average
- Monthly projection

**⚡ Performance:**
- Processing time trends
- Duration statistics
- Success/failure pie chart

**🎬 Video Gallery:**
- Browse generated videos
- Preview in browser
- View metadata (cost, duration, etc.)

### View Metrics Manually

```python
from src.utils import MetricsCollector

metrics = MetricsCollector()

# Get last 7 days
recent = metrics.get_recent_runs(days=7)
for run in recent:
    print(f"{run['date']} - {run['team']}: {'✅' if run['success'] else '❌'}")

# Cost summary
from datetime import datetime, timedelta
start = (datetime.now() - timedelta(days=7)).isoformat()
end = datetime.now().isoformat()

summary = metrics.get_cost_summary(start, end)
print(f"Total cost: ${summary['total']:.2f}")
print(f"Videos generated: {summary['runs']}")
```

---

## Understanding the Pipeline Flow

### Step-by-Step Breakdown

```
1. Data Fetching (5-10s)
   ├─ Fetch schedule for date
   ├─ Find team's game
   └─ Get detailed game data

2. Analysis (2-5s)
   ├─ Determine video type (series middle/end)
   ├─ Extract key insights
   └─ Find top performances

3. ML Prediction (1-2s)
   ├─ Load player stats
   ├─ Run LSTM model
   └─ Generate explanation

4. Script Generation (10-15s)
   ├─ Hydrate prompt template
   ├─ Call GPT-4o API
   └─ Validate script structure

5. Audio Synthesis (5-10s)
   ├─ Send script to ElevenLabs
   ├─ Download MP3
   └─ Track character usage

6. Chart Generation (2-3s)
   ├─ Create trend charts
   ├─ Generate matchup graphics
   └─ Save as PNG overlays

7. Video Assembly (20-30s)
   ├─ Load background video
   ├─ Overlay charts and text
   ├─ Mix audio track
   └─ Render final MP4

8. Upload (Optional, 30-60s)
   ├─ Authenticate with YouTube
   ├─ Upload video file
   └─ Set metadata

Total: ~60-120 seconds per video
```

### Cost Breakdown

**Per Video (Typical):**
- OpenAI (GPT-4o): $0.02 - $0.05
- ElevenLabs (TTS): $0.01 - $0.03
- **Total: $0.03 - $0.08 per video**

**Monthly (30 videos):**
- Estimated: $0.90 - $2.40/month

---

## Automation

### Daily Automation with Cron

```bash
# Edit crontab
crontab -e

# Add this line (runs daily at 6 AM)
0 6 * * * cd /Users/yunjaejung/Desktop/mlb-video-pipeline && ./mlb-env/bin/python main.py --team Yankees --upload >> logs/cron.log 2>&1
```

### GitHub Actions (Recommended)

Create `.github/workflows/daily-video.yml`:

```yaml
name: Daily MLB Video

on:
  schedule:
    - cron: '0 10 * * *'  # 10 AM UTC = 6 AM ET
  workflow_dispatch:  # Manual trigger

jobs:
  generate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Generate video
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ELEVENLABS_API_KEY: ${{ secrets.ELEVENLABS_API_KEY }}
        run: python main.py --team Yankees --upload
```

---

## Customization

### Change Video Style

Edit `src/video/video_assembler.py`:

```python
# Change background color
bg_clip = ColorClip(size=(1080, 1920), color=(10, 10, 30), duration=duration)

# Change text color
draw.text((x, y), text, fill="cyan")  # Try: "yellow", "white", "#00ff00"
```

### Modify Script Template

Edit `src/content/templates/series_middle.txt`:

```
You are a hype sports commentator for TikTok/Shorts.

Create a 60-second script with this structure:
1. HOOK (5s): Shocking stat or question
2. RECAP (30s): Game highlights
3. PREDICTION (20s): Tomorrow's game preview
4. CTA (5s): "Follow for daily MLB content!"

Tone: Energetic, fast-paced, use emojis sparingly.
```

### Add New Team

The system automatically supports all 30 MLB teams. Just use the team name:

```bash
python main.py --team "Mariners"
python main.py --team "Guardians"
```

---

## Troubleshooting

### "No games found"

**Cause:** No game scheduled for that team on that date.

**Solution:**
```bash
# Check schedule first
python -c "
from src.data import MLBDataFetcher
fetcher = MLBDataFetcher()
games = fetcher.get_schedule(start_date='2024-07-04', end_date='2024-07-04')
for g in games:
    print(f\"{g['away_name']} @ {g['home_name']}\")
"
```

### "Audio generation failed"

**Cause:** ElevenLabs API issue or quota exceeded.

**Solution:**
1. Check API key: `echo $ELEVENLABS_API_KEY`
2. Check quota at elevenlabs.io
3. View error in `logs/pipeline.log`

### "Video assembly failed"

**Cause:** MoviePy/PIL compatibility issue.

**Solution:**
```bash
# Already patched in video_assembler.py
# If issues persist, check:
pip list | grep -i pillow  # Should be 10.x
pip list | grep -i moviepy  # Should be 1.0.3
```

### Cost Alert Triggered

**Cause:** Daily spending exceeded $1.00 threshold.

**Action:**
1. Check dashboard for cost breakdown
2. Review `data/costs.jsonl`
3. Adjust threshold in `src/utils/alerts.py`:
   ```python
   self.alerts.check_cost_threshold(total_cost, threshold=2.00)  # Increase to $2
   ```

---

## Best Practices

### 1. Test Before Production

```bash
# Always dry-run first
python main.py --date 2024-07-04 --team Yankees --dry-run

# Check output
ls -lh outputs/videos/
```

### 2. Monitor Costs

```bash
# Check daily
streamlit run dashboard/app.py

# Or via CLI
python -c "
from src.utils import CostTracker
print(f'Total: ${CostTracker().get_total_cost():.2f}')
"
```

### 3. Backup Metrics

```bash
# Backup metrics weekly
cp data/metrics.jsonl backups/metrics-$(date +%Y%m%d).jsonl
cp data/costs.jsonl backups/costs-$(date +%Y%m%d).jsonl
```

### 4. Keep Videos Organized

```bash
# Archive old videos monthly
mkdir -p archive/2024-07
mv outputs/videos/2024-07-*.mp4 archive/2024-07/
```

---

## Next Steps

1. **Run your first video**: `python main.py --team Yankees`
2. **Check the dashboard**: `streamlit run dashboard/app.py`
3. **Set up automation**: Add cron job or GitHub Action
4. **Monitor costs**: Review dashboard weekly
5. **Customize**: Modify templates and styles to your preference

For issues, check `logs/pipeline.log` or create an issue on GitHub.
