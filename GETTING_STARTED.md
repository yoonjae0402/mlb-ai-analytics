# Getting Started with MLB Video Pipeline

Welcome! This guide will help you generate your first automated MLB video in under 10 minutes.

## Prerequisites

- Python 3.11 or higher
- Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))
- macOS, Linux, or Windows with WSL

## Step 1: Setup (5 minutes)

### 1.1 Navigate to Project

```bash
cd ~/Desktop/mlb-video-pipeline
```

### 1.2 Activate Virtual Environment

```bash
source mlb-env/bin/activate

# You should see (mlb-env) in your terminal prompt
```

### 1.3 Configure API Keys

```bash
# Copy the example environment file
cp .env.example .env

# Edit with your favorite editor
nano .env  # or: code .env, vim .env, etc.
```

Add your API key:
```bash
GEMINI_API_KEY=...your-key-here...
```

Note: Audio uses local Qwen3-TTS (no API key needed!)

Save and exit (Ctrl+X, then Y, then Enter in nano).

### 1.4 Verify Setup

```bash
# This should show "✅ Installation OK"
python -c "import src; print('✅ Installation OK')"

# Run tests (should see all passing)
./mlb-env/bin/pytest tests/ -v
```

## Step 2: Generate Your First Video (3 minutes)

### 2.1 Simple Test Run

```bash
# Generate a video for yesterday's Yankees game
python main.py --team Yankees
```

**What you'll see:**
```
2026-02-03 00:15:00 - INFO - Starting pipeline for Yankees on 2026-02-02
2026-02-03 00:15:05 - INFO - Step 1: Fetching game data...
2026-02-03 00:15:10 - INFO - Step 2: Determining video type...
2026-02-03 00:15:12 - INFO - Step 3: Analyzing game...
2026-02-03 00:15:14 - INFO - Step 4: Generating prediction...
2026-02-03 00:15:16 - INFO - Step 5: Generating script...
2026-02-03 00:15:30 - INFO - Step 6: Generating audio (local TTS)...
2026-02-03 00:15:40 - INFO - Step 7: Generating charts...
2026-02-03 00:15:45 - INFO - Step 8: Assembling video...
2026-02-03 00:16:15 - INFO - ✅ Pipeline completed successfully!
2026-02-03 00:16:15 - INFO - Total API cost: $0.0050
```

### 2.2 Find Your Video

```bash
# List generated videos
ls -lh outputs/videos/

# You should see something like:
# 2026-02-02_Yankees.mp4
```

### 2.3 Watch Your Video

```bash
# On macOS
open outputs/videos/2026-02-02_Yankees.mp4

# On Linux
xdg-open outputs/videos/2026-02-02_Yankees.mp4

# On Windows (WSL)
explorer.exe outputs/videos/2026-02-02_Yankees.mp4
```

## Step 3: View the Dashboard (2 minutes)

### 3.1 Install Dashboard Dependencies

```bash
pip install streamlit plotly
```

### 3.2 Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Your browser will automatically open to `http://localhost:8501`

### 3.3 Explore the Dashboard

- **Overview**: See your recent video generation
- **Cost Analysis**: Check how much you spent (~$0.001-0.01 typical)
- **Performance**: View processing time
- **Video Gallery**: Preview your video in the browser

## What Just Happened?

Your pipeline just:

1. ✅ Fetched yesterday's Yankees game data from MLB API
2. ✅ Analyzed the game for key insights
3. ✅ Generated an AI prediction for the next game
4. ✅ Created a 60-second script using Gemini 2.0 Flash
5. ✅ Synthesized professional voiceover with Qwen3-TTS (local, free)
6. ✅ Generated trend charts showing player stats
7. ✅ Assembled a vertical video (1080x1920) optimized for TikTok/Shorts
8. ✅ Logged metrics and costs
9. ✅ Saved the final video to `outputs/videos/`

**Total time:** ~60-120 seconds  
**Total cost:** ~$0.001-0.01

## Next Steps

### Try Different Teams

```bash
python main.py --team "Red Sox"
python main.py --team Dodgers
python main.py --team Cubs
```

### Generate for a Specific Date

```bash
python main.py --date 2024-07-04 --team Yankees
```

### Upload to YouTube

```bash
# First time: Set up YouTube credentials
# Download client_secrets.json from Google Cloud Console
# Place it in project root

python main.py --team Yankees --upload
```

### Automate Daily Videos

```bash
# Edit crontab
crontab -e

# Add this line (runs daily at 6 AM)
0 6 * * * cd /Users/yunjaejung/Desktop/mlb-video-pipeline && ./mlb-env/bin/python main.py --team Yankees --upload >> logs/cron.log 2>&1
```

## Common Issues

### "No games found"

**Problem:** The team didn't play on that date.

**Solution:** Check the MLB schedule or try a different date:
```bash
python main.py --date 2024-07-04 --team Yankees
```

### "Audio generation failed"

**Problem:** Qwen3-TTS model loading issue.

**Solution:**
1. Check if you have enough disk space (model is ~2GB)
2. Verify Python version is 3.11+
3. View detailed error in `logs/pipeline.log`
4. Try reinstalling: `pip install qwen-tts --force-reinstall`

### "ModuleNotFoundError"

**Problem:** Virtual environment not activated.

**Solution:**
```bash
source mlb-env/bin/activate
```

## Understanding Costs

### Per Video
- Gemini 2.0 Flash: ~$0.001 - $0.01
- Qwen3-TTS (local): $0.00 (free!)
- **Total: ~$0.001 - $0.01**

### Monthly (30 videos)
- **Total: ~$0.03 - $0.30**

### Cost Savings
- ~90% cheaper than OpenAI + ElevenLabs
- Audio is completely free (local processing)
- Check Gemini pricing: [ai.google.dev/pricing](https://ai.google.dev/pricing)

## Documentation

Now that you've generated your first video, explore these guides:

1. **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Comprehensive usage instructions
2. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Command cheat sheet
3. **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture diagrams
4. **[ROADMAP.md](ROADMAP.md)** - Strategic overview and future plans
5. **[dashboard/README.md](dashboard/README.md)** - Dashboard documentation

## Get Help

- **Logs:** Check `logs/pipeline.log` for detailed error messages
- **Metrics:** Review `data/metrics.jsonl` for run history
- **Costs:** Monitor `data/costs.jsonl` for API spending
- **Dashboard:** Launch `streamlit run dashboard/app.py` for visual monitoring

## Congratulations! 🎉

You've successfully:
- ✅ Set up the MLB Video Pipeline
- ✅ Generated your first automated video
- ✅ Viewed the monitoring dashboard
- ✅ Understood the costs

You're now ready to automate MLB video generation!

---

**Next:** Try generating videos for different teams, dates, or set up daily automation.
