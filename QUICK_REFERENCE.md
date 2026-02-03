# MLB Video Pipeline - Quick Reference

## Essential Commands

### Generate Video
```bash
# Basic usage
python main.py --team Yankees

# Specific date
python main.py --date 2024-07-04 --team "Red Sox"

# With YouTube upload
python main.py --date 2024-07-04 --team Yankees --upload

# Test run (no upload)
python main.py --date 2024-07-04 --team Yankees --dry-run
```

### Monitor Pipeline
```bash
# Launch dashboard
streamlit run dashboard/app.py

# View logs
tail -f logs/pipeline.log

# Check costs
python -c "from src.utils import CostTracker; print(f'${CostTracker().get_total_cost():.2f}')"
```

### Testing
```bash
# Run all tests
./mlb-env/bin/pytest tests/ -v

# Test specific module
./mlb-env/bin/pytest tests/video/ -v

# Verify Phase 5
./mlb-env/bin/python scripts/verify_phase5.py
```

---

## File Locations

| Type | Location |
|------|----------|
| Generated Videos | `outputs/videos/` |
| Audio Files | `outputs/audio/` |
| Charts | `data/charts/` |
| Metrics Log | `data/metrics.jsonl` |
| Cost Log | `data/costs.jsonl` |
| Pipeline Logs | `logs/pipeline.log` |
| Cached Data | `data/cache/` |

---

## Common Tasks

### Check Recent Videos
```bash
ls -lht outputs/videos/ | head -10
```

### View Metrics
```python
from src.utils import MetricsCollector

metrics = MetricsCollector()
runs = metrics.get_recent_runs(days=7)

for run in runs:
    status = "✅" if run['success'] else "❌"
    print(f"{status} {run['date']} - {run['team']} ({run['duration']:.1f}s)")
```

### Calculate Costs
```python
from src.utils import MetricsCollector
from datetime import datetime, timedelta

metrics = MetricsCollector()
start = (datetime.now() - timedelta(days=30)).isoformat()
end = datetime.now().isoformat()

summary = metrics.get_cost_summary(start, end)
print(f"Last 30 days: ${summary['total']:.2f} ({summary['runs']} videos)")
print(f"  OpenAI: ${summary['openai']:.2f}")
print(f"  ElevenLabs: ${summary['elevenlabs']:.2f}")
```

### Clear Cache
```bash
rm -rf data/cache/*
```

---

## Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=...

# Optional (Alerts)
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
ALERT_EMAIL_FROM=your-email@gmail.com
ALERT_EMAIL_TO=alerts@example.com
ALERT_EMAIL_PASSWORD=...
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# Optional (YouTube)
# client_secrets.json file required
```

---

## Pipeline Stages & Timing

| Stage | Duration | Cost |
|-------|----------|------|
| Data Fetching | 5-10s | Free |
| Analysis | 2-5s | Free |
| ML Prediction | 1-2s | Free |
| Script Generation | 10-15s | $0.02-0.05 |
| Audio Synthesis | 5-10s | $0.01-0.03 |
| Chart Generation | 2-3s | Free |
| Video Assembly | 20-30s | Free |
| YouTube Upload | 30-60s | Free |
| **Total** | **60-120s** | **$0.03-0.08** |

---

## Supported Teams

All 30 MLB teams supported. Use team name or city:

```bash
# American League East
python main.py --team Yankees
python main.py --team "Red Sox"
python main.py --team "Blue Jays"
python main.py --team Orioles
python main.py --team Rays

# National League West
python main.py --team Dodgers
python main.py --team Giants
python main.py --team Padres
python main.py --team Diamondbacks
python main.py --team Rockies

# ... etc for all 30 teams
```

---

## Troubleshooting Quick Fixes

| Issue | Solution |
|-------|----------|
| "No games found" | Check date, team plays that day |
| "Audio generation failed" | Check ElevenLabs API key & quota |
| "Video assembly failed" | Check logs, verify moviepy installed |
| "Cost alert triggered" | Check dashboard, adjust threshold |
| Import errors | `source mlb-env/bin/activate` |
| Permission denied | `chmod +x main.py` |

---

## Dashboard Pages

1. **📊 Overview**: Runs, success rate, costs, recent table
2. **💰 Cost Analysis**: Daily breakdown, projections
3. **⚡ Performance**: Duration trends, statistics
4. **🎬 Video Gallery**: Browse and preview videos

---

## Automation Setup

### Cron (Daily at 6 AM)
```bash
crontab -e
# Add:
0 6 * * * cd /path/to/mlb-video-pipeline && ./mlb-env/bin/python main.py --team Yankees --upload >> logs/cron.log 2>&1
```

### GitHub Actions
See `.github/workflows/daily-video.yml` example in USAGE_GUIDE.md

---

## Cost Optimization Tips

1. **Use caching**: Don't clear `data/cache/` unnecessarily
2. **Batch processing**: Generate multiple videos in one session
3. **Monitor daily**: Check dashboard for cost spikes
4. **Set alerts**: Configure email/Slack for $1+ threshold
5. **Test locally**: Use `--dry-run` before production runs

---

## Key Files to Customize

| File | Purpose |
|------|---------|
| `src/content/templates/series_middle.txt` | Script template for mid-series games |
| `src/content/templates/series_end.txt` | Script template for series finales |
| `src/video/video_assembler.py` | Video styling (colors, fonts, layout) |
| `src/utils/alerts.py` | Alert thresholds and messages |
| `dashboard/app.py` | Dashboard customization |

---

## Support

- **Logs**: Check `logs/pipeline.log` for errors
- **Metrics**: Review `data/metrics.jsonl` for run history
- **Costs**: Monitor `data/costs.jsonl` for API usage
- **Documentation**: See `USAGE_GUIDE.md` for detailed instructions
- **Roadmap**: See `ROADMAP.md` for architecture overview
