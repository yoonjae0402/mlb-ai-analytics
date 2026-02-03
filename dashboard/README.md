# MLB Video Pipeline Dashboard

Interactive monitoring dashboard for the MLB Video Pipeline.

## Features

- **📊 Overview**: Real-time metrics on pipeline runs, success rate, and costs
- **💰 Cost Analysis**: Daily breakdown of API spending (OpenAI + ElevenLabs)
- **⚡ Performance**: Processing time trends and bottleneck identification
- **🎬 Video Gallery**: Browse and preview generated videos

## Installation

```bash
# Install dashboard dependencies
pip install -r dashboard/requirements.txt
```

## Usage

```bash
# Launch the dashboard
streamlit run dashboard/app.py

# Or from project root
cd dashboard && streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

## Configuration

The dashboard reads from:
- `data/metrics.jsonl` - Pipeline execution logs
- `data/costs.jsonl` - API cost tracking
- `outputs/videos/` - Generated video files

## Screenshots

### Overview Page
- Total runs in selected time period
- Success rate percentage
- Average processing duration
- Total API costs
- Recent pipeline runs table

### Cost Analysis
- Stacked bar chart showing daily OpenAI vs ElevenLabs costs
- Cost per video average
- Monthly cost projection

### Performance
- Line chart of processing time over time
- Min/avg/max duration statistics
- Success vs failure pie chart

### Recent Videos
- List of successfully generated videos
- Video metadata (date, team, duration, cost)
- In-browser video preview

## Filters

Use the sidebar slider to adjust the time range (1-30 days).
