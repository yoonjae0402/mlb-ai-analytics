# MLB Video Pipeline - Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MLB VIDEO PIPELINE                           │
│                    (Fully Automated Video Generation)                │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  ENTRY POINT: main.py                                                │
│  Command: python main.py --date 2024-07-04 --team Yankees --upload  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR: src/pipeline.py                                       │
│  • Coordinates all modules                                           │
│  • Tracks timing and metrics                                         │
│  • Handles errors and alerts                                         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   PHASE 1    │  │   PHASE 2    │  │   PHASE 3    │
│ FOUNDATION   │  │  ML MODEL    │  │   CONTENT    │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                  │
       ▼                 ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ MLBDataFetch │  │ LSTM Model   │  │ GPT-4o       │
│ SeriesTrack  │  │ Explainer    │  │ ElevenLabs   │
│ Analyzer     │  │ Trainer      │  │ CostTracker  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                  │
       └─────────────────┴──────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │         DATA FLOW              │
        │  Game Data → Analysis →        │
        │  Prediction → Script → Audio   │
        └────────────────┬───────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
┌──────────────┐                  ┌──────────────┐
│   PHASE 4    │                  │   PHASE 5    │
│    VIDEO     │                  │  AUTOMATION  │
└──────┬───────┘                  └──────┬───────┘
       │                                 │
       ▼                                 ▼
┌──────────────┐                  ┌──────────────┐
│ ChartGen     │                  │ YouTube API  │
│ AssetMgr     │                  │ Scheduling   │
│ VideoAsm     │                  │ Monitoring   │
└──────┬───────┘                  └──────┬───────┘
       │                                 │
       └─────────────────┬───────────────┘
                         │
                         ▼
                ┌──────────────────┐
                │   PHASE 6        │
                │  MONITORING      │
                └────────┬─────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │  • MetricsCollector            │
        │  • AlertManager                │
        │  • Streamlit Dashboard         │
        └────────────────────────────────┘
```

---

## Data Flow Diagram

```
INPUT                    PROCESSING                      OUTPUT
─────                    ──────────                      ──────

Date + Team
    │
    ▼
┌─────────────┐
│ MLB API     │──────► Game Data (JSON)
└─────────────┘              │
                             ▼
                    ┌─────────────────┐
                    │ SeriesTracker   │──► Video Type
                    │ Analyzer        │──► Key Insights
                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ LSTM Model      │──► Prediction
                    │ Explainer       │──► Reasoning
                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ GPT-4o          │──► Script (text)
                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ ElevenLabs      │──► Audio (MP3)
                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ ChartGenerator  │──► Charts (PNG)
                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ VideoAssembler  │──► Video (MP4)
                    └─────────────────┘         │
                             │                  │
                             ▼                  ▼
                    ┌─────────────────┐   ┌──────────┐
                    │ YouTubeUploader │   │ Local    │
                    └─────────────────┘   │ Storage  │
                             │            └──────────┘
                             ▼
                        YouTube.com
```

---

## Module Dependencies

```
main.py
  └─► PipelineOrchestrator
       ├─► MLBDataFetcher ────────► MLB Stats API
       ├─► SeriesTracker
       ├─► MLBStatsAnalyzer
       ├─► PredictionDataProcessor
       ├─► PlayerPerformanceLSTM
       ├─► PredictionExplainer
       ├─► ScriptGenerator ───────► OpenAI API
       ├─► AudioGenerator ────────► ElevenLabs API
       ├─► ChartGenerator
       ├─► VideoAssembler
       ├─► AssetManager ──────────► MLB CDN
       ├─► CostTracker ───────────► data/costs.jsonl
       ├─► MetricsCollector ──────► data/metrics.jsonl
       └─► AlertManager ──────────► Email/Slack
```

---

## File System Layout

```
mlb-video-pipeline/
│
├── main.py                    # Entry point
├── src/
│   ├── pipeline.py            # Orchestrator
│   ├── data/                  # Phase 1
│   │   ├── fetcher.py
│   │   ├── series_tracker.py
│   │   └── predictor_data.py
│   ├── analysis/              # Phase 1
│   │   └── analyzer.py
│   ├── models/                # Phase 2
│   │   ├── classifier.py
│   │   ├── trainer.py
│   │   ├── explainer.py
│   │   └── dataset.py
│   ├── content/               # Phase 3
│   │   ├── script_generator.py
│   │   ├── audio_generator.py
│   │   └── templates/
│   ├── video/                 # Phase 4
│   │   ├── video_assembler.py
│   │   ├── chart_generator.py
│   │   └── asset_manager.py
│   ├── upload/                # Phase 5
│   │   └── youtube_uploader.py
│   └── utils/                 # Phase 6
│       ├── cost_tracker.py
│       ├── metrics.py
│       └── alerts.py
│
├── dashboard/                 # Phase 6
│   └── app.py
│
├── data/
│   ├── cache/                 # API responses
│   ├── charts/                # Generated charts
│   ├── metrics.jsonl          # Run logs
│   └── costs.jsonl            # Cost logs
│
├── outputs/
│   ├── audio/                 # MP3 files
│   └── videos/                # MP4 files
│
└── logs/
    └── pipeline.log           # Execution logs
```

---

## Technology Stack

```
┌─────────────────────────────────────────────┐
│           EXTERNAL SERVICES                 │
├─────────────────────────────────────────────┤
│ • MLB Stats API (Free)                      │
│ • OpenAI GPT-4o ($0.02-0.05/video)          │
│ • ElevenLabs TTS ($0.01-0.03/video)         │
│ • YouTube Data API (Free)                   │
│ • MLB CDN (Logos/Headshots, Free)           │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│           PYTHON LIBRARIES                  │
├─────────────────────────────────────────────┤
│ • PyTorch (ML Models)                       │
│ • MoviePy (Video Editing)                   │
│ • Matplotlib/Seaborn (Charts)               │
│ • Streamlit (Dashboard)                     │
│ • Plotly (Interactive Charts)               │
│ • Pillow (Image Processing)                 │
│ • Requests (HTTP)                           │
│ • MLB-StatsAPI (Data Fetching)              │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│           INFRASTRUCTURE                    │
├─────────────────────────────────────────────┤
│ • Python 3.11+                              │
│ • FFmpeg (Video Processing)                 │
│ • Git (Version Control)                     │
│ • Virtual Environment (Isolation)           │
└─────────────────────────────────────────────┘
```

---

## Execution Timeline

```
Time    Stage                   Activity
────    ─────                   ────────
0:00    Start                   User runs main.py
0:01    Data Fetching           Query MLB API
0:10    Analysis                Extract insights
0:12    ML Prediction           Run LSTM model
0:15    Script Generation       Call GPT-4o
0:30    Audio Synthesis         Call ElevenLabs
0:40    Chart Generation        Create PNG overlays
0:45    Video Assembly          Render with MoviePy
1:15    Upload (Optional)       YouTube API
1:45    Metrics Logging         Save to JSONL
1:46    Complete                Return video path
```

---

## Cost Breakdown

```
Component              Cost/Video    Monthly (30 videos)
─────────              ──────────    ───────────────────
OpenAI (GPT-4o)        $0.02-0.05    $0.60-1.50
ElevenLabs (TTS)       $0.01-0.03    $0.30-0.90
MLB API                Free          Free
YouTube Upload         Free          Free
Video Processing       Free          Free
─────────────────────────────────────────────────────
TOTAL                  $0.03-0.08    $0.90-2.40
```

---

## Monitoring Flow

```
Pipeline Execution
        │
        ├─► MetricsCollector.log_pipeline_run()
        │        │
        │        └─► data/metrics.jsonl
        │
        ├─► CostTracker.log_*()
        │        │
        │        └─► data/costs.jsonl
        │
        └─► AlertManager.check_cost_threshold()
                 │
                 ├─► Email (if > $1.00)
                 └─► Slack (if > $1.00)

Dashboard (Streamlit)
        │
        ├─► Read data/metrics.jsonl
        ├─► Read data/costs.jsonl
        └─► Display charts & tables
```

---

This architecture enables:
- ✅ Fully automated video generation
- ✅ Zero manual intervention required
- ✅ Cost-effective ($0.03-0.08 per video)
- ✅ Scalable to multiple teams
- ✅ Comprehensive monitoring
- ✅ Easy customization
