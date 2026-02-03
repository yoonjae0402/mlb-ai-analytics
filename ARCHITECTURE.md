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
│  Watch:   python main.py --watch --team Yankees --cinematic         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR: src/pipeline.py                                       │
│  • Coordinates all modules                                           │
│  • Tracks timing and metrics                                         │
│  • Handles errors and alerts                                         │
│  • Standard mode (run_for_date) or Cinematic mode                    │
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
│ MLBDataFetch │  │ LSTM Model   │  │ Gemini 2.0   │
│ SeriesTrack  │  │ Explainer    │  │ Google TTS   │
│ Analyzer     │  │ Trainer      │  │ Nano Banana  │
│ GameWatcher  │  │              │  │ CostTracker  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                  │
       └─────────────────┴──────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │         DATA FLOW              │
        │  Game Data → Analysis →        │
        │  Prediction → Script → Audio   │
        │  → Images (cinematic) → Video  │
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
│ AssetMgr     │                  │ GameWatcher  │
│ VideoAsm     │                  │ Monitoring   │
│ Cinematic    │                  │              │
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
        │  • CostTracker                 │
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
                    │ Gemini 2.0      │──► Script (text/JSON)
                    └─────────────────┘
                             │
                     ┌───────┴───────┐
                     ▼               ▼
            ┌──────────────┐  ┌──────────────┐
            │ Google TTS / │  │ Nano Banana  │
            │ Qwen3-TTS   │  │ (cinematic)  │
            └──────┬───────┘  └──────┬───────┘
                   │                 │
                   │    Audio (MP3)  │    Images (PNG)
                   └────────┬────────┘
                            ▼
                    ┌─────────────────┐
                    │ ChartGenerator  │──► Charts (PNG)
                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ VideoAssembler/ │──► Video (MP4)
                    │ CinematicEngine │
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
       ├─► ScriptGenerator ───────► Gemini API
       ├─► AudioGenerator ────────► Google Cloud TTS / Qwen3-TTS
       ├─► ImageGenerator ────────► Nano Banana API
       ├─► ChartGenerator
       ├─► VideoAssembler
       ├─► CinematicEngine
       ├─► AssetManager ──────────► MLB CDN
       ├─► CostTracker ───────────► data/costs.jsonl
       ├─► MetricsCollector ──────► data/metrics.jsonl
       └─► AlertManager ──────────► Email/Slack

  └─► GameWatcher (--watch mode)
       ├─► MLBDataFetcher ────────► MLB Stats API (polling)
       └─► PipelineOrchestrator ──► (triggered on game completion)
```

---

## File System Layout

```
mlb-video-pipeline/
│
├── main.py                    # Entry point
├── src/
│   ├── pipeline.py            # Orchestrator
│   ├── watcher.py             # Live game watcher
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
│   │   ├── image_generator.py
│   │   └── templates/
│   ├── audio/                 # Phase 3 (TTS)
│   │   ├── tts_engine.py
│   │   ├── google_tts.py
│   │   └── tts_cache.py
│   ├── video/                 # Phase 4
│   │   ├── video_assembler.py
│   │   ├── cinematic_engine.py
│   │   ├── chart_generator.py
│   │   ├── asset_manager.py
│   │   └── templates.py
│   ├── upload/                # Phase 5
│   │   ├── youtube.py
│   │   └── youtube_uploader.py
│   └── utils/                 # Phase 6
│       ├── cost_tracker.py
│       ├── metrics.py
│       ├── alerts.py
│       ├── logger.py
│       ├── exceptions.py
│       └── validators.py
│
├── dashboard/                 # Phase 6
│   ├── app.py
│   └── streamlit_app.py
│
├── data/
│   ├── cache/                 # API responses
│   ├── charts/                # Generated charts
│   ├── metrics.jsonl          # Run logs
│   └── costs.jsonl            # Cost logs
│
├── outputs/
│   ├── audio/                 # Audio files
│   ├── images/                # AI-generated images
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
│ • Google Gemini 2.0 Flash (Script gen)      │
│ • Google Cloud TTS Neural2 (Audio)          │
│ • Qwen3-TTS Local (Free audio fallback)     │
│ • Nano Banana API (Cinematic AI images)     │
│ • YouTube Data API (Free upload)            │
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
│ • google-genai (Gemini API)                 │
│ • google-cloud-texttospeech (TTS)           │
│ • MLB-StatsAPI (Data Fetching)              │
│ • Pydantic (Settings Validation)            │
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
0:15    Script Generation       Call Gemini 2.0 Flash
0:20    Audio + Assets          Google TTS / Qwen3-TTS + chart gen (parallel)
0:30    Video Assembly          Render with MoviePy
1:00    Upload (Optional)       YouTube API
1:30    Metrics Logging         Save to JSONL
1:31    Complete                Return video path
```

---

## Cost Breakdown

```
Component              Cost/Video    Monthly (30 videos)
─────────              ──────────    ───────────────────
Gemini 2.0 Flash       ~$0.001       ~$0.03
Google Cloud TTS       ~$0.01-0.02   ~$0.30-0.60
Nano Banana (cinematic) ~$0.00       ~$0.00
MLB API                Free          Free
YouTube Upload         Free          Free
Video Processing       Free          Free
─────────────────────────────────────────────────────
TOTAL                  ~$0.01-0.03   ~$0.03-0.90
```

Note: Costs vary by usage. Google TTS caching reduces repeat costs significantly.
Local Qwen3-TTS fallback is completely free.

---

## Monitoring Flow

```
Pipeline Execution
        │
        ├─► MetricsCollector.log_pipeline_run()
        │        │
        │        └─► data/metrics.jsonl
        │
        ├─► CostTracker.track_cost()
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
- Fully automated video generation
- Zero manual intervention required
- Cost-effective with smart caching
- Scalable to multiple teams
- Comprehensive monitoring
- Easy customization
