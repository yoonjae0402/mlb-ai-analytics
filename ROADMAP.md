# Real-Time MLB Cinematic Video Roadmap

## Goal
Detect end of MLB games via API and instantly generate cinematic video recaps using Nano Banana (Images), Gemini 2.0 Flash (Script), and MoviePy (Dynamic Motion).

---

## Phase 1: Real-Time Detection & Trigger logic
**Objective**: Eliminate manual intervention. The system watches games and starts itself.
- [x] **Game Watcher Component** (`src/watcher.py`)
    - Implement polling of MLB Stats API (`v1/schedule`) every 2 minutes.
    - Logic to detect status changes: `In Progress` -> `Final`.
    - Callback system to trigger the `PipelineOrchestrator`.

## Phase 2: High-Fidelity Content Generation
**Objective**: Replace generic assets with 8k AI-generated visuals and "Hype" commentary.
- [x] **Script Generation Upgrade** (`src/content/script_generator.py`)
    - [x] Integrate **Gemini 2.0 Flash** with specific "Real-Time MLB Video Producer" system prompt.
    - [x] Enforce rigid JSON output: `{"video_metadata": ..., "scenes": [...]}`.
    - [x] Ensure specific "motion_type" fields are generated for every scene.
- [x] **Image Generation Module** (`src/content/image_generator.py`)
    - [x] Build **Nano Banana** API client.
    - [x] Implement caching to prevent regenerating images for same prompts.
    - [x] Handling 8k resolution downloads and storage in `outputs/images/`.

## Phase 3: Cinematic Video Engine
**Objective**: "Fake Video" effect. Turn static images into dynamic scenes.
- [x] **Instant Recap Engine** (`src/video/cinematic_engine.py`)
    - [x] **Motion Logic**:
        - `zoom_in`: 100% -> 112% scale over duration.
        - `zoom_out`: 112% -> 100% scale.
        - `pan_left`/`pan_right`: Linear position interpolation.
    - [x] **Scene Assembly**:
        - Composite `ImageClip` (Background) + `TextClip` (Captions) + `AudioClip` (TTS).
        - Sync duration exactly to TTS audio length.

## Phase 4: Integration & Deployment
**Objective**: End-to-end automation from "Game Over" to YouTube.
- [x] **Pipeline Orchestration** (`src/pipeline.py`)
    - Updated with `run_cinematic_for_date(date, team)` flow:
      1. Fetch Game Data.
      2. Generate Script (Gemini - JSON).
      3. Generate Audio (Local Qwen3-TTS) + Generate Images (Nano Banana - Parallelized).
      4. Render Video (MoviePy Cinematic Engine).
      5. Upload to YouTube.
- [x] **Youtube Uploader**
    - `src/upload/youtube_uploader.py` handles authentication and metadata (Tags, Title, Description from Script JSON).

## Phase 5: Optimization
- [x] **Latency Reduction**: Parallelize Image Generation (5 images at once via ThreadPoolExecutor).
- [x] **Cost Tracking**: Monitor Nano Banana and Gemini usage via upgraded CostTracker.

## Phase 6: Audio Modernization (Google Cloud TTS)
**Objective**: Replace slow local Qwen3-TTS with ultra-fast, high-quality Google Cloud TTS while maintaining cost control.
- [x] **Google Cloud Integration**:
    - Build `GoogleTTSGenerator` with Neural2/Standard voice support.
    - Implement robust error handling and retries.
- [x] **Smart Caching System**:
    - Hash-based caching (Text + Voice ID) to eliminate 50%+ of calls.
    - Automatic expiration (90 days) to manage disk space.
- [x] **Cost Control**:
    - Daily/Monthly usage limits (e.g., $2.00/month cap).
    - Detailed CSV reporting for audit.

