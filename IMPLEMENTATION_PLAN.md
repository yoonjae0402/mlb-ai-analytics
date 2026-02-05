# MLB Video Pipeline Overhaul - Implementation Plan

## Overview
Fix 7 critical issues in the viral video pipeline: repetitive images, missing subtitles, generic placeholder text, robotic TTS, wrong scores, missing player names, and reused backgrounds.

---

## Step 1: Update TTS Config & Add SSML Support

**Files to modify:**
- `config/viral_video_config.py` - Fix TTS settings (speaking_rate 1.15->1.05, pitch 1.5->0.0)
- `src/audio/google_tts_generator.py` - Add SSML generation method with emphasis, pauses, prosody
- `src/audio/google_tts.py` - Add SSML input support to the `generate()` method

**Changes:**
- Reduce speaking rate from 1.15 to 1.05 (less robotic)
- Set pitch to 0.0 (natural, was 1.5 which sounds artificial)
- Change voice from `en-US-Neural2-J` to `en-US-Neural2-D` (deeper, more natural male voice)
- Add `generate_with_ssml()` method that converts script text to SSML with:
  - `<break>` tags at sentence boundaries
  - `<emphasis>` on team names, player names, and action words
  - `<prosody>` variations for excitement
- Add audio effects profile `headphone-class-device` for better quality
- Increase sample rate to 48000Hz

---

## Step 2: Fix Script Generator for Accuracy

**Files to modify:**
- `src/content/viral_script_generator.py` - Rewrite prompt to enforce real names/stats

**Changes:**
- Rewrite `VIRAL_SYSTEM_PROMPT` to:
  - Explicitly inject team names, player names, and scores into the prompt
  - Add CRITICAL RULES section forbidding placeholder text
  - Require exact score statement: "[Team A] [score], [Team B] [score]"
  - Require player names by name with their stats
- Rewrite `_create_prompt()` to pass all data more explicitly with labels like "USE THIS EXACT NAME"
- Add `_validate_script_accuracy()` method that checks generated script for:
  - Presence of actual team names (reject if "away team" or "home team" found)
  - Presence of player names from game data
  - Presence of actual score numbers
- Update `_fallback_script()` to use real game data instead of generic "Fallback content for {key}"

---

## Step 3: Remove Word Truncation for Viral Scripts

**Files to modify:**
- `src/content/audio_generator.py` - Increase `MAX_SCRIPT_WORDS` from 80 to 300

**Changes:**
- The current 80-word limit truncates 60s scripts (which need ~280 words)
- Change `MAX_SCRIPT_WORDS = 80` to `MAX_SCRIPT_WORDS = 300`
- This allows the full viral script narration to be spoken

---

## Step 4: Create Image Budget Manager

**New file:** `src/video/image_budget_manager.py`

**Purpose:** Manage 10-image limit with intelligent allocation and text overlay variants

**Key features:**
- `ImageBudgetManager` class with 10 named slots (hook, score, celebration, next_game, cta, player_1, player_2, key_moment, standings, prediction)
- `allocate_image(slot, path)` - Assign image to slot, enforce 10 max
- `get_scene_image(scene_type)` - Map scene types to image slots (multiple scenes can share an image)
- `create_text_overlay_variant(base_image, text, position)` - Create visual variety from same base image by adding different text overlays with dark gradient backgrounds
- Fallback chain: if slot empty, use celebration image; if that's empty, use tech gradient

---

## Step 5: Create Subtitle Generator

**New file:** `src/video/subtitle_generator.py`

**Purpose:** Burn subtitles into every frame so video works without audio

**Key features:**
- `SubtitleGenerator` class
- `generate_word_timings(script, audio_duration)` - Break script into 4-word chunks with start/end times
- `create_subtitle_clips(word_timings)` - Create MoviePy `TextClip` objects for each chunk
  - White text, black stroke (3px) for readability
  - Positioned at bottom of screen (y=1650)
  - Max width 900px with word wrap
  - 0.1s crossfade in/out between chunks
- `add_subtitles_to_video(video_clip, audio_path, script)` - Composite all subtitle clips onto the video

---

## Step 6: Enhance StatRenderer with Real Data

**Files to modify:**
- `src/video/stat_renderer.py` - Fix the `render_scoreboard` scene_type bug, pass real names

**Changes:**
- Fix bug at line 77: `scene_type == "scoreboard"` should also match `"score"` (the viral script uses `"score"` as scene_type, but renderer checks for `"scoreboard"`)
- Update `render_hook()` to use actual team name from game_data (e.g., "YANKEES UPSET!" instead of generic "HOOK TEXT")
- Update `render_player_spotlight()` to show real player name + stats from game_data
- Update `render_standings()` to show actual standings impact text from game_data
- Add `render_score()` method that creates a proper scoreboard with both team names and score

---

## Step 7: Integrate Everything into Viral Engine

**Files to modify:**
- `src/video/viral_engine.py` - Add subtitle system, use image budget manager

**Changes:**
- Import and instantiate `ImageBudgetManager` and `SubtitleGenerator`
- In `render_video()`:
  1. After generating AI backgrounds, allocate them to image budget slots
  2. After compositing all scenes, concatenate the full script narration text
  3. Add burnt-in subtitles using `SubtitleGenerator.add_subtitles_to_video()`
  4. Subtitles go on top of the final composited video before audio mixing
- Pass `game_data` through to subtitle generator for accurate text

---

## Step 8: Wire Into Pipeline

**Files to modify:**
- `src/pipeline.py` - Pass emphasis words to TTS, use SSML for viral mode

**Changes:**
- In `run_viral_for_date()`, extract team/player names from game_data for TTS emphasis
- Update `_generate_scene_audio()` to support SSML generation when in viral mode
- Collect full script text from all scenes for subtitle generation

---

## Files Summary

**New files (2):**
- `src/video/image_budget_manager.py`
- `src/video/subtitle_generator.py`

**Modified files (8):**
- `config/viral_video_config.py` - TTS settings
- `src/audio/google_tts_generator.py` - SSML support
- `src/audio/google_tts.py` - SSML input support
- `src/content/viral_script_generator.py` - Accurate prompts + validation
- `src/content/audio_generator.py` - Remove word truncation
- `src/video/stat_renderer.py` - Real data in overlays, fix score bug
- `src/video/viral_engine.py` - Subtitle + image budget integration
- `src/pipeline.py` - Wire SSML + emphasis into viral pipeline

---

## Verification

1. **Script accuracy**: Run viral script generation and verify output contains real team names, player names, and scores (no "away team" or "home team" placeholders)
2. **TTS quality**: Generate audio with SSML and verify speaking rate/pitch are natural
3. **Subtitle presence**: Render video and confirm subtitles appear at bottom of every frame
4. **Image variety**: Check that 10 unique images are allocated and scenes reuse them intelligently
5. **Score bug fix**: Verify `score` scene_type renders the scoreboard correctly
6. **End-to-end**: Run `python main.py --date 2024-07-04 --team Yankees --viral` and inspect output video
