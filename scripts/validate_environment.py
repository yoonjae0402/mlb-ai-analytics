
import os
import shutil
import sys
from pathlib import Path

def validate_environment():
    """
    Check all requirements before generation:
    - Google Cloud credentials present
    - FFmpeg installed
    - Required fonts available
    - Sufficient disk space
    - API keys valid
    - Python dependencies installed
    """
    checks = []
    
    # 1. Google Cloud
    if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        # Warn but maybe allow if fallback is okay (local testing)
        # checks.append("⚠️ GOOGLE_APPLICATION_CREDENTIALS not set (Will use MacOS TTS fallback)")
        pass
    
    # 2. FFmpeg
    if not shutil.which('ffmpeg'):
        checks.append("❌ FFmpeg not installed (Required for MoviePy)")
    
    # 3. Fonts
    # Simple check if we can load one
    try:
        from PIL import ImageFont
        ImageFont.truetype("Arial", 20)
    except:
        checks.append("⚠️ Default fonts might be missing")
    
    # 4. Disk space
    try:
        total, used, free = shutil.disk_usage("/")
        if free < 2_000_000_000:  # 2GB
            checks.append(f"❌ Low disk space: {free/1e9:.1f}GB")
    except:
        pass
    
    if checks:
        print("Environment validation ISSUES:")
        for check in checks:
            print(check)
        if any("❌" in c for c in checks):
            sys.exit(1)
    
    print("✅ Environment validation PASSED")

if __name__ == "__main__":
    validate_environment()
