#!/usr/bin/env python3
"""
TTS output using edge-tts (offline-capable, good voice quality).
"""

import asyncio
import io
import subprocess
import tempfile
from pathlib import Path


async def _speak_async(text: str, voice: str = "en-US-GuyNeural"):
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_path = f.name
    await communicate.save(tmp_path)
    # Play using ffplay (or fallback to mpg123/aplay)
    for player in ["ffplay -nodisp -autoexit", "mpg123", "mpv --no-video"]:
        cmd = player.split() + [tmp_path]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)
            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    Path(tmp_path).unlink(missing_ok=True)


def speak(text: str, voice: str = "en-US-GuyNeural", blocking: bool = True):
    """Speak text using edge-tts. blocking=True waits for completion."""
    print(f"[TTS] {text}")
    try:
        if blocking:
            asyncio.run(_speak_async(text, voice))
        else:
            # Fire and forget
            import threading
            t = threading.Thread(target=asyncio.run, args=(_speak_async(text, voice),),
                                 daemon=True)
            t.start()
    except Exception as e:
        print(f"[TTS error] {e}")
