"""
ComfyUI ML Nodes

A collection of ComfyUI custom nodes for:
- Saving images/videos without workflow metadata
- GPU-accelerated frame rate resampling (e.g., 25fps to 16fps)
- Multiple interpolation algorithms (blend, minterpolate, framestep)

Nodes included:
- ML Save Image (No Metadata)
- ML Save Image (Clean Metadata)
- ML Save Video (No Metadata)
- ML Frame Rate Resampler
- ML Frame Rate Resampler (GPU)
"""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

__author__ = "ML Custom Nodes"
__version__ = "0.2.0"
__license__ = "MIT"

from .src.comfyui_ML_nodes.nodes import NODE_CLASS_MAPPINGS
from .src.comfyui_ML_nodes.nodes import NODE_DISPLAY_NAME_MAPPINGS

# Verify ffmpeg availability on import (optional, non-blocking)
import subprocess
import sys

try:
    result = subprocess.run(
        ["ffmpeg", "-version"],
        capture_output=True,
        timeout=5
    )
    if result.returncode == 0:
        print("[ComfyUI-ML-Nodes] ffmpeg detected - video and frame rate nodes will work")
    else:
        print("[ComfyUI-ML-Nodes] Warning: ffmpeg may not be properly installed")
except (FileNotFoundError, subprocess.TimeoutExpired):
    print("[ComfyUI-ML-Nodes] Warning: ffmpeg not found. Video and frame rate nodes require ffmpeg.")
    print("[ComfyUI-ML-Nodes] Install: https://ffmpeg.org/download.html")
except Exception as e:
    # Don't block loading if check fails
    pass
