# Real-Time Screen NSFW Blur

A lightweight, silent Windows background overlay that automatically detects and blurs explicit content (both real-life and anime/2D) on your screen in real time. All detection runs **100% locally** on your machine—no screenshots or data are ever sent to external servers.

## Features

- **Dual AI Detection:** Uses **NudeNet** for real-life content and **`dghs-imgutils`** for anime/2D NSFW detection.
- **Ghost Mode:** Runs silently as a frameless, click-through overlay with no taskbar icon, no console window, and hidden from `Alt + Tab`.
- **Small & Multiple Item Detection:** Scans your screen using an overlapping 3×3 grid (plus a full-screen pass) to catch small thumbnails and multiple regions at once.
- **Low-Latency Performance:** Caches static tiles so only parts of the screen that actually change get rescanned, paired with a fast downscale/upscale blur.
- **Anti-Flicker & DPI Aware:** Excludes the overlay from Windows screen capture (`WDA_EXCLUDEFROMCAPTURE`) so blurs don't flicker, and handles Windows display scaling (125%, 150%, etc.) automatically.

---

## Requirements

- **OS:** Windows 10 / 11 (uses Win32 APIs for click-through, anti-flicker, and global hotkeys)
- **Python:** 3.10+ (tested on Python 3.13)
- **Internet Connection:** Required on the first run only to download the local ONNX models.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Uranium227/Screen-Blur
   cd Screen-Blur
   pip install requirements.txt
