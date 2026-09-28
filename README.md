# Any2VR 🥽
### Turn 2D photos and videos into 3D and VR180. Free, local, and private.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Models: open source](https://img.shields.io/badge/models-Apache--2.0-2ea44f.svg)](#-depth-models)

Any2VR converts regular photos, videos, and whole photo albums into stereoscopic 3D:
**VR180** for headsets (Meta Quest, Pico, Apple Vision Pro, Skybox, DeoVR), **side-by-side** for
3D TVs and monitors, and **anaglyph** for red/cyan glasses.

Everything runs on your own computer. No account, no upload, no watermark.

---

## ✨ Features

- 🔒 **Private by design.** Your files never leave your machine.
- 🧠 **Open-source depth AI.** Choose between fast and high-quality models, all Apache-2.0.
- 🎯 **Auto focus.** Anchors the background at screen depth and lets subjects pop forward, avoiding
  ghosted edges and eye strain.
- 👁️ **Wiggle 3D preview.** See the depth on a normal monitor before you convert.
- ⚡ **5-second samples.** Render a short clip from any point in a video to check it in your headset.
- 📁 **Photo albums.** Convert a whole folder in one go; unreadable files are skipped, not fatal.
- 🎞️ **Fast video pipeline.** Decoding, AI, and encoding run in parallel; NVIDIA NVENC is used
  automatically when available; audio is kept.
- 🌐 **Headset-ready files.** Output names (`_180_SBS`, `_3DH_SBS`, …) and stereo metadata let
  players detect the layout automatically.

---

## 🚀 Install (Windows)

1. Install **Python 3.10+** and **FFmpeg** (`winget install Gyan.FFmpeg`).
2. Get the code:
   ```powershell
   git clone https://github.com/mdelaguera/vr3d-studio.git
   cd vr3d-studio
   ```
3. Install PyTorch **with CUDA** for GPU speed (pick your CUDA version on [pytorch.org](https://pytorch.org/get-started/locally/)):
   ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   ```
4. Install the rest: `pip install -r requirements.txt`
5. Start: double-click `run_converter.bat` or run `python main.py`.

Models download automatically the first time you pick one and are cached by Hugging Face.

---

## 🕹️ Using the app

1. **Open a source:** drag a video, photo, or folder onto the window, or use **Open file** (`Ctrl+O`) / **Open folder** (`Ctrl+Shift+O`).
2. **Pick model & format:** VR180 for headsets, Full SBS for 3D TVs. Photos switch to Full SBS automatically.
3. **Tune the 3D effect:** start from a preset (Soft → Extreme), then adjust *Depth strength* and *Focus plane*.
4. **Preview:** **Wiggle 3D** shows depth without glasses. Keys `1`–`8` switch views, `F5` refreshes.
5. **Videos:** scrub to a scene and click **5 s sample** to test before the full render.
6. **Convert** (`Ctrl+Enter`). `Esc` cancels.

Settings are remembered between sessions.

---

## 🧠 Depth models

Every model we ship is licensed for commercial use.

| Model | Speed | Download | License | Best for |
|---|---|---|---|---|
| **Depth Anything V2 Small** (default) | fast | ~100 MB | Apache-2.0 | Video, everyday use |
| **DPT-Hybrid (MiDaS)** | fast | ~490 MB | Apache-2.0 | Smooth landscapes |
| **Marigold LCM** | slow | ~1.7 GB | Apache-2.0 | Detailed photos |
| **Hybrid** (Depth Anything + Marigold) | slow | ~1.8 GB | Apache-2.0 | Maximum quality photos |

`python main.py --list-models` prints the same list.

---

## ⌨️ Command line

```bash
python main.py video.mp4 -m vr180                      # video to VR180
python main.py "C:/Photos/Trip" -o "C:/Photos/Trip_3D" # whole album (Full SBS)
python main.py photo.jpg --model hybrid --strength 0.05
```

| Option | Meaning |
|---|---|
| `-m, --mode` | `vr180`, `sbs_full` (default), `sbs_half`, `anaglyph`, `depth_only` |
| `--model` | `da2-small` (default), `dpt-hybrid`, `marigold-lcm`, `hybrid` |
| `--strength` | 3D strength, default `0.035` ("Natural") |
| `--focus` | Fixed focus plane 0.1–0.9 (default: automatic) |
| `--fov` | VR180 field of view, default `110` |
| `--swap-eyes`, `--no-nvenc`, `--no-smoothing` | Switches |

---

## 🗂️ Project structure

```
engine/            conversion engine (no UI; used by the app, CLI, and future cloud workers)
  settings.py        output formats, stereo settings, file naming
  depth/             model catalog + backends (transformers, diffusers, hybrid)
  stereo.py          depth -> left/right eyes (two-layer warp, push-pull hole fill)
  vr180.py           equirectangular projection (cached remap tables)
  temporal.py        motion-adaptive depth smoothing for video
  media.py           image/video IO, threaded ffmpeg encoder, audio mux
  pipeline.py        Converter: frame / image / video / folder
  cli.py             command line
gui/app.py         desktop app (CustomTkinter)
tests/             pytest suite (`pytest`; `pytest -m model` also tests a real model)
```

---

## 🙏 Credits

Started as a fork of [grezoo/vr3d-studio](https://github.com/grezoo/vr3d-studio) and since rebuilt.
Depth models by the [Depth Anything](https://github.com/DepthAnything/Depth-Anything-V2),
[Marigold](https://marigoldmonodepth.github.io/), and [MiDaS/DPT](https://github.com/isl-org/MiDaS) teams.

## 📜 License

[MIT](LICENSE). Model weights are downloaded from Hugging Face under their own licenses (all Apache-2.0).
