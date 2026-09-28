# VR 3D Platform — Design

Status: approved 2026-09-27 (owner: mdelaguera). Product name TBD (rebrand pending).

## Goals

- Own the codebase: replace everything inherited from the upstream fork with
  our own modular, more efficient implementation.
- Free, local, private desktop app as the entry point; paid cloud credits as revenue.
- Paid users buy **one credit balance** usable for both cloud conversion (launch
  first) and cloud creation (prompt → VR image/video).
- Open-source models featured on equal footing with frontier/cloud models,
  wherever the economics work. Privacy-first stance is both ethics and marketing.
- Every model we ship or serve must be licensed for commercial use.

## Privacy stance (product rule, not just copy)

- Local processing is the default and is never gated behind an account.
- Cloud jobs: inputs/outputs deleted automatically after delivery (target 24 h),
  never used for training, never shared with third parties beyond the chosen provider.
- The UI always shows where a job runs (This PC / Our GPUs / Third-party API) and
  which model + license it uses.

## Sub-projects (each: design → plan → build)

| # | Sub-project | Depends on |
|---|---|---|
| 1 | Core engine rewrite (`engine/`) | — |
| 2 | Free desktop app polish (first run, model guidance, gallery) | 1 |
| 3 | Creator + provider layer (local / cloud API / self-hosted) | 1 |
| 4 | Cloud platform: accounts, credits, metering, limits, queue, GPU provisioning | 1, 3 |
| 5 | n8n ops automation (onboarding, low-credit, idle-GPU reaper, alerts, reports) | 4 |

Chosen building blocks for 4 (all permissive licenses; verify before adopting):
SkyPilot (Vast.ai GPU provisioning, per-model GPU selection), Autumn (credits,
entitlements, Stripe), Celery (job queue), Stripe (payments), n8n (ops).
Rejected: Lago, Flexprice, Unkey (AGPL).

---

## Sub-project 1: Core engine

### Why rewrite

- Upstream code is not ours; `core/depth_anything_v2/` vendors ~60 files of a
  research repo.
- Measured bottleneck: `cv2.inpaint` (Telea) took 1.3 s of a 2.4 s stereo render at
  1080p; float compositing another ~140 ms.
- Upstream NVENC fallback never triggers (`Popen` does not fail when the encoder is
  missing), ffmpeg errors are discarded, decode/compute/encode run serially.
- Model lineup included non-commercial weights (Depth Anything V2 Base/Large).

### Package layout

```
engine/
  __init__.py      public API re-exports
  settings.py      OutputFormat, StereoSettings, OutputSettings, output naming
  depth/
    __init__.py    model catalog (license, commercial_ok, size, speed) + load_model()
    hf.py          transformers backend (Depth Anything V2 Small, DPT-Hybrid)
    marigold.py    diffusers backend (Marigold LCM)
    hybrid.py      weighted fusion of two models
  stereo.py        depth → left/right eyes; push-pull hole fill; SBS/anaglyph compose
  vr180.py         equirectangular projection with cached remap tables
  temporal.py      motion-adaptive depth smoothing with scene-cut reset
  media.py         unicode-safe image IO, prefetching video reader,
                   threaded ffmpeg writer (probed NVENC), audio mux + stereo metadata
  pipeline.py      Converter: frame / image / video / folder; progress + cancel Event
  cli.py           command line entry
```

Rules: modules talk through plain numpy arrays and the dataclasses in
`settings.py`. Only `depth/` imports torch. Only `media.py` talks to ffmpeg.
GUI, CLI, and (later) the cloud worker all call `pipeline.Converter`.

### Depth contract

`DepthModel.estimate(bgr: uint8 HxWx3) -> float32 HxW in [0, 1]`, 1 = nearest.
Normalization uses the 1st/99th percentile (robust to outliers), not min/max.

Catalog (commercial-OK only):

| id | Model | License |
|---|---|---|
| `da2-small` (default) | depth-anything/Depth-Anything-V2-Small-hf | Apache-2.0 |
| `dpt-hybrid` | Intel/dpt-hybrid-midas | Apache-2.0 |
| `marigold-lcm` | prs-eth/marigold-depth-lcm-v1-0 | Apache-2.0 |
| `hybrid` | 60% da2-small + 40% marigold-lcm | Apache-2.0 |

Excluded: Depth Anything V2 Base/Large (CC-BY-NC-4.0), Apple DepthPro (research-only).
Future: Video Depth Anything Small (Apache-2.0) for temporally stable video depth.

### Stereo algorithm (behavior-compatible with current presets)

1. Stretch depth to its 2nd–98th percentile range (percentiles on a 4× downsample).
2. Zero-parallax anchor: 5th percentile (auto focus) or the user's focus value.
3. Disparity = max(0, depth − anchor) × width × strength (subjects pop forward,
   background stays put → no background ghosting).
4. Foreground = depth > anchor + 0.08, dilated; background plate = image with
   foreground removed and filled by **push-pull pyramid fill** (replaces Telea).
5. Each eye: warp foreground + soft alpha by ±½ disparity (`cv2.remap`), blend over
   background with `cv2.blendLinear`.

### Video pipeline

Reader thread (prefetch queue) → main thread depth + stereo → writer thread (ffmpeg
stdin). Encoder chosen by a one-time probe encode (NVENC HEVC → libx264 fallback).
ffmpeg stderr captured and surfaced on failure. Audio muxed from the source,
trimmed for samples; stereo metadata flags added; encoded video kept if muxing fails.

### Error handling

- Unreadable inputs raise `ValueError` with the path; the GUI shows it.
- Cancel is a `threading.Event`; partial temp files are removed on cancel.
- Album conversion continues past bad images and returns (ok, failed) lists.

### Testing

`tests/test_engine.py` (pytest), using a synthetic depth model so tests run without
downloads: hole fill correctness, stereo shift direction and swap, output shapes per
format, VR180 map caching, temporal scene-cut reset, video round trip through ffmpeg
(frame count + audio-less mux), folder conversion with one corrupt file, cancel.
One opt-in test loads the real default model (`-m model`).
