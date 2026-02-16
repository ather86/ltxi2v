# 🎬 LTX Video Studio — Image-to-Video Generation

A **Streamlit-powered** desktop application that turns scripts and images into fully stitched videos using the **LTX 2.0 Image-to-Video (i2v)** model via [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

Write a story, provide reference images, and let the app parse scenes, generate per-scene videos with AI, maintain visual continuity, and stitch everything into a single final video — all from one UI.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Image-to-Video** | Each scene is generated from a reference image + a text prompt describing motion/action |
| **Script Parsing** | Paste a full narrative script — the AI (Ollama or Gemini) splits it into timed scenes automatically |
| **Smart Continuity** | Extracts the best frame from each generated clip and feeds it into the next scene for visual consistency |
| **Character Management** | Upload per-character reference images; the app selects the right one for each scene |
| **Auto Stitching** | All scene clips are concatenated into a polished final video |
| **VRAM Management** | Built-in purge & cooldown controls to keep generation stable on consumer GPUs |
| **Multi-LLM Support** | Choose between a local **Ollama** model or the **Google Gemini** API for script parsing & prompt refinement |

---

## 🖼️ How It Works

```
┌────────────┐     ┌──────────────┐     ┌────────────────┐     ┌────────────┐
│  Paste a   │────▶│  AI parses   │────▶│  For each scene │────▶│  Stitch &  │
│  Script    │     │  into scenes │     │  Image + Prompt │     │  Export    │
│            │     │  (Ollama /   │     │  → ComfyUI i2v  │     │  final.mp4 │
│            │     │   Gemini)    │     │  → extract frame│     │            │
└────────────┘     └──────────────┘     └────────────────┘     └────────────┘
```

1. **Write / paste** your detailed script into the UI.
2. The app sends the script to an LLM which returns structured scene data (visual prompts, dialogue, characters).
3. For each scene the app selects the best reference image (uploaded, character-specific, or last-frame continuity) and sends it with a motion prompt to ComfyUI's LTX i2v workflow.
4. After each clip is generated, the **smart continuity system** scores multiple frames and picks the strongest one for the next scene.
5. All clips are stitched together with `ffmpeg` / OpenCV into a single output video.

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10+** | Tested on 3.10 / 3.11 |
| **ComfyUI** | Running locally at `http://127.0.0.1:8188` with the LTX 2.0 i2v model installed |
| **ffmpeg** | On your system PATH (used for video stitching) |
| **Ollama** *(optional)* | For local LLM script parsing — install from [ollama.com](https://ollama.com) |
| **Gemini API key** *(optional)* | For cloud-based script parsing via Google Gemini |

### Installation

```bash
# Clone the repository
git clone https://github.com/ather86/ltxi2v.git
cd ltxi2v

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install streamlit requests websocket-client opencv-python numpy python-dotenv

# Optional — for Ollama support
pip install ollama

# Optional — for Gemini support
pip install google-generativeai
```

### Configuration

1. **ComfyUI paths** — Open `app.py` and update the two `COMFYUI_REAL_*` variables to point to your ComfyUI installation's `input/` and `output/` directories.

2. **API keys** — Create a `.env` file in the project root (git-ignored):
   ```
   GEMINI_API_KEY=your_key_here
   ```

3. **Workflow** — The included ComfyUI workflow is at `workfllows/video_ltx2_i2v.json`. Import it into ComfyUI to verify all nodes are present before your first run.

### Run

```bash
streamlit run app.py
```

The app opens in your browser. Use the **sidebar** to select your LLM, configure video settings, and map workflow nodes. Then paste a script in the main panel and hit **Generate**.

---

## 🖼️ Image Input Methods

| Method | Best For |
|---|---|
| **Single reference image** | Upload one image used for every scene — maximum consistency |
| **Per-character images** | Upload a unique image per character; the app picks the right one per scene |
| **Auto-continuity (blank start)** | No upload needed — the system extracts the best frame from each clip for the next |
| **Per-scene upload** | Full manual control over every scene's starting image |

### Image Priority Order

When generating a scene the app resolves the input image in this order:

1. **Single uploaded reference** (if provided) → used for all scenes
2. **Last-frame continuity** from the previous scene
3. **Character-specific image** (if the character appears in the scene)
4. **Blank fallback** (black frame)

---

## 🧠 Smart Continuity System

Instead of always using the last frame (which may be a fade-out or transition), the app samples **5 frames** from each generated clip and scores them on:

| Signal | Weight | Why |
|---|---|---|
| Edge density | 40 % | Detects character silhouettes and detail |
| Non-black pixels | 20 % | Filters out blank / dark frames |
| Contrast / variance | 20 % | Prefers visually rich content |
| Center brightness | 20 % | Prefers frames with a centered subject |

The highest-scoring frame becomes the reference image for the next scene.

---

## 📁 Project Structure

```
ltxi2v/
├── app.py                       # Main Streamlit application
├── inspector.py                 # ComfyUI workflow node inspector utility
├── config.json                  # Non-secret runtime settings (git-ignored)
├── .env                         # API keys (git-ignored, create manually)
├── workfllows/
│   ├── video_ltx2_i2v.json      # LTX i2v ComfyUI workflow
│   ├── video_ltx2_i2v_CLEAN.json
│   └── purge_vram_workflow.json  # VRAM purge helper workflow
├── input/                       # Per-run image assets (git-ignored)
└── output/                      # Per-run generated videos (git-ignored)
```

---

## ⚙️ Sidebar Settings Reference

| Setting | Default | Description |
|---|---|---|
| LLM Provider | Ollama | Choose Ollama (local) or Gemini (cloud) |
| Scene Duration | 12 s | Length of each generated clip |
| Total Video Duration | 60 s | Target length; determines number of scenes |
| Frame Rate | 25 fps | Passed to the LTX conditioning node |
| Purge VRAM | Off | Run a VRAM-clearing workflow between scenes |
| Cooldown | 10 s | Pause between scenes to let VRAM settle |

---

## 🛠️ Troubleshooting

<details>
<summary><strong>ComfyUI is unreachable</strong></summary>

Make sure ComfyUI is running at `http://127.0.0.1:8188`. You can test with:
```bash
curl http://127.0.0.1:8188/queue
```
</details>

<details>
<summary><strong>"CRITICAL ERROR: i2v requires LoadImage, not EmptyImage"</strong></summary>

In the sidebar, change **Image Input Node** to **Load Image**. The i2v model cannot work without an input image.
</details>

<details>
<summary><strong>Out of VRAM / generation crashes</strong></summary>

- Enable **Purge VRAM between scenes** in the sidebar.
- Increase **Cooldown Between Scenes** to 30+ seconds.
- Reduce scene duration or resolution.
</details>

<details>
<summary><strong>Character consistency lost between scenes</strong></summary>

Use **Single image for all scenes** mode, or ensure each character has a dedicated uploaded image. The smart continuity system helps, but a consistent reference image is the strongest guarantee.
</details>

---

## 📄 License

This project is provided as-is for personal and educational use.

---

## 🙏 Acknowledgements

- [LTX Video](https://github.com/Lightricks/LTX-Video) by Lightricks — the i2v model powering generation
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — node-based inference backend
- [Streamlit](https://streamlit.io/) — the UI framework
- [Ollama](https://ollama.com/) — local LLM runtime
- [Google Gemini](https://ai.google.dev/) — cloud LLM API
