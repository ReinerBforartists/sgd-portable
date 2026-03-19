# SGD Portable v 1.0.1

🎧 A portable, self-contained web interface for music genre classification using MAEST.

Two variants available:
- **CUDA version**: GPU acceleration via CUDA (NVIDIA)  
- **CPU version**: Runs on any Windows machine without GPU support

You have downloaded the CPU version

No Python or system-wide installation required — everything runs from this folder.

---

## 📦 Installation Size

| Variant | Installed (after `start.bat`) |
|---------|---------------------------------|
| CPU     |    **~2 GB**                    |
| CUDA    |    **~6 GB**                    |

> ⚠️ MAEST model (~350 MB) downloads on first use — not included in the ZIP.

---

## ✅ Features

| Feature | Details |
|---------|---------|
| 📦 Portable | All-in-one folder: embedded Python 3.12.9, all dependencies pre-installed |
| 🖥️ No setup | Double-click `start.bat` → wait → launch. No admin rights needed. |
| 🚀 GPU support (CUDA) | Uses NVIDIA CUDA cu12.4 automatically when available; falls back to CPU otherwise |
| 🧠 Model | [MAEST](https://huggingface.co/mtg-upf/discogs-maest-10s-fs-129e): genre & subgenre classification (over 400 styles) |
| 🌐 Web UI | Gradio-based — dark/light theme, file upload, charts, tag cloud |

---

## ⚠️ License Notice

### Code
MIT License — see `LICENSE`.  
You may use, modify, and share this tool — even commercially.

### MAEST Model (CC BY-NC 4.0)
❌ Non-commercial only — attribution required  
✅ Allowed: research, education, personal use  
❌ Not allowed: commercial use of the model or its outputs  

The **tag cloud is SGD Portable’s own work**, not derived from MAEST — free to use.

Model is *not included* — downloads automatically on first use.  
See license: https://creativecommons.org/licenses/by-nc/4.0/

### Dependencies
All Python packages (`torch`, `gradio`, `transformers`, etc.) are open-source (BSD/Apache 2.0/MIT).  
Full details in `THIRD_PARTY_LICENSES.md`.

---

## 📥 Installation

1. Download the appropriate ZIP:
   - **CUDA version**: `SGD_Portable_CUDA.zip` *(NVIDIA GPU with CUDA required)*
   - **CPU version**: `SGD_Portable_CPU.zip` *(works anywhere, slower)*
> 💡 Note that the CUDA version runs in CPU mode in case it does not find a fitting CUDA version on the system.

2. Extract the folder.

3. Double-click `start.bat`.  
   → Downloads Python, installs ~1.5–6 GB of packages (depending on variant).  
   → Optionally downloads MAEST model (~350 MB).

4. Wait for confirmation → web UI starts automatically in your browser.

> 💡 Leave the terminal window open while using the app.

---

## 🎯 Usage
- Drag & drop audio files (MP3, WAV, etc.) into the upload zone  
- Click **▶ Analyze** to detect genre and subgenre  
- Toggle **Dark theme** / **Tag cloud**  
- View charts: main genres vs. detailed substyles
- To stop the software close the terminal window

You have to clear the result to be able to drop the next audio files. But you can append to the file list by using the file icon and choose other files.

> ⚠️ Results are probabilistic — AI makes mistakes. Always verify.

## 🎯 Uninstall
- Simply delete the folder. It is a virtual environment (Venv), and contains everything needed in its own folder.

---

## 🔧 System Requirements

| Requirement       | CUDA version            | CPU version            |
|-------------------|-------------------------|------------------------|
| OS                | Windows 10/11 (64-bit)  | Windows 10/11 (64-bit) |
| RAM               | ≥ 6 GB                  | ≥ 4 GB                 |
| GPU (optional)    | NVIDIA GPU + CUDA 12.x  | —                      |
| Storage           | ~6 GB free              | ~2 GB free             |

---

## 📜 Credits
- MAEST model: [MTG-UPF](https://mtg.upf.edu/) (Universitat Pompeu Fabra)  
- WebUI & tooling: SGD Portable  

Made with ❤️ by Reiner Prokein 
https://haizytiles.reinerstilesets.de/

---

> 📬 Feedback? Bugs? Contributions? Open an issue or PR at Github!
