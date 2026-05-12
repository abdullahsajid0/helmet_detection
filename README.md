# Helmet Detection Studio

Minimal exhibition dashboard for live helmet detection demos.

## Quick Start

```bash
cd "C:\Users\hp\Desktop\Edify\helmet Detection"
streamlit run app.py
```

## Features

- Fixed model: `model/bestone.pt`
- Auto-detect on image selection (demo folder, upload, URL, webcam)
- Auto-detect on video selection (demo folder, upload, URL, webcam live)
- YouTube live mode with two presets plus custom URL

## Notes

- The default YouTube links are read from `main.py` when available.
- Stop the Streamlit server with `Ctrl+C` in the terminal.
