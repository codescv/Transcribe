# Screen Audio Transcriber

A macOS application that monitors system audio and transcribes it locally using local ASR models (Faster-Whisper). Supports Chinese and English.

## Features
- **Zero Drivers**: Uses native `ScreenCaptureKit` — no need for BlackHole or Soundflower.
- **Local AI**: Transcribes using `faster-whisper` on your machine.
- **Multi-language**: Supports both English and Chinese auto-detection/transcription.

## Requirements
- macOS 12.3 or higher (for ScreenCaptureKit integration)
- Python 3.12+
- **Screen Recording Permission**: macOS will prompt you for this when capturing starts.

## Installation

1. Clone or download this repository.
2. Install dependencies with `uv`:
   ```bash
   uv sync
   ```

## Usage

Start transcribing system audio:

```bash
uv run transcribe start
```

### Options
- `--model-size`: Choose from `tiny`, `base` (default), `small`. Smaller is faster, larger is more accurate.
- `--output-file`: Path to save transcription (default: `transcription.txt`).
- `--interval`: Buffer duration in seconds before driving inference (default: `5.0`).

Example:
```bash
uv run transcribe start --model-size small --output-file my_meeting.txt
```

## Troubleshooting
- **No Sound / No Text**: Ensure you granted "Screen Recording" permissions to Python/Terminal in System Settings > Privacy & Security.
- **Audio Device Issues**: Sound Capture depends on the active display output. Ensure audio is playing through your main speakers/headphones.

## License
MIT
