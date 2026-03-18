import typer
import time
import threading
import numpy as np
from queue import Empty, Queue
import os
from Foundation import NSRunLoop, NSDate
from transcribe.model.model import get_model, download_model_files
from transcribe.audio.vad import VADTracker
from transcribe.audio.recorder import get_recorder
from transcribe.text_utils import remove_overlap


app = typer.Typer()
from transcribe.core import transcription_worker, summary_worker


@app.command()
def start(
    model_type: str = typer.Option("mlx-whisper", help="Model type: whisper, mlx-whisper, mlx-sensevoice"),

    model_size: str = typer.Option("base", help="Size of the ASR model (e.g., base, small for whisper; repo ID for mlx-whisper/mlx-sensevoice)"),
    output_file: str = typer.Option(None, help="Output file path for full transcription logs"),
    interval: float = typer.Option(5.0, help="Acculumation interval in seconds"),
    save_audio: str = typer.Option(None, help="Save raw audio to specified file"),
    summary_output: str = typer.Option(None, help="Save summary output to this file"),
    timestamp: bool = typer.Option(False, "--timestamp/--no-timestamp", help="Include timestamp in output file"),
    source: str = typer.Option("system", help="Audio source: system (output) or mic (input)"),
    summary_interval: float = typer.Option(0.0, "--summary-interval", help="Interval for continuous summarization in seconds (0 to disable)"),
):
    """
    Start capturing screen audio and transcribing.
    """

    print(f"Starting Screen Audio Transcriber...")
    print(f"Model: {model_size}")
    print(f"Output: {output_file}")
    print("Press Ctrl+C to stop.")

    # Handle default model size for mlx-whisper
    if model_type.lower() == "mlx-whisper" and model_size == "base":
        model_size = "mlx-community/whisper-large-v3-turbo"
    elif model_type.lower() == "mlx-sensevoice" and model_size == "base":
        model_size = "mlx-community/SenseVoiceSmall"

    # 0. Download/Verify Model BEFORE starting recorder to avoid concurrent download/recording issues
    # and ensure content is ready to transcribe immediately.
    # Uses try-except to allow running offline if already cached.
    try:
        download_model_files(model_type, model_size)
    except Exception as e:
        print(f"Warning/Error during model pre-download: {e}")
        print("Will attempt to proceed with loading later...")

    # 1. Initialize & Start Recorder
    recorder = get_recorder(source)
    recorder.start()

    text_queue = Queue() if summary_interval > 0 else None

    # 2. Start Worker (Loads model inside on background thread)
    worker_thread = threading.Thread(
        target=transcription_worker, 
        args=(recorder, model_type, model_size, output_file, interval, save_audio, timestamp, text_queue, summary_interval),
        daemon=True
    )
    worker_thread.start()

    stop_summary_event = None
    summary_thread = None
    if summary_interval > 0:
        stop_summary_event = threading.Event()
        summary_thread = threading.Thread(
            target=summary_worker,
            args=(text_queue, summary_interval, stop_summary_event, summary_output),
            daemon=True
        )
        summary_thread.start()

    loop = NSRunLoop.currentRunLoop()

    try:
        print("[DEBUG] Entering main run loop")
        while True:
            # Run loop to process events (e.g. main thread dispatch)
            loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(1.0))
    except KeyboardInterrupt:
        print("\nStopping transcription...")
        if stop_summary_event:
            stop_summary_event.set()
        recorder.stop()
        # Wait for worker to finish processing remaining queue
        worker_thread.join(timeout=3)
        if summary_thread:
            summary_thread.join(timeout=3)
        print("Done.")

        if summary_output and summary_interval <= 0:
            print("Generating summary...")
            try:
                from transcribe.summarize import generate_summary
                if output_file and os.path.exists(output_file):
                    with open(output_file, "r", encoding="utf-8") as f:
                        text_content = f.read()
                    
                    if text_content.strip():
                        summary_text = generate_summary(text_content)
                        print("\n=== Summary ===")
                        print(summary_text)
                        
                        with open(summary_output, "w", encoding="utf-8") as sf:
                            sf.write(summary_text)
                        print(f"Summary saved to {summary_output}")
                    else:
                        print("Transcription file is empty. Skipping summary.")
                else:
                    print("Transcription file not found. Skipping summary.")
            except Exception as e:
                print(f"Failed to generate summary: {e}")

if __name__ == "__main__":
    app()
