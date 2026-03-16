import typer
import time
import threading
import numpy as np
from queue import Empty
import os
from Foundation import NSRunLoop, NSDate
from transcribe.model.model import get_model
from transcribe.audio.vad import VADTracker
from transcribe.audio.recorder import ScreenAudioRecorder


app = typer.Typer()
def transcription_worker(recorder, model_type: str, model_type_size: str, output_file: str, interval: float = 5.0, save_audio: str = None):
    """
    Background worker that reads audio from recorder queue, aggregates, and transcribes.
    """
    print(f"Transcription worker started. Interval: {interval}s, Save Audio: {save_audio}")
    
    # Load Model (Load in background to keep main thread completely free for SCK)
    print(f"Loading Model {model_type_size} in background...")
    model = get_model(model_type=model_type, model_size=model_type_size)
    print("Model loaded.")

    print("Waiting for recording to start...")
    recorder.start_event.wait(timeout=10.0)
    if not recorder.is_recording:
        print("Error: Recording failed to start within timeout.")
        os._exit(1)
        
    print("Recording confirmed started. Beginning transcription loop.")

    tracker = VADTracker(max_duration_s=interval) # Use interval as max duration limit
    accumulated_np = np.array([], dtype=np.float32)
    
    # Open file for appending
    with open(output_file, "a", encoding="utf-8") as f:
        while recorder.is_recording or not recorder.delegate.queue.empty():
            try:
                # Get audio chunk
                chunk = recorder.delegate.queue.get(timeout=1.0)
                frame_np = np.frombuffer(chunk, dtype=np.float32)
                accumulated_np = np.concatenate([accumulated_np, frame_np])
                if save_audio:
                    with open(save_audio, "ab") as df:
                        df.write(chunk)
            except Empty:
                pass

            # Feed 512 sample frames to VAD
            while len(accumulated_np) >= 512:
                frame = accumulated_np[:512]
                accumulated_np = accumulated_np[512:]
                
                speech_segment = tracker.process_frame(frame)
                if speech_segment is not None:
                    try:
                        # Run ASR
                        text = model.transcribe(speech_segment)
                        if text:
                            print(f"[Captured]: {text}")
                            f.write(f"{time.strftime('%H:%M:%S')} - {text}\n")
                            f.flush()
                        else:
                            print('no text')
                    except Exception as e:
                        print(f"Error during transcription: {e}")

        # Flush remaining on exit
        if tracker.is_speaking and tracker.buffer:
            print("Flushing remaining audio to transcribe...")
            speech_segment = np.concatenate(tracker.buffer)
            if len(speech_segment) > 16000 * 0.5:
                try:
                    text = model.transcribe(speech_segment)
                    if text:
                        print(f"[Captured]: {text}")
                        f.write(f"{time.strftime('%H:%M:%S')} - {text}\n")
                        f.flush()
                except Exception as e:
                    print(f"Error during final transcription: {e}")


    print("Worker finished.")

@app.command()
def start(
    model_type: str = typer.Option("whisper", help="Model type: whisper, mlx-whisper"),
    model_size: str = typer.Option("base", help="Size of the ASR model (e.g., base, small for whisper; repo ID for mlx-whisper)"),
    output_file: str = typer.Option("transcription.txt", help="Output file path"),
    interval: float = typer.Option(5.0, help="Acculumation interval in seconds"),
    save_audio: str = typer.Option(None, help="Save raw audio to specified file"),
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

    # 1. Initialize & Start Recorder FIRST (avoid Metal conflict during SCK setup)
    recorder = ScreenAudioRecorder.alloc().init()
    recorder.start()

    # 2. Start Worker (Loads model inside on background thread)
    worker_thread = threading.Thread(
        target=transcription_worker, 
        args=(recorder, model_type, model_size, output_file, interval, save_audio),
        daemon=True
    )
    worker_thread.start()

    loop = NSRunLoop.currentRunLoop()

    try:
        print("[DEBUG] Entering main run loop")
        while True:
            # Run loop to process events (e.g. main thread dispatch)
            loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(1.0))
    except KeyboardInterrupt:
        print("\nStopping transcription...")
        recorder.stop()
        # Wait for worker to finish processing remaining queue
        worker_thread.join(timeout=3)
        print("Done.")

if __name__ == "__main__":
    app()
