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
def transcription_worker(recorder, model_type: str, model_type_size: str, output_file: str = None, interval: float = 5.0, save_audio: str = None, include_timestamp: bool = True, text_queue: Queue = None, summary_interval: float = 0.0):
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
    prev_text = ""
    
    f = open(output_file, "a", encoding="utf-8") if output_file else None
    try:
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
                            clean_text = remove_overlap(prev_text, text)
                            if clean_text:
                                if text_queue is not None:
                                    text_queue.put(clean_text)
                                if summary_interval <= 0:
                                    print(f"[Captured]: {clean_text}")
                                if f:
                                    if include_timestamp:
                                        f.write(f"{time.strftime('%H:%M:%S')} - {clean_text}\n")
                                    else:
                                        f.write(f"{clean_text}\n")
                                    f.flush()
                                prev_text = text
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
                        clean_text = remove_overlap(prev_text, text)
                        if clean_text:
                            if text_queue is not None:
                                text_queue.put(clean_text)
                            if summary_interval <= 0:
                                print(f"[Captured]: {clean_text}")
                            if f:
                                if include_timestamp:
                                    f.write(f"{time.strftime('%H:%M:%S')} - {clean_text}\n")
                                else:
                                    f.write(f"{clean_text}\n")
                                f.flush()
                except Exception as e:
                    print(f"Error during final transcription: {e}")
    finally:
        if f:
            f.close()

    print("Worker finished.")

def summary_worker(text_queue: Queue, interval: float, stop_event: threading.Event, summary_output: str = None):
    """
    Background worker that periodically updates a summary using incremental text.
    """
    print(f"Continuous summarization worker started. Interval: {interval}s")
    from transcribe.summarize import generate_incremental_summary
    current_summary = ""
    
    while not stop_event.is_set():
        next_run = time.time() + interval
        while time.time() < next_run and not stop_event.is_set():
            time.sleep(1)
        
        if stop_event.is_set():
            break
            
        print("\n--- Generating Continuous Summary ---")
        try:
            texts = []
            while not text_queue.empty():
                try:
                    texts.append(text_queue.get_nowait())
                except:
                    break
            
            incremental_text = " ".join(texts).strip()
            
            if incremental_text:
                new_summary = generate_incremental_summary(current_summary, incremental_text)
                if new_summary != current_summary:
                    current_summary = new_summary
                    print("\n=== Continuous Summary ===")
                    print(current_summary)
                    print("==========================\n")
                    
                    if summary_output:
                        try:
                            with open(summary_output, "w", encoding="utf-8") as fs:
                                fs.write(current_summary)
                        except Exception as e:
                            print(f"Failed to save summary to {summary_output}: {e}")
                else:
                    print("Summary unchanged.")
            else:
                print("No new dialogue to summarize.")
        except Exception as e:
            print(f"Error during continuous summarization: {e}")

@app.command()
def start(
    model_type: str = typer.Option("whisper", help="Model type: whisper, mlx-whisper, mlx-sensevoice"),
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
