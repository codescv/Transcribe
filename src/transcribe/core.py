import time
import threading
import numpy as np
from queue import Empty, Queue
import os
from transcribe.model.model import get_model
from transcribe.audio.vad import VADTracker
from transcribe.text_utils import remove_overlap
from transcribe.summarize import generate_incremental_summary

def transcription_worker(
    recorder, 
    model_type: str, 
    model_type_size: str, 
    output_file: str = None, 
    interval: float = 5.0, 
    save_audio: str = None, 
    include_timestamp: bool = True, 
    text_queue: Queue = None, 
    summary_interval: float = 0.0
):
    """
    Background worker that reads audio from recorder queue, aggregates, and transcribes.
    """
    print(f"Transcription worker started. Interval: {interval}s, Save Audio: {save_audio}")
    
    # Load Model (Load in background to keep main thread completely free)
    print(f"Loading Model {model_type_size} in background...")
    model = get_model(model_type=model_type, model_size=model_type_size)
    print("Model loaded.")

    print("Waiting for recording to start...")
    recorder.start_event.wait(timeout=10.0)
    if not recorder.is_recording:
        print("Error: Recording failed to start within timeout.")
        # In GUI or general core, we might not want os._exit(1).
        # Let's just return or set an error flag in recorder.
        return
        
    print("Recording confirmed started. Beginning transcription loop.")

    tracker = VADTracker(max_duration_s=interval)
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
                                    if len(clean_text) - len(set(clean_text)) > 50:
                                        # garbage text is usually repeated a lot of times
                                        print('drop possible garbage text:', clean_text)
                                    else:
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

def summary_worker(
    text_queue: Queue, 
    interval: float, 
    stop_event: threading.Event, 
    summary_output: str = None, 
    custom_prompt = None,
    callback = None
):
    """
    Background worker that periodically updates a summary using incremental text.
    Arguments:
        custom_prompt: String or callable returning string for the summary prompt.
        callback: Callable taking one argument (summary text) to update UI/logs.
    """
    print(f"Continuous summarization worker started. Interval: {interval}s")
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

            print('incremental text:', incremental_text, 'current summary:', current_summary)
            
            if incremental_text:
                current_prompt = custom_prompt() if callable(custom_prompt) else custom_prompt
                new_summary = generate_incremental_summary(current_summary, incremental_text, custom_prompt=current_prompt)
                if new_summary != current_summary:
                    current_summary = new_summary
                    print("\n=== Continuous Summary ===")
                    print(current_summary)
                    print("==========================\n")
                    
                    if callback:
                        callback(current_summary)
                        
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
