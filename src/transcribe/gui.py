import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import threading
import os
import time
from queue import Queue
from Foundation import NSRunLoop, NSDate

from transcribe.core import transcription_worker, summary_worker
from transcribe.audio.recorder import get_recorder
from transcribe.model.model import download_model_files
from transcribe.summarize import generate_summary

# Select color theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class TranscriptionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Screen Audio Transcriber")
        self.geometry("1000x700")

        # State
        self.recorder = None
        self.worker_thread = None
        self.summary_thread = None
        self.stop_summary_event = None
        self.text_queue = None
        self.is_running = False
        self.current_summary = ""

        self.setup_ui()

    def setup_ui(self):
        # Grid configuration
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # === Sidebar (Settings) ===
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.sidebar.grid_rowconfigure(15, weight=1) # Spacer


        self.title_label = ctk.CTkLabel(self.sidebar, text="Settings", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        # Model Type
        self.model_type_label = ctk.CTkLabel(self.sidebar, text="Model Type")
        self.model_type_label.grid(row=1, column=0, padx=20, pady=(10, 0), sticky="w")
        self.model_type_menu = ctk.CTkOptionMenu(
            self.sidebar, 
            values=["mlx-whisper", "whisper", "mlx-sensevoice"],
            command=self.on_model_type_change
        )

        self.model_type_menu.grid(row=2, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Model Size/Repo
        self.model_size_label = ctk.CTkLabel(self.sidebar, text="Size / Repo ID")
        self.model_size_label.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="w")
        self.model_size_entry = ctk.CTkEntry(self.sidebar, placeholder_text="mlx-community/whisper-large-v3-turbo")
        self.model_size_entry.grid(row=4, column=0, padx=20, pady=(0, 10), sticky="ew")
        self.model_size_entry.insert(0, "mlx-community/whisper-large-v3-turbo") # Default


        # Source
        self.source_label = ctk.CTkLabel(self.sidebar, text="Audio Source")
        self.source_label.grid(row=5, column=0, padx=20, pady=(10, 0), sticky="w")
        self.source_menu = ctk.CTkOptionMenu(self.sidebar, values=["system", "mic"])
        self.source_menu.grid(row=6, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Interval (ASR)
        self.interval_label = ctk.CTkLabel(self.sidebar, text="ASR Interval (s)")
        self.interval_label.grid(row=7, column=0, padx=20, pady=(10, 0), sticky="w")
        self.interval_entry = ctk.CTkEntry(self.sidebar, width=100)
        self.interval_entry.grid(row=8, column=0, padx=20, pady=(0, 10), sticky="w")
        self.interval_entry.insert(0, "5.0")

        # Summary interval
        self.summary_interval_label = ctk.CTkLabel(self.sidebar, text="Summary Interval (s)")
        self.summary_interval_label.grid(row=9, column=0, padx=20, pady=(10, 0), sticky="w")
        self.summary_interval_entry = ctk.CTkEntry(self.sidebar, width=100)
        self.summary_interval_entry.grid(row=10, column=0, padx=20, pady=(0, 10), sticky="w")
        self.summary_interval_entry.insert(0, "60.0") # Default continuous


        # === Main Panel ===
        self.main_panel = ctk.CTkFrame(self, corner_radius=10)
        self.main_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.main_panel.grid_columnconfigure(0, weight=1)
        self.main_panel.grid_rowconfigure(2, weight=1) # Summary area expands

        # Controls
        self.control_frame = ctk.CTkFrame(self.main_panel, fg_color="transparent")
        self.control_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        
        self.start_btn = ctk.CTkButton(
            self.control_frame, 
            text="Start Transcription", 
            command=self.toggle_start,
            fg_color="#28a745", 
            hover_color="#218838"
        )
        self.start_btn.pack(side="left", padx=0)

        self.status_label = ctk.CTkLabel(self.control_frame, text="Status: Idle", text_color="gray")
        self.status_label.pack(side="left", padx=20)

        # Custom Prompt
        self.prompt_frame = ctk.CTkFrame(self.main_panel, fg_color="transparent")
        self.prompt_frame.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="ew")
        
        self.prompt_label = ctk.CTkLabel(self.prompt_frame, text="Custom Summary Prompt:")
        self.prompt_label.pack(anchor="w")
        
        self.prompt_entry = ctk.CTkEntry(self.prompt_frame, placeholder_text="e.g., Summarize in bullet points, focus on actions...")
        self.prompt_entry.pack(fill="x", pady=(2, 0))

        # Summary Display
        self.summary_frame = ctk.CTkFrame(self.main_panel)
        self.summary_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        self.summary_frame.grid_columnconfigure(0, weight=1)
        self.summary_frame.grid_rowconfigure(1, weight=1)

        self.summary_title = ctk.CTkLabel(self.summary_frame, text="Summary (Markdown)", font=ctk.CTkFont(weight="bold"))
        self.summary_title.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.summary_text = ctk.CTkTextbox(self.summary_frame, wrap="word")
        self.summary_text.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self.summary_text.configure(state="disabled")

        # Actions Panel
        self.actions_frame = ctk.CTkFrame(self.main_panel, fg_color="transparent")
        self.actions_frame.grid(row=3, column=0, padx=20, pady=20, sticky="ew")

        self.save_btn = ctk.CTkButton(self.actions_frame, text="Save Summary", command=self.save_summary)
        self.save_btn.pack(side="right")

    def on_model_type_change(self, value):
        # Update default size based on type to help user
        current_size = self.model_size_entry.get()
        defaults = {
            "whisper": "base",
            "mlx-whisper": "mlx-community/whisper-large-v3-turbo",
            "mlx-sensevoice": "mlx-community/SenseVoiceSmall"
        }
        if current_size in defaults.values() or current_size == "":
            self.model_size_entry.delete(0, tk.END)
            self.model_size_entry.insert(0, defaults.get(value, "base"))
        elif value == "whisper" and current_size not in ["base", "tiny", "small", "medium", "large"]:
            # Fallback for whisper sizes just in case
            pass
        # No else, if user typed custom repo, we leave it


    def toggle_start(self):
        if self.is_running:
            self.stop_transcribing()
        else:
            self.start_transcribing()

    def start_transcribing(self):
        # Get config
        model_type = self.model_type_menu.get()
        model_size = self.model_size_entry.get()
        source = self.source_menu.get()
        
        try:
            interval = float(self.interval_entry.get())
            summary_interval = float(self.summary_interval_entry.get())
        except ValueError:
            self.update_status("Error: Invalid interval numbers", "red")
            return

        self.is_running = True
        self.start_btn.configure(text="Stop Transcription", fg_color="#dc3545", hover_color="#c82333")
        self.update_status("Starting...", "yellow")
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.configure(state="disabled")
        self.current_summary = ""

        # Start background thread
        threading.Thread(target=self.run_background_process, args=(model_type, model_size, source, interval, summary_interval), daemon=True).start()

    def run_background_process(self, model_type, model_size, source, interval, summary_interval):
        try:
            self.update_status("Downloading/Verifying Model...", "yellow")
            download_model_files(model_type, model_size)
            
            self.update_status("Initializing Recorder...", "yellow")
            self.recorder = get_recorder(source)
            self.recorder.start()

            self.text_queue = Queue()
            
            self.update_status("Starting Workers...", "yellow")
            self.worker_thread = threading.Thread(
                target=transcription_worker,
                args=(self.recorder, model_type, model_size, None, interval, None, False, self.text_queue, summary_interval),
                daemon=True
            )
            self.worker_thread.start()

            if summary_interval > 0:
                self.stop_summary_event = threading.Event()
                custom_prompt = lambda: self.prompt_entry.get()
                
                self.summary_thread = threading.Thread(
                    target=summary_worker,
                    args=(self.text_queue, summary_interval, self.stop_summary_event, None, custom_prompt, self.update_summary_ui),
                    daemon=True
                )
                self.summary_thread.start()

            self.update_status("Recording & Transcribing", "green")
            
            # NSRunLoop is needed on the main thread usually or where SCK runs.
            # However, in Tkinter, the main thread is the Tk event loop.
            # ScreenCaptureKit requires a run loop. if get_recorder starts a thread that manages it, great.
            # Let's assume get_recorder handles it as it did in CLI.
            # Wait, in CLI, loop.runUntilDate_ was on the MAIN thread.
            # If Tkinter runs its own loop, they might conflict, OR we need to run NSRunLoop periodically via tick.
            # Let's use after() to run NSRunLoop tick.
            self.run_ns_loop_tick()

        except Exception as e:
            self.update_status(f"Error: {e}", "red")
            self.stop_transcribing()

    def run_ns_loop_tick(self):
        if not self.is_running:
            return
            
        # Run loop to process events (e.g. main thread dispatch for SCK)
        loop = NSRunLoop.currentRunLoop()
        loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.1))
        
        # Schedule next tick
        self.after(100, self.run_ns_loop_tick)

    def stop_transcribing(self):
        self.is_running = False
        self.start_btn.configure(text="Start Transcription", fg_color="#28a745", hover_color="#218838")
        self.update_status("Stopping...", "yellow")

        if self.stop_summary_event:
            self.stop_summary_event.set()
        
        if self.recorder:
            self.recorder.stop()

        # If continuous summary is off, generate one last summary on stop
        summary_interval = float(self.summary_interval_entry.get())
        if summary_interval <= 0 and self.text_queue:
            self.update_status("Generating Final Summary...", "yellow")
            threading.Thread(target=self.generate_final_summary, daemon=True).start()
        else:
            self.update_status("Stopped", "gray")

    def generate_final_summary(self):
        try:
            # Collect all text from queue
            texts = []
            while not self.text_queue.empty():
                texts.append(self.text_queue.get_nowait())
            
            full_text = " ".join(texts).strip()
            if full_text:
                custom_prompt = self.prompt_entry.get()
                summary = generate_summary(full_text, custom_prompt=custom_prompt if custom_prompt else None)
                self.update_summary_ui(summary)
                self.update_status("Stopped (Summary Generated)", "gray")
            else:
                self.update_status("Stopped (No content)", "gray")
        except Exception as e:
            self.update_status(f"Summary Error: {e}", "red")

    def update_status(self, text, color="gray"):
        # Thread-safe UI update
        self.after(0, lambda: self.status_label.configure(text=f"Status: {text}", text_color=color))

    def update_summary_ui(self, summary):
        # Thread-safe UI update
        def do_update():
            self.summary_text.configure(state="normal")
            self.summary_text.delete("1.0", tk.END)
            self.summary_text.insert("1.0", summary)
            self.summary_text.configure(state="disabled")
            self.current_summary = summary
            
        self.after(0, do_update)

    def save_summary(self):
        if not self.current_summary:
             # Notify or flash button
             return
             
        filepath = filedialog.asksaveasfilename(
            defaultextension=".md",
            filetypes=[("Markdown", "*.md"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(self.current_summary)
                # Flash success?
            except Exception as e:
                 self.update_status(f"Save failed: {e}", "red")

def main():
    app = TranscriptionApp()
    app.mainloop()

if __name__ == "__main__":
    main()
