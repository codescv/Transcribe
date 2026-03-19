import sys
import os
import threading
from queue import Queue
from Foundation import NSRunLoop, NSDate, NSUserDefaults

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QFrame, QLabel, QPushButton, QLineEdit, QComboBox,
    QTextEdit, QFileDialog, QMessageBox
)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QObject
from PyQt6.QtGui import QFont

from transcribe.core import transcription_worker, summary_worker
from transcribe.audio.recorder import get_recorder
from transcribe.model.model import download_model_files
from transcribe.summarize import generate_summary

def get_system_theme():
    try:
        style = NSUserDefaults.standardUserDefaults().stringForKey_("AppleInterfaceStyle")
        return style == "Dark"
    except:
        return False

def get_stylesheet(is_dark):
    bg = "#0B0F19" if is_dark else "#F3F4F6"
    sidebar_bg = "#111827" if is_dark else "#FFFFFF"
    panel_bg = "#111827" if is_dark else "#FFFFFF"
    text = "#F9FAFB" if is_dark else "#111827"
    secondary_text = "#9CA3AF" if is_dark else "#6B7280"
    border = "#374151" if is_dark else "#E5E7EB"
    input_bg = "#1F2937" if is_dark else "#F9FAFB"
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    arrow_path = os.path.join(current_dir, "resources", "arrow_dark.svg" if is_dark else "arrow_light.svg").replace("\\", "/")
    
    return f"""
        QMainWindow {{
            background-color: {bg};
        }}
        QLabel {{
            color: {text};
            font-size: 13px;
        }}
        QLabel#section_title {{
            font-size: 18px;
            font-weight: bold;
            color: {text};
        }}
        QFrame#sidebar {{
            background-color: {sidebar_bg};
            border-radius: 16px;
            border-right: 1px solid {border};
        }}
        QFrame#main_panel {{
            background-color: {panel_bg};
            border-radius: 16px;
            border: 1px solid {border};
        }}
        QLineEdit, QComboBox {{
            background-color: {input_bg};
            color: {text};
            border: 1px solid {border};
            border-radius: 8px;
            padding: 8px;
            font-size: 13px;
        }}
        QLineEdit:focus, QComboBox:focus {{
            border: 1px solid #3B82F6;
        }}
        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 30px;
            border-left: 1px solid {border};
            border-top-right-radius: 8px;
            border-bottom-right-radius: 8px;
            background-color: transparent;
        }}
        QComboBox::down-arrow {{
            image: url({arrow_path});
            width: 10px;
            height: 6px;
        }}
        QComboBox QFrame {{
            background-color: {panel_bg};
            border: 1px solid {border};
            border-radius: 8px;
        }}
        QComboBox QAbstractItemView {{
            background-color: transparent;
            color: {text};
            padding: 4px;
        }}
        QComboBox QAbstractItemView::item {{
            background-color: transparent;
            color: {text};
            padding: 8px;
        }}
        QComboBox QAbstractItemView::item:selected {{
            background-color: #3B82F6;
            color: white;
        }}
        QPushButton {{
            background-color: #3B82F6;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 16px;
            font-weight: bold;
            font-size: 13px;
        }}
        QPushButton:hover {{
            background-color: #2563EB;
        }}
        QPushButton:pressed {{
            background-color: #1D4ED8;
        }}
        QPushButton[state="start"] {{
            background-color: #10B981;
        }}
        QPushButton[state="start"]:hover {{
            background-color: #059669;
        }}
        QPushButton[state="stop"] {{
            background-color: #EF4444;
        }}
        QPushButton[state="stop"]:hover {{
            background-color: #DC2626;
        }}
        QTextEdit {{
            background-color: {input_bg};
            color: {text};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 12px;
            font-size: 14px;
            line-height: 1.5;
        }}
    """

# Signal emitter for thread safety
class WorkerSignals(QObject):
    status_updated = pyqtSignal(str, str)
    summary_updated = pyqtSignal(str)

class TranscriptionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Screen Audio Transcriber")
        self.resize(1100, 750)
        
        # Style
        is_dark = get_system_theme()
        self.setStyleSheet(get_stylesheet(is_dark))
        
        # State
        self.recorder = None
        self.worker_thread = None
        self.summary_thread = None
        self.stop_summary_event = None
        self.text_queue = None
        self.is_running = False
        self.current_summary = ""
        
        self.signals = WorkerSignals()
        self.signals.status_updated.connect(self.update_status)
        self.signals.summary_updated.connect(self.update_summary_ui)
        
        self.setup_ui()
        
    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # === Sidebar ===
        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(260)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(20, 25, 20, 25)
        sidebar_layout.setSpacing(15)
        sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        main_layout.addWidget(self.sidebar)
        
        # Title
        settings_title = QLabel("Settings")
        settings_title.setObjectName("section_title")
        sidebar_layout.addWidget(settings_title)
        sidebar_layout.addSpacing(10)
        
        # Model Type
        sidebar_layout.addWidget(QLabel("Model Type"))
        self.model_type_menu = QComboBox()
        self.model_type_menu.addItems(["mlx-whisper", "whisper", "mlx-sensevoice"])
        self.model_type_menu.currentTextChanged.connect(self.on_model_type_change)
        self.model_type_menu.view().window().setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        sidebar_layout.addWidget(self.model_type_menu)
        
        # Model Size
        sidebar_layout.addWidget(QLabel("Size / Repo ID"))
        self.model_size_entry = QLineEdit("mlx-community/whisper-large-v3-turbo")
        sidebar_layout.addWidget(self.model_size_entry)
        
        # Source
        sidebar_layout.addWidget(QLabel("Audio Source"))
        self.source_menu = QComboBox()
        self.source_menu.addItems(["system", "mic"])
        self.source_menu.view().window().setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        sidebar_layout.addWidget(self.source_menu)
        
        # Intervals
        sidebar_layout.addWidget(QLabel("ASR Interval (s)"))
        self.interval_entry = QLineEdit("5.0")
        sidebar_layout.addWidget(self.interval_entry)
        
        sidebar_layout.addWidget(QLabel("Summary Interval (s)"))
        self.summary_interval_entry = QLineEdit("60.0")
        sidebar_layout.addWidget(self.summary_interval_entry)
        
        sidebar_layout.addStretch()
        
        # === Main Panel ===
        main_panel = QFrame()
        main_panel.setObjectName("main_panel")
        main_panel_layout = QVBoxLayout(main_panel)
        main_panel_layout.setContentsMargins(25, 25, 25, 25)
        main_panel_layout.setSpacing(20)
        main_layout.addWidget(main_panel)
        
        # Controls
        controls_frame = QFrame()
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        main_panel_layout.addWidget(controls_frame)
        
        self.start_btn = QPushButton("Start Transcription")
        self.start_btn.setProperty("state", "start")
        self.start_btn.clicked.connect(self.toggle_start)
        controls_layout.addWidget(self.start_btn)
        
        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("font-style: italic; color: gray;")
        controls_layout.addWidget(self.status_label)
        controls_layout.addStretch()
        
        # Prompt
        prompt_frame = QFrame()
        prompt_layout = QVBoxLayout(prompt_frame)
        prompt_layout.setContentsMargins(0, 0, 0, 0)
        prompt_layout.setSpacing(8)
        main_panel_layout.addWidget(prompt_frame)
        
        prompt_title = QLabel("Custom Summary Prompt")
        font = QFont()
        font.setBold(True)
        prompt_title.setFont(font)
        prompt_layout.addWidget(prompt_title)
        
        self.prompt_entry = QLineEdit()
        self.prompt_entry.setPlaceholderText("e.g., Summarize in bullet points, focus on actions...")
        prompt_layout.addWidget(self.prompt_entry)
        
        # Summary Display
        summary_frame = QFrame()
        summary_layout = QVBoxLayout(summary_frame)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.setSpacing(8)
        main_panel_layout.addWidget(summary_frame)
        
        summary_title = QLabel("Summary")
        summary_title.setObjectName("section_title")
        summary_layout.addWidget(summary_title)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        summary_layout.addWidget(self.summary_text)
        
        # Save Button
        save_layout = QHBoxLayout()
        main_panel_layout.addLayout(save_layout)
        save_layout.addStretch()
        self.save_btn = QPushButton("Save Summary")
        self.save_btn.clicked.connect(self.save_summary)
        save_layout.addWidget(self.save_btn)

    def on_model_type_change(self, value):
        current_size = self.model_size_entry.text()
        defaults = {
            "whisper": "base",
            "mlx-whisper": "mlx-community/whisper-large-v3-turbo",
            "mlx-sensevoice": "mlx-community/SenseVoiceSmall"
        }
        if current_size in defaults.values() or current_size == "":
            self.model_size_entry.setText(defaults.get(value, "base"))

    def toggle_start(self):
        if self.is_running:
            self.stop_transcribing()
        else:
            self.start_transcribing()

    def start_transcribing(self):
        model_type = self.model_type_menu.currentText()
        model_size = self.model_size_entry.text()
        source = self.source_menu.currentText()
        
        try:
            interval = float(self.interval_entry.text())
            summary_interval = float(self.summary_interval_entry.text())
        except ValueError:
            self.signals.status_updated.emit("Error: Invalid interval numbers", "red")
            return

        self.is_running = True
        self.start_btn.setText("Stop Transcription")
        self.start_btn.setProperty("state", "stop")
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)
        
        self.signals.status_updated.emit("Starting...", "orange")
        self.summary_text.clear()
        self.current_summary = ""

        # Timer for NSRunLoop
        self.timer = QTimer()
        self.timer.timeout.connect(self.run_ns_loop_tick)
        self.timer.start(100)

        threading.Thread(target=self.run_background_process, args=(model_type, model_size, source, interval, summary_interval), daemon=True).start()

    def run_background_process(self, model_type, model_size, source, interval, summary_interval):
        try:
            self.signals.status_updated.emit("Downloading/Verifying Model...", "orange")
            download_model_files(model_type, model_size)
            
            self.signals.status_updated.emit("Initializing Recorder...", "orange")
            self.recorder = get_recorder(source)
            self.recorder.start()

            self.text_queue = Queue()
            self.signals.status_updated.emit("Starting Workers...", "orange")
            
            self.worker_thread = threading.Thread(
                target=transcription_worker,
                args=(self.recorder, model_type, model_size, None, interval, None, False, self.text_queue, summary_interval),
                daemon=True
            )
            self.worker_thread.start()

            if summary_interval > 0:
                self.stop_summary_event = threading.Event()
                custom_prompt = lambda: self.prompt_entry.text()
                
                self.summary_thread = threading.Thread(
                    target=summary_worker,
                    args=(self.text_queue, summary_interval, self.stop_summary_event, None, custom_prompt, self.signals.summary_updated.emit),
                    daemon=True
                )
                self.summary_thread.start()

            self.signals.status_updated.emit("Recording & Transcribing", "green")
        except Exception as e:
            self.signals.status_updated.emit(f"Error: {e}", "red")
            self.signals.status_updated.emit("Stopped due to error", "red")

    def run_ns_loop_tick(self):
        if not self.is_running:
            self.timer.stop()
            return
        loop = NSRunLoop.currentRunLoop()
        loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.1))

    def stop_transcribing(self):
        self.is_running = False
        self.start_btn.setText("Start Transcription")
        self.start_btn.setProperty("state", "start")
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)
        
        self.signals.status_updated.emit("Stopping...", "orange")

        if self.stop_summary_event:
            self.stop_summary_event.set()
        
        if self.recorder:
            self.recorder.stop()

        try:
            summary_interval = float(self.summary_interval_entry.text())
        except ValueError:
            summary_interval = 0

        if summary_interval <= 0 and self.text_queue:
             self.signals.status_updated.emit("Generating Final Summary...", "orange")
             threading.Thread(target=self.generate_final_summary, daemon=True).start()
        else:
             self.signals.status_updated.emit("Stopped", "gray")

    def generate_final_summary(self):
        try:
            texts = []
            while not self.text_queue.empty():
                texts.append(self.text_queue.get_nowait())
            
            full_text = " ".join(texts).strip()
            if full_text:
                custom_prompt = self.prompt_entry.text()
                summary = generate_summary(full_text, custom_prompt=custom_prompt if custom_prompt else None)
                self.signals.summary_updated.emit(summary)
                self.signals.status_updated.emit("Stopped (Summary Generated)", "gray")
            else:
                self.signals.status_updated.emit("Stopped (No content)", "gray")
        except Exception as e:
            self.signals.status_updated.emit(f"Summary Error: {e}", "red")

    def update_status(self, text, color="gray"):
        self.status_label.setText(f"Status: {text}")
        colors = {"green": "#10B981", "orange": "#F59E0B", "red": "#EF4444", "gray": "#6B7280"}
        self.status_label.setStyleSheet(f"color: {colors.get(color, '#6B7280')}; font-style: italic;")

    def update_summary_ui(self, summary):
        self.summary_text.setMarkdown(summary)
        self.current_summary = summary

    def save_summary(self):
        if not self.current_summary:
             return
        filepath, _ = QFileDialog.getSaveFileName(
             self, "Save Summary", "", "Markdown (*.md);;Text files (*.txt);;All files (*.*)"
        )
        if filepath:
             try:
                 with open(filepath, "w", encoding="utf-8") as f:
                     f.write(self.current_summary)
             except Exception as e:
                 QMessageBox.warning(self, "Error", f"Save failed: {e}")

def main():
    app = QApplication(sys.argv)
    window = TranscriptionApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
