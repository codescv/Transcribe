import abc
import numpy as np
from typing import Iterator

class ASRModel(abc.ABC):
    """Abstract base class for ASR models."""
    
    @abc.abstractmethod
    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio data (numpy array, 16kHz, float32) to text."""
        pass

class FasterWhisperModel(ASRModel):
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "float32"):
        """
        Initialize Faster-Whisper model.
        
        Args:
            model_size: 'tiny', 'base', 'small', 'medium', 'large-v3'
            device: 'cpu' or 'auto' (MPS is not fully supported for all operations in whisper sometimes)
            compute_type: 'float32', 'int8', 'float16' (float16 requires CUDA usually)
        """
        from faster_whisper import WhisperModel
        print(f"Loading Faster-Whisper model: {model_size} on {device}...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio array.
        
        Args:
            audio_data: np.ndarray of shape (N,) containing float32 samples at 16000 Hz.
        """
        if audio_data.size == 0:
            return ""
            
        segments, info = self.model.transcribe(
            audio_data, 
            beam_size=5,
            language=None, # Auto-detect or configurable
            vad_filter=True, # Use VAD for better splitting
        )
        
        text_segments = []
        for segment in segments:
            text_segments.append(segment.text)
            
        return " ".join(text_segments).strip()

def get_model(model_type: str = "whisper", **kwargs) -> ASRModel:
    if model_type.lower() == "whisper":
        return FasterWhisperModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
