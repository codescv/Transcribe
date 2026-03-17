import abc
import numpy as np
from typing import Iterator

INITIAL_PRMOPT = '这是一段科技会议，包含中英文混用，例如：我们的新 Feature 表现非常 Cool。'

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

class MLXWhisperModel(ASRModel):
    

    def __init__(self, model_size: str = "mlx-community/whisper-large-v3-turbo"):
        """
        Initialize MLX-Whisper model.
        
        Args:
            model_size: HuggingFace repo ID or path, e.g., 'mlx-community/whisper-large-v3-turbo'
        """
        import mlx_whisper
        print(f"Loading MLX-Whisper model: {model_size}...")
        self.model_size = model_size

    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio array.
        
        Args:
            audio_data: np.ndarray of shape (N,) containing float32 samples at 16000 Hz.
        """
        if audio_data.size == 0:
            return ""
            
        import mlx_whisper
        result = mlx_whisper.transcribe(
            audio_data, 
            path_or_hf_repo=self.model_size,
            initial_prompt=INITIAL_PRMOPT
        )
        
        return result.get("text", "").strip()


class MLXSenseVoiceModel(ASRModel):
    def __init__(self, model_size: str = "mlx-community/SenseVoiceSmall"):
        """
        Initialize MLX-SenseVoice model.
        
        Args:
            model_size: HuggingFace repo ID, e.g., 'mlx-community/SenseVoiceSmall'
        """
        print(f"Loading MLX-SenseVoice model: {model_size}...")
        self.model_size = model_size
        self.model = None

    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio array.
        
        Args:
            audio_data: np.ndarray of shape (N,) containing float32 samples at 16000 Hz.
        """
        if audio_data.size == 0:
            return ""
            
        from mlx_audio.stt import load
        if self.model is None:
            self.model = load(self.model_size)
            
        result = self.model.generate(audio_data, language="auto")
        return result.text.strip()


def download_model_files(model_type: str, model_size: str) -> None:
    """
    Download/verify model files from remote repository without loading to memory/Metal.
    """
    print(f"Downloading/Verifying {model_type} model ({model_size}) in background (if needed)...")
    if model_type.lower() == "whisper":
        from faster_whisper import download_model
        download_model(model_size)
    elif model_type.lower() == "mlx-whisper":
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_size)
    elif model_type.lower() in ["mlx-sensevoice", "sensevoice"]:
        from mlx_audio.utils import get_model_path
        get_model_path(model_size)
    else:
        # We don't raise here if it might just be local paths or not supported but we can warn
        # Actually raising is safer if they pass nonsense
        raise ValueError(f"Unknown model type for download: {model_type}")

def get_model(model_type: str = "whisper", **kwargs) -> ASRModel:

    if model_type.lower() == "whisper":
        return FasterWhisperModel(**kwargs)
    elif model_type.lower() == "mlx-whisper":
        return MLXWhisperModel(**kwargs)
    elif model_type.lower() in ["mlx-sensevoice", "sensevoice"]:
        return MLXSenseVoiceModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

