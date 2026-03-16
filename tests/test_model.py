import pytest
from transcribe.model.model import get_model, MLXWhisperModel

def test_get_model_whisper():
    model = get_model(model_type="whisper")
    assert model.__class__.__name__ == "FasterWhisperModel"

def test_get_model_mlx_whisper():
    # Initializing MLXWhisperModel shouldn't download anything on __init__
    model = get_model(model_type="mlx-whisper")
    assert isinstance(model, MLXWhisperModel)
    assert model.model_size == "mlx-community/whisper-large-v3-turbo"
