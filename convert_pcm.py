import wave
import sys
import os

def convert_pcm_to_wav(pcm_path="debug_audio.pcm", wav_path="debug_audio.wav"):
    if not os.path.exists(pcm_path):
        print(f"Error: {pcm_path} not found.")
        return

    channels = 1        # Mono
    sample_width = 2   # 16-bit (2 bytes)
    sample_rate = 16000 # 16kHz

    import numpy as np

    print(f"Reading {pcm_path}...")
    with open(pcm_path, 'rb') as pcm:
        pcm_data = pcm.read()

    print(f"Converting Float32 to Int16 for WAV output...")
    # ScreenCaptureKit delivers float32
    audio_float = np.frombuffer(pcm_data, dtype=np.float32)
    # Scale to int16 range
    audio_int16 = (audio_float * 32767.0).astype(np.int16)

    print(f"Writing {wav_path}...")
    with wave.open(wav_path, 'wb') as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.tobytes())

    print(f"Successfully converted to {wav_path}")
    print(f"You can now play {wav_path} using QuickTime or any media player.")

if __name__ == "__main__":
    pcm = sys.argv[1] if len(sys.argv) > 1 else "debug_audio.pcm"
    wav = sys.argv[2] if len(sys.argv) > 2 else "debug_audio.wav"
    convert_pcm_to_wav(pcm, wav)
