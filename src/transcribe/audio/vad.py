import numpy as np
import onnxruntime
import warnings
import os

class OnnxWrapper:
    def __init__(self, path):
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"ONNX model not found at {path}")
            
        self.session = onnxruntime.InferenceSession(path, sess_options=opts)
        self.reset_states()
        self.sample_rates = [8000, 16000]

    def reset_states(self, batch_size=1):
        self._state = np.zeros((2, batch_size, 128), dtype=np.float32)
        self._context = np.zeros((batch_size, 0), dtype=np.float32)
        self._last_sr = 0
        self._last_batch_size = 0

    def _validate_input(self, x, sr: int):
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        if x.ndim > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.ndim}")
        if sr not in self.sample_rates:
             raise ValueError("Supported sampling rates: 8000, 16000")
        return x, sr

    def __call__(self, x, sr: int):
        x, sr = self._validate_input(x, sr)
        num_samples = 512 if sr == 16000 else 256
        if x.shape[-1] != num_samples:
            raise ValueError(f"Frame size must be {num_samples} samples")
            
        batch_size = x.shape[0]
        context_size = 64 if sr == 16000 else 32
        
        if not self._last_batch_size:
             self.reset_states(batch_size)
        
        if self._context.shape[1] == 0:
             self._context = np.zeros((batch_size, context_size), dtype=np.float32)
             
        x = np.concatenate([self._context, x], axis=1)
        
        ort_inputs = {
            'input': x,
            'state': self._state,
            'sr': np.array(sr, dtype='int64')
        }
        ort_outs = self.session.run(None, ort_inputs)
        out, state = ort_outs
        self._state = state
        self._context = x[:, -context_size:]
        self._last_batch_size = batch_size
        return out

class VADIterator:
    def __init__(self, model, threshold=0.5, sampling_rate=16000, min_silence_duration_ms=250, speech_pad_ms=30):
         self.model = model
         self.threshold = threshold
         self.sampling_rate = sampling_rate
         self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
         self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
         self.reset_states()

    def reset_states(self):
         self.model.reset_states()
         self.triggered = False
         self.temp_end = 0
         self.current_sample = 0

    def __call__(self, x):
        window_size_samples = x.shape[-1]
        self.current_sample += window_size_samples
        
        out = self.model(x, self.sampling_rate)
        speech_prob = out[0, 0] # Assume batch_size=1
        
        if (speech_prob >= self.threshold) and self.temp_end:
             self.temp_end = 0
             
        if (speech_prob >= self.threshold) and not self.triggered:
             self.triggered = True
             return {'start': max(0, self.current_sample - self.speech_pad_samples - window_size_samples)}
             
        if (speech_prob < self.threshold - 0.15) and self.triggered:
             if not self.temp_end:
                  self.temp_end = self.current_sample
             if self.current_sample - self.temp_end < self.min_silence_samples:
                  return None
             else:
                  speech_end = self.temp_end + self.speech_pad_samples - window_size_samples
                  self.temp_end = 0
                  self.triggered = False
                  return {'end': max(0, speech_end)}
        return None

import urllib.request

class VADTracker:
    def __init__(self, model_path=None, threshold=0.5, min_silence_ms=500, max_duration_s=15.0, speech_pad_ms=250, overlap_s=1.0):
        if model_path is None:
             cache_dir = os.path.expanduser("~/.cache/transcribe")
             os.makedirs(cache_dir, exist_ok=True)
             model_path = os.path.join(cache_dir, 'silero_vad.onnx')
             
             if not os.path.exists(model_path):
                  print(f"[VAD] Downloading Silero VAD model to {model_path}...")
                  url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
                  urllib.request.urlretrieve(url, model_path)
                  print("[VAD] Download complete.")
                  
        self.model = OnnxWrapper(model_path)
        self.iterator = VADIterator(self.model, threshold=threshold, min_silence_duration_ms=min_silence_ms, speech_pad_ms=speech_pad_ms)
        self.buffer = []
        self.is_speaking = False
        self.max_duration_s = max_duration_s
        self.overlap_s = overlap_s
        self.samples_recorded = 0
        
    def process_frame(self, frame_np):
         """
         frame_np: numpy array of shape (512,)
         Returns: numpy array of speech segment if segment finished, else None
         """
         res = self.iterator(frame_np)
         
         if res:
              if 'start' in res:
                   self.is_speaking = True
              elif 'end' in res:
                   self.is_speaking = False
                   seg = np.concatenate(self.buffer) if self.buffer else np.array([], dtype=np.float32)
                   self.buffer = []
                   self.samples_recorded = 0
                   if len(seg) > 16000 * 0.5: # Min 0.5s of speech
                        return seg
                        
         if self.is_speaking:
              self.buffer.append(frame_np)
              self.samples_recorded += len(frame_np)
              
              if self.samples_recorded >= 16000 * self.max_duration_s:
                   seg = np.concatenate(self.buffer)
                   
                   # Keep overlap for next segment
                   overlap_samples = int(16000 * self.overlap_s)
                   num_frames = overlap_samples // 512
                   if num_frames > 0 and len(self.buffer) > num_frames:
                       self.buffer = self.buffer[-num_frames:]
                   # NOTE: if we keep the buffer, we are still "speaking"
                   # self.is_speaking remains True
                   self.samples_recorded = len(self.buffer) * 512
                   return seg
                   
         return None
