import objc
import Foundation
from Foundation import NSObject, NSLog
import ScreenCaptureKit as sck
import CoreMedia as cm
import numpy as np
import threading
from queue import Queue
import time
import os

# Ensure we can look up classes if needed
# objc.lookUpClass('SCStream')

class AudioStreamOutput(NSObject):
    """Delegate for SCStream to receive audio sample buffers."""
    __pyobjc_protocols__ = [objc.protocolNamed('SCStreamOutput')] if hasattr(objc, 'protocolNamed') else []
    
    def init(self):
        self = objc.super(AudioStreamOutput, self).init()
        if self is None:
            return None
        self.queue = Queue()
        self.running = False
        return self

    # Explicit signature: void return, self, selector, stream (id), sampleBuffer (CMSampleBufferRef), type (NSInteger)
    @objc.signature(b'v@:@^{opaqueCMSampleBuffer=}q')
    def stream_didOutputSampleBuffer_ofType_(self, stream, sampleBuffer, type):
        if not self.running:
            return

        if type != 1: # SCStreamOutputTypeAudio
            return

        block_buffer = cm.CMSampleBufferGetDataBuffer(sampleBuffer)
        if not block_buffer:
            return

        length = cm.CMBlockBufferGetDataLength(block_buffer)
        if length <= 0:
            return

        try:
            data = bytearray(length)
            status = cm.CMBlockBufferCopyDataBytes(block_buffer, 0, length, data)
            status_code = status[0] if isinstance(status, tuple) else status
            if status_code == 0:
                actual_data = status[1] if isinstance(status, tuple) else data
                self.queue.put(bytes(actual_data))
        except Exception as e:
            pass

class ScreenAudioRecorder(NSObject):
    def init(self):
        self = objc.super(ScreenAudioRecorder, self).init()
        if self is None:
            return None
        self.delegate = AudioStreamOutput.alloc().init()
        self.stream = None
        self.is_recording = False
        self.start_event = threading.Event()
        return self

    def setupStream_(self, content):
        try:
            print("[DEBUG] setupStream_ entered")
            # Create stream configuration
            config = sck.SCStreamConfiguration.alloc().init()
            # Configure for audio only
            config.setCapturesAudio_(True)
            config.setExcludesCurrentProcessAudio_(True) # Don't capture own output if we speak
            config.setSampleRate_(16000)
            config.setChannelCount_(1)
            
            displays = content.displays()
            if not displays:
                print("No displays found.")
                self.start_event.set()
                return
            
            filter = sck.SCContentFilter.alloc().initWithDisplay_excludingApplications_exceptingWindows_(displays[0], [], [])
            
            self.stream = sck.SCStream.alloc().initWithFilter_configuration_delegate_(filter, config, self.delegate)
            
            print("[DEBUG] Adding stream output...")
            status = self.stream.addStreamOutput_type_sampleHandlerQueue_error_(self.delegate, 1, None, None)
            success = status[0] if isinstance(status, tuple) else status
            print(f"[DEBUG] addStreamOutput status: {status}, success: {success}")
            if success:
                self.delegate.running = True
                def completion_handler(err):
                    print(f"[DEBUG] completion_handler called with err: {err}")
                    if err:
                        print(f"Start capture err: {err}")
                    else:
                        self.is_recording = True
                        print("Recording started...")
                    self.start_event.set()
                print("[DEBUG] Calling startCaptureWithCompletionHandler_")
                # Keep alive to prevent garbage collection
                self._start_completion_handler = completion_handler
                self.stream.startCaptureWithCompletionHandler_(self._start_completion_handler)
            else:
                print("Failed to add stream output.")
                self.start_event.set()
        except Exception as e:
            print(f"Exception in setupStream_: {e}")
            self.start_event.set()

    def start(self):
        if self.is_recording:
            return
        
        self.start_event.clear()
        
        # Setup stream
        def handle_content(content, error):
            print(f"[DEBUG] handle_content called. Error: {error}")
            if error:
                print(f"Error getting content: {error}")
                self.start_event.set()
                return
            
            print("[DEBUG] Dispatching setupStream_ to main thread")
            # Dispatch to main thread
            self.performSelectorOnMainThread_withObject_waitUntilDone_("setupStream:", content, False)

        print("[DEBUG] Requesting shareable content...")
        self._handle_content_handler = handle_content
        sck.SCShareableContent.getShareableContentWithCompletionHandler_(self._handle_content_handler)
        print("[DEBUG] start() async chain initiated")

    def stop(self):
        if not self.is_recording:
            return
        if self.stream:
            self.stream.stopCaptureWithCompletionHandler_(lambda err: None)
        self.delegate.running = False
        self.is_recording = False
        print("Recording stopped.")

    def get_audio_data(self):
        if not self.delegate.queue.empty():
            return self.delegate.queue.get()
        return None

class MicAudioRecorder:
    def __init__(self):
        # Create a dummy delegate with a queue to match ScreenAudioRecorder structure
        class DummyDelegate:
            def __init__(self):
                self.queue = Queue()
        self.delegate = DummyDelegate()
        self.is_recording = False
        self.start_event = threading.Event()
        self.stream = None

    def start(self):
        if self.is_recording:
            return
            
        import sounddevice as sd
        
        def callback(indata, frames, time_info, status):
            if status:
                print(f"[Mic] Status: {status}")
            # indata is a numpy array of shape (frames, channels)
            if np.any(indata):
                # Put raw bytes into queue to match ScreenCaptureKit output
                self.delegate.queue.put(indata.tobytes())

        try:
            # 16000 Hz, 1 channel, float32 matches SCK configuration
            self.stream = sd.InputStream(samplerate=16000, channels=1, dtype='float32', callback=callback)
            self.stream.start()
            self.is_recording = True
            print("Microphone recording started...")
        except Exception as e:
            print(f"Error starting microphone recording: {e}")
        finally:
            self.start_event.set()

    def stop(self):
        if not self.is_recording:
            return
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_recording = False
        print("Microphone recording stopped.")

    def get_audio_data(self):
        if not self.delegate.queue.empty():
            return self.delegate.queue.get()
        return None

def get_recorder(source_type: str = "system"):
    """
    Factory to get the correct audio recorder.
    source_type: 'system' or 'output' for ScreenCaptureKit,
                 'mic' or 'input' for microphone via sounddevice.
    """
    if source_type.lower() in ["mic", "input"]:
        return MicAudioRecorder()
    else:
        return ScreenAudioRecorder.alloc().init()
