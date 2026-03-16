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
    
    def init(self):
        self = objc.super(AudioStreamOutput, self).init()
        if self is None:
            return None
        self.queue = Queue()
        self.running = False
        return self

    # Explicit signature: void return, self, selector, stream (id), sampleBuffer (id), type (NSInteger)
    @objc.signature(b'v@:@@q')
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
            
            status = self.stream.addStreamOutput_type_sampleHandlerQueue_error_(self.delegate, 1, None, None)
            success = status[0] if isinstance(status, tuple) else status
            if success:
                self.delegate.running = True
                def completion_handler(err):
                    if err:
                        print(f"Start capture err: {err}")
                    else:
                        self.is_recording = True
                        print("Recording started...")
                    self.start_event.set()
                self.stream.startCaptureWithCompletionHandler_(completion_handler)
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
            if error:
                print(f"Error getting content: {error}")
                self.start_event.set()
                return
            
            # Dispatch to main thread
            self.performSelectorOnMainThread_withObject_waitUntilDone_("setupStream:", content, False)

        sck.SCShareableContent.getShareableContentWithCompletionHandler_(handle_content)
        
        from Foundation import NSRunLoop, NSDate
        loop = NSRunLoop.currentRunLoop()
        timeout_time = time.time() + 10.0
        while not self.start_event.is_set() and time.time() < timeout_time:
            # Run loop for short burst to process performSelector
            loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.1))
        
        if not self.is_recording:
            print("Warning: Recording might not have started correctly (timeout or error)")

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
