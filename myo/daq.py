import queue
from threading import Lock, Thread

import myo
import numpy as np

class _MyoDaq(myo.DeviceListener):
    def __init__(self, samples_per_read):
        super().__init__()
        self.samples_per_read = samples_per_read
        self.data_queue = queue.Queue()
        self._hub = myo.Hub()
        self._lock = Lock()

    def start(self):
        self._thread = Thread(target=self._run)
        self._flag = True
        self._thread.start()

    def _run(self):
        with self._hub.run_in_background(self.on_event):
            while self._flag:
                pass

    def stop(self):
        self._flag = False
        self._hub.stop()

    def read(self):
        data = []
        while len(data) < self.samples_per_read:
            try:
                data.append(self.data_queue.get())
            except IndexError:
                print('pass')

        data = np.atleast_2d(np.asarray(data)).T
        return data[self.channels, :]

    def reset(self):
        self.data_queue.queue.clear()

class MyoDaqEMG(_MyoDaq):
    def __init__(self, channels, samples_per_read):
        super().__init__(samples_per_read)
        self.channels = channels

    def on_connected(self, event):
        event.device.stream_emg(True)

    def on_emg(self, event):
        with self._lock:
            self.data_queue.put(event.emg)

class MyoDaqIMU(_MyoDaq):
    def __init__(self, samples_per_read):
        super().__init__(samples_per_read)

    def on_connected(self, event):
        event.device.stream_imu(True) #TODO double-check this

    def on_emg(self, event):
        with self._lock:
            self.data_queue.put(event.orientation)
