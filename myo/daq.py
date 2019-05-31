from collections import deque
from threading import Lock, Thread
import time

import myo
import numpy as np


class _MyoListener(myo.DeviceListener):
    """Base simple Myo device listener."""
    def __init__(self):
        self.lock = Lock()
        self.data = []

    def get_data(self):
        with self.lock:
            return list(self.data)


class MyoEmgListener(_MyoListener):
    """Simple Myo EMG device listener."""
    def __init__(self):
        super().__init__()

    def on_connected(self, event):
        event.device.stream_emg(True)

    def on_emg(self, event):
        with self.lock:
            self.data = event.emg



class MyoIMUListener(_MyoListener):
    """Simple Myo IMU device listener."""
    def __init__(self):
        super().__init__()

    def on_connected(self, event):
        event.device.stream_emg(True)

    def on_orientation(self, event):
        with self.lock:
            self.data = event.orientation


class _MyoDaq(object):
    """Base Myo DAQ device."""
    def __init__(self, channels, samples_per_read):
        self.channels = np.asarray(channels)
        self.samples_per_read = samples_per_read
        self.listener = None
        self.rate = None
        self._total_channels = None

    def start(self):
        myo.init(sdk_path='Coding/myo-python/myo-sdk-win-0.9.0')
        self.hub = myo.Hub()

        self._flag = True
        self._thread = Thread(target=self._run)
        self._thread.start()

    def _run(self):
        with self.hub.run_in_background(self.listener.on_event):
            while self._flag:
                data = []
                while len(data) < self.samples_per_read:
                    cur_data = self.listener.get_data()
                    data.append(cur_data)
                    # Whenever queried, MYO will send data even if they haven't
                    # been updated. Therefore, we need to wait until next data
                    # become available so as to not read the same data twice.
                    time.sleep(1. / self.rate)

                data = np.asarray(data)
                if data.squeeze().shape == (self.samples_per_read,
                                            self._total_channels):
                    self.data = data[:, self.channels]
                else:
                    self.data = np.zeros((self.samples_per_read,self.channels.size))

    def read(self):
        return self.data

    def stop(self):
        self._flag = False
        self.hub.stop()

class MyoDaqEMG(_MyoDaq):
    """
    MYO EMG DAQ device.

    Data acquisition and updating is implemented using a Thread.

    Parameters
    ----------
    channels : list or tuple
        Sensor channels to use. Each sensor has a single EMG
        channel.

    samples_per_read : int
        Number of samples per channel to read in each read operation.

    Attributes
    ----------
    rate : int
        Sampling rate in Hz.
    """

    def __init__(self, channels, samples_per_read):
        super().__init__(channels=channels,
                                        samples_per_read=samples_per_read)
        self.listener = MyoEmgListener()
        self._total_channels = 8
        self._rate = 200.

class MyoDaqIMU(_MyoDaq):
    """
    MYO IMU DAQ device.

    Data acquisition and updating is implemented using a Thread.

    Parameters
    ----------
    channels : list or tuple
        Sensor channels to use. Each sensor has a single EMG
        channel.

    samples_per_read : int
        Number of samples per channel to read in each read operation.

    Attributes
    ----------
    rate : int
        Sampling rate in Hz.
    """
    def __init__(self, samples_per_read):
        super().__init__(channels=1,
                                        samples_per_read=samples_per_read)
        self.listener = MyoIMUCollector()
        self._total_channels = 1
        self._rate = 50.
