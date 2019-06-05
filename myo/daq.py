import queue
from threading import Lock, Thread
import time

import myo
import numpy as np


class _MyoDaq(myo.DeviceListener):
    """
    Base class for Myo daq devices

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    def __init__(self):
        super().__init__()
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
                time.sleep(1e-6)

    def stop(self):
        self._flag = False
        self._hub.stop()

    def read(self):
        raise NotImplementedError

    def reset(self):
        self.data_queue.queue.clear()


class MyoDaqEMG(_MyoDaq):
    """
    Myo armband EMG DAQ emulation.

    Requires the MyoConnect application to be running.

    Parameters
    ----------
    channels : list or tuple
        Sensor channels to use. Each sensor has a single EMG
        channel.
    samples_per_read : int
        Number of samples per channel to read in each read operation.

    Attributes
    ----------
    data_queue : Queue
        Fifo Queue to store incoming data.
    """

    def __init__(self, channels, samples_per_read):
        super().__init__()
        self.channels = channels
        self.samples_per_read = samples_per_read

    def on_connected(self, event):
        """Enables EMG streaming."""
        event.device.stream_emg(True)

    def on_emg(self, event):
        """Updates data queue when an EMG event happens."""
        with self._lock:
            self.data_queue.put(event.emg)

    def read(self):
        """
        Request a sample of data from the device.

        This is a blocking method, meaning it returns only once the requested
        number of samples are available.

        Returns
        -------
        data : ndarray, shape=(num_channels, num_samples)
            Data read from the device. Each channel is a row and each column
            is a point in time.
        """
        data = []
        while len(data) < self.samples_per_read:
            try:
                data.append(self.data_queue.get())
            except IndexError:
                pass

        data = np.atleast_2d(np.asarray(data)).T
        return data[self.channels, :]


class MyoDaqIMU(_MyoDaq):
    """
    Myo armband IMU DAQ emulation.

    Requires the MyoConnect application to be running.

    Parameters
    ----------
    samples_per_read : int
        Number of samples per channel to read in each read operation.

    Attributes
    ----------
    data_queue : Queue
        Fifo Queue to store incoming data.
    """

    def __init__(self, samples_per_read):
        super().__init__()
        self.samples_per_read = samples_per_read

    def on_connected(self, event):
        """Enables RSII streaming."""
        event.device.request_rssi()

    def on_orientation(self, event):
        """Updates data queue when an orientation event happens."""
        with self._lock:
            self.data_queue.put(event.orientation)

    def read(self):
        """
        Request a sample of data from the device.

        This is a blocking method, meaning it returns only once the requested
        number of samples are available.

        Returns
        -------
        data : ndarray, shape=(num_channels, num_samples)
            Data read from the device. Each channel is a row and each column
            is a point in time.
        """
        data = []
        while len(data) < self.samples_per_read:
            try:
                data.append(np.array(list(self.data_queue.get())))
            except IndexError:
                pass

        data = np.atleast_2d(data).T
        return data
