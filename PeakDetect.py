from ecgdetectors import Detectors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import heartpy as hp
from ecgdetectors import Detectors
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
import biosppy

dtype=np.float32

class PeakDetect:
    def __init__(self, signal):
        self.signal=signal

    def peak_detect(self):
        peaks, _ = find_peaks(self.signal, height=0.4*max(self.signal))

        return peaks
