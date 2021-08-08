import os
import argparse
import warnings
import scipy as sc
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
import cv2
import heartpy as hp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import biosppy

dtype=np.float32

class Filter():
    def __init__(self, signal,frequency):
        self.signal=signal
        self.frequency=frequency

    def baseline_wander(self):
        #fs = abs(hp.get_samplerate_mstimer(self.signal))
        #print("Czestotliwosc z klasy Filter: ", fs)
        #filtered = hp.remove_baseline_wander(self.signal, fs)

        filtered = hp.remove_baseline_wander(self.signal, self.frequency)
        return filtered
