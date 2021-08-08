import os
import argparse
import warnings
import scipy as sc
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import wfdb


from signal_loader import load_signals
from Filter import Filter

class ReadSignal():
    def __init__(self, signal_path, interval_index):
        self.signal_path = signal_path
        self.interval_index = interval_index


    def read_signal(self):
        warnings.filterwarnings('ignore')
        MITDB_PATH = self.signal_path  # scieżka do folderu z sygnałami nie do konkretnych sygnałów
        MITDB_FORMAT = 'dat'  # typ bazy e - zewnętrzne
        mitdbsvt = load_signals(MITDB_PATH, MITDB_FORMAT)
        mitdb = load_signals(MITDB_PATH, MITDB_FORMAT)
        start_index=self.interval_index[0]
        stop_index=self.interval_index[1]
        signal = mitdb[0]['signal'][start_index:stop_index]



        return signal



