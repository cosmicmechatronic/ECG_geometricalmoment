import matplotlib.pyplot as plt

from ecgdetectors import Detectors
import numpy as np
import pandas as pd
import heartpy as hp
from scipy.stats import linregress
import operator
from shapely.geometry import  LineString
from scipy.signal import find_peaks

dtype=np.float32

class ExportResults():
    def __init__(self, signal, peaks_indexes, result, value):
        self.signal = signal
        self.peaks_indexes = peaks_indexes
        self.result = result
        self.value = value


    def export_results(self):

        signal=self.signal
        y_value = 0.98 * max(self.signal)
        peaks_indexes = self.peaks_indexes
        results = self.result

        fig = plt.figure(figsize=(8,6))
        plt.plot(signal)
        for i in range (0, len(peaks_indexes), 1):
            plt.text(peaks_indexes[i],y_value, f"{results[i]}", color="r" )
        plt.savefig(str(self.value) + '_result.png', dpi=480, bbox_inches='tight', transparent=True, pad_inches=0)
        plt.show()
        plt.close('all')




