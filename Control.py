import os
import argparse
import warnings
import scipy as sc
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
import cv2

import numpy as np
import pandas as pd
import time
from ecgdetectors import Detectors
import heartpy as hp


from signal_loader import load_signals
from signal_loader_frequency import identify_frequency
import matplotlib.pyplot as plt
from utilspie import iterutils



from ReadSignal import ReadSignal
from Filter import Filter
from PeakDetect import PeakDetect
from FeatureExtract import FeatureExtract
from MomenntumCalculation import MomenntumCalculation
from ResultCalculation import ResultCalculation
from ExportResults import ExportResults
from MeshSignal import MeshSignal



class Control:
    def __init__(self, file_path, chunk_size):
        self.file_path = file_path
        self.chunk_size = chunk_size

    def chop_signal(self):
        warnings.filterwarnings('ignore')
        MITDB_PATH = self.file_path  # scieżka do folderu z sygnałami nie do konkretnych sygnałów
        MITDB_FORMAT = 'dat'  # typ bazy e - zewnętrzne

        mitdb = load_signals(MITDB_PATH, MITDB_FORMAT)
        signal = mitdb[0]['signal']
        signal_length = signal.size
        plt.close('all')
        #posiekanie sygnału na kwałki
        signal_list = np.arange(1,signal_length,1)

        result_list = list(iterutils.get_chunks(signal_list, self.chunk_size))

        #wyszukanie pierwszego i ostatniego elementu z listy kawałow
        first_last_element_list = []
        for i in range(0, len(result_list), 1):
            test_list = result_list[i]
            res = [ test_list[0], test_list[-1] ]
            first_last_element_list.append(res)


        return first_last_element_list

    def obtain_frequency(self):
        warnings.filterwarnings('ignore')
        MITDB_PATH = self.file_path  # scieżka do folderu z sygnałami nie do konkretnych sygnałów
        MITDB_FORMAT = 'dat'  # typ bazy e - zewnętrzne

        frequency = identify_frequency(MITDB_PATH, MITDB_FORMAT)


        return frequency


file_name='Sygnal_555'
interval_length = 1000
val=Control(file_name, interval_length)
#length=val.signal_length()
chunk=val.chop_signal()
print('ilosc interwalow: ', len(chunk))
print("Długość interwałów: ", chunk[0])
fs = val.obtain_frequency()
print("Czestotliwosc: ", fs)

i=3

value=i

signal = ReadSignal(file_name, chunk[i])
item = signal.read_signal()

# przewiltruj sygnal, wyrownaj baseline

signal_to_correction = Filter(item, fs)
corrected_signal = signal_to_correction.baseline_wander()

# plt.plot(fizz)
# plt.show()

# Znajdz piki, dostarcz liste pikow
signal_with_no_peaks = PeakDetect(corrected_signal)
peaks_indexes = signal_with_no_peaks.peak_detect()
print(peaks_indexes)

plt.plot(corrected_signal)
plt.plot(peaks_indexes, corrected_signal[peaks_indexes], "x")
plt.show()

# wyciecie fali |R|
cut_wave = FeatureExtract(corrected_signal, peaks_indexes)
signal_leaf = cut_wave.caught_wave()

print('Piki: ')
print(peaks_indexes)
"""
Cx_moments=[]
for i in range(0, len(peaks_indexes), 1):
    file=str(peaks_indexes[i]) + '.png'
    solve_moment = MomenntumCalculation(file)
    cx_momment=solve_moment.Calculate_moment_Cx()
    Cx_moments.append(cx_momment)

print('Pokaz Cx moments:')
print(Cx_moments)
grab_result = ResultCalculation( corrected_signal,peaks_indexes,Cx_moments  )
result = grab_result.result_calculation()
print('pokaz result')
print(result)

visualize = ExportResults(corrected_signal, peaks_indexes, result, value )
final_product=visualize.export_results()
"""
## szukanie wzorców
wave = MeshSignal(corrected_signal, peaks_indexes)
plot_wave = wave.plot_wave()



"""
for i in range(0, len(chunk)-1, 1):
    signal = ReadSignal(file_name, chunk[i])
    item = signal.read_signal()
    value=i

    # przewiltruj sygnal, wyrownaj baseline
    signal_to_correction = Filter(item,fs)
    corrected_signal = signal_to_correction.baseline_wander()

    # plt.plot(fizz)
    # plt.show()

    # Znajdz piki, dostarcz liste pikow
    signal_with_no_peaks = PeakDetect(corrected_signal)
    peaks_indexes = signal_with_no_peaks.peak_detect()
    print(peaks_indexes)

    #plt.plot(corrected_signal)
    #plt.plot(peaks_indexes, corrected_signal[peaks_indexes], "x")
    #plt.show()

    # wyciecie fali |R|
    cyff = FeatureExtract(corrected_signal, peaks_indexes)
    aff = cyff.caught_wave()

    Cx_moments=[]
    for i in range(0, len(peaks_indexes), 1):
        file=str(peaks_indexes[i]) + '.png'
        solve_moment = MomenntumCalculation(file)
        cx_momment=solve_moment.Calculate_moment_Cx()
        Cx_moments.append(cx_momment)

    grab_result = ResultCalculation( corrected_signal,peaks_indexes,Cx_moments  )
    result = grab_result.result_calculation()
    #print('pokaz result')
    #print(result)

    visualize = ExportResults(corrected_signal, peaks_indexes, result, value )
    final_product=visualize.export_results()

"""



