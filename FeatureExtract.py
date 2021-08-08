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

class FeatureExtract():

    def __init__(self, signal, peaks_indexes):
        self.signal = signal
        self.peaks_indexes = peaks_indexes


    def caught_wave(self):

        fs = abs(hp.get_samplerate_mstimer(self.signal))
        amount_indexes = len(self.peaks_indexes)
        signal = self.signal
        #print('ilosc rozpoznanych pikow: ',amount_indexes )
        signal_length=len(signal)
        #print('dlugosc sygnalu: ',signal_length )


        for i in range (0, amount_indexes, 1):
            #print('<-Feature Extraction - start->')
            peak_value = self.peaks_indexes[i]
            #print('Wartosc pika: ', i+1 ,' ',  peak_value)

            # calculate distance:
            if i < (amount_indexes-1):
                #wg Kamila
                #distance = 0.5 * np.diff(np.array[self.peaks_indexes[i+1]], np.array[self.peaks_indexes[i]])
                distance = 0.5 * (self.peaks_indexes[i+1] - self.peaks_indexes[i])
                #print('Dystans: ', int(distance))


            else:
                # wg Kamila

                distance = 0.5 * (self.peaks_indexes[i] - self.peaks_indexes[i-1])
                #print('Dystans: ', int(distance))


            #print('Indeksy start&stop: ')
            start_index = int(self.peaks_indexes[i] - distance)
            if start_index < 0:
                start_index =0

            stop_index = int(self.peaks_indexes[i] + distance)
            if stop_index > signal_length:
                stop_index=signal_length

            #print('Indeks do wyciecie pomiedzy pikami R-R')
            #print('Start index: ', start_index)
            #print('Stop index: ', stop_index)

            #wyciety sygnał pomiedzy dwoma pikami R-R
            caught_signal = self.signal[start_index:stop_index]


            length = caught_signal.size
            x_val = np.arange(0, length, 1)

            x_values = np.array([x_val[0], x_val[length-1]])
            y_values = np.array([caught_signal[0], caught_signal[length-1]])
            slope, intercept, r_value, p_value, std_err = linregress(x_values, y_values)
            #print("slope: %f    intercept: %f" % (slope, intercept))

            #gdzie szukamy pika
            peaks, _ = find_peaks(caught_signal, height=0.6 * max(caught_signal))

            print('znaleziony pik: ', peaks)

            val=peaks[0]
            cut_data_R=[]
            x_data_R=[]
            cut_data_l=[]
            x_data_l = []

            # podroz w lewo w szukaniu elementów
            #zchodzimy z pika w lewo (malejaco po x) do otoczenia linii przecinajacej
            for i in range (val,0, -1):
                #wg kamila
                #x = np.argwhere(caught_signal > (intercept + slope * np.arange(val, caught_signal.size) + 100)).min()
                #ValueError: operands could not be broadcast together
                #print('pkt wg Kamila', x)
                if caught_signal[i]> (intercept + slope * i)+100:
                    cut_data_l.append(caught_signal[i])
                    x_data_l.append(i)
                if caught_signal[i]< (intercept + slope * i)+100:
                    break

            #zmiana kierunku zapisywaia listy
            cut_data_L=[]
            for i in range(len(cut_data_l)-1, 0, -1):
                cut_data_L.append(cut_data_l[i])
            #print('cut_data_L: ', cut_data_L)

            x_data_L = []
            for i in range(len(x_data_l)-1, 0, -1):
                x_data_L.append(x_data_l[i])
            #print('cut_data_Left side: ', cut_data_L)
            #print('x_data_Left sie: ', x_data_L)

            # podroz w prawo w szukaniu elementów:
            # zchodzimy z pika w prawo (malejaco) do otoczenia linii przecinajacej
            for i in range (val, len(caught_signal), 1):
                #wg Kamila:
                #x = np.argwhere(caught_signal > (intercept + slope * np.arange(val, caught_signal.size)+ 100)).min()
                #ValueError: operands could not be broadcast together
                #print('pkt wg Kamila', x)
                if caught_signal[i]> (intercept + slope * i)+100:
                    cut_data_R.append(caught_signal[i])
                    x_data_R.append(i)

                if caught_signal[i]< (intercept + slope * i)+100:
                    break
            #print('cut_data_Right side: ', cut_data_R)
            #print('x_data_Right sie: ', x_data_R)
            ###
            #x =np.argwhere(caught_signal > (intercept + slope * np.arange(val, caught_signal.size) + 100)).min()
            ###


            cut_data =cut_data_L+cut_data_R
            x_data=x_data_L+x_data_R

            initial_point=x_data[0]
            end_point=x_data[-1]
            cut_data[0]=intercept + slope*x_data[0]
            cut_data[-1] = intercept + slope * x_data[-1]



            fig = plt.figure(figsize=(8, 6))
            plt.plot(caught_signal, label='original data')
            # plt.scatter(x_val,caught_signal, s=1,label='original data')
            plt.plot(x_val, intercept + slope * x_val, 'r', label='inresection line')
            plt.legend()
            #plt.show()
            plt.close('all')



            wave_y=[]
            for i in range (0, len(x_data), 1):
                wave_y.append(cut_data[0])


            fig = plt.figure( figsize=(8,6))
            plt.axis('off')
            plt.plot(x_val[0:initial_point],caught_signal[0:initial_point], color='w', linewidth=0.1)
            plt.plot(x_data, cut_data, color='k', linewidth=3)
            plt.plot(x_data,wave_y, color='k', linewidth=3 )
            plt.plot(x_val[end_point: -1],caught_signal[end_point: -1], color='w', linewidth=0.1)
            plt.plot(x_data, wave_y, color='k', linewidth=3)
            plt.fill_between(x_data,cut_data, wave_y, color='k')
            plt.savefig(str(peak_value) + '.png', dpi=720, bbox_inches='tight', transparent=True, pad_inches=0)
            #rplt.show()
            plt.close('all')


            print('<------------->')

