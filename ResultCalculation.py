import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


dtype=np.float32

class ResultCalculation():
    def __init__(self, signal, peaks_indexes, Cx_moments):
        self.signal = signal
        self.peaks_indexes = peaks_indexes
        self.Cx_moments = Cx_moments

    def result_calculation(self):

        #1700.00
        Mean_Value_Vetricular_Moment_Cx = 2350.00
        Vetricular_Moment_Cx_std = 50.00

        Mean_Value_Normal_Signal_Moment_Cx = 2149.00
        Normal_Signal_Moment_Cx_std = 150.00

        y_value = 1.2*max(self.signal)
        peaks_indexes=self.peaks_indexes
        Cx_moments=self.Cx_moments

        result = []

        for i in range(0, len(peaks_indexes), 1 ):
            if ((Mean_Value_Vetricular_Moment_Cx -Vetricular_Moment_Cx_std) <Cx_moments[i] ) and (Cx_moments[i] < (Mean_Value_Vetricular_Moment_Cx +Vetricular_Moment_Cx_std)):
                result.append('V')
            elif ((Mean_Value_Normal_Signal_Moment_Cx -Normal_Signal_Moment_Cx_std)<Cx_moments[i]   ) and (Cx_moments[i] < (Mean_Value_Normal_Signal_Moment_Cx + Normal_Signal_Moment_Cx_std)):
                result.append('N')

            else:
                result.append('NaN')

        return result