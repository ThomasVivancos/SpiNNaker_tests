# Code provenant de https://github.com/NeuralEnsemble/PyNN/blob/master/examples/simple_STDP.py le 05/04/2021

import neo
from quantities import ms
import numpy
import math

# Callback pour enregistrer les poids à intervals réguliers
class WeightRecorder(object):
    """
    Recording of weights is not yet built in to PyNN, so therefore we need
    to construct a callback object, which reads the current weights from
    the projection at regular intervals.
    """

    def __init__(self, sampling_interval, projection):
        self.interval = sampling_interval
        self.projection = projection
        self._weights = []

    def __call__(self, t):
        print(t)
        if type(self.projection) != list:
            self._weights.append(self.projection.get('weight', format='list', with_address=False))
        elif type(self.projection) == list:
            for proj in self.projection:
                self._weights.append(proj.get('weight', format='list', with_address=False))
        return t + self.interval
    
    #retourne la totalité des poids de chaque callbacks sur une projection
    def get_weights(self):
        signal = neo.AnalogSignal(self._weights, units='nA', sampling_period=self.interval * ms,
                                name="weight")
        signal.channel_index = neo.ChannelIndex(numpy.arange(len(self._weights[0])))
        return signal

    # Retourne l'écart-type des poids de chaque callbacks sur une projection
    def get_standard_deviations(self):
        standard_deviations = []
        for i in range(len(self._weights)-1):
            standard_deviations.append(numpy.std(self._weights[i]))
        return standard_deviations

        
