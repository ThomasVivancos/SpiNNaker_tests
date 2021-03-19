import os
import numpy as np
import pyNN.utility.plotting as plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tonic
import tonic.transforms as transforms
import torch
import matplotlib.pyplot as plt
import tonic
import tonic.transforms as transforms
from collections import Counter


testset = tonic.datasets.DVSGesture(save_to='./data', train=False)

number_of_event_series=11
testloader = tonic.datasets.DataLoader(testset, batch_size=number_of_event_series,
                                       collate_fn=tonic.utils.pad_events,
                                       shuffle=False)

events, target = next(iter(testloader))

targets_indexes=[]
targets=[0,1,2,10]
for t in targets:
    targets_indexes.extend([i for i, x in enumerate(target) if x == t])

len(targets_indexes)
target[targets_indexes[1]]
events = events[targets_indexes]

target=np.array(target)
target=target[targets_indexes]

N_pop = np.prod(testset.sensor_size)
N_pop

index_x = testset.ordering.find("x")
index_y = testset.ordering.find("y")
index_p = testset.ordering.find("p") 
index_t = testset.ordering.find("t")

delay=events.max()


events.max()
i=0
for i in range(0,len(events)):
    events[i, :, 3]=events[ i,:, 3]+(i*1.2e+7)

# this takes quite long -> could we optimize it?
#numpy array for events - legacy line --> to be cleaned
events_np = events
#map an event in one pixel to a spike in one source neuron
cellSourceSpikes = []
#iterate over all event series4
for i in range(N_pop):
    # i is number of neuron -> row and col are pixels
    row = i//(testset.sensor_size[0])
    col = i%(testset.sensor_size[0])
    spike_times=[]
    copy_of_spike_times=[]
    if(row>19 and col>15 and row<116 and col<112  and  row%2==0 and col%2==0):
        for index0 in range(len(events)):
            if(row==0 and col==0):
                break
            spike_idx = np.where((events_np[index0, :, index_x] == row) & (events_np[index0, :, index_y] == col) & (events_np[index0, :, index_p] == 1.0))[0]
            spike_times.extend([float(round(e)) for e in events_np[index0, spike_idx, index_t] * 1.e-3]) # in milliseconds
            for t in range(0,len(spike_times)-1):
                if ((spike_times[t+1] - spike_times[t])>10):
                    copy_of_spike_times.append(spike_times[t])
        cellSourceSpikes.append(copy_of_spike_times)

total_len=0
for elem in cellSourceSpikes:
    total_len+=len(elem)

total_len
input_n=len(cellSourceSpikes)
input_n

cellSourceSpikes = [list(elem) for elem in cellSourceSpikes]

for i in range(len(cellSourceSpikes)):
    cellSourceSpikes[i]= list(set(cellSourceSpikes[i]))

data_series = {}
sorted_list = sorted(target)
sorted_counted = Counter(sorted_list)
for i in range(0,11):
    data_series[i] = 0

for key, value in sorted_counted.items():
    data_series[key] = value

data_series = pd.Series(data_series)
x_values = data_series.index

plt.bar(x_values, data_series.values)




import pyNN.spiNNaker as sim
from pyNN.random import NumpyRNG, RandomDistribution
simulator = 'spinnaker'
sim.setup(timestep=0.1, min_delay=0.1, max_delay=2)
randoms=np.random.rand(100,1)

#defining network topology
#here we have an example of the input -> this to be changed to our input and the rest of topology might stay the same
Input =  sim.Population(
    input_n,
    sim.SpikeSourceArray(spike_times=cellSourceSpikes),
    label="Input"
)
Input.record("spikes")

#neuron types here are the same as in the attention layer, but the parameters are slightly different, we will have 60 neurons here
LIF_Intermediate = sim.IF_curr_exp()
Intermediate = sim.Population(64, LIF_Intermediate, label="Intermediate")
Intermediate.record(("spikes","v"))

LIF_Output = sim.IF_curr_exp()
Output = sim.Population(4, LIF_Output, label="Output")
Output.record(("spikes","v"))
#same type of neurons, 10 neurons, different params

python_rng = NumpyRNG(seed=98497627)

delay = 1            # (ms) synaptic time delay
#A_minus 
stdp_proj = sim.STDPMechanism(
                timing_dependence=sim.SpikePairRule(tau_plus=20.0, tau_minus=20.0,
                                                    A_plus=0.6, A_minus=0.2),
                weight_dependence=sim.AdditiveWeightDependence(w_min=0.1, w_max=6),
                weight=RandomDistribution('normal',(3.0,2.9),rng=python_rng),
                delay=RandomDistribution('normal',(0.1,0.02),rng=python_rng))

# first projection with stdp
Conn_input_inter = sim.Projection(
    Input, Intermediate,
    connector=sim.FixedProbabilityConnector(0.70,allow_self_connections=False),
    synapse_type=stdp_proj,
    receptor_type="excitatory",
    label="Connection input to intermediate"
)
# second projection with stdp
Conn_inter_output = sim.Projection(
            Intermediate, Output,   # pre and post population
            connector=sim.FixedProbabilityConnector(0.10,
                allow_self_connections=False), # no autapses
            synapse_type=stdp_proj,
            receptor_type="excitatory",
            label="Connection intermediate to output"
        )

#adding competition between neurons so they don't all learn the same patterns
FixedInhibitory_WTA = sim.StaticSynapse(delay=RandomDistribution('normal',(0.1,0.02),rng=python_rng), weight=6)
WTA_INT = sim.Projection(
    Intermediate, Intermediate,
    connector=sim.FixedProbabilityConnector(1,allow_self_connections=False), #more than 30
    synapse_type=FixedInhibitory_WTA,
    receptor_type="inhibitory",
    label="Connection WTA"
)

WTA_OUT = sim.Projection(
    Output,Output,
    connector=sim.FixedProbabilityConnector(1,
                allow_self_connections=False),
    synapse_type=FixedInhibitory_WTA,
    receptor_type="inhibitory",
    label="Connection WTA"
)

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
    def get_weights(self):
        signal = neo.AnalogSignal(self._weights, units='nA', sampling_period=self.interval * ms,
                                  name="weight")
        signal.channel_index = neo.ChannelIndex(numpy.arange(len(self._weights[0])))
        return signal

max_time= 36000.0
sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 255)

weight_recorder1 = WeightRecorder(sampling_interval=1.0, projection=Conn_input_inter)
weight_recorder2 = WeightRecorder(sampling_interval=1.0, projection=Conn_inter_output)

simtime = max_time
sim.run(simtime) #, callbacks=[weight_recorder1,weight_recorder2])
neo = Output.get_data(variables=["spikes", "v"])
spikes = neo.segments[0].spiketrains
print(spikes)
v = neo.segments[0].filter(name='v')[0]
print(v)

neo_in = Input.get_data(variables=["spikes"])
spikes_in = neo_in.segments[0].spiketrains
#print(spikes_in)


neo_intermediate = Intermediate.get_data(variables=["spikes", "v"])
spikes_intermediate = neo_intermediate.segments[0].spiketrains
print(spikes_intermediate)
v_intermediate = neo_intermediate.segments[0].filter(name='v')[0]
print(v_intermediate)

sim.end()

plot.Figure(
    # plot voltage for first ([0]) neuron
    plot.Panel(v_intermediate, ylabel="Membrane potential (mV)",data_labels=[Output.label], yticks=True, xlim=(0, simtime)),
    # plot spikes (or in this case spike)
    plot.Panel(spikes, yticks=True, markersize=5, xlim=(0, simtime)),
    title="Simple Example",
    annotations="Simulated with {}".format(sim.name()),
    dpi=4000
)

plt.show()

plot.Figure(
    # plot voltage for first ([0]) neuron
    plot.Panel(v, ylabel="Membrane potential (mV)",data_labels=[Output.label], yticks=True, xlim=(0, simtime)),
    # plot spikes (or in this case spike)
    plot.Panel(spikes, yticks=True, markersize=5, xlim=(0, simtime)),
    title="Simple Example",
    annotations="Simulated with {}".format(sim.name()),
    dpi=4000
)

plt.show()