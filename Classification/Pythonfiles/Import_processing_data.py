import numpy as np
import pandas as pd
import tonic
import tonic.transforms as transforms
import torch
import os
from collections import Counter



def importer_data(train):
    # Import du dataset test
    dl = not os.path.isfile('data/gesture.zip')
    testset = tonic.datasets.DVSGesture(save_to='./data', train=False)

    testloader = tonic.datasets.DataLoader(testset, batch_size=11,  collate_fn=tonic.utils.pad_events, shuffle=False)

    events, target = next(iter(testloader))



    # Réduction du dataset à 5 classes pour gagner du temps, à commenter si on veut passer tout le dataset
    # we here filter the videos to a subset of the classes, we will use 5 classes out of 10,
    # [0,1,2,3, 10 :arm roll, hand clap, left hand clockwise, left hand counter clockwise, other gesture]
    targets_indexes=[]
    targets_indexes=[i for i, x in enumerate(target) if (x==0 or x==1 or x==2 or x==3 or x==10)]

    len(targets_indexes)

    events = events[targets_indexes]
    target=np.array(target)
    target=target[targets_indexes]

    return testset, target, events


def order_events(events, target):
    # here we get the lengths of the videos so we can delay the next video with the needed amount
    # delay_times is used to setup the dealys of the videos to keep them in memory
    # start_end_lenghts_label is used in the information extraction at the end to control the output

    start_end_lenghts_label=[]
    start_end_lenghts_label.append((0,events[0].max(),events[0].max(),target[0]))
    delay_times=[]
    delay_times.append(0)
    for i in range(1,len(events)):
        d=delay_times[i-1] + 5.0e+5 + events[i-1].max()
        start_end_lenghts_label.append((d,d+events[i].max(),events[i].max(),target[i]))
        delay_times.append(d)

    max_time = delay_times[(len(delay_times)-1)].max() * 1.e-3

    # On modifie events pour avoir une séquence continue des vidéos du dataset
    # delay event times in events to have a sequence
    # event[0] delay =0
    # event[1] delay = previous_delay
    i=0
    for i in range(0,len(events)):
        events[i, :, 3]=events[ i,:, 3]+delay_times[i]
    
    return  start_end_lenghts_label, delay_times, events, max_time


def make_source_array(events, testset):
    N_pop = np.prod(testset.sensor_size)
    index_x = testset.ordering.find("x")
    index_y = testset.ordering.find("y")
    index_p = testset.ordering.find("p") 
    index_t = testset.ordering.find("t")
    # numpy array for events - legacy line --> to be cleaned
    events_np = events
    # map an event in one pixel to a spike in one source neuron
    cellSourceSpikes = []
    # iterate over all event series
    for i in range(N_pop):
        # i is number of neuron -> row and col are pixels
        row = i//(testset.sensor_size[0])
        col = i%(testset.sensor_size[0])
        spike_times=[]
        copy_of_spike_times=[]
        # réduction de la taille de l'image
        if (row > 19 and col > 15 and row < 116 and col < 112  and  row%3==0 and col%3==0):
            for index0 in range(len(events)):
                #pour chaque groupe de 3*3 pixel on ne récupère que les évènements d'un seul
                spike_idx = np.where((events_np[index0, :, index_x] == row) & (events_np[index0, :, index_y] == col) & (events_np[index0, :, index_p] == 1.0))[0]
                spike_times.extend([float(round(e)) for e in events_np[index0, spike_idx, index_t] * 1.e-3]) # arrondissement du timestamp des évènements en millisecondes

                # réduction du nombre d'évènements
                for t in range(0,len(spike_times)-1):
                    if ((spike_times[t+1] - spike_times[t])>10):
                        copy_of_spike_times.append(spike_times[t])
            cellSourceSpikes.append(copy_of_spike_times)
    
    # ToList => évite les ambiguïté dans PyNN
    cellSourceSpikes = [list(elem) for elem in cellSourceSpikes]

    # Suppression des doublons éventuels
    for i in range(len(cellSourceSpikes)):
        cellSourceSpikes[i]= list(set(cellSourceSpikes[i]))

    return cellSourceSpikes


