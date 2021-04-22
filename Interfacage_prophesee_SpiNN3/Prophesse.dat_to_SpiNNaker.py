import os
import pandas as pd
import numpy as np
import tonic
import tonic.transforms as transforms
from aertb.core import FileLoader
from pyNN.parameters import Sequence
import time as temps

datLoader = FileLoader('dat')

# ligne de commande pour la transformation des .raw en .dat
#metavision_raw_to_dat -i 'chemin_du_fichier_a_convertir.raw'

# Chemin d'accès à modifier selon son enregistrement
events = datLoader.load_events('./out_2021-02-04_16-59-26_cd.dat')
# création du tableau de spike, 32*32 pour avoir une popultion de 1024 neurones
sourceArray = [[float]] * (32*32)


# Réduction de données avec pandas 
df = pd.DataFrame.from_records(events)
# réduction de la taille de l'image
df = df[df["x"]<120]
df = df[df["y"]<120]
# filtre par polarité
df = df[df["p"]>0]
# réduction du nombre d'events avec arrondi à la milliseconde
df = df.groupby(df['ts']%.0001).max()
df.index.name = None
# Trie par ordre croissant de timestamp
df = df.sort_values(by=['ts'])


xmax = 120
ymax = 120
timemax = 0

# Guillaume à mis en place une prédiction de temps restant pour la création du SpikeSourceArray pour les grosse vidéo
pourcentage = int(len(df) / 100)
current_poucentage = 0
avant = temps.time()

# On rempli le spikesourcearray, et on réduit la résolution au passage
for i in range(0, len(df)-1):
    if((i%pourcentage) == 0):
        prediction = temps.time() - avant
        avant = temps.time()
        print(str(current_poucentage) + " %   prediction: " + str(prediction*(100-current_poucentage)))
        current_poucentage += 1
    if (df['x'].iloc[i]%4 == 0 & df['y'].iloc[i]%4 ==0):
        x = df['x'].iloc[i]
        y = df['y'].iloc[i]
        time = df['ts'].iloc[i]*1.e-3
        if(timemax < time):
            timemax = time
        index = y*x+x-2
        sourceArray[index] = sourceArray[index] + [time]

# Pour la desambiguisation sur SpiNNaker
sourceArray = [list(elem) for elem in sourceArray]

# lancement d'un simulation ultra simple pour vérifier l'acceptation du spikeSourceArray
import pyNN.spiNNaker as sim
simulator = 'spinnaker'


sim.setup(timestep=10,
          min_delay=20,
          max_delay=30)

sources = sim.SpikeSourceArray(spike_times=sourceArray)
spikeSource = sim.Population(1024, sources)    
spikeSource.record(['spikes'])

sim.run(simtime=10000)

spikeSources  = spikeSource.get_data()#.segments[0].spiketrains
S_spikes = spikeSources.segments[0].spiketrains

sim.end()
