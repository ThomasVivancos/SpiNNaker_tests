#Script

import Import_processing_data as ds
import SpiNNaker_sim_train as sim
import encoding_et_score_by_MLP as MLP
import numpy as np
import os
from numpy.random import random_sample
import SpiNNaker_sim_test as simTest
from neo import AnalogSignal, SpikeTrain
import pandas as pd


print("***Importation du dataset DVSGesture***")
# Fonction dans le fichier Import_processing_data.py

# Dataset d'entrainement
trainset, train_target, train_events = ds.importer_data(True)
# Dataset de test
testset, target, events = ds.importer_data(False)


print("***Ordonnecement des évènements pour faciliter la simulation SpiNNaker***")
# Fonction dans le fichier Import_processing_data.py

# Ordonancement du dataset d'entrainement
train_start_end_lenghts_label, train_delay_times, train_events, train_max_time = ds.order_events(train_events, train_target)
# Ordonancement du dataset de test
start_end_lenghts_label, delay_times, events, max_time = ds.order_events(events, target)


print("***Création du SpikeSourceArray pour PyNN***")
# Fonction dans le fichier Import_processing_data.py

# SpikeSourceArray de test 
cellSourceSpikes = ds.make_source_array(events, testset)
input_n=len(cellSourceSpikes)

# SpikeSourceArry d'entrainement
train_cellSourceSpikes = ds.make_source_array(train_events, trainset)
train_input_n=len(train_cellSourceSpikes)


print("***Lancement de la première simulation***")

# Paramètres que l'on veut tester, avec leur borne inférieure et supérieure
params = [("A-",1.0,20.0), ("A+",1.0,20.0), ("tau-",0.01,1.0), ("tau+",0.01,1.0), ("v_tresh", 1.0, 20.0) ]




## Algorithme hill climbing :

# la valeur de performance prend 0 au début
val_test = 0
# On se focalisera sur la performance de l'encodage de Yazan et Gabriele 
df = pd.DataFrame()
# on changera les valeurs 10 fois pour chaque paramètres à tester
n_iterations = 10
# liste des paramètres qui seront utilisé pour les simulations
params_values = {"time_step": 1.0, "input_n": 1024, "nb_neuron_int":64, "nb_neuron_out": 5, "delay" : 10, "p_conn_in_int": 0.25, "p_conn_int_out": 0.5,
                "A-": 0.6, "A+": 0.6, "tau-":12.0, "tau+":10.0, "v_tresh": 10.0}

# generate an initial point
path = "default"

# lancement d'un première simulation sans entrainement avec les valeurs par défault pour comparaison
v_test, spikes_test = simTest.lancement_sim(cellSourceSpikes, path, 0, 0, max_time=max_time, TIME_STEP=params_values["time_step"], input_n=params_values["input_n"],
                            nb_neuron_int=params_values["nb_neuron_int"], nb_neuron_out=params_values["nb_neuron_out"],
                            delay=params_values["delay"], p_conn_in_int=params_values["p_conn_in_int"],
                            p_conn_int_out=params_values["p_conn_int_out"], v_tresh=params_values["v_tresh"])

# Vérification des sorties de la simulation, si les formats ne sont pas bon, cela veut dire que la simulation plante avec ces paramètres
if((isinstance(v_test, AnalogSignal)) and (isinstance(spikes_test, list))):

        # récupération du score initial
        score_init = MLP.postProcess(delay_times, v_test, start_end_lenghts_label, params_values["time_step"], target, spikes_test)

        # Lancement d'une simulation d'entrainement
        v, spikes, weight_input, weight_inter = sim.lancement_sim(train_cellSourceSpikes, train_max_time, path, TIME_STEP=params_values["time_step"], input_n=params_values["input_n"],
                                nb_neuron_int=params_values["nb_neuron_int"], nb_neuron_out=params_values["nb_neuron_out"],
                                delay=params_values["delay"], p_conn_in_int=params_values["p_conn_in_int"],
                                p_conn_int_out=params_values["p_conn_int_out"], a_minus=params_values["A-"], a_plus=params_values["A+"],
                                tau_minus=params_values["tau-"], tau_plus=params_values["tau+"], v_tresh=params_values["v_tresh"])# evaluate the initial point

        # revérification des sorties pour que la simulation de test puisse se passer correctement
        if(weight_input != 0 and weight_inter != 0):

                # Mise en forme des Tuples de poids
                # Amélioration à faire : utiliser map au lieu des boucles for ou un toTuples si existe
                final_weight_input =[]
                for i in range(len(weight_input)):
                        aux = (weight_input[i][0], weight_input[i][1], weight_input[i][2], params_values["delay"])
                        final_weight_input.append(aux)
                final_weight_inter = []
                for j in range(len(weight_inter)):
                        aux = (weight_inter[j][0], weight_inter[j][1], weight_inter[j][2], params_values["delay"])
                        final_weight_inter.append(aux)
                
                # lancement de la simulation test avec les poids finaux de l'entrainement
                v_test, spikes_test = simTest.lancement_sim(cellSourceSpikes, path, final_weight_input, final_weight_inter, max_time=max_time, TIME_STEP=params_values["time_step"], input_n=params_values["input_n"],
                                        nb_neuron_int=params_values["nb_neuron_int"], nb_neuron_out=params_values["nb_neuron_out"],
                                        delay=params_values["delay"], p_conn_in_int=params_values["p_conn_in_int"],
                                        p_conn_int_out=params_values["p_conn_int_out"], v_tresh=params_values["v_tresh"])

                # récupération du score
                score_post_train = MLP.postProcess(delay_times, v_test, start_end_lenghts_label, params_values["time_step"], target, spikes_test)

                val_test = score_post_train

                # Enreistrement des performances dans un dataframe
                df = pd.DataFrame(['default', score_post_train, 'succed'], columns=['Parameter', 'score', 'status'])
        else:
                print("La simulation n'est pas supportée avec ces paramètres :")
                print(params_values)
                df = pd.DataFrame([['default'], [0], ['failed']], columns=['Parameter', 'score', 'status'])

else:
        print("La simulation n'est pas supportée avec ces paramètres :")
        print(params_values)


# loop on all choosen parameters for the hill climbing
rng = np.random.default_rng(12345)
for i in range(len(params)):
        for j in range(n_iterations):
                # take a step
                present_param = (params[i][2] - params[i][1]) * random_sample() + params[i][1]
                print("***Lancement du hill climbing sur le paramètre " + str(params[i][0]) + "avec pour valeurs : " + str(present_param))
                present_parram_name = params[i][0]
                present_path = present_parram_name + str(present_param)
                present_params_values = params_values
                present_params_values[present_parram_name] = present_param

                # entrainement
                present_v, present_spikes, present_weight_input, present_weight_inter = sim.lancement_sim(cellSourceSpikes, max_time, path, TIME_STEP=params_values["time_step"],
                                                        input_n=params_values["input_n"], nb_neuron_int=params_values["nb_neuron_int"],
                                                        nb_neuron_out=params_values["nb_neuron_out"], delay=params_values["delay"],
                                                        p_conn_in_int=params_values["p_conn_in_int"], p_conn_int_out=params_values["p_conn_int_out"],
                                                        a_minus=params_values["A-"], a_plus=params_values["A+"],
                                                        tau_minus=params_values["tau-"], tau_plus=params_values["tau+"], v_tresh=params_values["v_tresh"])


                # vérification des sorties pour que la simulation de test puisse se passer correctement
                if(weight_input != 0 and weight_inter != 0):

                        # Mise en forme des Tuples de poids
                        # Amélioration à faire : utiliser map au lieu des boucles for ou un toTuples si existe
                        final_weight_input =[]
                        for i in range(len(present_weight_input)):
                                aux = (present_weight_input[i][0], present_weight_input[i][1], present_weight_input[i][2], params_values["delay"])
                                print(aux)
                                final_weight_input.append(aux)
                        final_weight_inter = []
                        for j in range(len(present_weight_inter)):
                                aux = (present_weight_inter[j][0], present_weight_inter[j][1], present_weight_inter[j][2], params_values["delay"])
                                final_weight_inter.append(aux)

                        
                        # Simulation avec les poids finaux de l'entrainement
                        present_v, present_spikes = simTest.lancement_sim(cellSourceSpikes, final_weight_input, final_weight_inter, max_time=max_time, TIME_STEP=params_values["time_step"], input_n=params_values["input_n"],
                                        nb_neuron_int=params_values["nb_neuron_int"], nb_neuron_out=params_values["nb_neuron_out"],
                                        delay=params_values["delay"], p_conn_in_int=params_values["p_conn_in_int"],
                                        p_conn_int_out=params_values["p_conn_int_out"], v_tresh=params_values["v_tresh"])

                        # evaluate candidate point
                        candidate_eval_test = MLP.postProcess(delay_times, present_v, start_end_lenghts_label, params_values["time_step"], target, present_spikes)
                        # check if we should keep the new point
                        aux_df = pd.DataFrame([path, candidate_eval_test], columns=['Parameter', 'score'])
                        df = df.append(aux_df)
                        if candidate_eval_test > val_test:
                                # store the new point
                                params_values, val_test = present_params_values, candidate_eval_test
                                # report progress
                                print('>%d f(%s) = %.5f' % (i, params_values, val_test))
                
                else:
                        aux_df = pd.DataFrame([path, 0, 'failed'], columns=['Parameter', 'score', 'status'])
                        df = df.append(aux_df)

# sauvegarde du dataFrame des scores sous csv
df.to_csv('./Generated_data/Scores.csv', index=False)
print("********** Meilleur score = " + str(val_test) + "***********")
