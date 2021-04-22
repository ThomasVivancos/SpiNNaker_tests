import numpy as np
import pyNN.spiNNaker as sim
from pyNN.random import NumpyRNG, RandomDistribution
import pyNN.utility.plotting as plot
from weightRecorder import WeightRecorder
from math import ceil
from neo import *
import matplotlib.pyplot as plt
import cv2 as cv


TIME_STEP=10.0          # {1, 2, 5, 10, 20,}
input_n=1024            # {512, 1024, 2048, 3800, 4096}
nb_neuron_int = 64      # {32, 64, 128, 256}
nb_neuron_out = 5       # {5, 10, 15}
delay = 2.0             # {0.2, 1, 2, 5, 10}
p_conn_in_int = 0.25    # {0.25, 0.5, 0.75, 1}
p_conn_int_out = 0.5    # {0.25, 0.5, 0.75, 1}

# Globalement la même chose que le train, les différence seront commentées
def lancement_sim(cellSourceSpikes, max_time=800000, path="default", TIME_STEP=TIME_STEP, input_n=input_n, nb_neuron_int=nb_neuron_int,
                nb_neuron_out=nb_neuron_out, delay=delay, p_conn_in_int=p_conn_in_int,
                p_conn_int_out=p_conn_int_out, a_minus=0.6, a_plus=0.6, tau_minus=12.0, tau_plus=10.0, v_tresh=10.0):

    simulator = 'spinnaker'
    sim.setup(timestep=TIME_STEP, min_delay=delay, max_delay=delay*2)
    randoms=np.random.rand(100,1)

    

    lif_curr_exp_params = {
        'cm': 1.0,          # The capacitance of the LIF neuron in nano-Farads
        'tau_m': 20.0,      # The time-constant of the RC circuit, in milliseconds
        'tau_refrac': 5.0,  # The refractory period, in milliseconds
        'v_reset': -65.0,   # The voltage to set the neuron at immediately after a spike
        'v_rest': -65.0,    # The ambient rest voltage of the neuron
        'v_thresh': -(v_tresh),  # The threshold voltage at which the neuron will spike
        'tau_syn_E': 5.0,   # The excitatory input current decay time-constant
        'tau_syn_I': 5.0,   # The inhibitory input current decay time-constant
        'i_offset': 0.0,    # A base input current to add each timestep
    }
    Input =  sim.Population(
        input_n,
        sim.SpikeSourceArray(spike_times=cellSourceSpikes),
        label="Input"
    )
    Input.record("spikes")

    LIF_Intermediate = sim.IF_curr_exp(**lif_curr_exp_params)
    Intermediate = sim.Population(nb_neuron_int, LIF_Intermediate, label="Intermediate")
    Intermediate.record(("spikes","v"))

    LIF_Output = sim.IF_curr_exp(**lif_curr_exp_params)
    Output = sim.Population(nb_neuron_out, LIF_Output, label="Output")
    Output.record(("spikes","v"))

    LIF_delayer = sim.IF_curr_exp(**lif_curr_exp_params)
    Delay_n = sim.Population(1, LIF_delayer, label="Delay")
    Delay_n.record(("spikes","v"))

    python_rng = NumpyRNG(seed=98497627)

    delay = delay            # (ms) synaptic time delay

    # Définition du fonctionnement de la stdp
    stdp_proj = sim.STDPMechanism(
                    timing_dependence=sim.SpikePairRule(tau_plus=tau_plus, tau_minus=tau_minus,
                                                        A_plus=a_plus, A_minus=a_minus),
                    weight_dependence=sim.AdditiveWeightDependence(w_min=0.1, w_max=6),
                    weight=RandomDistribution('normal',(3,2.9),rng=python_rng),
                    delay=delay)

    Conn_input_inter = sim.Projection(
        Input, Intermediate,
        connector=sim.FixedProbabilityConnector(p_conn_in_int,allow_self_connections=False),
        # synapse type set avec la définition de la stdp pour l'aprentsissage 
        synapse_type=stdp_proj,
        receptor_type="excitatory",
        label="Connection input to intermediate"
    )

    # second projection with stdp
    Conn_inter_output = sim.Projection(
        Intermediate, Output,   # pre and post population
        connector=sim.FixedProbabilityConnector(p_conn_int_out,allow_self_connections=False),
        synapse_type=stdp_proj,
        receptor_type="excitatory",
        label="Connection intermediate to output"
            )



    FixedInhibitory_WTA = sim.StaticSynapse(weight=6)
    WTA_INT = sim.Projection(
        Intermediate, Intermediate,
        connector=sim.AllToAllConnector(allow_self_connections=False),
        synapse_type=FixedInhibitory_WTA,
        receptor_type="inhibitory",
        label="Connection WTA"
    )

    WTA_OUT = sim.Projection(
        Output,Output,
        connector=sim.AllToAllConnector(allow_self_connections=False),
        synapse_type=FixedInhibitory_WTA,
        receptor_type="inhibitory",
        label="Connection WTA"
    )

    FixedInhibitory_delayer = sim.StaticSynapse(weight=2)
    Delay_out = sim.Projection(
    Delay_n, Output,
    connector=sim.AllToAllConnector(allow_self_connections=False),
    synapse_type=FixedInhibitory_delayer,
    receptor_type="inhibitory",
    label="Connection WTA"
    )

    Delay_inter = sim.Projection(
        Intermediate, Delay_n, 
        connector=sim.AllToAllConnector(allow_self_connections=False),
        synapse_type=FixedInhibitory_delayer,
        receptor_type="inhibitory",
        label="Connection WTA"
    )





    sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 255)


    # Définition des callbacks pour la récupération de l'écart-type sur les connexions entrée-intermédiaire, intermédiaire-sortie
    weight_recorder1 = WeightRecorder(sampling_interval=1000.0, projection=Conn_input_inter)
    weight_recorder2 = WeightRecorder(sampling_interval=1000.0, projection=Conn_inter_output)

    simtime = ceil(max_time)

    # Initialisation des tableaux pour la récupération des poids
    weights_int = []
    weights_out = []

    try:
        sim.run(simtime, callbacks=[weight_recorder1, weight_recorder2])
        neo = Output.get_data(variables=["spikes", "v"])
        spikes = neo.segments[0].spiketrains
        #print(spikes)
        v = neo.segments[0].filter(name='v')[0]
        weights_int = Conn_input_inter.get(["weight"], format="list")

        neo_in = Input.get_data(variables=["spikes"])
        spikes_in = neo_in.segments[0].spiketrains
        #print(spikes_in)


        neo_intermediate = Intermediate.get_data(variables=["spikes", "v"])
        spikes_intermediate = neo_intermediate.segments[0].spiketrains
        #print(spikes_intermediate)
        v_intermediate = neo_intermediate.segments[0].filter(name='v')[0]
        #print(v_intermediate)
        weights_out = Conn_inter_output.get(["weight"], format="list")

        sim.reset()
        sim.end()
    except:
        v = 0
        spikes = 0
    

    if(isinstance(spikes, list) and isinstance(v, AnalogSignal)):
        # Récupération des écart-types
        standard_deviation_out = weight_recorder2.get_standard_deviations()
        standard_deviation_int = weight_recorder1.get_standard_deviations()
        t = np.arange(0., max_time, 1.)

        # Création et sauvegarde des graphs sur les spikes et écart-types
        savePath = "./Generated_data/training/" + path + "/intermediate_layer_standard_deviation.png"
        plt.plot(standard_deviation_int)
        plt.xlabel("callbacks tick (1s)")
        plt.ylabel("standard deviation of the weights( wmax=6, wmin=0.1 )")
        plt.savefig(savePath)
        plt.clf()

        savePath = "./Generated_data/training/" + path + "/output_layer_standard_deviation.png"
        plt.plot(standard_deviation_out)
        plt.xlabel("callbacks tick")
        plt.ylabel("standard deviation ( wmax=6, wmin=0.1 )")
        plt.savefig(savePath)
        plt.clf()


        savePath = "./Generated_data/training/" + path + "/output_layer_membrane_voltage_and_spikes.png"
        plot.Figure(
            # plot voltage for first ([0]) neuron
            plot.Panel(v, ylabel="Membrane potential (mV)",
                    data_labels=[Output.label], yticks=True, xlim=(0, simtime)),
            # plot spikes (or in this case spike)
            plot.Panel(spikes, yticks=True, markersize=5, xlim=(0, simtime)),
            title="Spiking activity of the output layer during training",
            annotations="Simulated with {}".format(sim.name())
        ).save(savePath)


        savePath = "./Generated_data/training/" + path + "/intermediate_layer_membrane_voltage_and_spikes.png"
        plot.Figure(
            # plot voltage for first ([0]) neuron
            plot.Panel(v_intermediate, ylabel="Membrane potential (mV)",
                    data_labels=[Output.label], yticks=True, xlim=(0, simtime)),
            # plot spikes (or in this case spike)
            plot.Panel(spikes_intermediate, yticks=True, markersize=5, xlim=(0, simtime)),
            title="Spiking activity of the intermediate layer during training",
            annotations="Simulated with {}".format(sim.name())
        ).save(savePath)

        return v, spikes, weights_int, weights_out

    else:
        print("simulation failed with parmaters parameters : (l'affichage des paramètres ayant causés le disfonctionnement de la simulation sera traitée à une date ultérieur, merci!)")
        return 0, 0, 0, 0


