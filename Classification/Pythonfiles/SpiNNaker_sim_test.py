import numpy as np
import pyNN.spiNNaker as sim
from pyNN.random import NumpyRNG, RandomDistribution
import pyNN.utility.plotting as plot
import matplotlib.pyplot as plt
from pyNN.standardmodels import StandardSynapseType
from pyNN.spiNNaker import Projection
from weightRecorder import WeightRecorder
from math import ceil
import neo
from neo import Segment, AnalogSignal, SpikeTrain


TIME_STEP = 10.0        # {1, 2, 5, 10, 20,}
input_n = 1024          # {512, 1024, 2048, 3800, 4096}
nb_neuron_int = 64      # {32, 64, 128, 256}
nb_neuron_out = 5       # {5, 10, 15}
delay = 2.0             # {0.2, 1, 2, 5, 10}
p_conn_in_int = 0.25    # {0.25, 0.5, 0.75, 1}
p_conn_int_out = 0.5    # {0.25, 0.5, 0.75, 1}
v_tresh = 10

def lancement_sim(cellSourceSpikes, path, weight_input=0, weight_inter=0, max_time=800000, TIME_STEP=TIME_STEP, input_n=input_n, nb_neuron_int=nb_neuron_int,
                nb_neuron_out=nb_neuron_out, delay=delay, p_conn_in_int=p_conn_in_int,
                p_conn_int_out=p_conn_int_out, v_tresh=v_tresh):



    simulator = 'spinnaker'
    # le max_delay doit être inférieur à 14*time_step
    sim.setup(timestep=TIME_STEP, min_delay=delay, max_delay=delay*2)
    randoms=np.random.rand(100,1)

    #defining network topology
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

    # Population d'entrée avec comme source le SpikeSourceArray en paramètre
    Input =  sim.Population(
        input_n,
        sim.SpikeSourceArray(spike_times=cellSourceSpikes),
        label="Input"
    )
    Input.record("spikes")

    # Définition des types de neurones et des couches intermédiaire, de sortie, ainsi que celle contenant le neurone de l'attention
    LIF_Intermediate = sim.IF_curr_exp(**lif_curr_exp_params)
    Intermediate = sim.Population(nb_neuron_int, LIF_Intermediate, label="Intermediate")
    Intermediate.record(("spikes","v"))

    LIF_Output = sim.IF_curr_exp(**lif_curr_exp_params)
    Output = sim.Population(nb_neuron_out, LIF_Output, label="Output")
    Output.record(("spikes","v"))

    LIF_delayer = sim.IF_curr_exp(**lif_curr_exp_params)
    Delay_n = sim.Population(1, LIF_delayer, label="Delay")
    Delay_n.record(("spikes","v"))

    # set the stdp mechanisim parameters, we are going to use stdp in both connections between (input-intermediate) adn (intermediate-output)
    python_rng = NumpyRNG(seed=98497627)

    delay = delay            # (ms) synaptic time delay
    #A_minus 
    
    # définition des connexions entre couches de neurones entrée <=> intermédiaire, intermédiaire <=> sortie
    # vérificatio pour savoir si on est dans le cas de la première simulation par défault ou si on doit injecter les poids
    if((weight_input != 0) or (weight_inter != 0)):
        # cas ou l'on inject les poids
        Conn_input_inter = sim.Projection(
            Input, Intermediate,
            # Le fromListConnector pour injecter les poids
            connector=sim.FromListConnector(weight_input),
            receptor_type="excitatory",
            label="Connection input to intermediate",
            # des synapses static pour "suprimer" l'apprentissage
            synapse_type=sim.StaticSynapse()
        )


        Conn_inter_output = sim.Projection(
            Intermediate, Output,   # pre and post population
            connector=sim.FromListConnector(weight_inter),
            receptor_type="excitatory",
            label="Connection intermediate to output",
            synapse_type=sim.StaticSynapse()
            )
        
    else:
        # cas par défault
        Conn_input_inter = sim.Projection(
            Input, Intermediate,
            connector=sim.FixedProbabilityConnector(p_conn_in_int,allow_self_connections=False),
            synapse_type=sim.StaticSynapse(weight=RandomDistribution('normal',(3,2.9),rng=python_rng)),
            receptor_type="excitatory",
            label="Connection input to intermediate"
        )
        Conn_inter_output = sim.Projection(
            Intermediate, Output,   # pre and post population
            connector=sim.FixedProbabilityConnector(p_conn_int_out,allow_self_connections=False),
            synapse_type=sim.StaticSynapse(weight=RandomDistribution('normal',(3,2.9),rng=python_rng)),
            receptor_type="excitatory",
            label="Connection intermediate to output"
                )


    # définition des connexions inhibitrices des couches intermédiaire et de sortie
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

    # Connexion avec le neurone de l'attention
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




    # On précise le nombre de neurone par coeurs au cas ou
    sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 255)

    # on arrondie le temps de simulation, sinon avec les callbacks, on a une boucle infinie pour des temps d'arrêts plus précis que la fréquence des callbacks
    simtime = ceil(max_time)

    try:
        #lancement de la simulation
        sim.run(simtime)
        #récupération des infos sur les spike des trois couches
        neo = Output.get_data(variables=["spikes", "v"])
        spikes = neo.segments[0].spiketrains
        #print(spikes)
        v = neo.segments[0].filter(name='v')[0]

        neo_in = Input.get_data(variables=["spikes"])
        spikes_in = neo_in.segments[0].spiketrains
        #print(spikes_in)


        neo_intermediate = Intermediate.get_data(variables=["spikes", "v"])
        spikes_intermediate = neo_intermediate.segments[0].spiketrains
        #print(spikes_intermediate)
        v_intermediate = neo_intermediate.segments[0].filter(name='v')[0]
        #print(v_intermediate)

        sim.reset()
        sim.end()
    except:    
        # Si la simulation fail, on set ces deux variables à zéros pour gérer l'erreur dans le script principal
        v = 0
        spikes = 0
    

    # Création et sauvegarde des graphs des graphs si la simluation s'est bien passée, + envoie des sorties de la fonction
    if(isinstance(spikes, list) and isinstance(v, AnalogSignal)):
        
        plot.Figure(
            # plot voltage for first ([0]) neuron
            plot.Panel(v, ylabel="Membrane potential (mV)",
                    data_labels=[Output.label], yticks=True, xlim=(0, simtime)),
            # plot spikes (or in this case spike)
            plot.Panel(spikes, yticks=True, markersize=5, xlim=(0, simtime)),
            title="Spiking activity of the output layer during test",
            annotations="Simulated with {}".format(sim.name())
        ).save("./Generated_data/tests/" + path + "/output_layer_membrane_voltage_and_spikes.png")


        plot.Figure(
            # plot voltage for first ([0]) neuron
            plot.Panel(v_intermediate, ylabel="Membrane potential (mV)",
                    data_labels=[Output.label], yticks=True, xlim=(0, simtime)),
            # plot spikes (or in this case spike)
            plot.Panel(spikes_intermediate, yticks=True, markersize=5, xlim=(0, simtime)),
            title="Spiking activity of the intermediate layer during test",
            annotations="Simulated with {}".format(sim.name())
        ).save("./Generated_data/tests/" + path + "/intermediate_layer_membrane_voltage_and_spikes.png")

        return v, spikes
    else:
        print("simulation failed with parameters : (l'affichage des paramètres ayant causés le disfonctionnement de la simulation sera traitée à une date ultérieur, merci!)")
        return 0, 0





