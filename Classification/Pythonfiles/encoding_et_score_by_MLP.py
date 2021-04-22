import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

def postProcess(delay_times, v, start_end_lenghts_label, TIME_STEP, target, spikes):
    d_times=np.array(delay_times)
    d_times=d_times/1000
    l=np.where(d_times< 10000000000)
    l[0].max()
    d_times


    #iterate over outputs based on video
    which_video=0
    step=0
    v_by_video=dict()
    for mv in v:
        step+=10
        vid_index=np.where(d_times<step)[0]
        if(len(vid_index)>0):
            which_video=vid_index.max()
        if which_video in v_by_video:
            v_by_video[which_video]= np.add(v_by_video.get(which_video),mv)
        else:
            v_by_video[which_video]=mv


    v_by_video_average=[]
    for i in range(len(v_by_video)):
        number_of_timesteps=start_end_lenghts_label[i][2]/(TIME_STEP * 1000)
        v_by_video_average.append(np.divide(v_by_video[i], number_of_timesteps))

    voltage_df = pd.DataFrame(v_by_video_average, columns = ['n1_v','n2_v','n3_v','n4_v','n5_v'])
    voltage_df['target']=target


    #iterate over outputs based on video
    which_video=0
    number_of_spikes_by_video=[]
    for video in start_end_lenghts_label:
        video_spikes=[]
        for neuron in spikes:
            video_spikes.append(len(np.where((neuron>video[0]/1000) & (neuron<video[1]/1000))[0]))
        number_of_spikes_by_video.append(video_spikes)


    spike_df = pd.DataFrame(number_of_spikes_by_video, columns = ['n1_total','n2_total','n3_total','n4_total','n5_total'])
    spike_df['target']=target

    #iterate over outputs based on video
    which_video=0
    rank_by_video=[]
    for video in start_end_lenghts_label:
        neuron_rank=[]
        for neuron in spikes:
            neuron_rank.append(neuron[np.where(neuron<video[1]/1000)[0][0]])
        rank_by_video.append(neuron_rank)


    rank_df = pd.DataFrame(rank_by_video, columns = ['n1_t_to_spike','n2_t_to_spike','n3_t_to_spike','n4_t_to_spike','n5_t_to_spike'])
    rank_df['target']=target

    interm_df=pd.DataFrame.merge(spike_df,voltage_df,left_index=True, right_index=True)
    final_df=pd.DataFrame.merge(interm_df,rank_df,left_index=True, right_index=True)


    final_df=final_df.drop([ 'target_x','target_y' ,'n1_t_to_spike','n2_t_to_spike','n3_t_to_spike','n4_t_to_spike','n5_t_to_spike'],axis=1)


    X_train, X_test, y_train, y_test = train_test_split(final_df.loc[:, final_df.columns != 'target'], final_df['target'], test_size=0.2, random_state=42)


    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    clf.fit(X_train, y_train)

    return clf.score(X_test,y_test)


    
