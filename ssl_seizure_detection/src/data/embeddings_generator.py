#!/usr/bin/env python
import pandas as pd
import numpy as np
import mne
from mne_connectivity import spectral_connectivity_time, spectral_connectivity_epochs
from scipy.signal import welch,coherence
import sys
import os
import logging
import matplotlib.pyplot as plt
import torch

from .preprocess import new_grs, create_tensordata_new, convert_to_Data
import traceback

# hyper parameters 
freq = 256
ws = int(2*freq)
step = int(0.125*ws)
dim = 22
data_theshold = 100000
n_subepochs = 4
overlap = 0.5
freq_bands = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma1": (30, 40),
}
fmin = 5
fmax = 40
nperseg = 128 # for wetch method
selected_channels = channels = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8-1",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
    "P7-T7",
    "T7-FT9",
    "FT9-FT10",
    "FT10-T8",
]
# bipolar montages -- check the data

# def get_stats(fcns, logger):
#     feature_map = {
#         0: "ones",
#         1: "correlation",
#         2: "connectivity",
#         3: "phase lock val"
#     }
#     print("PLV == Connectivity:", np.all(fcns[0][2] == fcns[0][3]))
#     feature_matrices = [fcn[1].flatten() for fcn in fcns]  # Flatten each graph matrix
#     all_values = np.concatenate(feature_matrices)  # Combine all graphs into one array
    
#     mean_value = np.mean(all_values)
#     std_value = np.std(all_values)
    
#     logger.info(f"samples correlation mean is {mean_value} and variance value is {std_value}...\n")

# def get_stats_2(fcn, feature, logger):
#     logger.info(f"{feature} stats {pd.Series(fcn.flatten()).describe()}")


def get_logger(data_log):
    logger = logging.getLogger("GraphGeneration")
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(os.path.join(data_log, "graph_generation.log"))
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

# def plot_energies(band_energy_matrix,data_log_dir):
#     save_dir = os.path.join(data_log_dir, "channel_power_plots")
#     os.makedirs(save_dir, exist_ok=True)
#     for i in range(band_energy_matrix.shape[0]):
#         plt.figure()
#         plt.plot(band_energy_matrix[i,:], marker= 'o', linestyle='-')
#         plt.xlabel("band")
#         plt.ylabel("power sum")
#         plt.title(f"channel {i} plot")
#         save_path = os.path.join(save_dir, f"channel_{i}")
#         plt.savefig(save_path)
#         plt.close()

def get_band_energies(data, dim,data_log_dir):

    data = np.nan_to_num(data)  # Replace NaN/Inf with finite numbers
    if np.var(data) == 0:
        raise ValueError("Input data has zero variance. PSD cannot be computed.")
    freqs, psd = welch(data, fs=freq, nperseg=nperseg, axis=-1)
    n_bands = len(freq_bands)
    band_energy_matrix = np.zeros((dim, n_bands))

    for i,(band_name, (fmin, fmax)) in enumerate(freq_bands.items()):
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        band_energy_matrix[:, i] = np.sum(psd[:, band_mask], axis=1)
    
    # plot_energies(data,data_log_dir)

    return band_energy_matrix

# why use 3 metrics - analyse which metric is discriminating
# GAF for EEG 
# torch EEG 
# use for band passing.
def compute_coherence_matrix(segment,logger,type,indexes,stat_log):
    sfreq = freq
    nperseg=64
    """
    Computes the spectral coherence matrix for an EEG segment.
    
    Parameters:
    - segment: np.ndarray of shape (n_channels, n_times)
    - sfreq: Sampling frequency (Hz)
    - fmin, fmax: Frequency range for coherence
    - nperseg: Number of samples per FFT segment

    Returns:
    - coherence_matrix: np.ndarray of shape (n_channels, n_channels)
    """
    # get_segement_stats(segment)
    n_channels = segment.shape[0]
    coherence_matrix = np.zeros((n_channels, n_channels))
    # plt.plot(segment[0], label=f'Channel {0}')
    # plt.show()
    # plt.plot(segment[1], label=f'Channel {3}')
    # plt.show()
    # f, coh = coherence(segment[0], segment[1], fs=freq, nperseg=128)  # Example for first two channels

    # # Plot the coherence spectrum
    # plt.plot(f, coh)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Coherence')
    # plt.title('Coherence between Channel 0 and Channel 1')
    # plt.show()
    # Compute coherence for each pair of channels
    for i in range(n_channels):
        for j in range(i + 1, n_channels):  # Avoid duplicate computations
            f, coh = coherence(segment[i], segment[j], fs=sfreq, nperseg=nperseg)
            
            # Select coherence values only within the specified frequency range
            freq_mask = (f >= fmin) & (f <= fmax)
            mean_coh = np.mean(coh[freq_mask])  # Average coherence over selected frequencies
            
            coherence_matrix[i, j] = mean_coh
            coherence_matrix[j, i] = mean_coh  # Ensure symmetry

    # get_stats_2(coherence_matrix, "connect")
    # if type == "non_ictal":
    #     plot(coherence_matrix, f"{type}_coh_matrix_scipy",stat_log, indexes[0])
    # else: 
    #     plot(coherence_matrix, f"{type}_coh_matrix_scipy",stat_log, indexes[1])
    return coherence_matrix
def generate_coh_matrix_time_avg(segment, logger
# ,coh ,type,indexes
):
    # segment = np.nan_to_num(segment)
    segment = np.array(segment)
    sh = segment.shape
    # Create artificial epochs (e.g., splitting into 4 sub-epochs)
    # logger.info(f"segment shape: {segments.shape}")
    segment = segment.reshape(1,sh[0], sh[1])
    freqs = np.linspace(fmin, fmax, 50) 
    try:
        con = spectral_connectivity_time(
            segment,
            freqs = freqs,
            method='coh',
            average=True,
            sfreq=freq,
            fmin=fmin,
            fmax=fmax,
            faverage=True,
            sm_times=0.05, 
            mode='multitaper',                     
            verbose=False        
        )
        coherence_matrix = con.get_data().reshape(dim, dim)
        coherence_matrix = coherence_matrix + coherence_matrix.T - np.diag(np.diag(coherence_matrix))
        # logger.info(f"coherence mat: {coherence_matrix}")
        # coh += coherence_matrix
        # if type == "non_ictal":
        #     plot(coherence_matrix, f"{type}_coh_matrix_time_avg",stat_log, indexes[0])
        # else: 
        #     plot(coherence_matrix, f"{type}_coh_matrix_time_avg",stat_log, indexes[1])
        return coherence_matrix
    except Exception as e:
            logger.error(f"Error in coh the data point as {e}")  

    # get_stats_2(coherence_matrix, "connectivity")
    # coherence_matrix = np.nan_to_num(coherence_matrix
    

# def generate_plv_matrix_time_avg(segment,logger
# # ,plv,type,indexes
# ):
#     segment = np.array(segment)
#     freqs = np.linspace(fmin, fmax, 100) 
#     sh = segment.shape
#     # Create artificial epochs (e.g., splitting into 4 sub-epochs)
#     # logger.info(f"segment shape: {segments.shape}")
#     segment = segment.reshape(1,sh[0], sh[1])

#     con = spectral_connectivity_time(
#         segment, 
#         freqs=freqs,
#         method='plv', 
#         average=True,
#         fmin=fmin,
#         fmax=fmax,
#         faverage=True,
#         sm_times=0.1,
#         mode='multitaper',
#         sfreq=freq,
#         verbose=False
#     )
#     # get_stats_2(con.get_data(), "plv")
    
#     plv_matrix = con.get_data().reshape(dim, dim)
#     # plv += plv_matrix
#     # if type == "non_ictal":
#     #     plot(plv_matrix, f"{type}_plv_matrix_time_avg",stat_log,indexes[0])
#     # else:
#     #     plot(plv_matrix, f"{type}_plv_matrix_time_avg",stat_log,indexes[1])
#     return plv_matrix
# def generate_coh_matrix_time_locked(segment, 
# # coh,
#  logger
# # ,type,indexes
# ):
#     # segment = np.nan_to_num(segment)
#     segment = np.array(segment)
#     sh = segment.shape
#     epochs = 2
#     epoch_len = sh[1]//epochs
#     # Create artificial epochs (e.g., splitting into 4 sub-epochs)
#     # logger.info(f"segment shape: {segments.shape}")
#     segment = segment.reshape(epochs,sh[0], epoch_len)
#     try:
#         con = spectral_connectivity_epochs(
#             segment,
#             method='coh',
#             sfreq=freq,
#             fmin=fmin,
#             fmax=fmax,
#             faverage=True, 
#             mode='multitaper',                     
#             verbose=False        
#         )
#         coherence_matrix = con.get_data().reshape(dim, dim)
#         # coh += coherence_matrix
#         # if type == "non_ictal":
#             # plot(coherence_matrix,f"{type}_coh_matrix_time_locked",stat_log,indexes[0])
#         # else:
#             # plot(coherence_matrix,f"{type}_coh_matrix_time_locked",stat_log,indexes[1])
#         return coherence_matrix
#     except Exception as e:
#             logger.error(f"Error in coh the data point as {e}")  

#     # get_stats_2(coherence_matrix, "connectivity")
#     # coherence_matrix = np.nan_to_num(coherence_matrix
    

# def generate_plv_matrix_time_locked(segment
# # plv
# # ,logger,type,indexes
# ):
#     segment = np.array(segment)
#     sh = segment.shape
#     # Create artificial epochs (e.g., splitting into 4 sub-epochs)
#     # logger.info(f"segment shape: {segments.shape}")
#     epochs = 2
#     epoch_len = sh[1]//epochs
#     # Create artificial epochs (e.g., splitting into 4 sub-epochs)
#     # logger.info(f"segment shape: {segments.shape}")
#     segment = segment.reshape(epochs,sh[0], epoch_len)

#     con = spectral_connectivity_epochs(
#         segment, 
#         method='plv',  
#         fmin=fmin,
#         fmax=fmax,
#         faverage=True,
#         mode='multitaper',
#         sfreq=freq,
#         verbose=False
#     )
#     # get_stats_2(con.get_data(), "plv")
    
#     plv_matrix = con.get_data().reshape(dim, dim)
#     # plv += plv_matrix
#     # if type == "non_ictal":
#         # plot(plv_matrix, f"{type}_plv_matrix_time_locked",stat_log,indexes[0])
#     # else: 
#         # plot(plv_matrix, f"{type}_plv_matrix_time_locked",stat_log,indexes[1])
#     return plv_matrix
def generate_cor_matrix(data,logger,
# corr, 
# ,type,indexes
):
    try:
        corr_mat = np.corrcoef(data)
        # corr += corr_mat
        # if type == "non_ictal":
            # plot(corr_mat, f"{type}_cor_matrix",stat_log,indexes[0])
        # else: plot(corr_mat, f"{type}_cor_matrix",stat_log,indexes[1])
    except Exception as e:
        logger.error(f"Error in cor the data point as {e}")  
    return corr_mat
def generate_fcns(
    data, dim, logger,
    # corr,
    # coh_time_avg,
    # plv_time_avg,coh_time_locked,plv_time_locked,
    type,
    indexes,
    stat_log
    ):
    fcns = []
    i = 0
    while i < len(data):
        try:
            fcn = []
            # All ones
            fcn.append(np.ones((dim, dim)))
            # Correlation
            fcn.append(generate_cor_matrix(data[i],
            logger
            # ,corr,type,indexes
            ))
            # Coherence
            fcn.append(compute_coherence_matrix(data[i],logger,type,indexes,stat_log
            )) 
            fcns.append(fcn)     
            if type == "non_ictal":
                indexes[0]+=1
            else: indexes[1] +=1 

        except Exception as e:
            logger.error(f"Skipping the data point as {e}")  
        i+=1
    return fcns

def generate_node_features(data, dim,log_dir):
    features = []
    # all ones
    features.append(np.ones((dim, dim)))
    # avg energy 
    features.append(np.mean(data**2, axis=1, keepdims=True))
    # band energies
    features.append(get_band_energies(data,dim,log_dir)) 
    return features

def generate_graphs(data, 
                    fcns, 
                    logger,log_dir):
    i = 0
    graphs = []
    while i < len(data):
        try:
            NF = generate_node_features(data[i], dim,log_dir)
            graphs.append([
                fcns[i], 
                NF,
                np.expand_dims(fcns[i], axis=-1)
                ])
        except Exception as e:
                logger.error(f"Skipping the data point as {e}")  
        i += 1
    return graphs

def generate_segements(data):
     # print(data.shape)
    dim, num = data.shape
    i = 0
    segments = []
    while i < num:
        if num - i > ws:
            curr_win = data[:, i:i+ws]
        if np.var(curr_win) == 0:
            logging.info(f"Skipping: Data has zero variance....{curr_win}.")
        else :
            segments.append(curr_win)
        i += step
    
    return segments
# returns train and test graph embeddings

def split_graphs(data, train_ratio, test_ratio):
    n = len(data)

    print(n)

    n_train = int(n*train_ratio)
    n_test = int(n*test_ratio)

    assert(n_train + n_test <= n)

    train_samples = data[0:n_train]
    test_samples = data[n_train:n]

    return train_samples,test_samples

def get_pyg_grs(num_electrodes, new_data_train,logger):
    return create_tensordata_new(logger,num_nodes=num_electrodes, data_list=new_data_train, complete=True, save=False, logdir=None)

def get_segement_stats(segement, logger):
    # random_samples = random.sample(segements, 1000)
    # for sample in random_samples:
    logger.info(f"Segment Mean: {np.mean(segement)}")
    logger.info(f"Segment Std:, {np.std(segement)}")
    logger.info(f"Min:, {np.min(segement)}, Max:, {np.max(segement)}")
    # Plot a few EEG channels
    plt.figure(figsize=(10, 5))
    for i in range(5):  # Plot 5 random channels
        plt.plot(segement[i, :], label=f'Channel {i}')
    plt.legend()
    plt.title("Raw EEG Signals")
    plt.show()
    # break

def generate_embeddings_util(
                            data_log_dir,
                            non_ictal_mat,
                            ictal_mat,
                            file_index,
                            logger,
                            stat_log,
                            # non_ictal_corr,
                            # ictal_corr,
                            # non_ictal_coh_time_avg,
                            # ictal_coh_time_avg,
                            # non_ictal_plv_time_avg,
                            # ictal_plv_time_avg,
                            # non_ictal_coh_time_locked,
                            # ictal_coh_time_locked,
                            # non_ictal_plv_time_locked,
                            # ictal_plv_time_locked,
                            indexes
                            ):

    if non_ictal_mat is not None:
        logger.info(f"generating non_ictal segments for ...")
        try:
            non_ictal_segments = generate_segements(non_ictal_mat)
        except Exception as e:
            logger.error(f"error in generating non_ictal segments:- {e}")
            sys.exit(1)
        logger.info(f"generating non_ictal fcns for {data_log_dir}/{file_index}...")

        # get_segement_stats(non_ictal_segments)

        try:
            non_ictal_fcns = generate_fcns(non_ictal_segments, dim, logger,
            # non_ictal_corr,non_ictal_coh_time_avg,non_ictal_plv_time_avg,non_ictal_coh_time_locked,non_ictal_plv_time_locked,
            "non_ictal",
            indexes,
            stat_log
            )
        except Exception as e: 
            logger.error(f"error in generating non_ictal fcns:- {e}")
            sys.exit(1)
        logger.info(f"generating non_ictal graphs for {data_log_dir}/{file_index}...")
        # get_stats(non_ictal_fcns)

        # adjacency matrix
        try:  
            non_ictal_graphs = generate_graphs(non_ictal_segments, 
                                               non_ictal_fcns, 
                                               logger,stat_log)
        except Exception as e:
            logger.error(f"error in generating non_ictal graphs:- {e}")
            sys.exit(1)

        logger.info("generating non_ictal train embeddings...")
        try:
            data_non_ictal = new_grs(non_ictal_graphs, type="non_ictal")
        except Exception as e:
            logger.info(f"error generating graphs for non_ictal data:- {e}")
            sys.exit(e)
        
    if ictal_mat is not None:
        logger.info(f"generating ictal segments for {data_log_dir}/{file_index}...")
        try: 
            ictal_segments = generate_segements(ictal_mat)
        except Exception as e: 
            logger.error(f"error in generating ictal segments :- {e}")
            sys.exit(1)
        logger.info(f"generating ictal fcns for {data_log_dir}/{file_index}...")
        # get_segement_stats(generate_segements)
        try:
            ictal_fcns = generate_fcns(ictal_segments, dim, logger,
            # ,ictal_corr,ictal_coh_time_avg,ictal_plv_time_avg,ictal_coh_time_locked,ictal_plv_time_locked,
            "ictal",indexes,stat_log
            )
        except Exception as e: 
            logger.error(f"error in generating ictal fcns :- {e}")
            sys.exit(1)

        # # get_stats(ictal_fcns)
        try:
            ictal_graphs = generate_graphs(ictal_segments,
                                            ictal_fcns,
                                              logger,stat_log)
        except Exception as e:
            logger.error(f"error in generating ictal graphs:- {e}")
            sys.exit(1)

        logger.info("generating ictal train embeddings...")
        try:
            data_ictal = new_grs(ictal_graphs, type="ictal")
        except Exception as e:
            logger.info(f"error generating graphs for ictal data:- {e}")
            sys.exit(e)


    if non_ictal_mat is not None and ictal_mat is not None:
        logger.info("appending graphs for ictal and non_ictal data...")
        try:
            new_data = data_non_ictal + data_ictal
        except Exception as e:
            logger.info(f"error appending graphs for ictal and non_ictal data {e}")
            sys.exit(e)
    elif non_ictal_mat is not  None: 
        try:
            new_data = data_non_ictal
        except Exception as e:
            logger.info(f"error appending graphs for ictal and non_ictal data {e}")
            sys.exit(e)
    elif ictal_mat is not  None:
        try:
            new_data = data_ictal
        except Exception as e:
            logger.info(f"error appending graphs for ictal and non_ictal data {e}")
            sys.exit(e)

    num_electrodes = new_data[0][0][0].shape[0]

    logger.info("generating pyg format graphs...")
    try: 
        pyg_grs = get_pyg_grs(num_electrodes,new_data,logger)
    except Exception as e:
        logger.info(f"error generating pyg format graphs {e}")

    pyg_Data_path = f"{data_log_dir}/jh101_pyg_Data_{file_index}.pt"

    convert_to_Data(pyg_grs, logger, save=True, logdir=pyg_Data_path)

def get_filtered_data(raw):
    raw.notch_filter([60], fir_design='firwin') 
    raw.filter(0.5, 50, fir_design='firwin')
    return raw

def normalise_data(data):
    return ((data - np.mean(data, axis=1, keepdims=True)) / np.std(data,axis=1, keepdims=True))

def plot(matrix, title, base_log_path,index):
    log_path = os.path.join(base_log_path, title)

    # Ensure the directory exists
    os.makedirs(log_path, exist_ok=True)

    # Create figure without displaying
    fig, ax = plt.subplots(figsize=(6,5))
    cax = ax.imshow(matrix, cmap="plasma", interpolation='bilinear')
    ax.set_title(title)
    fig.colorbar(cax, label="Value")

    # Save the figure with a proper extension
    save_path = os.path.join(log_path, f"{index}.png")
    fig.savefig(save_path, format="png", dpi=300)

    plt.close(fig)  # Close the specific figure
    print(f"Figure saved to {save_path}")
  
def generate_embeddings(data_path,label_path,index,data_log, stat_log):
    # non_ictal_corr = np.zeros((22,22))
    # ictal_corr = np.zeros((22, 22))
    # non_ictal_coh_time_avg = np.zeros((22,22))
    # ictal_coh_time_avg = np.zeros((22, 22))
    # non_ictal_plv_time_avg = np.zeros((22,22))
    # ictal_plv_time_avg= np.zeros((22, 22))
    # non_ictal_coh_time_locked = np.zeros((22,22))
    # ictal_coh_time_locked = np.zeros((22, 22))
    # non_ictal_plv_time_locked = np.zeros((22,22))
    # ictal_plv_time_locked = np.zeros((22, 22))

    indexes = [0,0]
    
  
    logger = get_logger(stat_log)
    
    logger.info("generating matrices...")

    df = pd.read_csv(label_path)
    file_index = 0
    data_log_dir = f"{data_log}/chb{index}"

    if not os.path.exists(data_log_dir):
        os.makedirs(data_log_dir)

    for index, row in df.iterrows():
        # if file_index == 16: break

        file_name = row['File_names']
        label = row['Labels']
        start = row['Start_time']*freq
        end = row['End_time']*freq
        file_path = f"{data_path}/{file_name}"
        try:
            raw = mne.io.read_raw_edf(file_path,preload=True)
        except Exception as e:
            logger.error(f"Error reading .edf file at {file_path}")
        raw.pick(selected_channels)
        raw = get_filtered_data(raw)
        data, times = raw[:]
        logger.info(data.shape)
        data = normalise_data(data)

        if label == 0: 
            non_ictal_list = data
            generate_embeddings_util(   
                                        data_log_dir,
                                        data,
                                        None,
                                        file_index,logger,
                                        stat_log,
                                        # non_ictal_corr,
                                        # ictal_corr,
                                        # non_ictal_coh_time_avg,
                                        # ictal_coh_time_avg,
                                        # non_ictal_plv_time_avg,
                                        # ictal_plv_time_avg,
                                        # non_ictal_coh_time_locked,
                                        # ictal_coh_time_locked,
                                        # non_ictal_plv_time_locked,
                                        # ictal_plv_time_locked,
                                        indexes
                                    )
        else:
            non_ictal_list = []
            non_ictal_list.append(data[:, 0:start])
            if end < len(data)-1:
                non_ictal_list.append(data[:, end+1:len(data)])
            ictal_list = data[:, start:end]
            generate_embeddings_util( 
                                        data_log_dir,
                                        np.hstack(non_ictal_list),
                                        ictal_list,
                                        file_index,
                                        logger,
                                        stat_log,
                                        # non_ictal_corr,
                                        # ictal_corr,
                                        # non_ictal_coh_time_avg,
                                        # ictal_coh_time_avg,
                                        # non_ictal_plv_time_avg,
                                        # ictal_plv_time_avg,
                                        # non_ictal_coh_time_locked,
                                        # ictal_coh_time_locked,
                                        # non_ictal_plv_time_locked,
                                        # ictal_plv_time_locked,
                                        indexes
                                    )
        file_index += 1
    
    # plot(non_ictal_corr, "non_ictal_correlation",stat_log)
    # plot(ictal_corr, "ictal_correlation",stat_log)
    # plot(non_ictal_coh_time_avg, "non_ictal_coherence_time_averaged",stat_log)
    # plot(ictal_coh_time_avg,"ictal_coherence_time_averaged",stat_log)
    # plot(non_ictal_plv_time_avg, "non_ictal_plv_time_averaged",stat_log)
    # plot(ictal_plv_time_avg,"ictal_plv_time_averaged",stat_log)
    # plot(non_ictal_coh_time_locked,"non_ictal_coherence_time_locked",stat_log)
    # plot(ictal_coh_time_locked,"ictal_coherence_time_locked",stat_log)
    # plot(non_ictal_plv_time_locked, "non_ictal_plv_time_locked",stat_log)
    # plot(ictal_plv_time_locked, "ictal_plv_time_locked",stat_log)


if __name__ == "__main__":
    index_str = "03"
    data_base_path = "/Users/dentira/anomaly-detection/1.0.0"
    label_base_path = "/Users/dentira/anomaly-detection/epilepsy-detection/parsed_labels"
    p_name = f"chb{index_str}"
    data_path = os.path.join(data_base_path,p_name)
    label_path = os.path.join(label_base_path,f"{p_name}_labels.csv")
    data_log = "/Users/dentira/anomaly-detection/epilepsy-detection/ssl_seizure_detection/data/patient_gr"
    stat_log = "/Users/dentira/anomaly-detection/epilepsy-detection/ssl_seizure_detection/logs"
   
    generate_embeddings(data_path,label_path,index_str,data_log, stat_log)


   

    

    





    

    




    