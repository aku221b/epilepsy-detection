#!/usr/bin/env python
import pandas as pd
import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs
from scipy.signal import welch,detrend,coherence
import pickle
import sys
from pathlib import Path    
PROJECT_ROOT = Path(__file__).resolve().parents[3] 
sys.path.append(str(PROJECT_ROOT / "ssl_seizure_detection/src/data"))
from preprocess import new_grs, create_tensordata_new, convert_to_Data, pseudo_data, convert_to_PairData, convert_to_TripletData
import os
import random
import logging
import matplotlib.pyplot as plt

# Set up a global logger
logger = logging.getLogger("GraphGeneration")
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler(str(PROJECT_ROOT / "ssl_seizure_detection/run_logs/graph_generation.log"))
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(file_handler)

# hyper parameters 
freq = 256
ws = int(2*freq)
step = int(0.125*ws)
dim = channels_to_take = 22
data_theshold = 100000
n_subepochs = 4
overlap = 0.5

def get_band_energies(data, dim):
    freq_bands = {
        "Delta": (1, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma1": (30, 70),
        "Gamma2": (70, 100),
        "Gamma3": (100, 120),
        "Gamma4": (120, 128),
    }
    data = np.nan_to_num(data)  # Replace NaN/Inf with finite numbers
    if np.var(data) == 0:
        raise ValueError("Input data has zero variance. PSD cannot be computed.")
    nperseg = min(len(data), 1024)  # Ensure nperseg <= signal length
    freqs, psd = welch(data, fs=freq, nperseg=nperseg, axis=-1)
    n_bands = len(freq_bands)
    band_energy_matrix = np.zeros((dim, n_bands))

    for i,(band_name, (fmin, fmax)) in enumerate(freq_bands.items()):
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        band_energy_matrix[:, i] = np.sum(psd[:, band_mask], axis=1)

    return band_energy_matrix
def get_stats_2(fcn, feature):
    logger.info(f"{feature} stats {pd.Series(fcn.flatten()).describe()}")

def compute_coherence_matrix(segment):
    sfreq = freq
    fmin=1
    fmax=128
    nperseg=128
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
    return coherence_matrix
def generate_coh_array(segment):
    # segment = np.nan_to_num(segment)
    segment = np.array(segment)
    # Create artificial epochs (e.g., splitting into 4 sub-epochs)
    window_size = 128
    overlap = 64  # 50% overlap
    step_size = window_size - overlap
    segments = []

    # Loop through each channel
    for channel in segment:
        channel_segments = []
        for start in range(0, len(channel) - window_size + 1, step_size):
            channel_segments.append(channel[start:start + window_size])
        segments.append(np.array(channel_segments))

    # Convert to numpy array for final shape
    segments = np.array(segments)
    # Replace NaN/Inf with finite numbers
    segments_transposed = np.transpose(segments, (1, 0, 2))
    logger.info(f"segment shape: {segments.shape}")
    con = spectral_connectivity_epochs(
        segments_transposed,
        method='coh',
        mode='multitaper',
        fmin=20,
        fmax=120,                           # Use FFT for spectral estimation
        sfreq=freq,
        tmin=0,                  # Start coherence computation from 1 second
        tmax=None,                 # Use until the end of the data
        faverage=True,            # Do not average frequencies
        verbose=False           # Print detailed logs
    )
    
    coherence_matrix = con.get_data().reshape(dim, dim)
    get_stats_2(coherence_matrix, "connectivity")
    # coherence_matrix = np.nan_to_num(coherence_matrix)
    return coherence_matrix

def generate_plv_array(segment):
    segment = np.array(segment)
    window_size = 128
    overlap = 64  # 50% overlap
    step_size = window_size - overlap
    segments = []

    # Loop through each channel
    for channel in segment:
        channel_segments = []
        for start in range(0, len(channel) - window_size + 1, step_size):
            channel_segments.append(channel[start:start + window_size])
        segments.append(np.array(channel_segments))

    # Convert to numpy array for final shape
    segments = np.array(segments)
    segments_transposed = np.transpose(segments, (1, 0, 2))
    # logger.info(segments.shape)
    con = spectral_connectivity_epochs(
        segments_transposed, method='plv', fmin=20,
        fmax=120,  mode='multitaper', sfreq=freq,
        faverage=True, tmin=0, tmax=None, verbose=False
    )
    get_stats_2(con.get_data(), "plv")
    plv_matrix = con.get_data().reshape(dim, dim)
    plv_matrix = np.nan_to_num(plv_matrix)
    return plv_matrix
def generate_fcns(data, dim):
    fcns = []
    i = 0
    while i < len(data):
        try:
            fcn = []
            # all ones
            fcn.append(np.ones((dim, dim)))
            # correlation
            fcn.append(np.corrcoef(data[i]))
            # coherence
            fcn.append(generate_coh_array(data[i]))       
            # PLV
            fcn.append(generate_plv_array(data[i]))

            fcns.append(fcn)
        except Exception as e:
            logger.error(f"Skipping the data point as {e}")  
        i+=1
    return fcns

def generate_node_features(data, dim):
    features = []
    # all ones
    features.append(np.ones((dim, dim)))
    # avg energy 
    features.append(np.mean(data**2, axis=1, keepdims=True))
    # band energies
    features.append(get_band_energies(data,dim)) 
    return features

def generate_graphs(data, fcns):
    i = 0
    graphs = []
    while i < len(data):
        try:
            NF = generate_node_features(data[i], dim)
            graphs.append([fcns[i], NF, np.expand_dims(fcns[i], axis=-1)])
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

def get_pyg_grs(num_electrodes, new_data_train):
    return create_tensordata_new(num_nodes=num_electrodes, data_list=new_data_train, complete=True, save=False, logdir=None)

def get_stats(fcns):
    feature_map = {
        0: "ones",
        1: "correlation",
        2: "connectivity",
        3: "phase lock val"
    }
    print("PLV == Connectivity:", np.all(fcns[0][2] == fcns[0][3]))
    feature_matrices = [fcn[1].flatten() for fcn in fcns]  # Flatten each graph matrix
    all_values = np.concatenate(feature_matrices)  # Combine all graphs into one array
    
    mean_value = np.mean(all_values)
    std_value = np.std(all_values)
    
    logger.info(f"samples correlation mean is {mean_value} and variance value is {std_value}...\n")




def get_segement_stats(segement):
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

def generate_embeddings_util(data_log_dir,preictal_mat,ictal_mat,file_index):

    if preictal_mat is not None:
        logger.info(f"generating preictal segments for ...")
        try:
            preictal_segments = generate_segements(preictal_mat)
        except Exception as e:
            logger.error(f"error in generating preictal segments:- {e}")
            sys.exit(1)
        logger.info(f"generating preictal fcns for {data_log_dir}/{file_index}...")

        # get_segement_stats(preictal_segments)

        try:
            precital_fcns = generate_fcns(preictal_segments, dim)
        except Exception as e: 
            logger.error(f"error in generating preictal fcns:- {e}")
            sys.exit(1)
        logger.info(f"generating preictal graphs for {data_log_dir}/{file_index}...")

        # get_stats(precital_fcns)

        try:  
            preictal_graphs = generate_graphs(preictal_segments, precital_fcns)
        except Exception as e:
            logger.error(f"error in generating preictal graphs:- {e}")
            sys.exit(1)
        logger.info("generating preictal train embeddings...")
        try:
            data_preictal = new_grs(preictal_graphs, type="preictal")
        except Exception as e:
            logger.info(f"error generating graphs for pretical data:- {e}")
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
            ictal_fcns = generate_fcns(ictal_segments, dim)
        except Exception as e: 
            logger.error(f"error in generating ictal fcns :- {e}")
            sys.exit(1)
        logger.info("generating ictal graphs...")

        # get_stats(ictal_fcns)
        try:
            ictal_graphs = generate_graphs(ictal_segments, ictal_fcns)
        except Exception as e:
            logger.error(f"error in generating ictal graphs:- {e}")
            sys.exit(1)
        logger.info("generating ictal train embeddings...")
        try:
            data_ictal = new_grs(ictal_graphs, type="ictal")
        except Exception as e:
            logger.info(f"error generating graphs for ictal data:- {e}")
            sys.exit(e)

    if preictal_mat is not None and ictal_mat is not None:
        logger.info("appending graphs for ictal and preictal data...")
        try:
            new_data = data_preictal + data_ictal
        except Exception as e:
            logger.info(f"error appending graphs for ictal and preictal data {e}")
            sys.exit(e)
    else: 
        try:
            new_data = data_preictal
        except Exception as e:
            logger.info(f"error appending graphs for ictal and preictal data {e}")
            sys.exit(e)

    num_electrodes = new_data[0][0][0].shape[0]

    logger.info("generating pyg format graphs...")
    try: 
        pyg_grs = get_pyg_grs(num_electrodes,new_data)
    except Exception as e:
        logger.info(f"error generating pyg format graphs {e}")

    pyg_Data_path = f"{data_log_dir}/jh101_pyg_Data_{file_index}.pt"

    convert_to_Data(pyg_grs, logger, save=True, logdir=pyg_Data_path)
  
def generate_embeddings(data_path,label_path,index,data_log):
    logger.info("generating matrices...")

    df = pd.read_csv(label_path)
    file_index = 0
    data_log_dir = f"{data_log}/chb{index}"
    if not os.path.exists(data_log_dir):
        os.makedirs(data_log_dir)
    for index, row in df.iterrows():
        file_name = row['File_names']
        label = row['Labels']
        start = row['Start_time']*freq
        end = row['End_time']*freq
        file_path = f"{data_path}/{file_name}"
        try:
            raw = mne.io.read_raw_edf(file_path)
        except Exception as e:
            logger.error(f"Error reading .edf file at {file_path}")
        data, times = raw[:]
        data = data[:channels_to_take,:]
        if label == 0: 
            preictal_list = data
            generate_embeddings_util(data_log_dir,data,None,file_index)
        else:
            preictal_list = []
            preictal_list.append(data[:, 0:start])
            if end < len(data)-1:
                preictal_list.append(data[:, end+1:len(data)])
            ictal_list = data[:, start:end]
            generate_embeddings_util(data_log_dir, np.hstack(preictal_list),ictal_list,file_index)
        file_index += 1

   

    

    





    

    




    