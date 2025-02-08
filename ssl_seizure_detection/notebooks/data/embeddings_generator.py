#!/usr/bin/env python
import pandas as pd
import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs
from scipy.signal import welch,detrend
import pickle
import sys
from pathlib import Path    
PROJECT_ROOT = Path(__file__).resolve().parents[3] 
sys.path.append(str(PROJECT_ROOT / "ssl_seizure_detection/src/data"))
from preprocess import new_grs, create_tensordata_new, convert_to_Data, pseudo_data, convert_to_PairData, convert_to_TripletData


freq = 256
ws = int(1*freq)
step = int(0.125*256)
dim = channels_to_take = 22

def get_band_energies(data, dim):
    freq_bands = {
        "Delta": (1, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma1": (30, 70),
        "Gamma2": (70, 100),
        "Gamma3": (30, 70),
        "Gamma4": (70, 100),
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
def generate_coh_array(segment):
    n_channels = segment.shape[0]
    n_times = segment.shape[1]
    segment = np.nan_to_num(segment)
    segment = np.array(segment)

    # Create artificial epochs (e.g., splitting into 4 sub-epochs)
    segment_split = segment.reshape(1, n_channels, n_times)
    # Replace NaN/Inf with finite numbers
    con = spectral_connectivity_epochs(
        segment_split,
        method='coh',
        mode='fourier',
        fmin=5,
        fmax=100,                           # Use FFT for spectral estimation
        sfreq=freq,
        tmin=0,                  # Start coherence computation from 1 second
        tmax=None,                 # Use until the end of the data
        faverage=True,            # Do not average frequencies
        verbose=False           # Print detailed logs
    )
    coherence_matrix = con.get_data().reshape(dim, dim)
    coherence_matrix = np.nan_to_num(coherence_matrix)
    return coherence_matrix

def generate_plv_array(segment):
    n_channels = segment.shape[0]
    n_times = segment.shape[1]
    segment = np.nan_to_num(segment)
    segment= np.array(segment)
    # Create artificial epochs (e.g., splitting into 4 sub-epochs)
    segments_split = segment.reshape(1  , n_channels, n_times)
    con = spectral_connectivity_epochs(
        segments_split, method='plv', fmin=5,
        fmax=100,  mode='multitaper', sfreq=freq,
        faverage=True, tmin=0, tmax=None, verbose=False
    )
    plv_matrix = con.get_data().reshape(dim, dim)
    plv_matrix = np.nan_to_num(plv_matrix)
    return plv_matrix
def generate_fcns(data, dim):
    fcns = []
    i = 0
    while i < len(data):
        fcn = []
        # all ones
        fcn.append(np.ones((dim, dim)))
        # correlation
        fcn.append(np.corrcoef(data[i]))
        # coherence
        fcn.append(generate_coh_array(data[i]))       
        # PLV
        fcn.append(generate_coh_array(data[i]))

        fcns.append(fcn)
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

def get_data_matrices(data_path, label_path):
    df = pd.read_csv(label_path)
    preitcal_list = []
    ictal_list = []
    for index, row in df.iterrows():
        file_name = row['File_names']
        label = row['Labels']
        start = row['Start_time']*256
        end = row['End_time']*256
        file_path = f"{data_path}/{file_name}"
        raw = mne.io.read_raw_edf(file_path)
        data, times = raw[:]
        data = data[:channels_to_take,:]
        if label == 0: 
            preitcal_list.append(data)
        else:
            preitcal_list.append(data[:, 0:start])
            if end < len(data)-1:
                preitcal_list.append(data[:, end+1:len(data)])
            ictal_list.append(data[:, start:end])
    preictal_mat = np.hstack(preitcal_list)
    ictal_mat = np.hstack(ictal_list)
    if preictal_mat.shape[1] >= 100000:
        return preictal_mat[:, :100000], ictal_mat
    else:
        return preictal_mat, ictal_mat

def generate_graphs(data, fcns):
    i = 0
    graphs = []
    while i < len(data):
        NF = generate_node_features(data[i], dim)
        graphs.append([fcns[i], NF, np.expand_dims(fcns[i], axis=-1)])
        i += 1
    return graphs

def generate_segements(data):
     # print(data.shape)
    dim, num = data.shape
    i = 0
    segments = []
    while i < num:
        if num - i > ws:
            curr_win = data[:, i:i+256]
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

def generate_embeddings_util(preictal_data, ictal_data, index):
    base_path = "/Users/dentira/anomaly-detection/epilepsy-detection/ssl_seizure_detection/data/supervised/"
    file_path_preictal = f"{base_path}preictal_{index}.pkl"
    file_path_ictal = f"{base_path}ictal_{index}.pkl"

    with open(file_path_preictal,'wb') as file:
        pickle.dump(preictal_data, file)

    with open(file_path_ictal,'wb') as file:
        pickle.dump(ictal_data, file)

    with open(file_path_preictal, 'rb') as f:
        data_preictal= pickle.load(f)
    
    with open(file_path_ictal, 'rb') as f:
        data_ictal = pickle.load(f)

    data_preictal = new_grs(data_preictal, type="preictal")
    data_ictal = new_grs(data_ictal, type="ictal")

    new_data = data_preictal + data_ictal

    num_electrodes = new_data[0][0][0].shape[0]

    pyg_grs = get_pyg_grs(num_electrodes,new_data)

    pyg_Data_path = f"/Users/dentira/anomaly-detection/epilepsy-detection/ssl_seizure_detection/data/patient_gr/jh101_pyg_Data_{index}.pt"

    convert_to_Data(pyg_grs, save=True, logdir=pyg_Data_path)


def generate_embeddings(data_path,label_path,index):
    print("generating matrices...")
    preictal_mat, ictal_mat = get_data_matrices(data_path, label_path)
    print(preictal_mat.shape)
    print(ictal_mat.shape)

    print("generating segments...")
    preictal_segments = generate_segements(preictal_mat)
    ictal_segments = generate_segements(ictal_mat)

    print("generating fcns...")
    ictal_fcns = generate_fcns(ictal_segments, dim)
    precital_fcns = generate_fcns(preictal_segments, dim)

    print("generating graphs...")
    preictal_graphs = generate_graphs(preictal_segments, precital_fcns)
    ictal_graphs = generate_graphs(ictal_segments, ictal_fcns)

    print("generating train embeddings...")
    generate_embeddings_util(preictal_graphs, ictal_graphs, index)

# if __name__ == "__main__":
#     generate_embeddings(sys.argv[1], sys.argv[2], sys.argv[3])


    

    





    

    




    