# Installation 

Ensure you have the latest conda version.

### Step 1: Download the CHB-MIT Scalp Dataset 
you can either download the zip file or using gsutil
```bash
gsutil -m -u YOUR_PROJECT_ID cp -r gs://chbmit-1.0.0.physionet.org DESTINATION
```
### Step 2: Clone the Repository
```bash
git clone https://github.com/aku221b/epilepsy-detection.git
cd ssl-seizure-detection
```
### Step 3 Create Conda Environment
```bash
conda create --name ssl-seizure-detection python=3.10
conda activate ssl-seizure-detection
```
### Step 4: Install PyTorch
For PC or Linux:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
For Mac:
```bash
conda install pytorch::pytorch torchvision torchaudio -c pytorch
```
### Step 5: Install mne
```bash
conda create --channel=conda-forge --strict-channel-priority --name=mne mne
```
### Step 6: Install mne-connectivity
```bash
conda install -c conda-forge mne-connectivity
```
### Step 7: Install PyTorch Geometric
```bash
pip install torch_geometric
```
### Step 8: Install Additional Packages
```bash
conda install scikit-learn pandas scipy
```

