#!/usr/bin/env python
from embeddings_generator import generate_embeddings 
import sys
import argparse
from pathlib import Path    
PROJECT_ROOT = Path(__file__).resolve().parents[3] 
sys.path.append(str(PROJECT_ROOT / "ssl_seizure_detection/src/train"))
sys.path.append(str(PROJECT_ROOT / "ssl_seizure_detection/src/config"))

from train import train
from config import TrainConfig
label_base_path = str(PROJECT_ROOT / "parsed_labels")
def generate_all_embeddings(total_particpants, data_base_path,label_base_path, data_log):
    if total_particpants >= 4:
        total_particpants += 1
    for index in range(total_particpants):
        if index == 3: # skipping participant 3 as data it has problematic data 
            continue
        index = str(index+1).zfill(2)
        p_name = f"chb{index}"
        data_path = f"{data_base_path}/{p_name}"
        label_path = f"{label_base_path}/{p_name}_labels.csv"

        generate_embeddings(data_path,label_path,index, data_log)

def LOO_training(data_path, logdir, index):

    model_config = {
        "num_node_features":9,
        "num_edge_features":3,
        "hidden_channels":[32, 16, 16],
        "batch_norm":True,
        "classify":"binary",
        "head" : "linear",
        "dropout": True,
        "p": 0.1
    }  

    loss_config = None

    train_config = TrainConfig(
        data_path=data_path,
        logdir=logdir,
        patient_id="dummy",
        epochs=20,
        data_size=1.0,
        train_ratio=1.0,
        val_ratio=0.7,
        test_ratio=0.3,
        batch_size=32,
        num_workers=4,
        lr=[1e-3, 0.02],
        weight_decay=1e-3,
        model_id="supervised",
        timing=True,
        project_id="Test_Bay",
        patience=20,
        eta_min=0.002,
        run_type="all",
        exp_id= "dummy_1",
        datetime_id=None,
        requires_grad=True,
        classify= "binary",
        head="linear"
    )


    train(train_config, model_config, loss_config, index, logdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser for preprocess and training script")
    parser.add_argument("-dp", "--data_base_path", type=str, help ="path to your data directory", required=True)
    parser.add_argument("-dl", "--data_log", type=str, help ="path to dump your graph embeddings", required=True)
    parser.add_argument("-sl", "--stat_log", type=str, help ="path to dump your training stats", required=True)
    parser.add_argument("-p", "--total_participants", type=int, help ="number of participants data to train on", required=True)

    args = parser.parse_args()
    generate_all_embeddings(args.total_participants,args.data_base_path,label_base_path, args.data_log)

    # for i in range(args.total_participants):
    #     LOO_training(args.data_log,args.stat_log,i)

    

    


        
            

