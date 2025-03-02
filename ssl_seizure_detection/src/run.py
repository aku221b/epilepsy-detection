#!/usr/bin/env python
from .data.embeddings_generator import generate_embeddings 
from .train.train import train
from .config.config import TrainConfig
import argparse
from pathlib import Path   
import multiprocessing
import time
import logging
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2] 
label_base_path = str(PROJECT_ROOT / "parsed_labels")
participants_to_avoid = [0,2,4,5,6,7,8,9,10,11]

def get_logger(data_log):
    logger = logging.getLogger("GraphGeneration")
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(os.path.join(data_log, "graph_generation.log"))
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def run_multiprocessing(num_core, participants):
    """
    Runs multiprocessing with num_core parallel processes
    and total_participants in batches.
    """
    with multiprocessing.Pool(processes=num_core) as pool:
        results = pool.map(worker, participants)  # Process in parallel
    return results

def worker(participant):
    """ Wrapper function for multiprocessing """
    data_path, label_path, index_str, data_log, stat_log = participant
    return generate_embeddings(data_path, label_path, index_str, data_log, stat_log)

def generate_all_embeddings(total_particpants, data_base_path,label_base_path, data_log, stat_log):
    index = 0
    count = 0
    participants=[]
    logger = get_logger(stat_log)

    num_cores = multiprocessing.cpu_count()

    while count < total_particpants:
        if index in participants_to_avoid:
            index+=1 # skipping participant 4 as data it has problematic data 
            continue
        index_str = str(index+1).zfill(2)
        p_name = f"chb{index_str}"
        data_path = f"{data_base_path}/{p_name}"
        label_path = f"{label_base_path}/{p_name}_labels.csv"
        participants.append((data_path,label_path,index_str,data_log,stat_log))
        count+=1
        index+=1

    start_time = time.time()
    run_multiprocessing(num_cores, participants)
    end_time = time.time()

    logger.info(f"\nTotal Time Taken: {end_time - start_time:.2f} seconds")

def LOO_training(data_path, logdir, index):

    model_config = {
        "num_node_features":6,
        "num_edge_features":2,
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
        val_ratio=1.0,
        test_ratio=0,
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
    parser.add_argument("-sl", "--stat_log", type=str, help ="path to dump your logs and stats", required=True)
    parser.add_argument("-p", "--total_participants", type=int, help ="number of participants data to train on", required=True)

    args = parser.parse_args()
    generate_all_embeddings(args.total_participants,args.data_base_path,label_base_path, args.data_log,args.stat_log)

    # for i in range(args.total_participants):
    #     LOO_training(args.data_log,args.stat_log,i)

    

    


        
            

