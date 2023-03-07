import pickle
import argparse
from os.path import join as pjoin

import yaml
import torch
from tqdm import tqdm
from attrdict import AttrDict

from env.sumo.sumo_env import parallelize_sumo


if __name__ == "__main__":
    # Argument Passing
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="sumo_3by3")
    parser.add_argument("--root", type=str, default='.')
    args = parser.parse_args()

    dtype = torch.float32
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    
    if args.task.startswith("sumo"):
        # os.environ["SUMO_HOME"] = "/usr/share/sumo"
        num_worker = 5
        task, subtask = args.task.split("_")

        # Settings
        settings = yaml.load(open(pjoin(args.root, "config", task, subtask, "preprocess_settings.yaml"), "r"),
                             Loader=yaml.SafeLoader)
        torch.manual_seed(settings["seed"])

        # 1. data for training
        raw_data = AttrDict()
        raw_data.x = torch.zeros(size=(settings["num_scenarios"], settings["num_samples_per_scenario"], settings["raw_input_dim"]))
        raw_data.y = torch.zeros(size=(settings["num_scenarios"], settings["num_samples_per_scenario"], settings["raw_output_dim"]))
        
        for scenario in tqdm(range(settings["num_scenarios"])):
            x_task, y_task = parallelize_sumo(args, scenario, settings, num_worker, 'train')
            raw_data.x[scenario] = x_task
            raw_data.y[scenario] = y_task
        
        with open(f"./data/{task}/{subtask}/input_output_pair.pkl", "wb") as f:
            pickle.dump(raw_data, f)

        # 2. data for validation
        raw_data_val = AttrDict()
        raw_data_val.x = torch.zeros(size=(settings["num_val_scenario"], settings["num_samples_per_scenario"], settings["raw_input_dim"]))
        raw_data_val.y = torch.zeros(size=(settings["num_val_scenario"], settings["num_samples_per_scenario"], settings["raw_output_dim"]))

        for scenario in tqdm(range(settings["num_val_scenario"])):
            x_task, y_task = parallelize_sumo(args, scenario, settings, num_worker, 'valid')
            raw_data_val.x[scenario] = x_task
            raw_data_val.y[scenario] = y_task

        with open(f"./data/{task}/{subtask}/input_output_pair_valid.pkl", "wb") as f:
            pickle.dump(raw_data_val, f)

        # 3. normalize data
        data = AttrDict()
        data.x = raw_data.x
        data.y = torch.sum(raw_data.y, dim=-1, keepdim=True)

        data_val = AttrDict()
        data_val.x = raw_data_val.x
        data_val.y = torch.sum(raw_data_val.y, dim=-1, keepdim=True)

        data.y_min = data_val.y_min = min(data.y.min(), data_val.y.min())
        data.y_max = data_val.y_max = max(data.y.max(), data_val.y.max())

        data.y = (data.y - data.y_min) / (data.y_max - data.y_min)
        data_val.y = (data_val.y - data_val.y_min) / (data_val.y_max - data_val.y_min)

        with open(f"./data/{task}/{subtask}/traffic_data.pkl", "wb") as f:
            pickle.dump(data, f)

        with open(f"./data/{task}/{subtask}/traffic_data_valid.pkl", "wb") as f:
            pickle.dump(data_val, f)
