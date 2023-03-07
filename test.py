import os
from os.path import join as pjoin
import pickle
import argparse
import pickle
import random

import yaml
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement

from utils.log import get_logger
from utils.misc import load_module


if __name__ == "__main__":
    # Argument Passing
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="sumo_3by3")
    parser.add_argument("--model", type=str, default='anp')
    parser.add_argument("--exp_id", type=str, default="debug")

    # only for SUMO
    parser.add_argument("--scenario_id", type=int, default=0)
    
    args = parser.parse_args()

    dtype = torch.double
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    if args.task.startswith("sumo"):
        from env.sumo.sumo_env import SumoEnv
        task = args.task.replace("_", "/")

    root = pjoin("results", task, args.model, args.exp_id)
    if not os.path.isdir(root):
        os.makedirs(root)

    # Set Path for loading model, saving results, and logger
    model_load_path = pjoin(root, "ckpt.tar")
    if args.task.startswith("sumo"):
        results_save_path = pjoin(root, f"results_{args.scenario_id}.pkl")
        log_path = pjoin(root, f"test_{args.scenario_id}.log")
        # os.environ["SUMO_HOME"] = "/usr/share/sumo"
    logger = get_logger(log_path)

    # Set Hyperparameters
    settings = yaml.load(open(pjoin("config", task, "test_settings.yaml"), "r"), Loader=yaml.SafeLoader)

    if args.task.startswith("sumo"):
        env = SumoEnv(args.task, args.scenario_id, run_type='test')
        dim, bounds, equality_constraints = env.get_constraints(dtype, device)

    # Load Data
    data_path = pjoin("data", task, "traffic_data.pkl")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    y_min = data.y_min
    y_max = data.y_max  

    # Test
    test_results_overall = defaultdict(list)
    for num_test in range(settings["num_tests"]):
        test_results = {}
        
        # Set Seed for Reproduction
        seed = settings["seed"] + num_test
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        if args.model == "anp":
            # Load Model
            model_cls = getattr(load_module(pjoin("models", f"{args.model}.py")), args.model.upper())
            with open(pjoin("results", task, args.model, args.exp_id, "model.yaml"), "r") as f:
                config = yaml.safe_load(f)
            model = model_cls(**config).to(dtype=dtype, device=device)

            ckpt = torch.load(model_load_path)
            model.load_state_dict(ckpt.model)

            x_init = env.get_init_points(settings["init_num_points"], dim, seed=seed)
            x_init = x_init.to(dtype=dtype, device=device)
            _, y_init = env.evaluate(x_init)
            y_init = y_init.to(dtype=dtype, device=device)

            model.eval()
            for trial in tqdm(range(settings["test_num_trials"])):
                model.x_init = x_init
                model.y_init = (y_init - y_min) / (y_max - y_min)
                acqf = ExpectedImprovement(model, best_f=model.y_init.min(), maximize=False)
                x_cand, _ = optimize_acqf(acqf, bounds=bounds,
                                          q=1, num_restarts=10, raw_samples=512)

                x_cand_transform, y_cand = env.evaluate(x_cand)

                x_init = torch.cat([x_init, x_cand], dim=0)
                y_init = torch.cat([y_init, y_cand], dim=0)

                logger.info(f"[{trial+1}/{settings['test_num_trials']}]\nAction: {x_cand_transform}\nPerformance: {y_cand.item():4f}")

        test_results["actions"] = x_init.cpu().detach().numpy()
        test_results["performance"] = y_init.cpu().detach().numpy().flatten()

        test_results_overall["actions"].append(test_results["actions"])
        test_results_overall["performance"].append(test_results["performance"])

        logger.info(f'Best Performance: {test_results_overall["performance"][-1].min().item():.4f}')
    
    # Save results
    with open(results_save_path, "wb") as f:
        pickle.dump(test_results_overall, f)
