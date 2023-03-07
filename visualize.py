from os.path import join as pjoin
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')


def preprocess_log(log_path):
    with open(log_path, "rb") as f:
        data = pickle.load(f)
    return np.stack(data["performance"])


if __name__ == "__main__":

    # Argument Passing
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="sumo_3by3")
    parser.add_argument("--exp_id", type=str, default="debug")

    args = parser.parse_args()

    model_names = ["random", "cma-es", "gp", "learned_gp", "rgpe", "ablr", "anp"]
    colors = ["firebrick", "deepskyblue", "royalblue", "chocolate", "darkgreen", "hotpink", "mediumpurple"]
    labels = ["Random", "CMA-ES", "GP-BO", "Learned GP-BO", "RGPE", "ABLR", "Ours"]

    if args.task.startswith("sumo"):
        task, subtask = args.task.split("_")
        root = pjoin("results", task, subtask)
        scenarios = range(10)

    plt.figure(figsize=(8.0, 8.0), dpi=300)

    performance_comparison = {}
    for i, model_name in enumerate(model_names):
        overall_performance = []
        for scenario in scenarios:
            log_path = pjoin(root, model_name, args.exp_id, f"results_{scenario}.pkl")
            performance = preprocess_log(log_path)
            if args.task.startswith("sumo"):
                performance = np.minimum.accumulate(performance, axis=1)
            else:
                performance = np.maximum.accumulate(performance, axis=1)

            overall_performance.append(performance)
        overall_performance = np.stack(overall_performance).mean(axis=0)

        overall_performance_mean = np.stack(overall_performance).mean(axis=0)[10-1:]
        overall_performance_std = np.stack(overall_performance).std(axis=0)[10-1:]

        plt.plot(range(len(overall_performance_mean)), overall_performance_mean, label=labels[i], linewidth=2.0, color=colors[i])
        plt.fill_between(range(len(overall_performance_mean)), overall_performance_mean - overall_performance_std,
                         overall_performance_mean + overall_performance_std, alpha=0.2, color=colors[i])
        performance_comparison[model_name] = [overall_performance_mean[-1], overall_performance_std[-1]]

    plt.xlim(0, len(overall_performance_mean)-1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Number of Trials", fontsize=20, labelpad=12)
    plt.ylabel("Average Number of Waiting Vehicles (veh/30min)", fontsize=20, labelpad=12)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               fancybox=True, shadow=True, ncol=4, fontsize="large")
    plt.tight_layout()
    plt.savefig(pjoin(root, f"{args.exp_id}_performance.png"))

    for model_name in model_names:
        print(f"Model Name: {model_name}\tPerformance: {performance_comparison[model_name][0]:.2f} / {performance_comparison[model_name][1]:.2f}")
