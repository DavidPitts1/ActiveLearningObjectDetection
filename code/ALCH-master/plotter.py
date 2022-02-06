import sys
import os

from pathlib import Path

import json
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from util.plot_utils import *
from pathlib import Path, PurePath

import argparse

def plot_by_seed(seed, metric="", f_name="run_data.json"):
    fig, ax = plt.subplots()

    log_dirs = [pth for pth in (Path.cwd() / "logs").iterdir()]
    for log in log_dirs:
        try:
            path = os.path.join(log, f_name)
            with open(path) as f:
                data = json.load(f)

                if "args" not in data:
                    continue
                if data['args']['seed'] != seed:
                    print(5)
                    continue

            #val_size = data['al_data']['val_size']
            val_size = 0
            queries = data['queries_data']
            n_labeled = [q['n_total_labeled'] for q in queries]
            if not "random" in path:
                n_labeled = [q['n_total_labeled'] + val_size for q in queries]

            if "mAP" in metric:
                ewm_col = 0
                # dropna()?
                y = [q['eval_stats']["test_coco_eval_bbox"] for q in queries]
                y = np.array(y)
                coco_eval = pd.DataFrame(
                   y[:, 1]
                ).ewm(com=ewm_col).mean()

                y = np.array(coco_eval).flatten()
            else:
                y = [q['eval_stats'][metric] for q in queries]

            #label = log.name.split("/")[-1].split("_")[0]
            label = log.name

            if "random" in path:
                ax.plot(n_labeled, y, c="r", marker='o', markersize=3, label=label)
            else:
                #ax.plot(n_labeled, y, marker='o', c='b', markersize=3, label="AL_classs_hardness")
                ax.plot(n_labeled, y, marker='o', markersize=3, label=label)

            plt.title("{}, {}".format(metric, seed))
        except OSError:
            pass

    leg = ax.legend()
    ax.set_xlabel("Number of labeled images")
    ax.set_ylabel(metric)

    plt.show()

def plot_eval(log_dirs, metric="", f_name="run_data.json"):
    fig, ax = plt.subplots()

    for log in log_dirs:
        path = os.path.join(log, f_name)
        with open(path) as f:
            data = json.load(f)

        #val_size = data['al_data']['val_size']
        val_size = 0
        queries = data['queries_data']
        n_labeled = [q['n_total_labeled'] for q in queries]
        if not "random" in path:
            n_labeled = [q['n_total_labeled'] + val_size for q in queries]

        if "mAP" in metric:
            ewm_col = 0
            # dropna()?
            y = [q['eval_stats']["test_coco_eval_bbox"] for q in queries]
            y = np.array(y)
            coco_eval = pd.DataFrame(
               y[:, 1]
            ).ewm(com=ewm_col).mean()

            y = np.array(coco_eval).flatten()
        else:
            y = [q['eval_stats'][metric] for q in queries]

        #label = log.name.split("/")[-1].split("_")[0]
        label = log.name

        if "random" in path:
            ax.plot(n_labeled, y, c="r", marker='o', markersize=3, label=label)
        else:
            ax.plot(n_labeled, y, marker='o', c='b', markersize=3, label="AL_classs_hardness")

        plt.title(metric)

    leg = ax.legend()
    ax.set_xlabel("Number of labeled images")
    ax.set_ylabel(metric)

    plt.show()


parser = argparse.ArgumentParser(description='')
parser.add_argument("--d1", type=str, default="23.04")
parser.add_argument("--d2", type=str, default="23.04")
args = parser.parse_args()


base_path = os.getcwd()
print(base_path)
#date = args.date

log_dirs = [pth for pth in (Path.cwd() / "logs").iterdir() if args.d1 in pth.name]
log_dirs += [pth for pth in (Path.cwd() / "logs").iterdir() if args.d2 in pth.name]
#log_dirs += [pth for pth in (Path.cwd() / "logs").iterdir() if "random_25.05"in pth.name]
#log_dirs += [pth for pth in (Path.cwd() / "logs").iterdir() if "random_14.04"in pth.name]
#log_dirs += [pth for pth in (Path.cwd() / "logs").iterdir() if "random_13.05" in pth.name]
#print(log_dirs)

# fields = ['class_error', 'loss_bbox_unscaled', 'mAP']
fields = ['class_error', 'loss_bbox', 'mAP']
metrics = ["test_" + f for f in fields]

for metric in metrics:
    plot_eval(log_dirs, metric=metric)

#for metric in metrics:
#    plot_by_seed(100, metric=metric)


