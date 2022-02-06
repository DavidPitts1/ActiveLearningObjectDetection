import json
import os
import torch
import pickle
import datetime

class Logger:
    """ Logs data after every query
    This class has a directory to which it queries and various files
    # TODO: set ouptut_dir=None probably for all of DETR functions
    and let this class handle it
    """
    def __init__(self, al, output_dir, args):

        # self.run_data = {"dataset_name": "",  "dataset_size": 0, "model_name": "", "initial_size": 0, "query_size": 0,
        #                  "queries_data": []}

        # class data for ActiveLearner
        serializable = [int, str]
        self.al_learner_data = vars(al) # outputs a dict of the variables, which are setted in the constructor
        self.al_learner_data = {k: v for k, v in self.al_learner_data.items() if type(v) in serializable}
        self.args_data = {k: v for k, v in vars(args).items() if type(v) in serializable}

        self.run_data = {"dataset_name": "", "al_data": self.al_learner_data,
                "queries_data": [], "args": self.args_data, "val_ids": []}

        # query data - number of query, class error, mAP, loss_bbox, queried image ids, n_labeled - number of total
        # labeled images inclusive until now from the start of the run
        # per AL cycle
        # self.query_data = {"n_q": -1, "class_error": 0, "mAP": 0, "loss_bbox": 0, "q_img_ids": [],
        #                    "n_labeled": 0, "eval_stats": {}}



        self.pred_box = None
        self.pred_classes = None

        self.output_dir = output_dir
        self.file = os.path.join(output_dir, "run_data.json")

        self.sets_file = os.path.join(output_dir, "sets.py")

        self.q_model_info_dir = output_dir / "q_model_info"

        self.dfs_dir = output_dir / "dfs"


        self.q_model_info_dir.mkdir(parents=True, exist_ok=True)

        self.dfs_dir.mkdir(parents=True, exist_ok=True)


    def log_q_info(self, obj, al_cycle, obj_desc):
        print("[LOGGER]: saved {}".format(obj_desc))
        f_name = "{}_{}".format(al_cycle, obj_desc)
        path = str(self.q_model_info_dir) + "/" + f_name
        torch.save(obj, path)

    def log_df(self, df, al_cycle):
        print("[LOGGER]: saved {}".format("df"))
        f_name = "{}_{}".format(al_cycle, "cls_losses")
        path = str(self.dfs_dir) + "/" + f_name
        df.to_csv(path)

    def update(self, al_iter, eval_stats, n_total_labeled, n_sampled_query, img_ids, n_epochs, val_ids, sets):
        q_data = {"n_query": al_iter, "eval_stats": eval_stats, "n_total_labeled": n_total_labeled,
                  "n_sampled_query": n_sampled_query, "img_ids": img_ids, "n_epochs": n_epochs}

        if al_iter == 0:
            run_data = self.run_data
            q_data["val_ids"] = val_ids

            with open(self.sets_file, 'wb') as f:
                print("[LOGGER] dumped sets")
                pickle.dump(sets, f)
        else:
            with open(self.file, 'r') as f:
                run_data = json.load(f)

        run_data["queries_data"].append(q_data)

        with open(self.file, 'w') as f:
            json.dump(run_data, f)

        print("[LOGGER] Updated the run_data log")

    # def write(self):
    #     # maybe here just write not tensor data
    #     with open(self.file, 'w') as f:
    #         json.dump(self.q_data, f)
