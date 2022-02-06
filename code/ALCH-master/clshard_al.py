from cls_ops import *
from torch.utils.data import DataLoader, DistributedSampler, Subset, ConcatDataset

import util.misc as utils
from torch.utils.data import Dataset, DataLoader
from numpy.random import dirichlet

import time
import math
import random


class ClsHardnessAL:
    def __init__(self, model, criterion, device, query_size, num_workers,
                 loader_pool, loader_val, pool_set, logger, cycle, args, n_classes=21,
                 with_noise="False", dir_inv_var=1e10):
        self.model = model
        self.criterion = criterion
        self.device = device

        self.args = args

        self.n_classes = n_classes

        # classification threshold
        self.cls_threshold = 0.5

        self.query_size = query_size

        self.with_noise = with_noise
        self.dir_inv_var = dir_inv_var
        self.num_workers = num_workers


        self.pool_set = pool_set
        self.loader_pool = loader_pool
        self.loader_val = loader_val

        self.cycle = cycle
        self.logger = logger

    @torch.no_grad()
    def eval_pool(self):
        print("[CL_AL] Running model on the pool")
        start_time = time.time()

        img_ids = [0] * len(self.loader_pool)

        cls2img = [[] for i in range(self.n_classes)]

        count_classified = 0
        self.model.eval()
        for i, (samples, targets) in tqdm(enumerate(self.loader_pool), total=len(self.loader_pool)):
            samples = samples.to(self.device)
            outputs = self.model(samples)

            img_ids[i] = int(targets[0]["image_id"])

            # dataloader has only 1 batch size, so index it directly
            # take the max in every
            probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]  # last class output is not needed
            high_conf_idx = probas.max(-1).values >= self.cls_threshold
            if high_conf_idx.sum() > 0:
                conf_preds = probas[high_conf_idx]  # (Reduced Queries) x C
                cls_preds = conf_preds.argmax(dim=1).unique()  # (Reduced Queries) : the predicted class in each query

                for cls in cls_preds:
                    cls2img[cls].append((self.pool_set[i], img_ids[i]))  # (raw dataset image index, image_id)

                count_classified = count_classified + cls_preds.size()[0]


        print("[cls_hard_al] count_classified={}".format(count_classified))
        print("Time taken to eval on pool: {}".format((time.time() - start_time) / 60))

        print("Time taken(minutes): {}".format((time.time() - start_time)/60))

        return cls2img


    def get_idx_from_pool(self):
        """
        :return: the indices from the pool to query from the oracle
        """
        cls_op = ClsOperator(self.model, n_classes=self.n_classes, args=self.args)
        cls_losses, df = cls_op.cls_eval(self.criterion, self.loader_val, self.device)

        cls_weights = cls_op.get_cls_weights().numpy()

        df["wanted"] = np.nan
        df["actual"] = np.nan

        df["cls_weights"] = cls_weights # TODO: verify numpy

        if self.with_noise == "True":
            eps = 1e-8
            print("[cls_hard] dir_inv_var", self.dir_inv_var)
            alpha = (cls_weights + eps) * self.dir_inv_var
            print("[cls_hard] alpha:")
            print(alpha)
            cls_weights = dirichlet(alpha)
            print("[cls_hard] noisy weights")
            print(cls_weights)

        else:
            print("[cls_hard] weights: ")
            print(cls_weights)

        n_cls_samples = [int(math.ceil(w * self.query_size)) for w in cls_weights]

        # evaluate the model on all of the pool to get their estimated classes
        cls2img = self.eval_pool()  # list of lists

        print("[CL_AL] taking images from the pool")
        start_time = time.time()
        to_query_idx = []
        # choose what images to take from the pool
        for i, (n_samples, idx_id_c) in enumerate(zip(n_cls_samples, cls2img)):
            # idx = list(set(idx_c.copy()))

            # idx_id - list of lists of pairs (img_index, img_id)
            # remove duplicates
            idx_id = list(set(idx_id_c.copy()))
            # remove taken images
            idx_id = [pair for pair in idx_id if pair not in to_query_idx]

            if n_samples <= len(idx_id):
                # Added
                to_query_idx += idx_id[:n_samples]

            else:
                print("[CL_AL] [!] less pseudo labels then required samples for class:", i, cls_op.CLASSES[i])
                to_query_idx += idx_id[:n_samples]

            print("[CL_AL] # of target added examples for class ({}, {}) : {}"
                  .format(i, cls_op.CLASSES[i], min(len(idx_id_c), n_samples)))

            print("[CL_AL] wanted to take from class ({}, {}) : {}, took : {}"
                  .format(i, cls_op.CLASSES[i], n_samples, min(len(idx_id_c), n_samples)))

            df.at[i, "wanted"] = n_samples
            df.at[i, "actual"] = min(len(idx_id_c), n_samples)

        random.shuffle(to_query_idx)

        unpacked = list(zip(*to_query_idx))
        if len(unpacked) > 0:
            # cut queried indices if queried too much
            q_idx, q_img_id = list(unpacked[0])[:self.query_size], list(unpacked[1])[:self.query_size]

        else:
            print("[CL_AL] to_query_idx size = 0")
            q_idx = []
            q_img_id = []

        print("[CL_AL] total samples queried: {}".format(len(q_idx)))
        print("[CL_AL] total samples needed to query after weighting: {}".format(sum(n_cls_samples)))
        if sum(n_cls_samples) != self.query_size:
            print("[CL_AL] samples needed to query after weighting is different then the configuration. "
                  "\n[CL_AL] query size: {}, samples_weighted_query: {}".format(self.query_size, sum(n_cls_samples)))
        print("[CL_AL] Time taken for deciding on the queried images(minutes): {}".format((time.time() - start_time)/60))

        print("[CL_AL] logging df: ")
        self.logger.log_df(df, al_cycle=self.cycle)

        return q_idx, q_img_id




