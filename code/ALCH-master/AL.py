import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import util.misc as utils

import pickle

import datetime

import Logger
from tqdm import tqdm

import random

from torch.utils.data import DataLoader, DistributedSampler, Subset, ConcatDataset

from torch.utils.data.sampler import SubsetRandomSampler




class ActiveLearner:
    def __init__(self, model, criterion, optimizer, al_method, model_train, dataset_complete, dataset_test, device,
                 evaluator, lr_scheduler, args, param_dicts, n_iters=15, pool_prop=0.2, initial_size=2500, query_size=2500,
                 epochs_per_query=1, distributed=False, num_workers=1, out_val=True,
                 model_name="DETR", n_classes=21, with_noise="False", dir_inv_var=1e10):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.al_method = al_method
        self.model_name = model_name
        self.distributed = distributed
        self.num_workers = num_workers
        self.evaluator = evaluator
        self.lr_scheduler = lr_scheduler
        self.n_iters = n_iters
        self.n_classes = n_classes

        self.pool_prop = pool_prop

        self.dataset = dataset_complete

        if args.dataset_file == "pvoc":
            # this is for pvoc
            self.POOL_SZ = 10000
            self.LABELED_INIT_SZ = 1000
        elif args.dataset_file == "kitti":
            # for kitti
            self.POOL_SZ = 4500
            self.LABELED_INIT_SZ = 500


        self.VAL_SZ = args.val_sz

        #indices = list(range(int(self.pool_prop * len(dataset_complete))))
        indices = list(range(len(dataset_complete)))
        random.shuffle(indices)

        self.labeled_set = indices[:self.LABELED_INIT_SZ]
        # removing labeled set on the way
        self.pool_set = indices[self.LABELED_INIT_SZ: self.POOL_SZ]
        self.val_set = indices[self.POOL_SZ: self.POOL_SZ + self.VAL_SZ]

        self.sets = [self.labeled_set, self.pool_set, self.val_set]

        ####

        self.with_noise = with_noise
        self.dir_inv_var = dir_inv_var

        self.total_train_size = self.POOL_SZ

        self.dataset_test = dataset_test
        self.batch_size = args.batch_size
        # a functions that trains the model for one epoch
        self.model_train = model_train
        self.device = device

        self.initial_size = initial_size
        self.current_size = initial_size
        self.query_size = query_size
        self.epochs_per_query = epochs_per_query

        #self.val_prop = 0.2  # validation set proportion in the sampled dev set
        #self.val_size = int(self.val_prop * self.initial_size)

        self.max_to_label = 100000

        self.debug = True
        self.val_is_test = args.val_is_test

        self.args = args
        self.out_val = out_val
        self.param_dicts = param_dicts

        print("[AL] total pool size: {}".format(self.total_train_size))

    def print_dbg(self, s):
        if self.debug:
            print(s)

    class SubsetSequentialSampler(torch.utils.data.Sampler):
        r"""Samples elements sequentially from a given list of indices, without replacement.
        Arguments:
            indices (sequence): a sequence of indices
        
        credit: 
        https://github.com/Mephisto405/Learning-Loss-for-Active-Learning/blob/
        3c11ff7cf96d8bb4596209fe56265630fda19da6/data/sampler.py#L3
        """

        def __init__(self, indices):
            self.indices = indices

        def __iter__(self):
            return (self.indices[i] for i in range(len(self.indices)))

        def __len__(self):
            return len(self.indices)

    def init_ds(self, sz=-1):
        if sz == -1:
            sz = self.initial_size

        temp_loader = self.make_seq_loader(self.dataset, subset=self.labeled_set, batch_size=1, flag=True)
        img_id_to_query = [0] * len(self.labeled_set)

        print("[AL] Initializing dataset")
        for j, (samples, targets) in tqdm(enumerate(temp_loader), total=len(temp_loader)):
            img_id_to_query[j] = int(targets[0]["image_id"])


        temp_loader_val = self.make_seq_loader(self.dataset, subset=self.val_set, batch_size=1, flag=True)
        print("[AL] Getting validation set ids")
        val_set_ids = [0] * len(self.val_set)
        for j, (samples, targets) in tqdm(enumerate(temp_loader_val), total=len(temp_loader_val)):
            val_set_ids[j] = int(targets[0]["image_id"])

        return self.labeled_set, img_id_to_query, val_set_ids

    def make_seq_loader(self, dataset, subset, batch_size=2, flag=False):
        if self.model_name == "DETR":
            if flag:
                data_loader = DataLoader(dataset, batch_size, sampler=self.SubsetSequentialSampler(subset),
                                             drop_last=False, collate_fn=utils.collate_fn, num_workers=0)
            else:
                data_loader = DataLoader(dataset, batch_size, sampler=self.SubsetSequentialSampler(subset),
                                             drop_last=False, collate_fn=utils.collate_fn)
        return data_loader

    def make_train_loader(self, dataset_train, subset):
        batch_size = self.args.batch_size
        data_loader_train = DataLoader(dataset=dataset_train,
                                       sampler=SubsetRandomSampler(indices=subset),
                                       collate_fn=utils.collate_fn, num_workers=1, batch_size=batch_size)
        return data_loader_train

    def query_oracle(self, idx):
        print("[AL] in query_oracle")
        print("[AL] idx size: {}".format(len(idx)))

        self.labeled_set = list(self.labeled_set) + list(idx)
        self.pool_set = [i for i in self.pool_set if i not in idx]

        return self.labeled_set

    @torch.no_grad()
    def save_q_info(self, logger, idx, al_cycle):
        data_loader = self.make_seq_loader(self.dataset, subset=idx, batch_size=1, flag=True)

        self.model.eval()
        # iterate over the fetched queries
        total_outputs = []
        for i, (samples, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
            samples = samples.to(self.device)
            output = self.model(samples)
            total_outputs.append(output)

        logger.log_q_info(total_outputs, al_cycle, "output_preds")


    def get_random_idx(self, sz=-1, without=None):
        """
        For random sampling from the pool
        :return: indices, image ids of the indices
        """
        if sz == -1:
            sz = self.query_size

        if without != None:
            subset = [i for i in self.pool_set if i not in without]
            idx = np.random.choice(subset, size=sz, replace=False)
        else:
            idx = np.random.choice(self.pool_set, size=sz, replace=False)

        temp_loader = self.make_seq_loader(self.dataset, subset=idx, batch_size=1, flag=True)

        img_id_to_query = [0] * len(idx)

        for i, (samples, targets) in tqdm(enumerate(temp_loader), total=len(temp_loader)):
            img_id_to_query[i] = int(targets[0]["image_id"])

        print("[AL] Fetched random indices")
        print("[AL] in get_random_idx, idx size:{}".format(len(idx)))

        return idx, img_id_to_query

    def train_al(self, logger: Logger):
        # train the model with active learning
        loader_test = DataLoader(self.dataset_test, batch_size=2,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=0)
        n_labeled = 0

        # AL loop
        i = 0
        while n_labeled < self.max_to_label:
            # train on the current dataset
            if i == 0:
                _ , img_id_to_query, val_set_ids = self.init_ds()

                n_cur_sampled = len(self.labeled_set)
                self.print_dbg("[AL] Created Initial Dataset")
            else:
                # query oracle and and expand the current training set
                self.print_dbg("[AL] Running Active Learning Method")
                if self.al_method == "random":
                    print("[AL] Random sampling AL, fetching random indices from oracle")
                    idx_to_query, img_id_to_query = self.get_random_idx()
                else:
                    self.loader_pool = self.make_seq_loader(self.dataset, subset=self.pool_set, batch_size=1)
                    if self.val_is_test == "True":
                        print("[AL] YES val_is_test")
                        self.loader_val = DataLoader(self.dataset_test, batch_size=1,
                                       drop_last=False, collate_fn=utils.collate_fn, num_workers=0)
                    else:
                        print("[AL] NOT val_is_test")
                        self.loader_val = self.make_seq_loader(self.dataset, subset=self.val_set, batch_size=1)

                    al_method = self.al_method(model=self.model, criterion=self.criterion, device=self.device,
                                               query_size=self.query_size, loader_pool=self.loader_pool,
                                               loader_val=self.loader_val, pool_set=self.pool_set,
                                               num_workers=self.num_workers, args=self.args,
                                               n_classes=self.n_classes,
                                               with_noise=self.with_noise,
                                               dir_inv_var=self.dir_inv_var, cycle=i,logger=logger)

                    idx_to_query, img_id_to_query = al_method.get_idx_from_pool()
                    if len(idx_to_query) < self.query_size:
                        diff = self.query_size - len(idx_to_query)
                        print("[AL] completing the query, diff={}".format(diff))
                        idx_2, img_id_2 = self.get_random_idx(sz=diff, without=idx_to_query)
                        idx_to_query = np.concatenate((idx_to_query, idx_2)).astype(int)
                        img_id_to_query += img_id_2


                if self.args.save_preds == "True":
                    print("[AL] saving preds")
                    self.save_q_info(logger, idx_to_query, al_cycle=i)


                if self.args.save_model == "True":
                    print("[AL] saving model")
                    logger.log_q_info(self.model, i, "model.pt")

                new_labeled_set = self.query_oracle(idx_to_query)
                n_cur_sampled = len(idx_to_query)

                self.print_dbg("[AL] Added labeled data")

            # total labeled images
            n_labeled = len(self.labeled_set)
            # without validation set
            print("[AL] Labeled a total of {} images".format(len(self.labeled_set))) 
            print("[AL] training")


            print("[AL] labeled set size:", len(self.labeled_set))
            print("[AL] pool set size:", len(self.pool_set))
            print("[AL] val set size:", len(self.val_set))


            loader_train = self.make_train_loader(self.dataset, subset=self.labeled_set)

            self.print_dbg("[AL] Training on a dataset of size: {}".format(len(self.labeled_set)))

            param_dicts = [
                {"params": [p for n, p in self.model.named_parameters() if
                            "backbone" not in n and p.requires_grad]},
                {
                    "params": [p for n, p in self.model.named_parameters() if
                               "backbone" in n and p.requires_grad],
                    "lr": self.args.lr_backbone,
                },
            ]
            optimizer = torch.optim.AdamW(param_dicts, lr=self.args.lr,
                                          weight_decay=self.args.weight_decay)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.args.lr_drop)

            prev_train_loss = 1000000
            eps = 0.0009  
            patience = 2
            hits = 0
            for j in range(self.epochs_per_query):
                train_stats = self.model_train(self.model, self.criterion, loader_train, optimizer, self.device,
                                               epoch=j)
                print(train_stats)
                cur_train_loss = train_stats['loss']
                if abs(cur_train_loss - prev_train_loss) < eps:
                    hits += 1
                    if hits == patience:
                        print("[AL]: stopping early after {} epochs".format(j))
                        print("[AL]: diff=", cur_train_loss - prev_train_loss)
                        break
                    else:
                        print("[AL] hits={}".format(hits))
                        print("[AL]: diff=", cur_train_loss - prev_train_loss)

                else:
                    print("[AL] Continuing training")
                    print("[AL]: diff=", cur_train_loss - prev_train_loss)
                prev_train_loss = cur_train_loss
                lr_scheduler.step()

            self.print_dbg("[AL] Trained the model on dataset_sub for {} epochs".format(j))

            print("[AL] Trained on: {}, current pool set size: {}".format(len(self.labeled_set), len(self.pool_set)))

            print("[AL] Evaluating performance on test set")
            eval_stats = self.evaluator.wrap_evaluate(loader_test, epoch=i, train_stats=train_stats)
            logger.update(i, eval_stats, n_labeled, n_cur_sampled, img_id_to_query, j,
                          val_ids=val_set_ids, sets=self.sets)
            print("[AL] Done evaluating performance")

            i += 1

        print("[AL]: done, ")


