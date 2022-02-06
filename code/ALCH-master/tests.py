import torch
from tqdm import tqdm

import util.misc as utils
import random

from torch.utils.data import DataLoader, DistributedSampler, Subset, ConcatDataset

def check_joint(dataset1, dataset2):
    sampler1 = torch.utils.data.SequentialSampler(dataset1)
    sampler2 = torch.utils.data.SequentialSampler(dataset2)
    bs = 1
    dl1 = DataLoader(dataset1, batch_size=bs, sampler=sampler1,
                                   collate_fn=utils.collate_fn,
                                   drop_last=False)
    dl2 = DataLoader(dataset2, batch_size=bs, sampler=sampler2,
                                 collate_fn=utils.collate_fn,
                                 drop_last=False)


    for i, (samples_1, targets_1) in tqdm(enumerate(dl1), total=len(dl1)):
        for j, (samples_2, targets_2) in (enumerate(dl2)):
            if torch.equal(samples_1.tensors[0], samples_2.tensors[0]):
                print("[!] duplicate images")
                return False

    return True





