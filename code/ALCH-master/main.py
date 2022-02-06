# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model

from AL import ActiveLearner
from clshard_al import ClsHardnessAL
import sys

import time
from datetime import datetime

import Logger

from tests import *
import os


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Active Learning parameters
    parser.add_argument('--al_method', type=str, default="random",
                        help="The Active Learning method")
    parser.add_argument('--val_is_test', type=str, default="False",
                        help="if to estimate class hardness directly from the test set")
    # val size for class hardness
    parser.add_argument('--val_sz', default=2000, type=int)


    # logs path
    parser.add_argument("--logs_dir", default="/cs/++/usr/segal/ALCH/logs/", type=str)



    # noise parameters
    parser.add_argument('--with_noise', type=str, default="False",
                        help="with noise on the weights")
    parser.add_argument('--inv_var', type=int, default=1e10,
                        help="the noise on the class weights")


    parser.add_argument('--save_preds', type=str, default="True",
                        help="if to save prediction tensors")
    parser.add_argument('--save_model', type=str, default="True",
                        help="if to save the model after every cycle")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)




    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def load_pretrained_total(new_model):
    model_pre = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)

    new_model_params = new_model.state_dict()

    for name, param in model_pre.named_parameters():
        if "encoder" in name or "backbone" in name:
            new_model_params[name] = param
            #print("Copied:", name)

    new_model.load_state_dict(new_model_params)

    return new_model

def load_pretrained(new_model, freeze=False):
    #path_pre = "./pretrained_models/detr_pre.pth"
    model_pre = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)

    new_model_params = new_model.state_dict()

    if len(model_pre.state_dict()) != len(new_model_params):
        print("ERROR: pretrained and not-trained model don't have the same architecture")
        sys.exit()

    # copy the parameters of the encoder and the backbone

    for name, param in model_pre.named_parameters():
        if "encoder" in name or "backbone" in name:
            new_model_params[name] = param
            #print("Copied:", name)

    new_model.load_state_dict(new_model_params)
    if freeze:
        for name, param in new_model.named_parameters():
            if "encoder" in name or "backbone" in name:
                param.requires_grad = False


    print("loaded the encoder and backbone from pretrained model")


def main(args, output_dir):


    # Determinism
    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    epochs_per_query = args.epochs  # TODO: modify later if needed, maybe more, depends on the model and such
    initial_size = 1000

    pool_prop = 1.0

    if args.dataset_file == "pvoc":
        n_classes=21
        query_size = 1000
    elif args.dataset_file == "coco":
        n_classes=91
        query_size = 2500
    elif args.dataset_file == "kitti":
        query_size = 500
        n_classes=9

    print("[main] dataset={}".format(args.dataset_file))
    print("[main] query size={}".format(query_size))

    # small number - high variance, high noise, large number - low variance
    # w_inv_vars = [1e10, 1e4, 1e2, 1e1, 5, 1, 0.5]
    # dir_inv_var = w_inv_vars[0]
    dir_inv_var = args.inv_var
    with_noise = args.with_noise

    args.distributed = False  # maybe also this

    serializable = [int, str]


    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("[MAIN]: building model")
    model, criterion, postprocessors = build_model(args)


    print("[MAIN] USING MODEL PRETRAINED ON COCO")
    model = load_pretrained_total(model)

    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]


    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)



    base_ds = get_coco_api_from_dataset(dataset_val)


    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    evaluator = Evaluator(model, criterion, postprocessors, base_ds, device, output_dir)

    if args.al_method == "random":
        al_method = "random"
    elif args.al_method == "cls_hardness":
        al_method = ClsHardnessAL
    else:
        al_method = "random"


    al = ActiveLearner(model, criterion, optimizer, al_method, train_one_epoch, dataset_complete=dataset_train,
                       dataset_test=dataset_val, with_noise=with_noise, dir_inv_var=dir_inv_var, device=device, evaluator=evaluator,
                       lr_scheduler=lr_scheduler, args=args, param_dicts=param_dicts,
                       n_iters=12, n_classes=n_classes, initial_size=initial_size, query_size=query_size,
                       epochs_per_query=epochs_per_query, pool_prop=pool_prop)

    logger = Logger.Logger(al, output_dir, args)
    logger.run_data["dataset_name"] = "COCO"

    print("[MAIN] epochs/query: ", epochs_per_query)
    al.train_al(logger)


class Evaluator:
    def __init__(self, model, criterion, postprocessors, base_ds, device, output_dir):
        self.model = model
        self.criterion = criterion
        self.postprocessors = postprocessors
        self.base_ds = base_ds
        self.device = device
        self.output_dir = output_dir

    def wrap_evaluate(self, data_loader_val, epoch, train_stats):
        test_stats, coco_evaluator = evaluate(
            self.model, self.criterion, self.postprocessors, data_loader_val, self.base_ds, self.device, self.output_dir
        )
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch}

        if self.output_dir and utils.is_main_process():
            with (self.output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (self.output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   self.output_dir / "eval" / name)

        print("[EVALUATOR] Done evaluating")
        return log_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    print("[MAIN] AL method: ", args.al_method)
    
    base_path = args.logs_dir

    date_str = datetime.now().strftime("%d.%m_%H:%M")

    if args.val_is_test == "True":
        log_str = "{}{}_{}_{}_log".format(base_path, args.al_method, "val_is_test", date_str)
    else:
        log_str = "{}{}_{}_log".format(base_path, args.al_method, date_str)

    output_dir = Path(log_str)
    print("[MAIN] log_str:", log_str)

    # for our logging
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    main(args, output_dir)
