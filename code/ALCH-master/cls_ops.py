import torch
import torch.nn.functional as F
from torch import nn
from util import box_ops

from tqdm import tqdm

import pandas as pd
import numpy as np
import pickle

class ClsOperator:
    def __init__(self, model, n_classes, args):
        self.model = model
        self.debug = True


        self.CLASSES_COCO = [
            'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
            'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
            'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

        self.CLASSES_PVOC = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
                             "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
                             "sheep", "sofa", "train",  "tvmonitor", "N/A"]

        # TODO: hardcoded this

        self.CLASSES_KITTI = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc", "N/A"]

        if args.dataset_file == "pvoc":
            self.CLASSES = self.CLASSES_PVOC
        elif args.dataset_file == "kitti":
            self.CLASSES = self.CLASSES_KITTI


        print("[CLS_OPS] {} classes".format(n_classes))


        self.cls_losses = torch.zeros(n_classes, 4)

    @torch.no_grad()
    def get_cls_loss(self, outputs, targets, criterion, cls_losses):
        "get per class loss"
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = criterion.matcher(outputs_without_aux, targets)

        src_idx = criterion._get_src_permutation_idx(indices)

        # order the labels by the indices
        target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # BOXES
        src_logits = outputs['pred_logits'][src_idx] # (BOXES) X C

        # order the bounding boxes by the indices
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0) # BOXES
        src_boxes = outputs['pred_boxes'][src_idx] # BOXES

        classes = torch.unique(target_classes)
        for cls in classes:
            idx = torch.where(target_classes == cls)[0]

            loss_ce = F.cross_entropy(src_logits[idx], target_classes[idx], reduction='sum')

            loss_bbox = F.l1_loss(src_boxes[idx], target_boxes[idx], reduction='sum')

            loss_giou = (1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes[idx]),
                box_ops.box_cxcywh_to_xyxy(target_boxes[idx])))).sum()

            losses = torch.tensor([len(idx), loss_ce, loss_bbox, loss_giou])
            cls_losses[cls] += losses

        return cls_losses

    def get_cls_weights(self):
        weighted_losses = self.cls_losses[:, -1]
        cls_weights = weighted_losses / weighted_losses.sum()
        if self.debug:
            print("-"*20)
            print("Class Weights: ", cls_weights)
            print("-"*20)
        return cls_weights

    def create_df(self, cls_losses):
        # csv_file = "cls_losses_{}.csv".format(self.cycle)

        df = pd.DataFrame(cls_losses.numpy())
        col_names = ["count", "loss_ce", "loss_bbox", "loss_giou", "weighted_loss"]
        mapping = {i: col_names[i] for i, _ in enumerate(col_names)}

        df = df.rename(columns=mapping)

        # df = df.reindex(columns=["cls_name"] + col_names)
        #
        # df["cls_name"] = self.CLASSES

        # df.to_csv(csv_file)
        return df

    @torch.no_grad()
    def cls_eval(self, criterion, data_loader, device):
        # evaluates the model per class
        print("[CLS_OPS] Evaluating model on the classes")

        self.model.eval()

        for i, (samples, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = self.model(samples)
            self.get_cls_loss(outputs, targets, criterion, self.cls_losses)


        # loss_ce, loss_bbox, loss_giou
        loss_weights = torch.tensor([1, 5, 2]).type(self.cls_losses.type())

        eps = torch.tensor(1e-8).type(self.cls_losses.type())
        weighted_loss = torch.matmul(self.cls_losses[:, 1: 1 + len(loss_weights)], loss_weights) / (self.cls_losses[:, 0] + eps)

        self.cls_losses = torch.cat((self.cls_losses, weighted_loss.unsqueeze(0).T), dim=1)

        df = self.create_df(self.cls_losses)
        #np.save("cls_losses.mat", cls_losses.numpy(), pickle=True)

        print(self.cls_losses)

        print("[CLS_OPS] finished evaluating class hardness")

        return self.cls_losses, df






