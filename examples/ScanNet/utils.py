# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch, numpy as np
import plyfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from tensorboardX import SummaryWriter
import pdb

def get_ori_label():
        ori_label = [1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]
        return ori_label


# color palette for nyu40 labels
def create_color_palette():
        return [
             (0, 0, 0),
             (174, 199, 232),     # wall
             (152, 223, 138),     # floor
             (31, 119, 180),      # cabinet
             (255, 187, 120),     # bed
             (188, 189, 34),      # chair
             (140, 86, 75),       # sofa
             (255, 152, 150),     # table
             (214, 39, 40),       # door
             (197, 176, 213),     # window
             (148, 103, 189),     # bookshelf
             (196, 156, 148),     # picture
             (23, 190, 207),      # counter
             (178, 76, 76),
             (247, 182, 210),     # desk
             (66, 188, 102),
             (219, 219, 141),     # curtain
             (140, 57, 197),
             (202, 185, 52),
             (51, 176, 203),
             (200, 54, 131),
             (92, 193, 61),
             (78, 71, 183),
             (172, 114, 82),
             (255, 127, 14),      # refrigerator
             (91, 163, 138),
             (153, 98, 156),
             (140, 153, 101),
             (158, 218, 229),     # shower curtain
             (100, 125, 154),
             (178, 127, 135),
             (120, 185, 128),
             (146, 111, 194),
             (44, 160, 44),       # toilet
             (112, 128, 144),     # sink
             (96, 207, 209),
             (227, 119, 194),     # bathtub
             (213, 92, 176),
             (94, 106, 211),
             (82, 84, 163),       # otherfurn
             (100, 85, 144)
        ]

def to_origianl_label(labels):
    labelMapping = get_ori_label()
    oriLabel = np.zeros([labels.shape[0]], dtype=np.int32)
    for i in range(labels.shape[0]):
        label = labels[i]
        if label >= 0:
            oriLabel[i] = labelMapping[label]
    return oriLabel


def label2color(labels):
    oriLabel = get_ori_label()
    color_palette = create_color_palette()
    color = np.zeros([labels.shape[0],3])
    for i in range(labels.shape[0]):
        label = labels[i]
        if label >= 0:
            oL = oriLabel[label]
            color[i,0] = color_palette[oL][0]
            color[i,1] = color_palette[oL][1]
            color[i,2] = color_palette[oL][2]
        else:
            color[i,0] = 0
            color[i,1] = 0
            color[i,2] = 0
    color = color / 255
    return color

def visualize_label(batch,predictions,rep):
        pred_ids = predictions.max(1)[1]
        index_list = batch['idxs']
        index_list = torch.cat(index_list,0)
        [locs, feats, normals] = batch['x']
        fns = batch['pth_file']
        fn2s = []
        fn3s = []
        plyFiles = []
        for fn in fns:
                fn2 = fn[:-11]+'.ply'
                fn3 = fn[:-3]+'labels.predict.ply'
                print(fn2, fn3)
                a = plyfile.PlyData().read(fn2)
                fn2s.append(fn2)
                fn3s.append(fn3)
                plyFiles.append(a)
        oriLabel = get_ori_label()
        color_palette = create_color_palette()

        new_pos = np.cumsum(index_list) - 1
        point_start_cnt = 0
        for point_cloud_id,plyFile in enumerate(plyFiles):
                point_num = len(plyFile.elements[0]['red'])
                for i in range(point_num):
                        pred_id = pred_ids[new_pos[point_start_cnt + i]]
                        if(pred_id >= 0):
                                w=oriLabel[pred_id]
                                r = color_palette[w][0]
                                g = color_palette[w][1]
                                b = color_palette[w][2]
                        else:
                                r = 0
                                g = 0
                                b = 0
                        plyFile.elements[0]['red'][i]   = r
                        plyFile.elements[0]['green'][i] = g
                        plyFile.elements[0]['blue'][i]  = b

                point_start_cnt = point_start_cnt + point_num
                print("write file: ", fn3s[point_cloud_id])
                plyFile.write(fn3s[point_cloud_id])

SELECTED_LABEL_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]

#Predictions will all be in the set {0,1,...,19}
VALID_CLASS_IDS = range(0, len(SELECTED_LABEL_IDS))

#label id to label name mapping: http://dovahkiin.stanford.edu/scannet-public/v1/tasks/scannet-labels.combined.tsv
LABEL_ID_TO_LABEL_NAME = {
        1:  'wall',
        2:  'chair',
        3:  'floor',
        4:  'table',
        5:  'door',
        6:  'couch',
        7:  'cabinet',
        8:  'shelf',
        9:  'desk',
        10: 'office chair',
        11: 'bed',
        12: 'trashcan',
        13: 'pillow',
        14: 'sink',
        15: 'picture',
        16: 'window',
        17: 'toilet',
        18: 'bookshelf',
        19: 'monitor',
        20: 'computer',
        21: 'curtain',
        22: 'book',
        23: 'armchair',
        24: 'coffee table',
        25: 'drawer',
        26: 'box',
        27: 'refrigerator',
        28: 'lamp',
        29: 'kitchen cabinet',
        30: 'dining chair',
        31: 'towel',
        32: 'clothes',
        33: 'tv',
        34: 'nightstand',
        35: 'counter',
        36: 'dresser',
        37: 'countertop',
        38: 'stool',
        39: 'cushion',
}
METRIC_ID_TO_NAME = {0: 'iou',
                     1: 'tp',
                     2: 'denom',
                     3: 'fp',
                     4: 'fn',
                     }
#Classes relabelled
"""
CLASS_LABELS = []
for i, x in enumerate(SELECTED_LABEL_IDS):
        # print(i, LABEL_ID_TO_LABEL_NAME[x])
        CLASS_LABELS.append(LABEL_ID_TO_LABEL_NAME[x])
"""
CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
INSTANCE_LABELS = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']


def confusion_matrix(pred_ids, gt_ids, class_num):
        assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
        idxs= gt_ids>=0
        return np.bincount(pred_ids[idxs]*class_num+gt_ids[idxs],minlength=class_num*class_num).reshape((class_num,class_num)).astype(np.ulonglong)

def get_iou(label_id, confusion):
        # true positives
        tp = np.longlong(confusion[label_id, label_id])
        # false negatives
        fp = np.longlong(confusion[label_id, :].sum()) - tp
        fn = np.longlong(confusion[:, label_id].sum()) - tp
        denom = (tp + fp + fn)
        if denom == 0:  
                return False
        return (float(tp) / denom, tp, denom, fp, fn)

def evaluate_scannet(pred_ids,gt_ids,train_writer,iter_id,class_num,topic = 'valid'):
    print('evaluating', gt_ids.size, 'points...')
    confusion=confusion_matrix(pred_ids,gt_ids,class_num)
    class_ious = {}
    for i in range(len(VALID_CLASS_IDS)):
            label_name = CLASS_LABELS[i]
            label_id = VALID_CLASS_IDS[i]
            class_iou = get_iou(label_id, confusion)
            if class_iou is not False:
                    class_ious[label_name] = get_iou(label_id, confusion)

    sum_iou = 0
    for label_name in class_ious:
            sum_iou+=class_ious[label_name][0]
    mean_iou = sum_iou/len(class_ious)

    if topic == 'valid':
        print('classes          IoU         tp  denom   fp  fn')
        print('----------------------------')
        for i in range(len(VALID_CLASS_IDS)):
                label_name = CLASS_LABELS[i]
                if label_name in class_ious:
                    print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d}/{4:>6d}/{5:<6d})'.format(
                        label_name,
                        class_ious[label_name][0],
                        class_ious[label_name][1],
                        class_ious[label_name][2],
                        class_ious[label_name][3],
                        class_ious[label_name][4]))
                else:
                    print('{0:<14s}: {1}'.format(label_name, 'missing'))
                for cate_id in range(5):
                    if(label_name in class_ious):
                        train_writer.add_scalar("{}/category/{}/{}".format(topic,label_name.replace(' ', '_'),
                                                                           METRIC_ID_TO_NAME[cate_id]),
                                                class_ious[label_name][cate_id], global_step=iter_id)

        print('mean IOU', mean_iou)


    train_writer.add_scalar(topic+"/overall_iou", mean_iou, iter_id)
    return mean_iou

def evaluate_single_scan(pred_ids,gt_ids,train_writer,iter_id,class_num,topic = 'valid'):
    confusion=confusion_matrix(pred_ids,gt_ids,class_num)
    class_ious = {}
    for i in range(len(VALID_CLASS_IDS)):
            label_name = CLASS_LABELS[i]
            label_id = VALID_CLASS_IDS[i]
            class_iou = get_iou(label_id, confusion)
            if class_iou is not False:
                    class_ious[label_name] = get_iou(label_id, confusion)

    sum_iou = 0
    count = 0

    for label_name in class_ious:
        if class_ious[label_name][0] > 0.01:
            sum_iou+=class_ious[label_name][0]
            count = count + 1
    mean_iou = sum_iou/count



    if topic == 'valid':
        print('classes          IoU         tp  denom   fp  fn')
        print('----------------------------')
        for i in range(len(VALID_CLASS_IDS)):
                label_name = CLASS_LABELS[i]
                if label_name in class_ious:
                    print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d}/{4:>6d}/{5:<6d})'.format(
                        label_name,
                        class_ious[label_name][0],
                        class_ious[label_name][1],
                        class_ious[label_name][2],
                        class_ious[label_name][3],
                        class_ious[label_name][4]))
                else:
                    print('{0:<14s}: {1}'.format(label_name, 'missing'))
        print('mean IOU', mean_iou)

    if train_writer is not None:
        if not topic == 'valid':
            for i in range(len(VALID_CLASS_IDS)):
                label_name = CLASS_LABELS[i]
                if label_name in class_ious:
                    for cate_id in range(5):
                        if train_writer is not None:
                            train_writer.add_scalar("{}/category/{}/{}".format(topic,label_name.replace(' ', '_'),
                                                                               METRIC_ID_TO_NAME[cate_id]),
                                                    class_ious[label_name][cate_id], global_step=iter_id)
        train_writer.add_scalar(topic+"/overall_iou", mean_iou, iter_id)
    return mean_iou



class stanford_params:
    def __init__(self):
        self.class_freq = np.asarray([19.203, 16.566, 27.329,
                                        2.428, 2.132, 2.123, 5.494, 3.25,
                                        4.079, 0.488, 4.726, 1.264, 10.918, 100.0])
        self.class_weights = -np.log(self.class_freq / 100.0)
        self.num_classes = len(self.class_freq)
        self.color_map = [  [128, 128, 128], # ceiling (red)
                            [124, 152, 0], # floor (green)
                            [255, 225, 25], # walls (yellow)
                            [0,   130, 200], # beam (blue)
                            [245, 130,  48], # column (orange)
                            [145,  30, 180], # window (purple)
                            [0, 130, 200], # door (cyan)
                            [0, 0, 128], # table (black)
                            [128, 0, 0], # chair (maroon)
                            [250, 190, 190], # sofa (pink)
                            [170, 110, 40], # bookcase (teal)
                            [0, 0, 0], # board (navy)
                            [170, 110,  40], # clutter (brown)
                            [128, 128, 128]] # stairs (grey)
        self.class_name = ['ceiling','floor','walls','beam','column','window','door','table','chair','sofa','bookcase','board','clutter','stairs']


def evaluate_stanford3D(pred_ids,gt_ids,train_writer,iter_id, class_num = 20,topic = 'valid'):
        print('evaluating', gt_ids.size, 'points...')
        confusion=confusion_matrix(pred_ids,gt_ids,class_num)
        class_ious = {}
        dataset = stanford_params()
        num_classes = dataset.num_classes
        for i in range(num_classes):
                label_name = dataset.class_name[i]
                label_id = i
                tp = np.longlong(confusion[label_id, label_id])
                fp = np.longlong(confusion[label_id, :].sum()) - tp
                not_ignored = [l for l in range(num_classes) if not l == label_id]
                fn = np.longlong(confusion[not_ignored, label_id].sum())
                denom = (tp + fp + fn)
                if denom > 0 and (tp + fn) > 0:
                    class_ious[label_name] = (float(tp) / denom, tp, denom, fp, fn)

        sum_iou = 0
        for label_name in class_ious:
                sum_iou+=class_ious[label_name][0]
        mean_iou = sum_iou/len(class_ious)

        print('classes          IoU          fp          fn')
        print('----------------------------')
        for i in range(num_classes):
                label_name = dataset.class_name[i]
                if label_name in class_ious:
                        print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d}/{4:>6d}/{5:<6d})'.format(label_name, class_ious[label_name][0], class_ious[label_name][1], class_ious[label_name][2],class_ious[label_name][3],class_ious[label_name][4]))
                else:
                        print('{0:<14s}: {1}'.format(label_name, 'missing'))
        print('mean IOU', mean_iou)

        train_writer.add_scalar(topic+"/overall_iou", mean_iou, iter_id)
        return mean_iou


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.weight = [0.3005,0.2700,0.0418,0.0275,0.0810,0.0254,0.0462,0.0418,0.0297,0.0277,0.0061,0.0065,0.0194,0.0150,0.0060,0.0036,0.0029,0.0025,0.0029,0.0434]
    def forward(self, p, target):
        idx = target >= 0
        p1 = p[idx]
        target = target[idx]
        prob = torch.exp(p1)
        pt = torch.gather(prob, 1, target.view(-1, 1))
        pt = torch.div(pt.view(-1, 1), torch.sum(prob, 1).view(-1, 1))
        modulator = (1 - pt) ** 2
        loss = -(modulator * torch.log(pt)).mean()
        return loss


def cost2color(prob,target):
    target = torch.from_numpy(target)
    idx = target >= 0
    idx = torch.from_numpy(np.array(idx, dtype=np.uint8))
    p1 = prob[idx]
    target = target[idx]
    x_class = torch.gather(p1, 1, target.view(-1, 1))
    loss = -x_class + torch.logsumexp(p1, 1).view(-1, 1)
    vmin = None
    vmax = None
    viridis = cm.get_cmap('viridis', 512)
    norm = plt.Normalize(vmin, vmax)
    colors = viridis(norm(loss.cpu().data.numpy()))
    return colors[:,0,0:3]


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self,weight):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight
    def forward(self, p, target):
        # weight is independent to traing, backward propogation
        idx = target >= 0
        p1 = p[idx]
        target = target[idx]
        with torch.no_grad():
            predicted = p1.max(1)[1]
            pw = torch.gather(self.weight.view(-1, 1), 0, predicted.view(-1, 1))
            pt = torch.gather(self.weight.view(-1, 1), 0, target.view(-1, 1))
            weight = 1 / (0.01 + torch.min(pw, pt))
            weight = weight.detach()
        x_class = torch.gather(p1, 1, target.view(-1, 1))
        loss = -x_class + torch.logsumexp(p1, 1).view(-1, 1)
        loss = weight * loss

        return loss.mean()

