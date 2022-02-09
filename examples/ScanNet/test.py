# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Options
import glob
import sys
import iou
import os
import torch
import numpy as np
import math
import time
import sparseconvnet as scn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import scipy.io as sio
from config import get_args
from model import ThreeVoxelKernel,FourVoxelKernel,FiveVoxelKernel
import logging
from datasets import ScanNet
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('val logger')


def evaluate(model ,config, save_ply=True):
    SELECTED_LABEL_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]

    LABEL_ID_TO_CLASS_ID = {
        #    9: 3 #force 'desk' to be the same class_id as 'table'
    }
    for i, label_id in enumerate(SELECTED_LABEL_IDS):
        if label_id not in LABEL_ID_TO_CLASS_ID:
            LABEL_ID_TO_CLASS_ID[label_id] = i
    CLASS_ID_TO_LABEL={}
    for k,v in LABEL_ID_TO_CLASS_ID.items():
        CLASS_ID_TO_LABEL[v]=k
    # with torch.no_grad():
    #     model.eval()
    #     store = torch.zeros(config['valOffsets'][-1], 20)
    #     scn.forward_pass_multiplyAdd_count = 0
    #     scn.forward_pass_hidden_states = 0
    #     for rep in range(1, 1+config['val_reps']):
    #         for i, batch in enumerate(config['val_data_loader']):
    #             logger.info(i)
    #             batch['x'][1] = batch['x'][1].cuda()
    #             # batch['y'] = batch['y'].cuda()
    #             predictions = model(batch['x'])
    #             predictions = predictions.cpu()
    #             # store.index_add_(0, batch['point_ids'], predictions)
    #
    #             predictions = predictions.argmax(1).numpy()
    #             predictions = [CLASS_ID_TO_LABEL[i]  for i in predictions]
    #             # print(predictions.shape[0]==batch['y'].numpy().shape[0])
    #             # print(batch['pth_file'],':',predictions)
    #             dir_name=args.test_result_path
    #             basename=os.path.basename(batch['pth_file'][0])
    #             latter='_vh_clean_2_normal_test.pth'
    #             with open(os.path.join(dir_name,basename[0:-len(latter)]+'.txt'),"w") as f:
    #                 for i in range(len(predictions)):
    #                     if i==len(predictions)-1:
    #                         print(predictions[i],file=f,end='')
    #                     else:
    #                         print(predictions[i],file=f)
    #         # iou.evaluate(predLabels, config['valLabels'])
    #         # sio.savemat('predLabels.mat', {'pl':predLabels,'vl':config['valLabels'],'score':store.data.cpu().numpy()})
    with torch.no_grad():
        model.eval()
        store = torch.zeros(config['valOffsets'][-1], 20)
        scn.forward_pass_multiplyAdd_count = 0
        scn.forward_pass_hidden_states = 0
        for rep in range(1, 1+config['val_reps']):
            for i, batch in enumerate(config['val_data_loader']):
                logger.info(i)
                batch['x'][1] = batch['x'][1].cuda()
                # batch['y'] = batch['y'].cuda()
                predictions = model(batch['x'])
                predictions = predictions.cpu()
                store.index_add_(0, batch['point_ids'], predictions)
        predLabels = store.max(1)[1].numpy()
        for i, batch in enumerate(config['val_data_loader']):
            logger.info(i)
            # predictions = predictions.argmax(1).numpy()
            predictions = [CLASS_ID_TO_LABEL[predLabels[j]]  for j in batch['point_ids']]
            # print(predictions.shape[0]==batch['y'].numpy().shape[0])
            # print(batch['pth_file'],':',predictions)
            dir_name=args.test_result_path
            basename=os.path.basename(batch['pth_file'][0])
            latter='_vh_clean_2_normal_test.pth'
            with open(os.path.join(dir_name,basename[0:-len(latter)]+'.txt'),"w") as f:
                for i in range(len(predictions)):
                    if i==len(predictions)-1:
                        print(predictions[i],file=f,end='')
                    else:
                        print(predictions[i],file=f)
                # iou.evaluate(predLabels, config['valLabels'])
                # s

if __name__ == '__main__':
    args=get_args()
    config = {}
    config['m'] = args.m
    config['residual_blocks'] = args.residual_blocks
    config['block_reps'] = args.block_reps
    config['scale'] = args.scale
    config['val_reps'] = args.val_reps
    config['dimension'] = args.dimension
    config['full_scale'] = args.full_scale
    config['unet_structure'] = [args.m, 2 * args.m, 3 * args.m, 4 * args.m, 5 * args.m, 6 * args.m, 7 * args.m]
    config['kernel_size'] = args.kernel_size
    config['use_normal']=args.use_normal
    config['use_elastic']=args.use_elastic
    config['rotation_guide_level']=0

    if config['kernel_size'] == 3:
        Model = ThreeVoxelKernel
    else:
        raise NotImplementedError

    assert args.batch_size == 1
    assert hasattr(args,'test_path')
    assert args.test_path!=''
    assert args.test_result_path!=''
    if args.dataset == 'scannet':
        dataset = ScanNet(train_pth_path="./datasets/scannet/val/*.pth",
                          val_pth_path=args.test_path,
                          scale=args.scale,  # Voxel size = 1/scale
                          val_reps=args.val_reps,  # Number of test views, 1 or more
                          batch_size=args.batch_size,
                          dimension=args.dimension,
                          full_scale=args.full_scale,
                          use_normal=config['use_normal'],
                          use_elastic=config['use_elastic']
                          )
    else:
        raise NotImplementedError

    config['valOffsets'], config['train_data_loader'], config['val_data_loader'], config['valLabels'], config[
        'trainLabels'] = dataset.load_data()

    if not os.path.exists(args.test_result_path):
        os.mkdir(args.test_result_path)

    net=Model(config)
    if args.load:
        net.load_state_dict(torch.load(args.load))
        logger.info('Model loaded from {}'.format(args.load))
    else:
        logger.error("pls specify the model file")
        raise RuntimeError

    net = net.cuda()
    evaluate(net,  config=config, save_ply=True)
