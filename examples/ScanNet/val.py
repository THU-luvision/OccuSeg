from .datasets import ScanNet
from .utils import evaluate_scannet, evaluate_stanford3D,WeightedCrossEntropyLoss, FocalLoss
from .model import ThreeVoxelKernel
from .config import get_args
from .config import ArgsToConfig

import sparseconvnet as scn

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter

import sys, os, time
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('training logger')
logger.setLevel(logging.DEBUG)
Model = ThreeVoxelKernel


def evaluate(net, config, global_iter):
    valOffsets = config['valOffsets']
    val_data_loader = config['val_data_loader']
    valLabels = config['valLabels']
    with torch.no_grad():
        # net.eval()
        store = torch.zeros(valOffsets[-1], config['class_num'])
        scn.forward_pass_multiplyAdd_count = 0
        scn.forward_pass_hidden_states = 0
        time_val_start = time.time()
        for rep in range(1, 1 + dataset.val_reps):
            locs = None
            pth_files = []
            for i, batch in enumerate(val_data_loader):
                batch['x'][1] = batch['x'][1].cuda()
                batch['y'] = batch['y'].cuda()
                predictions = net(batch['x'])
                predictions = predictions.cpu()
                store.index_add_(0, batch['point_ids'], predictions)
                # loop for all val set every snap shot is tooooo slow, pls use val.py to check individually
                # if save_ply:
                #     print("evaluate data: ", i)
                #     iou.visualize_label(batch, predictions, rep, save_dir=config['checkpoints_dir'] +)
            predLabels = store.max(1)[1].numpy()
            topic = 'valid_' + str(rep)
            iou_evaluate(predLabels, valLabels, train_writer, global_iter, topic=topic)
            train_writer.add_scalar("valid/MegaMulAdd", scn.forward_pass_multiplyAdd_count / len(dataset.val) / 1e6,
                                    global_step=global_iter)
            train_writer.add_scalar("valid/MegaHidden", scn.forward_pass_hidden_states / len(dataset.val) / 1e6,
                                    global_step=global_iter)

            print('infer', rep, 'Val MegaMulAdd=', scn.forward_pass_multiplyAdd_count / len(dataset.val) / 1e6,
                        'MegaHidden', scn.forward_pass_hidden_states / len(dataset.val) / 1e6, 'time=',
                        time.time() - time_val_start,
                        's')

        predLabels = store.max(1)[1].numpy()
        iou_evaluate(predLabels, valLabels, train_writer, global_iter, topic='valid')
        train_writer.add_scalar("valid/time", time.time() - time_val_start, global_step=global_iter)


def train_net(net, config):
    if config['optim'] == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=config['lr'])
    elif config['optim'] == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=config['lr'])

    if config['loss'] == 'cross_entropy':
        criterion = nn.functional.cross_entropy
    elif config['loss'] == 'focal':
        criterion = FocalLoss()
    # elif config['loss'] == 'weighted_cross_entropy':
    #     criterion = WeightedCrossEntropyLoss(weight)
    else:
        raise NotImplementedError

    for epoch in range(config['checkpoint'], config['max_epoch']):
        net.train()
        stats = {}
        scn.forward_pass_multiplyAdd_count = 0
        scn.forward_pass_hidden_states = 0
        start = time.time()
        train_loss = 0
        epoch_len = len(config['train_data_loader'])

        pLabel = []
        tLabel = []
        for i, batch in enumerate(config['train_data_loader']):
            # checked
            # logger.debug("CHECK RANDOM SEED(torch seed): sample id {}".format(batch['id']))
            optimizer.zero_grad()
            batch['x'][1] = batch['x'][1].cuda()
            batch['y'] = batch['y'].cuda()
            p1 = net(batch['x'])
            loss = criterion(p1, batch['y'])
            train_writer.add_scalar("train/loss", loss.item(), global_step=epoch_len * epoch + i)    
            predict_label = p1.cpu().max(1)[1].numpy()
            true_label = batch['y']            
            pLabel.append(torch.from_numpy(predict_label))
            tLabel.append((true_label))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        # add some log item
        
        pLabel = torch.cat(pLabel,0).cpu().numpy()
        tLabel = torch.cat(tLabel,0).cpu().numpy()
        mIOU = iou_evaluate(pLabel, tLabel, train_writer ,epoch, 'train')
        train_writer.add_scalar("train/epoch_avg_loss", train_loss / epoch_len, global_step=epoch_len * (epoch + 1))
        train_writer.add_scalar("train/MegaMulAdd", scn.forward_pass_multiplyAdd_count / len(dataset.train) / 1e6,
                                global_step=epoch_len * (epoch + 1))
        train_writer.add_scalar("train/MegaHidden", scn.forward_pass_hidden_states / len(dataset.train) / 1e6,
                                global_step=epoch_len * (epoch + 1))
        train_writer.add_scalar("train/time", time.time() - start, global_step=epoch_len * (epoch + 1))
        train_writer.add_scalar("train/lr", config['lr'], global_step=epoch_len * (epoch + 1))
        print(epoch, 'Train loss', train_loss / (i + 1),'/  ',mIOU, 'MegaMulAdd=',
                    scn.forward_pass_multiplyAdd_count / len(dataset.train) /
                    1e6, 'MegaHidden', scn.forward_pass_hidden_states / len(dataset.train) / 1e6, 'time=',
                    time.time() - start, 's')

        # evaluate every config['snapshot'] epoch, and save model at the same time.
        if ((epoch + 1) % config['snapshot'] == 0) or (epoch in [0,1,2,4,8]):
            # net.eval()
            evaluate(net=net, config=config, global_iter=epoch_len * (epoch + 1))
            torch.save(net.state_dict(), config['checkpoints_dir'] + 'Epoch{}.pth'.format(epoch + 1))

        if config['gamma'] != 0 and (epoch + 1) % config['step_size'] == 0:
            config['lr'] = config['lr'] * config['gamma']
            if config['optim'] == 'SGD':
                optimizer = optim.SGD(net.parameters(), lr=config['lr'] * config['gamma'], momentum=0.9,
                                      weight_decay=0.0005)
            elif config['optim'] == 'Adam':
                optimizer = optim.Adam(net.parameters(), lr=config['lr'], weight_decay=0.00005)

        # if scn.is_power2(epoch) or (epoch % eval_epoch == 0 if eval_epoch else False) or epoch == training_epochs:
        #         evaluate(unet,valOffsets,val_data_loader,valLabels,save_ply=(eval_save_ply and scn.is_power2(epoch)),
        #                  prefix="epoch_{epoch}_".format(epoch=epoch))


if __name__ == '__main__':
    torch.manual_seed(100)  # cpu
    torch.cuda.manual_seed(100)  # gpu
    np.random.seed(100)  # numpy
    torch.backends.cudnn.deterministic = True  # cudnn

    # setup config
    args = get_args()
    config = ArgsToConfig(args)

    # choose kernel size
    if config['kernel_size'] == 3:
        Model = ThreeVoxelKernel
    else:
        raise NotImplementedError

    train_writer = SummaryWriter(comment=args.taskname)

    if os.path.exists(config['checkpoints_dir']) is False:
        os.makedirs(config['checkpoints_dir'])

    # choose dataset
    if args.dataset == 'scannet':
        if args.all_to_train == True:
            train_dataset_mid_dir = 'all'
        else:
            train_dataset_mid_dir = 'train'
        # use uniform data
        # if args.use_normal == True:
        #     pth_reg_exp = '*_2_normal_depth.pth'
        # else:
        #     pth_reg_exp = '*_2.pth'
        pth_reg_exp = '*_sl50.pth'
        train_pth_path='./datasets/scannet/{}/{}'.format(train_dataset_mid_dir, pth_reg_exp)
        val_pth_path='./datasets/scannet/val/{}'.format(pth_reg_exp)
        if config['simple_train'] == True:
            train_pth_path = './datasets/scannet/simple/*_sl50.pth'
            val_pth_path = './datasets/scannet/simple/*_sl50.pth'
        dataset = ScanNet(train_pth_path=train_pth_path,
                          val_pth_path=val_pth_path,
                          scale=args.scale,  # Voxel size = 1/scale
                          val_reps=args.val_reps,  # Number of test views, 1 or more
                          batch_size=args.batch_size,
                          dimension=args.dimension,
                          full_scale=args.full_scale,
                          use_normal=config['use_normal'],
                          use_elastic=config['use_elastic'],
                          use_feature=config['use_feature'],
                          use_rotation_noise = config['use_rotation_noise']
                          )
        config['class_num'] = 20
        iou_evaluate = evaluate_scannet
    elif args.dataset == 'stanford3d':

        config['class_num'] = 14
        iou_evaluate = evaluate_stanford3D
    else:
        raise NotImplementedError

    # log the config to tensorboard
    tmp_config_str = ''
    for k, v in config.items():
        if isinstance(v, str) or isinstance(v, int) or isinstance(v, float):
            train_writer.add_text('config', '{:<8}   :    {:>8}\n'.format(k, v), global_step=0)
        else:
            train_writer.add_text('config', '{:<8}   :    {:>8}\n'.format(k, json.dumps(v)), global_step=0)

    config['valOffsets'], \
    config['train_data_loader'], \
    config['val_data_loader'], \
    config['valLabels'] = dataset.load_data()

    net = Model(config)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    # pls use GPU to train
    assert torch.cuda.is_available()
    net.cuda()

    for key in config.keys():
        print(key,[config[key]])

    try:
        evaluate(net=net, config=config,global_iter=0)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logger.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
