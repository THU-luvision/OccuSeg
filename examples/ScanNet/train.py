from datasets import ScanNet
from utils import evaluate_scannet, evaluate_stanford3D,WeightedCrossEntropyLoss, FocalLoss, label2color,evaluate_single_scan,cost2color
from model import ThreeVoxelKernel
from model import DenseUNet
from config import get_args
from config import ArgsToConfig
from scipy.io import savemat
import pdb
import sys
#sys.path.insert(0, '../../extra/KNN_CUDA/')
#from knn_cuda import KNN
import sparseconvnet as scn

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter

import open3d
import sys, os, time
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('training logger')
logger.setLevel(logging.DEBUG)
Model = ThreeVoxelKernel


def custom_draw_geometry_with_key_callback(pcd_ori_color, pcd_gt_color, pcd_predict_color):
    vis = open3d.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd_ori_color)
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()
    vis.destroy_window()

#    open3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)



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
                predictions, feature = net(batch['x'])
                predictions = predictions.cpu()
                store.index_add_(0, batch['point_ids'], predictions)
                #visualize based on predictions, use open3D for convinence
                if config['evaluate']:
                    tbl = batch['id']
                    for k,idx in enumerate(tbl):
                        index = batch['x'][0][:,3] == k
                        pos = batch['x'][0][index,0:3].data.cpu().numpy()
                        color = batch['x'][1][index,0:3].data.cpu().numpy() + 0.5
                        label = batch['y'][index].data.cpu().numpy()
                        if(label.shape[0] > 0):
                            predicted_label = predictions[index,:].max(1)[1]
                            prob = predictions[index,:]
                            cost_color = cost2color(prob,label)
                            predicted_colors = label2color(predicted_label)
                            label_colors = label2color(label)
                            ref = torch.from_numpy(pos).float().cuda()
                            pcd_ori = open3d.geometry.PointCloud()
                            pcd_ori.points = open3d.Vector3dVector(pos)
                            pcd_ori.colors = open3d.Vector3dVector(cost_color)
                            pcd_gt_label = open3d.geometry.PointCloud()
                            gt_pos = pos
                            gt_pos[:,0] = pos[:,0] + 400
                            pcd_gt_label.points = open3d.Vector3dVector(gt_pos)
                            pcd_gt_label.colors = open3d.Vector3dVector(label_colors)
                            pcd_predict_label = open3d.geometry.PointCloud()
                            predict_pos = gt_pos
                            predict_pos[:,0] = predict_pos[:,0] + 400
                            pcd_predict_label.points = open3d.Vector3dVector(predict_pos)
                            pcd_predict_label.colors = open3d.Vector3dVector(predicted_colors)
                            current_iou = iou_evaluate(predicted_label.numpy(), label, train_writer, global_iter, topic='valid')
                            #predictions
                            if config['dataset'] == 'stanford3d':
                                fileName = batch['pth_file'][k][0:batch['pth_file'][k].rfind('/')] + '_' + "%.2f" % current_iou
                            elif config['dataset'] == 'scannet':
                                fileName = batch['pth_file'][k][0:-4] + '_' + "%.2f" % current_iou
                            fori = fileName + '_ori.pcd'
                            fgt = fileName + '_gt.pcd'
                            fpredict = fileName + '_predict.pcd'
                            open3d.write_point_cloud(fori, pcd_ori)
                            open3d.write_point_cloud(fgt, pcd_gt_label)
                            open3d.write_point_cloud(fpredict, pcd_predict_label)

#                            open3d.visualization.draw_geometries([pcd_ori, pcd_gt_label, pcd_predict_label])

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
            if config['evaluate']:
                savemat('pred.mat', {'p':store.numpy(),'val_offsets':np.hstack(valOffsets), 'v': valLabels})

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
    elif config['loss'] == 'weighted_cross_entropy':
        if config['dataset'] == 'stanford3d':
            weight = torch.from_numpy(np.hstack([0.1861, 0.1586,0.2663,0.0199,0.0039,0.0210,0.0210,0.0575,0.0332,0.0458,0.0052,0.0495 ,0.0123,0.1164,0.0032]))
        elif config['dataset'] == 'scannet':
            weight = torch.from_numpy(np.hstack([0.3005,0.2700,0.0418,0.0275,0.0810,0.0254,0.0462,0.0418,0.0297,0.0277,0.0061,0.0065,0.0194,0.0150,0.0060,0.0036,0.0029,0.0025,0.0029,0.0434]))
        weight = weight.cuda().float()
        criterion = WeightedCrossEntropyLoss(weight)
    else:
        raise NotImplementedError

    for epoch in range(config['checkpoint'], config['max_epoch']):
        net.train()
        stats = {}
        scn.forward_pass_multiplyAdd_count = 0
        scn.forward_pass_hidden_states = 0
        start = time.time()
        train_loss = 0
        consistent_loss_average = 0
        epoch_len = len(config['train_data_loader'])

        pLabel = []
        tLabel = []
        for i, batch in enumerate(config['train_data_loader']):
            # checked
            # logger.debug("CHECK RANDOM SEED(torch seed): sample id {}".format(batch['id']))

            optimizer.zero_grad()
            batch['x'][1] = batch['x'][1].cuda()
            batch['y'] = batch['y'].cuda()
#            print(batch['pth_file'])
#            print(batch['x'][0].shape)
#            print('before backward ', batch['pth_file'], batch['x'][0].shape)
            p1, feature = net(batch['x'])

            tbl = batch['id']
            point_ids = torch.arange(0,batch['x'][0].shape[0])
            random_samples = []
            for count,idx in enumerate(tbl):
                index = (batch['x'][0][:,config['dimension']] == count)
                current_point_ids = point_ids[index]
                samples = np.random.permutation(current_point_ids.data.cpu().numpy())
                random_samples.append(torch.from_numpy(samples))
            random_samples = torch.cat(random_samples,dim=0)
            consistent_feature = torch.cat([feature,feature[random_samples,:]],dim=1)
            consistent_label = (batch['y']  == batch['y'][random_samples]).long()
#            consistent_loss = torch.nn.functional.cross_entropy(net.linear_regularize(consistent_feature), consistent_label)
            consistent_loss = torch.nn.functional.cross_entropy(net.similarity(feature,feature[random_samples,:]), consistent_label)

            loss = criterion(p1, batch['y']) + consistent_loss
            train_writer.add_scalar("train/loss", loss.item(), global_step=epoch_len * epoch + i)
            predict_label = p1.cpu().max(1)[1].numpy()
            true_label = batch['y']
            pLabel.append(torch.from_numpy(predict_label))
            tLabel.append((true_label.detach()))
            train_loss += loss.item()
            consistent_loss_average += consistent_loss.item()

            loss.backward()
#            print('after backward')

            optimizer.step()

        # for memory efficient purpose:
#        del p1
#        del loss
#        del true_label
#        torch.cuda.empty_cache()
        pLabel = torch.cat(pLabel,0).cpu().numpy()
        tLabel = torch.cat(tLabel,0).cpu().numpy()
        mIOU = iou_evaluate(pLabel, tLabel, train_writer ,epoch, 'train')
        train_writer.add_scalar("train/epoch_avg_consistent_loss", (consistent_loss_average) / epoch_len, global_step=epoch_len * (epoch + 1))
        train_writer.add_scalar("train/epoch_avg_class_loss", (train_loss - consistent_loss_average) / epoch_len, global_step=epoch_len * (epoch + 1))
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
        if ((epoch + 1) % config['snapshot'] == 0) or (epoch in [1,4]):
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

    if config['use_dense_model']:
        Model = DenseUNet

    train_writer = SummaryWriter(comment=args.taskname)

    if os.path.exists(config['checkpoints_dir']) is False:
        os.makedirs(config['checkpoints_dir'])


    # choose dataset
    if args.dataset == 'scannet':
        config['class_num'] = 20
        iou_evaluate = evaluate_scannet
        dataset_dir = 'scannet_data'
        pth_reg_exp = '*_sl50n40.pth'
    elif args.dataset == 'stanford3d':
        config['class_num'] = 14
        iou_evaluate = evaluate_stanford3D
        dataset_dir = 'stanford_data'
        pth_reg_exp = '*/scan_normal_s50.pth'
    else:
        raise NotImplementedError


    if True:
        if args.all_to_train == True:
            train_dataset_mid_dir = 'full_train'
            test_dataset_mid_dir = 'full_val'
        else:
            train_dataset_mid_dir = 'train'
            test_dataset_mid_dir = 'val'
        if args.evaluate:
            train_dataset_mid_dir = 'simple'
            test_dataset_mid_dir = 'val'
        train_pth_path='datasets/{}/{}/{}'.format(dataset_dir, train_dataset_mid_dir, pth_reg_exp)
        val_pth_path='datasets/{}/{}/{}'.format(dataset_dir, test_dataset_mid_dir, pth_reg_exp)
        if config['simple_train'] == True:
            train_pth_path = 'datasets/' + dataset_dir + '/simple/' + pth_reg_exp
            val_pth_path = 'datasets/' + dataset_dir  + '/simple/' + pth_reg_exp
        dataset = ScanNet(train_pth_path=train_pth_path,
                          val_pth_path=val_pth_path,
                          config = config
                          )


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
    print('#classifer parameters', sum([x.nelement() for x in net.parameters()]))
    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    # pls use GPU to train
    assert torch.cuda.is_available()
    net.cuda()

    for key in config.keys():
        print(key,[config[key]])

    try:

        if(args.evaluate):
            evaluate(net=net, config=config,global_iter=0)
        else:
            train_net(net=net, config=config)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logger.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
