
import open3d
from config import get_args
from config import ArgsToConfig
from scipy.io import savemat
import pdb
import sys

from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
import scipy.stats as stats
import scipy
from discriminative import DiscriminativeLoss
#from knn_cuda import KNN
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
from utils import evaluate_single_scan,evaluate_scannet,to_origianl_label
import sys, os, time
import json
import matplotlib.cm as cm
import math
import numpy as np
import operator
import torch
from torch import exp, sqrt
import multiprocessing as mp
import point_cloud_utils as pcu
from torch_scatter import scatter_mean,scatter_std,scatter_add,scatter_max

#label id to label name mapping: http://dovahkiin.stanford.edu/scannet-public/v1/tasks/scannet-labels.combined.tsv
CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
VALID_CLASS_IDS = np.array([1,2,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
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
config = {}
config['bw_relax_factor'] = 1.0
config['visualize'] = False
config['visualize_part'] = False
config['occupancy_ratio_threshold'] = 0.3
config['occupancy_ratio_threshold_up'] = 3
config['use_merge'] = False
semantic_embedding_len = 64

print(config)

def weighted_mean(embedding, weight):
    expanded_weight = weight.view(-1,1).expand(-1,embedding.shape[1])
    return (embedding * expanded_weight).sum(0)  / expanded_weight.sum(0)

def gaussian(d, bw): return exp(-0.5*((d/bw))**2) / (bw * math.sqrt(2*math.pi))

def gaussian_weighted_region_mean(embedding, region):
    [region_index, region_mask] = np.unique(region, False, True)
    point_num = embedding.shape[0]
    feature_dim = embedding.shape[1]
    center_num = region_mask.max() + 1
    region_mask = torch.from_numpy(region_mask).cuda()
    centers = scatter_mean(embedding, region_mask, dim = 0)
    center_to_point_index = region_mask.view(-1,1).expand(embedding.shape)
    std = torch.gather(torch.norm(scatter_std(embedding, region_mask, dim = 0) + 1e-8 , dim = 1).view(-1,1), dim = 0, index = region_mask.view(-1,1))
    diff = embedding - torch.gather(centers, dim = 0, index = center_to_point_index)
    weight = torch.exp(- torch.div(torch.norm(diff, dim = 1) , std.view(-1)) ** 2)
    weights = scatter_add(weight, region_mask, dim = 0).view(-1,1).expand([center_num, feature_dim])
    weighted_mean = scatter_add(embedding * weight.view(-1,1).expand([point_num, feature_dim]), region_mask, dim = 0) / weights
    return weighted_mean


def gaussian_weighted_mean(embedding, bw, region_points):
    centers = torch.mean(embedding, dim = 0).view(1,-1)
    point_num = embedding.shape[0]
    feature_dim = embedding.shape[1]
    diff = torch.norm(embedding - centers,dim= 1)
    weight = gaussian(diff,bw * 2) * region_points
    centroid = weighted_mean(embedding, weight)
    return centroid


def gaussian_weight(d1,d2,b1, b2):

    d1 = torch.norm(d1,dim=1)
    d2 = torch.norm(d2,dim=1)
    p1 = exp(-((d1/b1))**2)
    p2 = exp(-((d2/b2))**2)
    p = exp(-((d1/b1))**2 - ((d2/b2))**2) / (2 * math.pi * b1 * b2)
    return p

def cross_modal_gaussian(d1,d2,b1, b2):
    d1 = torch.norm(d1,dim=1) * b1
    d2 = torch.norm(d2,dim=1) * b2
    p = exp(-d1*d1-d2*d2)
#    p = exp(-d2*d2)
    return p


def merge(center_num,  sv_groups, region_point_num,pred_center_embedding_device,
          pose_center_embedding_device , pred_bw_device, occupancy_center):
    merge_count = 0
    similarity_map = torch.zeros(center_num, center_num).cuda()
    region_ptn = torch.zeros(center_num).cuda()
    region_ocup = torch.zeros(center_num).cuda()
    valid_center_index = torch.zeros(center_num)
    bandwidth = torch.zeros(center_num, pred_bw_device.shape[1]).cuda()
    pred_centers = torch.zeros(center_num,pred_center_embedding_device.shape[1]).cuda()
    pose_centers = torch.zeros(center_num,pose_center_embedding_device.shape[1]).cuda()
    for i in range(center_num):

        new_clusters = torch.LongTensor(region_point_num[sv_groups[i]])
        region_ptn[i] = torch.sum(region_point_num[new_clusters])
        region_ocup[i] = torch.mean(occupancy_center[new_clusters])
        bandwidth[i,:] = torch.mean(pred_bw_device[new_clusters,:], dim = 0).view(1,-1)
        pred_centers[i,:] = gaussian_weighted_mean(pred_center_embedding_device[new_clusters,:], bandwidth[i,0], region_point_num[new_clusters])
        pose_centers[i,:] = gaussian_weighted_mean(pose_center_embedding_device[new_clusters,:], bandwidth[i,1], region_point_num[new_clusters])

        dist_semantic = pred_centers[i:i+1,:] - pred_centers
        dist_pose = pose_centers[i:i+1,:] - pose_centers
        bw1 = (bandwidth[i:i+1,0] * region_ptn[i] + bandwidth[:,0] * region_ptn) / (region_ptn[i] + region_ptn)
        bw2 = (bandwidth[i:i+1,1] * region_ptn[i] + bandwidth[:,1] * region_ptn) / (region_ptn[i] + region_ptn)
        occupancy_ratio = torch.clamp((region_ptn[i] + region_ptn) / torch.exp(region_ocup), min = 0.5)
        prob = cross_modal_gaussian(dist_semantic,dist_pose,bw1*config['bw_relax_factor'] , bw2*config['bw_relax_factor'] ) / occupancy_ratio
        similarity_map[i, :] = prob

    similar_pair = torch.argmax(similarity_map)
    similarity_value = similarity_map[similar_pair / center_num,similar_pair % center_num].clone()
    while(similarity_value > 0.5):
        similar_x = torch.max(similar_pair % center_num, similar_pair / center_num)
        similar_y = torch.min(similar_pair % center_num, similar_pair / center_num)
        occupancy_ratio_x = (torch.sum(region_point_num[sv_groups[similar_x]])) / torch.exp(torch.mean(occupancy_center[sv_groups[similar_x]]))
        occupancy_ratio_y = (torch.sum(region_point_num[sv_groups[similar_y]])) / torch.exp(torch.mean(occupancy_center[sv_groups[similar_y]]))
        if(occupancy_ratio_x + occupancy_ratio_y < 2):
            merge_count += 1
    #        print(similarity_map[1161,1153],similarity_map[1153,1161],valid_center_index[1161],valid_center_index[1153])
            remove_last_element = min(sv_groups[similar_x])
            similarity_map[remove_last_element, :] = -1
            similarity_map[:, remove_last_element] = -1
            sv_groups[similar_y] += (sv_groups[similar_x])
            valid_center_index[remove_last_element] = False
            sv_groups[similar_x] = [-1]
            new_clusters = torch.LongTensor(sv_groups[similar_y])
    # turns out here is a problem, run use of previous result!
            #
            bandwidth[similar_y,:] = torch.mean(pred_bw_device[new_clusters,:], dim = 0).view(1,-1)
            pred_centers[similar_y,:] = gaussian_weighted_mean(pred_center_embedding_device[new_clusters,:], bandwidth[similar_y,0], region_point_num[new_clusters])
            pose_centers[similar_y,:] = gaussian_weighted_mean(pose_center_embedding_device[new_clusters,:], bandwidth[similar_y,1], region_point_num[new_clusters])
            region_ptn[similar_y] = torch.sum(region_point_num[new_clusters])
            region_ocup[similar_y] = torch.mean(occupancy_center[new_clusters])
            i = similar_y
            dist_semantic = pred_centers[i:i+1,:] - pred_centers
            dist_pose = pose_centers[i:i+1,:] - pose_centers
            bw1 = (bandwidth[i:i+1,0] * region_ptn[i] + bandwidth[:,0] * region_ptn) / (region_ptn[i] + region_ptn)
            bw2 = (bandwidth[i:i+1,1] * region_ptn[i] + bandwidth[:,1] * region_ptn) / (region_ptn[i] + region_ptn)
            occupancy_ratio = torch.clamp((region_ptn[i] + region_ptn) / torch.exp(region_ocup), min = 0.5)

            prob = cross_modal_gaussian(dist_semantic,dist_pose,bw1*config['bw_relax_factor'] , bw2*config['bw_relax_factor'] ) / occupancy_ratio
            similarity_map[i,:] = prob
            similarity_map[:,i] = prob
            similarity_map[~valid_center_index,i] = -1
            similarity_map[i,~valid_center_index] = -1
            similarity_map[i,i] = -1
            similar_pair = torch.argmax(similarity_map)
            similarity_value = similarity_map[similar_pair / center_num,similar_pair % center_num].clone()
        else:
            similarity_map[similar_pair / center_num,similar_pair % center_num] = -1
            similar_pair = torch.argmax(similarity_map)
            similarity_value = similarity_map[similar_pair / center_num,similar_pair % center_num].clone()
    return sv_groups


def region_based_cross_modal_meanshift_merging(exsiting_offsets, pred_embedding,pred_semantic, pred_displacements, xyz, pred_bw, regions, occupancy_size):

    instance_count = 0
    expected_instance_num = 300
    minimum_instance_size = 0
    point_num = pred_embedding.shape[0]
    feature_dim = pred_embedding.shape[1]
    pred_embedding_device = torch.tensor(pred_embedding).cuda()
    pose_embedding_device = torch.tensor(xyz - pred_displacements).cuda()
    #pred_embedding_device = torch.cat([pred_embedding_device, pose_embedding_device], dim = 1)
    [region_index, region_mask] = np.unique(regions, False, True)
    region_mask = torch.from_numpy(region_mask).cuda()
    center_num = region_index.shape[0]

    pose_device = torch.tensor(xyz).cuda()
    region_centers = scatter_mean(pose_device,region_mask,dim = 0)
    region_range = torch.norm(pose_device - torch.gather(region_centers, dim = 0, index = region_mask.view(-1,1).expand(-1,3)),dim=1)
    region_range,max_index_value = scatter_max(region_range, region_mask, dim = 0)
    region_point_num = scatter_add(torch.ones([point_num]).cuda(), region_mask, dim = 0)

    similarity_map = torch.zeros(center_num, center_num)
    pred_center_embedding_device = gaussian_weighted_region_mean(pred_embedding_device, regions)
    pose_center_embedding_device = gaussian_weighted_region_mean(pose_embedding_device, regions)
    pred_bw_device = gaussian_weighted_region_mean(torch.tensor(pred_bw).cuda(), regions)
    clustering_centers = torch.zeros(center_num, feature_dim)
    #compute the probability of two arbitrary supervoxels

    # First, group features from bottom to top
    center_semantic = torch.zeros((center_num,))
    for i in range(center_num):
        center_semantic[i] = int(stats.mode(pred_semantic[regions == region_index[i]])[0])
    valid_center_index = torch.ones([center_num])
    valid_center_index = (valid_center_index == 1)

    sv_groups = []
    for i in range(center_num):
        sv_groups.append([])
        sv_groups[i].append(i)
    pred_centers = (pred_center_embedding_device).clone()
    pose_centers = (pose_center_embedding_device).clone()
    bandwidth = pred_bw_device.clone()
    region_ptn = region_point_num.clone()
    occupancy_size_device = torch.from_numpy(occupancy_size).cuda()
    occupancy_center = scatter_mean(occupancy_size_device,region_mask,dim = 0)
    region_ocup = occupancy_center.clone()
    for i in range(center_num):
        dist_semantic = pred_centers[i:i+1,:] - pred_centers
        dist_pose = pose_centers[i:i+1,:] - pose_centers
        bw1 = (pred_bw_device[i:i+1,0] * region_point_num[i] + pred_bw_device[:,0] * region_point_num) / (region_point_num[i] + region_point_num)
        bw2 = (pred_bw_device[i:i+1,1] * region_point_num[i] + pred_bw_device[:,1] * region_point_num) / (region_point_num[i] + region_point_num)
        # estimate the occupancy size:
        occupancy_ratio = (region_ptn[i] + region_ptn) / torch.exp(region_ocup)
        similarity_map[i,:] = cross_modal_gaussian(dist_semantic,dist_pose,bw1*config['bw_relax_factor'] , bw2*config['bw_relax_factor'] )
        similarity_map[i,i] = -1
    for i in range(center_num):
        if(center_semantic[i] < 2):
            similarity_map[i,:] = -1
            similarity_map[:,i] = -1
            valid_center_index[i] = False
            sv_groups[i] = [-1]

    merge_count = 0

    similar_pair = torch.argmax(similarity_map)
    similarity_value = similarity_map[similar_pair / center_num,similar_pair % center_num].clone()
    while(similarity_value > 0.6):
        merge_count += 1
        similar_x = torch.max(similar_pair % center_num, similar_pair / center_num)
        similar_y = torch.min(similar_pair % center_num, similar_pair / center_num)
#        print(similarity_map[1161,1153],similarity_map[1153,1161],valid_center_index[1161],valid_center_index[1153])
        remove_last_element = min(sv_groups[similar_x])
        similarity_map[remove_last_element, :] = -1
        similarity_map[:, remove_last_element] = -1
        sv_groups[similar_y] += (sv_groups[similar_x])
        valid_center_index[remove_last_element] = False
        sv_groups[similar_x] = [-1]

        new_clusters = torch.LongTensor(sv_groups[similar_y])
# turns out here is a problem, run use of previous result!
        #
        bandwidth[similar_y,:] = torch.mean(pred_bw_device[new_clusters,:], dim = 0).view(1,-1)
        pred_centers[similar_y,:] = gaussian_weighted_mean(pred_center_embedding_device[new_clusters,:], bandwidth[similar_y,0], region_point_num[new_clusters])
        pose_centers[similar_y,:] = gaussian_weighted_mean(pose_center_embedding_device[new_clusters,:], bandwidth[similar_y,1], region_point_num[new_clusters])
        region_ptn[similar_y] = torch.sum(region_point_num[new_clusters])
        region_ocup[similar_y] = torch.mean(occupancy_center[new_clusters])
        i = similar_y
        dist_semantic = pred_centers[i:i+1,:] - pred_centers
        dist_pose = pose_centers[i:i+1,:] - pose_centers
        bw1 = (bandwidth[i:i+1,0] * region_ptn[i] + bandwidth[:,0] * region_ptn) / (region_ptn[i] + region_ptn)
        bw2 = (bandwidth[i:i+1,1] * region_ptn[i] + bandwidth[:,1] * region_ptn) / (region_ptn[i] + region_ptn)
#        occupancy_ratio = (region_ptn[i] + region_ptn) / torch.exp(region_ocup)
        prob = cross_modal_gaussian(dist_semantic,dist_pose,bw1*config['bw_relax_factor'] , bw2*config['bw_relax_factor'] )
        similarity_map[i,:] = prob
        similarity_map[:,i] = prob
        similarity_map[~valid_center_index,i] = -1
        similarity_map[i,~valid_center_index] = -1
        similarity_map[i,i] = -1
        similar_pair = torch.argmax(similarity_map)
        similarity_value = similarity_map[similar_pair / center_num,similar_pair % center_num].clone()

    #update all probabilities based on the predicted instance segmentation

    new_sv_groups = []
    new_center_num = 0
    for i in range(center_num):
        if(valid_center_index[i]):
            new_sv_groups.append([])
            new_sv_groups[new_center_num].append(sv_groups[i])
            new_center_num += 1
    sv_groups = merge(new_center_num,  new_sv_groups, region_point_num,pred_center_embedding_device,
          pose_center_embedding_device , pred_bw_device, occupancy_center)

    center_index = torch.zeros([center_num]) - 1
    for i in range(center_num):
        if(len(sv_groups[i]) > 1):
            center_index[torch.LongTensor(sv_groups[i])] = i
    center_index = center_index.long()
    instance_id, instance_mask = np.unique(center_index.numpy(), False, True)

    pred_instance_points = torch.gather(torch.LongTensor(instance_mask).cuda(), dim = 0, index = region_mask)
    return pred_instance_points.cpu().numpy()


    """
    for i in range(center_num):
        new_center = pred_center_embedding_device[i,:]
        for k in range(10):
            new_center = gaussian_weighted_mean(pred_center_embedding_device, new_center)
        clustering_centers[i,:] = new_center
    """
#    pdb.set_trace()
    pred_embedding_device = torch.cat([pred_center_embedding_device, pose_center_embedding_device], dim = 1)
    pred_instance_points = torch.zeros((point_num,), dtype=torch.int32).cuda()

    pred_instance = torch.zeros((center_num,), dtype=torch.int32).cuda()
    background_mask = torch.tensor(np.ones((center_num,), dtype=np.int)).cuda() > 0


    exsiting_offsets_device = torch.tensor(exsiting_offsets).cuda()
    exsiting_offsets_device, max_index_value = scatter_max(exsiting_offsets_device, region_mask, dim = 0)
    instance_flag = (torch.tensor(center_semantic).cuda() < 2) * background_mask
    exsiting_offsets_device[instance_flag] = 0
    background_mask[instance_flag] = 0
    pred_instance[instance_flag] = -1
    # consider to remove ground and floor for scannet dataset?
    while instance_count <  expected_instance_num and background_mask.sum(0) > minimum_instance_size * 0.5:
        keypoint = torch.argmax(exsiting_offsets_device)
        pre_centroid = pred_embedding_device[keypoint,:]
        b = pred_bw_device[keypoint,:] * config['bw_relax_factor']
        #first iteration, only use the selected point feature as embedding feature
        embedding_distance = (pred_embedding_device - pre_centroid)
        prob = cross_modal_gaussian(embedding_distance[:,0:semantic_embedding_len],embedding_distance[:,semantic_embedding_len:semantic_embedding_len+3], b[0], b[1])
        instance_flag = (prob > 0.5) * background_mask
        local_embedding = pred_embedding_device[instance_flag]
        weight = gaussian(embedding_distance.norm(dim = 1),b.mean(dim=0))[instance_flag] * region_point_num[instance_flag]
        centroid = weighted_mean(local_embedding, weight)
        pre_centroid = centroid
        for k in range(10):
            embedding_distance = (pred_embedding_device - pre_centroid)
            b = pred_bw_device[instance_flag,:].mean(dim = 0) * config['bw_relax_factor']
            prob = cross_modal_gaussian(embedding_distance[:,0:semantic_embedding_len],embedding_distance[:,semantic_embedding_len:semantic_embedding_len+3], b[0], b[1])
            instance_flag = (prob > 0.5) * background_mask
            local_embedding = pred_embedding_device[instance_flag]
            weight = gaussian(embedding_distance.norm(dim = 1),b.mean(dim=0))[instance_flag] * region_point_num[instance_flag]
            centroid = weighted_mean(local_embedding, weight)
            mean_shift_vector = pre_centroid - centroid
            pre_centroid = centroid
            if(mean_shift_vector.norm(dim = 0) < 1e-5):
                break
#        print(instance_flag.sum().item())

        vis_flag = (torch.gather(instance_flag, dim = 0, index = region_mask).sum() == 2986)
        if config['visualize_part']:
            shiftX = np.zeros((xyz.shape[0],3), dtype = np.float32)
            shiftX[:,0] += 6
            shiftY = np.zeros((xyz.shape[0],3), dtype = np.float32)
            shiftY[:,1] += 6

            pcd_center = open3d.geometry.PointCloud()
            pcd_center.points = open3d.utility.Vector3dVector(xyz)
            gt_color = torch.zeros([xyz.shape[0],3],dtype=torch.float32)
            point_instance_flag = torch.gather(instance_flag, dim = 0, index = region_mask)
            gt_color[point_instance_flag,1] = 1
            pcd_center.colors = open3d.utility.Vector3dVector(cm.hot(exsiting_offsets_device.data.cpu().numpy())[:,0,0:3])


            pcd_gt = open3d.geometry.PointCloud()
            pcd_gt.points = open3d.utility.Vector3dVector(xyz + shiftX)
            gt_color = torch.zeros([xyz.shape[0],3],dtype=torch.float32)
            gt_color[point_instance_flag,1] = 1
            pcd_gt.colors = open3d.utility.Vector3dVector(gt_color.data.numpy())

            pcd_pred_instance = open3d.geometry.PointCloud()
            pcd_pred_instance.points = open3d.utility.Vector3dVector(xyz + shiftX*2)
            gt_color = cm.hot(torch.gather(prob, dim = 0, index = region_mask).data.cpu().numpy())[:,0:3]
            pcd_pred_instance.colors = open3d.utility.Vector3dVector(gt_color)


            d1 = torch.norm(embedding_distance[:,0:semantic_embedding_len],dim=1)
            d2 = torch.norm(embedding_distance[:,semantic_embedding_len:semantic_embedding_len+3],dim=1)
            p1 = exp(-((d1/ b[0]))**2)
            p2 = exp(-((d2/ b[1]))**2)

            pcd_prob1 = open3d.geometry.PointCloud()
            pcd_prob1.points = open3d.utility.Vector3dVector(xyz + shiftX*3)
            gt_color = cm.hot(torch.gather(p1, dim = 0, index = region_mask).data.cpu().numpy())[:,0:3]
            pcd_prob1.colors = open3d.utility.Vector3dVector(gt_color)

            pcd_prob2 = open3d.geometry.PointCloud()
            pcd_prob2.points = open3d.utility.Vector3dVector(xyz + shiftX*4)
            gt_color = cm.hot(torch.gather(p2, dim = 0, index = region_mask).data.cpu().numpy())[:,0:3]
            pcd_prob2.colors = open3d.utility.Vector3dVector(gt_color)

            open3d.visualization.draw_geometries([pcd_gt,pcd_pred_instance,pcd_prob1,pcd_prob2])
#            open3d.visualization.draw_geometries([pcd_center, pcd_gt,pcd_pred_instance,pcd_prob1,pcd_prob2])
    #            print(k,mean_shift_vector.norm(dim = 0).item(), instance_flag.sum(0).item())

        background_mask[instance_flag] = 0
        exsiting_offsets_device[instance_flag] = 0

        if False:
            cluster_poses = pose_device[instance_flag,:]
            detailed_instances = DBSCAN(eps = 0.1, min_samples = 10 , n_jobs = 1).fit(cluster_poses.cpu().numpy()).labels_
            detailed_instances = torch.tensor(detailed_instances)
            pos_index = torch.tensor(np.arange(0,instance_flag.shape[0]))[instance_flag]
            pcds = []
            for k in range(torch.max(detailed_instances)+1):
                if((detailed_instances == k).sum(0) > minimum_instance_size):
                    instance_count += 1
                    pred_instance[pos_index[detailed_instances == k]] = instance_count
                else:
                    pred_instance[pos_index[detailed_instances == k]] = -1
                """
                pcd_gt = open3d.geometry.PointCloud()
                pcd_gt.points = open3d.utility.Vector3dVector(pose_device[instance_flag,:][detailed_instances==k,:].cpu().numpy())
                pcd_gt.colors = open3d.utility.Vector3dVector(np.random.rand(1,3).repeat((detailed_instances==k).sum(0),axis=0))
                pcds.append(pcd_gt)
                """
#            open3d.visualization.draw_geometries(pcds)
            pred_instance[pos_index[detailed_instances == -1]] = -1
        else:
            instance_count += 1
            if(instance_flag.sum(0) > minimum_instance_size ):
                pred_instance[instance_flag] = instance_count
            else:
                pred_instance[instance_flag] = -1
    pred_instance_points = torch.gather(pred_instance, dim = 0, index = region_mask)
    return pred_instance_points.cpu().numpy()


def occupancy_aware_region_based_cross_modal_meanshift_clustering(exsiting_offsets, pred_embedding,pred_semantic, pred_displacements, xyz, pred_bw, regions, occupancy_size):

    instance_count = 0
    expected_instance_num = 300
    minimum_instance_size = 0
    point_num = pred_embedding.shape[0]
    feature_dim = pred_embedding.shape[1]
    pred_embedding_device = torch.tensor(pred_embedding).cuda()
    pose_embedding_device = torch.tensor(xyz - pred_displacements).cuda()
    #pred_embedding_device = torch.cat([pred_embedding_device, pose_embedding_device], dim = 1)
    [region_index, region_mask] = np.unique(regions, False, True)
    region_mask = torch.from_numpy(region_mask).cuda()
    center_num = region_index.shape[0]

    region_point_num = scatter_add(torch.ones([point_num]).cuda(), region_mask, dim = 0)
    pose_device = torch.tensor(xyz).cuda()
    region_centers = scatter_mean(pose_device,region_mask,dim = 0)
    region_range = torch.norm(pose_device - torch.gather(region_centers, dim = 0, index = region_mask.view(-1,1).expand(-1,3)),dim=1)
    region_range,max_index_value = scatter_max(region_range, region_mask, dim = 0)

    similarity_map = torch.zeros(2,center_num, center_num)
    pred_center_embedding_device = gaussian_weighted_region_mean(pred_embedding_device, regions)
    pose_center_embedding_device = gaussian_weighted_region_mean(pose_embedding_device, regions)
    clustering_centers = torch.zeros(center_num, feature_dim)
    """
    for i in range(center_num):
        new_center = pred_center_embedding_device[i,:]
        for k in range(10):
            new_center = gaussian_weighted_mean(pred_center_embedding_device, new_center)
        clustering_centers[i,:] = new_center
    """
#    pdb.set_trace()
    pred_embedding_device = torch.cat([pred_center_embedding_device, pose_center_embedding_device], dim = 1)
    pred_instance_points = torch.zeros((point_num,), dtype=torch.int32).cuda()

    pred_instance = torch.zeros((center_num,), dtype=torch.int32).cuda()
    background_mask = torch.tensor(np.ones((center_num,), dtype=np.int)).cuda() > 0
    pred_bw_device = gaussian_weighted_region_mean(torch.tensor(pred_bw).cuda(), regions)
    center_semantic = torch.zeros((center_num,))

    occupancy_size_device = torch.from_numpy(occupancy_size).cuda()
    occupancy_center = scatter_mean(occupancy_size_device,region_mask,dim = 0)
    for i in range(center_num):
        center_semantic[i] = int(stats.mode(pred_semantic[regions == region_index[i]])[0])


    exsiting_offsets_device = torch.tensor(exsiting_offsets).cuda()
    region_point_num = scatter_add(torch.ones([point_num]).cuda(), region_mask, dim = 0)
    exsiting_offsets_device, max_index_value = scatter_max(exsiting_offsets_device, region_mask, dim = 0)
    instance_flag = (torch.tensor(center_semantic).cuda() < 2) * background_mask
    exsiting_offsets_device[instance_flag] = 0
    background_mask[instance_flag] = 0
    pred_instance[instance_flag] = -1
    # consider to remove ground and floor for scannet dataset?
    while instance_count <  expected_instance_num and background_mask.sum(0) > minimum_instance_size * 0.5:
        keypoint = torch.argmax(exsiting_offsets_device)
        pre_centroid = pred_embedding_device[keypoint,:]
        b = pred_bw_device[keypoint,:] * config['bw_relax_factor']
        #first iteration, only use the selected point feature as embedding feature
        embedding_distance = (pred_embedding_device - pre_centroid)
        prob = cross_modal_gaussian(embedding_distance[:,0:semantic_embedding_len],embedding_distance[:,semantic_embedding_len:semantic_embedding_len+3], b[0], b[1])
        instance_flag = (prob > 0.5) * background_mask
        local_embedding = pred_embedding_device[instance_flag]
        weight = gaussian(embedding_distance.norm(dim = 1),b.mean(dim=0))[instance_flag] * region_point_num[instance_flag]
        centroid = weighted_mean(local_embedding, weight)
        pre_centroid = centroid
        for k in range(10):
            embedding_distance = (pred_embedding_device - pre_centroid)
            b = pred_bw_device[instance_flag,:].mean(dim = 0) * config['bw_relax_factor']
            prob = cross_modal_gaussian(embedding_distance[:,0:semantic_embedding_len],embedding_distance[:,semantic_embedding_len:semantic_embedding_len+3], b[0], b[1])
            instance_flag = (prob > 0.5) * background_mask
            local_embedding = pred_embedding_device[instance_flag]
            weight = gaussian(embedding_distance.norm(dim = 1),b.mean(dim=0))[instance_flag] * region_point_num[instance_flag]
            centroid = weighted_mean(local_embedding, weight)
            mean_shift_vector = pre_centroid - centroid
            pre_centroid = centroid
            if(mean_shift_vector.norm(dim = 0) < 1e-5):
#                print(torch.exp(torch.mean(occupancy_center[instance_flag])).item(),torch.exp(torch.median(occupancy_center[instance_flag])).item())
#                print(torch.sum(region_point_num[instance_flag]).item())
#                pdb.set_trace()
                break
#        print(instance_flag.sum().item())

        vis_flag = (torch.gather(instance_flag, dim = 0, index = region_mask).sum() == 2986)
        if config['visualize_part']:
            shiftX = np.zeros((xyz.shape[0],3), dtype = np.float32)
            shiftX[:,0] += 6
            shiftY = np.zeros((xyz.shape[0],3), dtype = np.float32)
            shiftY[:,1] += 6

            pcd_center = open3d.geometry.PointCloud()
            pcd_center.points = open3d.utility.Vector3dVector(xyz)
            gt_color = torch.zeros([xyz.shape[0],3],dtype=torch.float32)
            point_instance_flag = torch.gather(instance_flag, dim = 0, index = region_mask)
            gt_color[point_instance_flag,1] = 1
            pcd_center.colors = open3d.utility.Vector3dVector(cm.hot(exsiting_offsets_device.data.cpu().numpy())[:,0,0:3])


            pcd_gt = open3d.geometry.PointCloud()
            pcd_gt.points = open3d.utility.Vector3dVector(xyz + shiftX)
            gt_color = torch.zeros([xyz.shape[0],3],dtype=torch.float32)
            gt_color[point_instance_flag,1] = 1
            pcd_gt.colors = open3d.utility.Vector3dVector(gt_color.data.numpy())

            pcd_pred_instance = open3d.geometry.PointCloud()
            pcd_pred_instance.points = open3d.utility.Vector3dVector(xyz + shiftX*2)
            gt_color = cm.hot(torch.gather(prob, dim = 0, index = region_mask).data.cpu().numpy())[:,0:3]
            pcd_pred_instance.colors = open3d.utility.Vector3dVector(gt_color)


            d1 = torch.norm(embedding_distance[:,0:semantic_embedding_len],dim=1)
            d2 = torch.norm(embedding_distance[:,semantic_embedding_len:semantic_embedding_len+3],dim=1)
            p1 = exp(-((d1/ b[0]))**2)
            p2 = exp(-((d2/ b[1]))**2)

            pcd_prob1 = open3d.geometry.PointCloud()
            pcd_prob1.points = open3d.utility.Vector3dVector(xyz + shiftX*3)
            gt_color = cm.hot(torch.gather(p1, dim = 0, index = region_mask).data.cpu().numpy())[:,0:3]
            pcd_prob1.colors = open3d.utility.Vector3dVector(gt_color)

            pcd_prob2 = open3d.geometry.PointCloud()
            pcd_prob2.points = open3d.utility.Vector3dVector(xyz + shiftX*4)
            gt_color = cm.hot(torch.gather(p2, dim = 0, index = region_mask).data.cpu().numpy())[:,0:3]
            pcd_prob2.colors = open3d.utility.Vector3dVector(gt_color)

            open3d.visualization.draw_geometries([pcd_gt,pcd_pred_instance,pcd_prob1,pcd_prob2])
#            open3d.visualization.draw_geometries([pcd_center, pcd_gt,pcd_pred_instance,pcd_prob1,pcd_prob2])
    #            print(k,mean_shift_vector.norm(dim = 0).item(), instance_flag.sum(0).item())

        background_mask[instance_flag] = 0
        exsiting_offsets_device[instance_flag] = 0

        if False:
            cluster_poses = pose_device[instance_flag,:]
            detailed_instances = DBSCAN(eps = 0.1, min_samples = 10 , n_jobs = 1).fit(cluster_poses.cpu().numpy()).labels_
            detailed_instances = torch.tensor(detailed_instances)
            pos_index = torch.tensor(np.arange(0,instance_flag.shape[0]))[instance_flag]
            pcds = []
            for k in range(torch.max(detailed_instances)+1):
                if((detailed_instances == k).sum(0) > minimum_instance_size):
                    instance_count += 1
                    pred_instance[pos_index[detailed_instances == k]] = instance_count
                else:
                    pred_instance[pos_index[detailed_instances == k]] = -1
                """
                pcd_gt = open3d.geometry.PointCloud()
                pcd_gt.points = open3d.utility.Vector3dVector(pose_device[instance_flag,:][detailed_instances==k,:].cpu().numpy())
                pcd_gt.colors = open3d.utility.Vector3dVector(np.random.rand(1,3).repeat((detailed_instances==k).sum(0),axis=0))
                pcds.append(pcd_gt)
                """
#            open3d.visualization.draw_geometries(pcds)
            pred_instance[pos_index[detailed_instances == -1]] = -1
        else:
            instance_count += 1
            if(instance_flag.sum(0) > minimum_instance_size ):
                pred_instance[instance_flag] = instance_count
            else:
                pred_instance[instance_flag] = -1
    pred_instance_points = torch.gather(pred_instance, dim = 0, index = region_mask)
    return pred_instance_points.cpu().numpy()

def region_based_cross_modal_meanshift_clustering(exsiting_offsets, pred_embedding,pred_semantic, pred_displacements, xyz, pred_bw, regions):

    instance_count = 0
    expected_instance_num = 300
    minimum_instance_size = 0
    point_num = pred_embedding.shape[0]
    feature_dim = pred_embedding.shape[1]
    pred_embedding_device = torch.tensor(pred_embedding).cuda()
    pose_embedding_device = torch.tensor(xyz - pred_displacements).cuda()
    #pred_embedding_device = torch.cat([pred_embedding_device, pose_embedding_device], dim = 1)
    [region_index, region_mask] = np.unique(regions, False, True)
    region_mask = torch.from_numpy(region_mask).cuda()
    center_num = region_index.shape[0]

    pose_device = torch.tensor(xyz).cuda()
    region_centers = scatter_mean(pose_device,region_mask,dim = 0)
    region_range = torch.norm(pose_device - torch.gather(region_centers, dim = 0, index = region_mask.view(-1,1).expand(-1,3)),dim=1)
    region_range,max_index_value = scatter_max(region_range, region_mask, dim = 0)

    similarity_map = torch.zeros(2,center_num, center_num)
    pred_center_embedding_device = gaussian_weighted_region_mean(pred_embedding_device, regions)
    pose_center_embedding_device = gaussian_weighted_region_mean(pose_embedding_device, regions)
    clustering_centers = torch.zeros(center_num, feature_dim)
    """
    for i in range(center_num):
        new_center = pred_center_embedding_device[i,:]
        for k in range(10):
            new_center = gaussian_weighted_mean(pred_center_embedding_device, new_center)
        clustering_centers[i,:] = new_center
    """
#    pdb.set_trace()
    pred_embedding_device = torch.cat([pred_center_embedding_device, pose_center_embedding_device], dim = 1)
    pred_instance_points = torch.zeros((point_num,), dtype=torch.int32).cuda()

    pred_instance = torch.zeros((center_num,), dtype=torch.int32).cuda()
    background_mask = torch.tensor(np.ones((center_num,), dtype=np.int)).cuda() > 0
    pred_bw_device = gaussian_weighted_region_mean(torch.tensor(pred_bw).cuda(), regions)
    center_semantic = torch.zeros((center_num,))
    for i in range(center_num):
        center_semantic[i] = int(stats.mode(pred_semantic[regions == region_index[i]])[0])


    exsiting_offsets_device = torch.tensor(exsiting_offsets).cuda()
    region_point_num = scatter_add(torch.ones([point_num]).cuda(), region_mask, dim = 0)
    exsiting_offsets_device, max_index_value = scatter_max(exsiting_offsets_device, region_mask, dim = 0)
    instance_flag = (torch.tensor(center_semantic).cuda() < 2) * background_mask
    exsiting_offsets_device[instance_flag] = 0
    background_mask[instance_flag] = 0
    pred_instance[instance_flag] = -1
    # consider to remove ground and floor for scannet dataset?

    candidates = torch.zeros((point_num, 100),dtype = torch.int64).cuda()
    probabilities = torch.zeros((point_num, 100),dtype = torch.float32).cuda()
    pred_point_embedding_device = torch.cat([pred_center_embedding_device, pose_center_embedding_device], dim = 1)


    while instance_count <  expected_instance_num and background_mask.sum(0) > minimum_instance_size * 0.5:
        keypoint = torch.argmax(exsiting_offsets_device)
        pre_centroid = pred_embedding_device[keypoint,:]
        b = pred_bw_device[keypoint,:] * config['bw_relax_factor']
        #first iteration, only use the selected point feature as embedding feature
        embedding_distance = (pred_embedding_device - pre_centroid)
        prob = cross_modal_gaussian(embedding_distance[:,0:semantic_embedding_len],embedding_distance[:,semantic_embedding_len:semantic_embedding_len+3], b[0], b[1])
        instance_flag = (prob > 0.5) * background_mask
        local_embedding = pred_embedding_device[instance_flag]
        weight = gaussian(embedding_distance.norm(dim = 1),b.mean(dim=0))[instance_flag] * region_point_num[instance_flag]
        centroid = weighted_mean(local_embedding, weight)
        pre_centroid = centroid
        for k in range(10):
            embedding_distance = (pred_embedding_device - pre_centroid)
            b = pred_bw_device[instance_flag,:].mean(dim = 0) * config['bw_relax_factor']
            prob = cross_modal_gaussian(embedding_distance[:,0:semantic_embedding_len],embedding_distance[:,semantic_embedding_len:semantic_embedding_len+3], b[0], b[1])
            instance_flag = (prob > 0.5) * background_mask
            local_embedding = pred_embedding_device[instance_flag]
            weight = gaussian(embedding_distance.norm(dim = 1),b.mean(dim=0))[instance_flag] * region_point_num[instance_flag]
            centroid = weighted_mean(local_embedding, weight)
            mean_shift_vector = pre_centroid - centroid
            pre_centroid = centroid
            if(mean_shift_vector.norm(dim = 0) < 1e-5):
                break
#        print(instance_flag.sum().item())
        instance_semantic_label = int(stats.mode(center_semantic[instance_flag].cpu().numpy())[0])
        instance_flag = instance_flag * (center_semantic.cuda() == instance_semantic_label)
        vis_flag = (torch.gather(instance_flag, dim = 0, index = region_mask).sum() == 2986)
        if config['visualize_part']:
            shiftX = np.zeros((xyz.shape[0],3), dtype = np.float32)
            shiftX[:,0] += 6
            shiftY = np.zeros((xyz.shape[0],3), dtype = np.float32)
            shiftY[:,1] += 6

            pcd_center = open3d.geometry.PointCloud()
            pcd_center.points = open3d.utility.Vector3dVector(xyz)
            gt_color = torch.zeros([xyz.shape[0],3],dtype=torch.float32)
            point_instance_flag = torch.gather(instance_flag, dim = 0, index = region_mask)
            gt_color[point_instance_flag,1] = 1
            pcd_center.colors = open3d.utility.Vector3dVector(cm.hot(exsiting_offsets_device.data.cpu().numpy())[:,0,0:3])


            pcd_gt = open3d.geometry.PointCloud()
            pcd_gt.points = open3d.utility.Vector3dVector(xyz + shiftX)
            gt_color = torch.zeros([xyz.shape[0],3],dtype=torch.float32)
            gt_color[point_instance_flag,1] = 1
            pcd_gt.colors = open3d.utility.Vector3dVector(gt_color.data.numpy())

            pcd_pred_instance = open3d.geometry.PointCloud()
            pcd_pred_instance.points = open3d.utility.Vector3dVector(xyz + shiftX*2)
            gt_color = cm.hot(torch.gather(prob, dim = 0, index = region_mask).data.cpu().numpy())[:,0:3]
            pcd_pred_instance.colors = open3d.utility.Vector3dVector(gt_color)


            d1 = torch.norm(embedding_distance[:,0:semantic_embedding_len],dim=1)
            d2 = torch.norm(embedding_distance[:,semantic_embedding_len:semantic_embedding_len+3],dim=1)
            p1 = exp(-((d1/ b[0]))**2)
            p2 = exp(-((d2/ b[1]))**2)

            pcd_prob1 = open3d.geometry.PointCloud()
            pcd_prob1.points = open3d.utility.Vector3dVector(xyz + shiftX*3)
            gt_color = cm.hot(torch.gather(p1, dim = 0, index = region_mask).data.cpu().numpy())[:,0:3]
            pcd_prob1.colors = open3d.utility.Vector3dVector(gt_color)

            pcd_prob2 = open3d.geometry.PointCloud()
            pcd_prob2.points = open3d.utility.Vector3dVector(xyz + shiftX*4)
            gt_color = cm.hot(torch.gather(p2, dim = 0, index = region_mask).data.cpu().numpy())[:,0:3]
            pcd_prob2.colors = open3d.utility.Vector3dVector(gt_color)

            open3d.visualization.draw_geometries([pcd_gt,pcd_pred_instance,pcd_prob1,pcd_prob2])
#            open3d.visualization.draw_geometries([pcd_center, pcd_gt,pcd_pred_instance,pcd_prob1,pcd_prob2])
    #            print(k,mean_shift_vector.norm(dim = 0).item(), instance_flag.sum(0).item())

        background_mask[instance_flag] = 0
        exsiting_offsets_device[instance_flag] = 0

        if False:
            cluster_poses = pose_device[instance_flag,:]
            detailed_instances = DBSCAN(eps = 0.1, min_samples = 10 , n_jobs = 1).fit(cluster_poses.cpu().numpy()).labels_
            detailed_instances = torch.tensor(detailed_instances)
            pos_index = torch.tensor(np.arange(0,instance_flag.shape[0]))[instance_flag]
            pcds = []
            for k in range(torch.max(detailed_instances)+1):
                if((detailed_instances == k).sum(0) > minimum_instance_size):
                    instance_count += 1
                    pred_instance[pos_index[detailed_instances == k]] = instance_count
                else:
                    pred_instance[pos_index[detailed_instances == k]] = -1
                """
                pcd_gt = open3d.geometry.PointCloud()
                pcd_gt.points = open3d.utility.Vector3dVector(pose_device[instance_flag,:][detailed_instances==k,:].cpu().numpy())
                pcd_gt.colors = open3d.utility.Vector3dVector(np.random.rand(1,3).repeat((detailed_instances==k).sum(0),axis=0))
                pcds.append(pcd_gt)
                """
#            open3d.visualization.draw_geometries(pcds)
            pred_instance[pos_index[detailed_instances == -1]] = -1
        else:
            instance_count += 1
            if instance_flag.sum(0) > minimum_instance_size:
                pred_instance[instance_flag] = instance_count
            else:
                pred_instance[instance_flag] = -1
    pred_instance_points = torch.gather(pred_instance, dim = 0, index = region_mask)
    return pred_instance_points.cpu().numpy()



def joint_semantic_instance_segmentation(pred_semantic_probability, true_semantic,exsiting_offsets, pred_embedding,pred_semantic, pred_displacements, xyz, pred_bw, regions):

    instance_count = 0
    expected_instance_num = 10000
    minimum_instance_size = 0
    point_num = pred_embedding.shape[0]
    feature_dim = pred_embedding.shape[1]
    class_num = pred_semantic_probability.shape[1]
    pred_embedding_device = torch.tensor(pred_embedding).cuda()
    pose_embedding_device = torch.tensor(xyz - pred_displacements).cuda()
    #pred_embedding_device = torch.cat([pred_embedding_device, pose_embedding_device], dim = 1)
    [region_index, region_mask] = np.unique(regions, False, True)
    region_mask = torch.from_numpy(region_mask).cuda()
    center_num = region_index.shape[0]

    pred_semantic_probability_device = torch.tensor(pred_semantic_probability).cuda()
    pred_semantic_probability_device = torch.exp(pred_semantic_probability_device) / torch.sum(torch.exp(pred_semantic_probability_device), dim = 1).view(-1,1).expand(point_num, 20)
    center_semantic_probability_device = scatter_mean(pred_semantic_probability_device,region_mask,dim = 0)
    pose_device = torch.tensor(xyz).cuda()
    region_centers = scatter_mean(pose_device,region_mask,dim = 0)
    region_range = torch.norm(pose_device - torch.gather(region_centers, dim = 0, index = region_mask.view(-1,1).expand(-1,3)),dim=1)
    region_range,max_index_value = scatter_max(region_range, region_mask, dim = 0)

    similarity_map = torch.zeros(2,center_num, center_num)
    pred_center_embedding_device = gaussian_weighted_region_mean(pred_embedding_device, regions)
    pose_center_embedding_device = gaussian_weighted_region_mean(pose_embedding_device, regions)
    clustering_centers = torch.zeros(center_num, feature_dim)
    """
    for i in range(center_num):
        new_center = pred_center_embedding_device[i,:]
        for k in range(10):
            new_center = gaussian_weighted_mean(pred_center_embedding_device, new_center)
        clustering_centers[i,:] = new_center
    """
#    pdb.set_trace()
    pred_embedding_device = torch.cat([pred_center_embedding_device, pose_center_embedding_device], dim = 1)
    pred_instance_points = torch.zeros((point_num,), dtype=torch.int32).cuda()

    pred_instance = torch.zeros((center_num,), dtype=torch.int32).cuda()
    background_mask = torch.tensor(np.ones((center_num,), dtype=np.int)).cuda() > 0
    pred_bw_device = gaussian_weighted_region_mean(torch.tensor(pred_bw).cuda(), regions)
    center_semantic = torch.zeros((center_num,))
    for i in range(center_num):
        center_semantic[i] = int(stats.mode(pred_semantic[regions == region_index[i]])[0])


    exsiting_offsets_device = torch.tensor(exsiting_offsets).cuda()
    region_point_num = scatter_add(torch.ones([point_num]).cuda(), region_mask, dim = 0)
    exsiting_offsets_device, max_index_value = scatter_max(exsiting_offsets_device, region_mask, dim = 0)
    pred_semantic_label = torch.ones([center_num], dtype=torch.int32).cuda()
    # consider to remove ground and floor for scannet dataset?



    while instance_count <  expected_instance_num and background_mask.sum(0) > minimum_instance_size * 0.5:
        keypoint = torch.argmax(exsiting_offsets_device)
        pre_centroid = pred_embedding_device[keypoint,:]
        b = pred_bw_device[keypoint,:] * config['bw_relax_factor']
        #first iteration, only use the selected point feature as embedding feature
        embedding_distance = (pred_embedding_device - pre_centroid)
        prob = cross_modal_gaussian(embedding_distance[:,0:semantic_embedding_len],embedding_distance[:,semantic_embedding_len:semantic_embedding_len+3], b[0], b[1])
        prob[prob < 0.3] = 0
        instance_flag = (prob > 0.5)

        if config['visualize_part']:
            shiftX = np.zeros((xyz.shape[0],3), dtype = np.float32)
            shiftX[:,0] += 8
            shiftY = np.zeros((xyz.shape[0],3), dtype = np.float32)
            shiftY[:,1] += 8

            pcd_center = open3d.geometry.PointCloud()
            pcd_center.points = open3d.utility.Vector3dVector(xyz)
            gt_color = torch.zeros([xyz.shape[0],3],dtype=torch.float32)
            point_instance_flag = torch.gather(instance_flag, dim = 0, index = region_mask)
            gt_color[point_instance_flag,1] = 1
            pcd_center.colors = open3d.utility.Vector3dVector(cm.hot(exsiting_offsets_device.data.cpu().numpy())[:,0,0:3])


            pcd_gt = open3d.geometry.PointCloud()
            pcd_gt.points = open3d.utility.Vector3dVector(xyz + shiftX)
            gt_color = torch.zeros([xyz.shape[0],3],dtype=torch.float32)
            gt_color[point_instance_flag,1] = 1
            pcd_gt.colors = open3d.utility.Vector3dVector(gt_color.data.numpy())

            pcd_pred_instance = open3d.geometry.PointCloud()
            pcd_pred_instance.points = open3d.utility.Vector3dVector(xyz + shiftX*2)
            gt_color = cm.hot(torch.gather(prob, dim = 0, index = region_mask).data.cpu().numpy())[:,0:3]
            pcd_pred_instance.colors = open3d.utility.Vector3dVector(gt_color)


            d1 = torch.norm(embedding_distance[:,0:semantic_embedding_len],dim=1)
            d2 = torch.norm(embedding_distance[:,semantic_embedding_len:semantic_embedding_len+3],dim=1)
            p1 = exp(-((d1/ b[0]))**2)
            p2 = exp(-((d2/ b[1]))**2)

            pcd_prob1 = open3d.geometry.PointCloud()
            pcd_prob1.points = open3d.utility.Vector3dVector(xyz + shiftX*3)
            gt_color = cm.hot(torch.gather(p1, dim = 0, index = region_mask).data.cpu().numpy())[:,0:3]
            pcd_prob1.colors = open3d.utility.Vector3dVector(gt_color)

            pcd_prob2 = open3d.geometry.PointCloud()
            pcd_prob2.points = open3d.utility.Vector3dVector(xyz + shiftX*4)
            gt_color = cm.hot(torch.gather(p2, dim = 0, index = region_mask).data.cpu().numpy())[:,0:3]
            pcd_prob2.colors = open3d.utility.Vector3dVector(gt_color)

            pcd_semantic = open3d.geometry.PointCloud()
            pcd_semantic.points = open3d.utility.Vector3dVector(xyz + shiftX*5)
            pcd_semantic.colors  = open3d.utility.Vector3dVector(label2color(pred_semantic))


            open3d.visualization.draw_geometries([pcd_gt,pcd_pred_instance,pcd_prob1,pcd_prob2,pcd_semantic])
#            open3d.visualization.draw_geometries([pcd_center, pcd_gt,pcd_pred_instance,pcd_prob1,pcd_prob2])
    #            print(k,mean_shift_vector.norm(dim = 0).item(), instance_flag.sum(0).item())

        background_mask[keypoint] = 0
        exsiting_offsets_device[keypoint] = 0
        instance_count += 1
        instance_influence_weight = (region_point_num * prob)
        pred_semantic_label[keypoint] = torch.argmax(torch.sum((center_semantic_probability_device * instance_influence_weight.view(-1,1).expand([-1,class_num])), dim = 0) / torch.sum(instance_influence_weight))
        if(instance_flag.sum(0) > minimum_instance_size ):
            pred_instance[instance_flag] = instance_count
        else:
            pred_instance[instance_flag] = -1
    pred_instance_points = torch.gather(pred_instance, dim = 0, index = region_mask)
    joint_pred_semantic_labels = torch.gather(pred_semantic_label, dim = 0, index = region_mask)

    # Visualization
    if False:
        original_colors = label2color(pred_semantic)
        joint_pred_colors = label2color(joint_pred_semantic_labels.cpu().numpy())
        gt_colors = label2color(true_semantic)

        shiftX = np.zeros((xyz.shape[0],3), dtype = np.float32)
        shiftX[:,0] += 8
        shiftY = np.zeros((xyz.shape[0],3), dtype = np.float32)
        shiftY[:,1] += 0
        pcd_gt = open3d.geometry.PointCloud()
        pcd_gt.points = open3d.utility.Vector3dVector(xyz + shiftX)
        pcd_gt.colors = open3d.utility.Vector3dVector(gt_colors)

        pcd_joint_pred = open3d.geometry.PointCloud()
        pcd_joint_pred.points = open3d.utility.Vector3dVector(xyz + shiftX * 2)
        pcd_joint_pred.colors = open3d.utility.Vector3dVector(joint_pred_colors)

        pcd_ori = open3d.geometry.PointCloud()
        pcd_ori.points = open3d.utility.Vector3dVector(xyz + shiftX * 3)
        pcd_ori.colors = open3d.utility.Vector3dVector(original_colors)
#        evaluate_single_scan(joint_pred_semantic_labels.cpu().numpy(),true_semantic,None, 0, 20, topic = 'valid')
#        evaluate_single_scan(pred_semantic,true_semantic,None, 0, 20, topic = 'valid')
        open3d.visualization.draw_geometries([pcd_gt,pcd_joint_pred, pcd_ori])


    return joint_pred_semantic_labels.cpu().numpy(), pred_semantic, true_semantic



def cross_modal_meanshift_clustering_simple(exsiting_offsets, pred_embedding,pred_semantic, pred_displacements, xyz, pred_bw):

    instance_count = 0
    expected_instance_num = 300
    minimum_instance_size = 200
    exsiting_offsets_device = torch.tensor(exsiting_offsets).cuda()
    pred_embedding_device = torch.tensor(pred_embedding).cuda()
    pose_device = torch.tensor(xyz).cuda()
    pose_embedding_device = torch.tensor(xyz - pred_displacements).cuda()
    pred_embedding_device = torch.cat([pred_embedding_device, pose_embedding_device], dim = 1)
    total_points_num = pose_device.shape[0]
    pred_instance = torch.zeros((total_points_num,), dtype=torch.int32).cuda()
    background_mask = torch.tensor(np.ones((total_points_num,), dtype=np.int)).cuda() > 0
    pred_bw_device = torch.tensor(pred_bw).cuda()
    instance_flag = (torch.tensor(pred_semantic).cuda() < 2) * background_mask
    background_mask[instance_flag] = 0
    exsiting_offsets_device[instance_flag] = 0
    pred_instance[instance_flag] = -1
    bw = 1.2
    while instance_count <  expected_instance_num and background_mask.sum(0) > minimum_instance_size * 0.5:
        keypoint = torch.argmax(exsiting_offsets_device)
        pre_centroid = pred_embedding_device[keypoint,:]
        b = pred_bw_device[keypoint,:] * config['bw_relax_factor']
        #first iteration, only use the selected point feature as embedding feature
        embedding_distance = (pred_embedding_device - pre_centroid)
        prob = cross_modal_gaussian(embedding_distance[:,0:semantic_embedding_len],embedding_distance[:,semantic_embedding_len:semantic_embedding_len+3], b[0], b[1])
        instance_flag = (prob > 0.5) * background_mask
        local_embedding = pred_embedding_device[instance_flag]
        weight = gaussian(embedding_distance.norm(dim = 1),b.mean(dim=0))[instance_flag]
        centroid = weighted_mean(local_embedding, weight)
        pre_centroid = centroid
        for k in range(10):
            embedding_distance = (pred_embedding_device - pre_centroid)
            b = pred_bw_device[instance_flag,:].mean(dim = 0) * config['bw_relax_factor']
            prob = cross_modal_gaussian(embedding_distance[:,0:semantic_embedding_len],embedding_distance[:,semantic_embedding_len:semantic_embedding_len+3], b[0], b[1])
            instance_flag = (prob > 0.5) * background_mask
            local_embedding = pred_embedding_device[instance_flag]
            weight = gaussian(embedding_distance.norm(dim = 1),b.mean(dim=0))[instance_flag]
            centroid = weighted_mean(local_embedding, weight)
            mean_shift_vector = pre_centroid - centroid
            pre_centroid = centroid
            if(mean_shift_vector.norm(dim = 0) < 1e-5):
                break

        background_mask[instance_flag] = 0
        exsiting_offsets_device[instance_flag] = 0

        instance_count += 1
        if(instance_flag.sum(0) > minimum_instance_size ):
            pred_instance[instance_flag] = instance_count
        else:
            pred_instance[instance_flag] = -1
    return pred_instance.cpu().numpy()


def cross_modal_meanshift_clustering(exsiting_offsets, pred_embedding,pred_semantic, pred_displacements, xyz, pred_bw):

    instance_count = 0
    expected_instance_num = 300
    minimum_instance_size = 200
    exsiting_offsets_device = torch.tensor(exsiting_offsets).cuda()
    pred_embedding_device = torch.tensor(pred_embedding).cuda()
    pose_device = torch.tensor(xyz).cuda()
    pose_embedding_device = torch.tensor(xyz - pred_displacements).cuda()
    pred_embedding_device = torch.cat([pred_embedding_device, pose_embedding_device], dim = 1)
    background_points = pred_offsets.shape[0]
    pred_instance = torch.zeros((pred_offsets.shape[0],), dtype=torch.int32).cuda()
    background_mask = torch.tensor(np.ones((pred_offsets.shape[0],), dtype=np.int)).cuda() > 0
    pred_bw_device = torch.tensor(pred_bw).cuda()
    instance_flag = (torch.tensor(pred_semantic).cuda() < 2) * background_mask
    background_mask[instance_flag] = 0
    exsiting_offsets_device[instance_flag] = 0
    pred_instance[instance_flag] = -1

    # consider to remove ground and floor for scannet dataset?

    #remove floor and wall, the first two categories
    bw = 1.2
    while instance_count <  expected_instance_num and background_mask.sum(0) > minimum_instance_size * 0.5:
        keypoint = torch.argmax(exsiting_offsets_device)
        pre_centroid = pred_embedding_device[keypoint,:]
        b = pred_bw_device[keypoint,:] * config['bw_relax_factor']
        #first iteration, only use the selected point feature as embedding feature
        embedding_distance = (pred_embedding_device - pre_centroid)
        prob = cross_modal_gaussian(embedding_distance[:,0:semantic_embedding_len],embedding_distance[:,semantic_embedding_len:semantic_embedding_len+3], b[0], b[1])
        instance_flag = (prob > 0.5) * background_mask
        local_embedding = pred_embedding_device[instance_flag]
        weight = gaussian(embedding_distance.norm(dim = 1),b.mean(dim=0))[instance_flag]
        centroid = weighted_mean(local_embedding, weight)
        pre_centroid = centroid
        for k in range(10):
            embedding_distance = (pred_embedding_device - pre_centroid)
            b = pred_bw_device[instance_flag,:].mean(dim = 0) * config['bw_relax_factor']
            prob = cross_modal_gaussian(embedding_distance[:,0:semantic_embedding_len],embedding_distance[:,semantic_embedding_len:semantic_embedding_len+3], b[0], b[1])
            instance_flag = (prob > 0.5) * background_mask
            local_embedding = pred_embedding_device[instance_flag]
            weight = gaussian(embedding_distance.norm(dim = 1),b.mean(dim=0))[instance_flag]
            centroid = weighted_mean(local_embedding, weight)
            mean_shift_vector = pre_centroid - centroid
            pre_centroid = centroid
            if(mean_shift_vector.norm(dim = 0) < 1e-5):
                break

        if config['visualize_part'] :
            shiftX = np.zeros((xyz.shape[0],3), dtype = np.float32)
            shiftX[:,0] += 6
            shiftY = np.zeros((xyz.shape[0],3), dtype = np.float32)
            shiftY[:,1] += 6

            pcd_center = open3d.geometry.PointCloud()
            pcd_center.points = open3d.utility.Vector3dVector(xyz)
            gt_color = torch.zeros([xyz.shape[0],3],dtype=torch.float32)
            gt_color[instance_flag,1] = 1
            pcd_center.colors = open3d.utility.Vector3dVector(cm.hot(exsiting_offsets_device.data.cpu().numpy())[:,0,0:3])


            pcd_gt = open3d.geometry.PointCloud()
            pcd_gt.points = open3d.utility.Vector3dVector(xyz + shiftX)
            gt_color = torch.zeros([xyz.shape[0],3],dtype=torch.float32)
            gt_color[instance_flag,1] = 1
            pcd_gt.colors = open3d.utility.Vector3dVector(gt_color.data.numpy())

            pcd_pred_instance = open3d.geometry.PointCloud()
            pcd_pred_instance.points = open3d.utility.Vector3dVector(xyz + shiftX*2)
            gt_color = cm.hot(prob.data.cpu().numpy())[:,0:3]
            pcd_pred_instance.colors = open3d.utility.Vector3dVector(gt_color)


            d1 = torch.norm(embedding_distance[:,0:semantic_embedding_len],dim=1)
            d2 = torch.norm(embedding_distance[:,semantic_embedding_len:semantic_embedding_len+3],dim=1)
            p1 = exp(-((d1/ b[0]))**2)
            p2 = exp(-((d2/ b[1]))**2)

            pcd_prob1 = open3d.geometry.PointCloud()
            pcd_prob1.points = open3d.utility.Vector3dVector(xyz + shiftX*3)
            gt_color = cm.hot(p1.data.cpu().numpy())[:,0:3]
            pcd_prob1.colors = open3d.utility.Vector3dVector(gt_color)

            pcd_prob2 = open3d.geometry.PointCloud()
            pcd_prob2.points = open3d.utility.Vector3dVector(xyz + shiftX*4)
            gt_color = cm.hot(p2.data.cpu().numpy())[:,0:3]
            pcd_prob2.colors = open3d.utility.Vector3dVector(gt_color)

            open3d.visualization.draw_geometries([pcd_pred_instance,pcd_prob1,pcd_prob2])
#            open3d.visualization.draw_geometries([pcd_center, pcd_gt,pcd_pred_instance,pcd_prob1,pcd_prob2])
    #            print(k,mean_shift_vector.norm(dim = 0).item(), instance_flag.sum(0).item())

        background_mask[instance_flag] = 0
        exsiting_offsets_device[instance_flag] = 0

        if False:
            cluster_poses = pose_device[instance_flag,:]
            detailed_instances = DBSCAN(eps = 0.1, min_samples = 10 , n_jobs = 1).fit(cluster_poses.cpu().numpy()).labels_
            detailed_instances = torch.tensor(detailed_instances)
            pos_index = torch.tensor(np.arange(0,instance_flag.shape[0]))[instance_flag]
            pcds = []
            for k in range(torch.max(detailed_instances)+1):
                if((detailed_instances == k).sum(0) > minimum_instance_size):
                    instance_count += 1
                    pred_instance[pos_index[detailed_instances == k]] = instance_count
                else:
                    pred_instance[pos_index[detailed_instances == k]] = -1
                """
                pcd_gt = open3d.geometry.PointCloud()
                pcd_gt.points = open3d.utility.Vector3dVector(pose_device[instance_flag,:][detailed_instances==k,:].cpu().numpy())
                pcd_gt.colors = open3d.utility.Vector3dVector(np.random.rand(1,3).repeat((detailed_instances==k).sum(0),axis=0))
                pcds.append(pcd_gt)
                """
#            open3d.visualization.draw_geometries(pcds)
            pred_instance[pos_index[detailed_instances == -1]] = -1
        else:
            instance_count += 1
            if(instance_flag.sum(0) > minimum_instance_size ):
                pred_instance[instance_flag] = instance_count
            else:
                pred_instance[instance_flag] = -1
    return pred_instance.cpu().numpy()



def meanshift_clustering(exsiting_offsets, pred_embedding,pred_semantic, pred_displacements, xyz):

    instance_count = 0
    expected_instance_num = 300
    minimum_instance_size = 200
    exsiting_offsets_device = torch.tensor(exsiting_offsets).cuda()
    pred_embedding_device = torch.tensor(pred_embedding).cuda()
    pose_device = torch.tensor(xyz).cuda()
    pose_embedding_device = torch.tensor(xyz - pred_displacements).cuda()
    pred_embedding_device = torch.cat([pred_embedding_device, pose_embedding_device], dim = 1)
    background_points = pred_offsets.shape[0]
    pred_instance = torch.zeros((pred_offsets.shape[0],), dtype=torch.int32).cuda()
    background_mask = torch.tensor(np.ones((pred_offsets.shape[0],), dtype=np.int)).cuda() > 0

    instance_flag = (torch.tensor(pred_semantic).cuda() < 2) * background_mask
    background_mask[instance_flag] = 0
    exsiting_offsets_device[instance_flag] = 0
    pred_instance[instance_flag] = -1

    # consider to remove ground and floor for scannet dataset?

    #remove floor and wall, the first two categories
    bw = 1.2
    while instance_count <  expected_instance_num and background_mask.sum(0) > minimum_instance_size * 0.5:
        keypoint = torch.argmax(exsiting_offsets_device)
        pre_centroid = pred_embedding_device[keypoint,:]
        #first iteration, only use the selected point feature as embedding feature
        embedding_distance = torch.norm(pred_embedding_device - pre_centroid, dim = 1)
        weight = gaussian(embedding_distance, bw)
        instance_flag = (embedding_distance  < bw) * background_mask
        local_embedding = pred_embedding_device[instance_flag]
        local_weight = weight[instance_flag]
        centroid = (local_weight.view(-1,1).expand(-1,pred_embedding_device.shape[1]) * local_embedding).sum(0)  / local_weight.sum(0)
        mean_shift_vector = centroid - pre_centroid
        for k in range(5):
            embedding_distance = torch.norm(pred_embedding_device - pre_centroid, dim = 1)
            weight = gaussian(embedding_distance, bw)
            instance_flag = (embedding_distance  < bw) * background_mask
            local_embedding = pred_embedding_device[instance_flag]
            local_weight = weight[instance_flag]
            centroid = (local_weight.view(-1,1).expand(-1,pred_embedding_device.shape[1]) * local_embedding).sum(0)  / local_weight.sum(0)
            mean_shift_vector = centroid - pre_centroid
            pre_centroid = centroid
#            print(mean_shift_vector.norm(dim = 0).item(), instance_flag.sum(0).item())

        background_mask[instance_flag] = 0
        exsiting_offsets_device[instance_flag] = 0

        if False:
            cluster_poses = pose_device[instance_flag,:]
            detailed_instances = DBSCAN(eps = 0.1, min_samples = 10 , n_jobs = 1).fit(cluster_poses.cpu().numpy()).labels_
            detailed_instances = torch.tensor(detailed_instances)
            pos_index = torch.tensor(np.arange(0,instance_flag.shape[0]))[instance_flag]
            pcds = []
            for k in range(torch.max(detailed_instances)+1):
                if((detailed_instances == k).sum(0) > minimum_instance_size):
                    instance_count += 1
                    pred_instance[pos_index[detailed_instances == k]] = instance_count
                else:
                    pred_instance[pos_index[detailed_instances == k]] = -1
                """
                pcd_gt = open3d.geometry.PointCloud()
                pcd_gt.points = open3d.utility.Vector3dVector(pose_device[instance_flag,:][detailed_instances==k,:].cpu().numpy())
                pcd_gt.colors = open3d.utility.Vector3dVector(np.random.rand(1,3).repeat((detailed_instances==k).sum(0),axis=0))
                pcds.append(pcd_gt)
                """
#            open3d.visualization.draw_geometries(pcds)
            pred_instance[pos_index[detailed_instances == -1]] = -1
        else:
            instance_count += 1
            if(instance_flag.sum(0) > minimum_instance_size ):
                pred_instance[instance_flag] = instance_count
            else:
                pred_instance[instance_flag] = -1
    return pred_instance.cpu().numpy()



def write_results(instance_segmentation_info ):

    file = instance_segmentation_info['file']
    proposals = instance_segmentation_info['proposals']
    pred_semantic = instance_segmentation_info['pred_semantic']
    volumetric_sizes_per_category = instance_segmentation_info['volumetric_sizes_per_category']
    scene_id = file[file.find('scene'):file.find('scene')+12]

    instance_index = 0
    path = 'predictions'
    sub_dir = 'predicted_masks'
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path + '/' + sub_dir):
        os.mkdir(path + '/' + sub_dir)
    with open(path + '/' + scene_id + '.txt','w') as f:
        for label, proposal_per_class in enumerate(proposals):
            for proposal in proposal_per_class:
                class_label_nyu = VALID_CLASS_IDS[label]
                confidence = 1 #(pred_semantic[proposal] == label).sum(0) / proposal.sum(0)
                if(confidence > 0.0):
                    f.write(sub_dir + '/' + scene_id + '_' + str(instance_index) + '.txt ' + str(class_label_nyu) + ' ' + str(confidence) + '\n')
                    np.savetxt(path + '/' + sub_dir + '/'  + scene_id + '_' + str(instance_index) + '.txt', proposal, fmt='%d')
                    instance_index += 1


def joint_instance_semantic_prediction(files, args):

    ori_pts = []
    gt_pts = []
    joint_pts = []
    train_writer = SummaryWriter(comment="joint_instance_segmentation")
    path = 'semantic_prediction'
    if not os.path.exists(path):
        os.mkdir(path)
    for file_index,file in enumerate(files):
        print(files)
        results = np.load(file)
        pred_semantic = results['pred_semantic']
        pred_embedding = results['pred_embedding']
        true_semantic = results['true_semantic']
        true_instance = results['true_instance']
        true_offsets = results['true_offsets']
        pred_offsets = results['pred_offsets']
        true_displacements = results['true_displacements']
        pred_displacements = results['pred_displacements']
        pred_bw = results['pred_bw']
        regions = results['regions']
        pred_occupancy_size = results['occupancy']
        true_occupancy_size = results['true_occupancy']
        pred_semantic_probability = results['pred_semantic_probability']

        cost = torch.nn.MSELoss()(torch.tensor(true_offsets), torch.tensor(pred_offsets))
        """
        true_offset_distance = np.sqrt(-np.log(true_offsets) * (sigma ** 2))
        pred_offset_distance = np.sqrt(-np.log(pred_offsets) * (sigma ** 2))
        true_offset_distance[np.isinf(true_offset_distance)] = 2.0
        pred_offset_distance[np.isinf(pred_offset_distance)] = 2.0
        true_offset_distance[true_offset_distance > 2.0] = 2.0
        pred_offset_distance[pred_offset_distance > 2.0] = 2.0

        dist = np.abs(true_offset_distance - pred_offset_distance ).mean()
        print("distance: ", dist, "    cost: ", cost.item())
        """
        xyz = results['xyz'] / 50
        features = results['feature']

        exsiting_offsets  = pred_offsets.copy()
        instance_count = 0
        joint, ori, gt  = joint_semantic_instance_segmentation(pred_semantic_probability,true_semantic,exsiting_offsets, pred_embedding, pred_semantic, pred_displacements, xyz, pred_bw,regions)

        pred_semantic_label = to_origianl_label(joint)
        scene_id = file[file.find('scene'):file.find('scene')+12]
        if args.evaluate:
            print('save semantic prediction: ', path + '/' + scene_id + '.txt')
            np.savetxt(path + '/' + scene_id + '.txt', pred_semantic_label, fmt='%d')
        else:
            iou_joint = evaluate_single_scan(joint,gt,None, file_index, 20, topic = 'single')
            iou_pred = evaluate_single_scan(ori,gt,None, file_index, 20, topic = 'single')
            train_writer.add_scalar("joint_opt/gain", iou_joint - iou_pred, file_index)
            print(file_index, " ", file,   "predictions joint/pred: ", iou_joint, iou_pred)
            gt_pts.append(gt)
        joint_pts.append(joint)
        ori_pts.append(ori)

    # joint_pts = np.concatenate(joint_pts)
    # ori_pts = np.concatenate(ori_pts)

    # gt_pts = np.concatenate(gt_pts)
    # print("joint optimization:")
    # evaluate_single_scan(joint_pts,gt_pts,train_writer, 0, 20)
    # print("pure semantic:")
    # evaluate_single_scan(ori_pts,gt_pts,train_writer, 0, 20)

if __name__ == '__main__':

    torch.manual_seed(100)  # cpu
    torch.cuda.manual_seed(100)  # gpu
    np.random.seed(100)  # numpy
    torch.backends.cudnn.deterministic = True  # cudnn
    args = get_args()
    # choose dataset
    if args.dataset == 'scannet':
        candidate_path = './datasets/scannet_data/instance/test/*.npz'

        # if args.evaluate:
        #     candidate_path = './datasets/scannet_data/instance/test/*.npz'
        # else:
        #     candidate_path = './datasets/scannet_data/instance/test/*.npz'
        class_num = 20
        volumetric_sizes_per_category = np.loadtxt("sizes_scannet.txt")
    elif args.dataset == 'stanford3d':
        candidate_path = './datasets/stanford_data/instance/full/Area_5_*.npz'
        volumetric_sizes_per_category = np.loadtxt("sizes.txt")
        class_num = 14
    else:
        raise NotImplementedError

    files = sorted(glob.glob(candidate_path))
    print(candidate_path,files)
    sigma = args.regress_sigma
    total = np.zeros(class_num )

    fps   = [ [] for i in range(class_num )]
    tps   = [ [] for i in range(class_num )]


    fps_1   = [ [] for i in range(class_num )]
    tps_1   = [ [] for i in range(class_num )]

    scene_instance_segmentation_info = []


    proposal_info = [[] for i in range(class_num )]
    instance_info = [[] for i in range(class_num )]

    pts_per_class = np.zeros([len(files),class_num,1000]) - 1
    predict_pts_per_class = np.zeros([len(files),class_num,1000]) - 1
    statistics = np.zeros([len(files),class_num,5,1000]) - 1 # initialized as -1, future process

    joint_instance_semantic_prediction(files,args)
    for file_index,file in enumerate(files):
        #file = files[76]
        print("processing: ", file)
        results = np.load(file)
        pred_semantic = results['pred_semantic']
        pred_embedding = results['pred_embedding']
        true_semantic = results['true_semantic']
        true_instance = results['true_instance']
        true_offsets = results['true_offsets']
        pred_offsets = results['pred_offsets']
        true_displacements = results['true_displacements']
        pred_displacements = results['pred_displacements']
        pred_bw = results['pred_bw']
        regions = results['regions']
        pred_occupancy_size = results['occupancy']
        true_occupancy_size = results['true_occupancy']
        pred_semantic_probability = results['pred_semantic_probability']

        cost = torch.nn.MSELoss()(torch.tensor(true_offsets), torch.tensor(pred_offsets))
        """
        true_offset_distance = np.sqrt(-np.log(true_offsets) * (sigma ** 2))
        pred_offset_distance = np.sqrt(-np.log(pred_offsets) * (sigma ** 2))
        true_offset_distance[np.isinf(true_offset_distance)] = 2.0
        pred_offset_distance[np.isinf(pred_offset_distance)] = 2.0
        true_offset_distance[true_offset_distance > 2.0] = 2.0
        pred_offset_distance[pred_offset_distance > 2.0] = 2.0

        dist = np.abs(true_offset_distance - pred_offset_distance ).mean()
        print("distance: ", dist, "    cost: ", cost.item())
        """
        xyz = results['xyz'] / 50
        features = results['feature']

        exsiting_offsets  = pred_offsets.copy()
        instance_count = 0
        pred_semantic, ori, gt  = joint_semantic_instance_segmentation(pred_semantic_probability,true_semantic,exsiting_offsets, pred_embedding, pred_semantic, pred_displacements, xyz, pred_bw,regions)
#        evaluate_single_scan(pred_semantic,gt,None, 0, 20)
#        evaluate_single_scan(ori,gt,None, 0, 20)
#        pred_instance = select_instances(exsiting_offsets,pred_embedding, pred_semantic)
        if args.dataset == 'scannet':
            if config['use_merge'] == True:
                pred_instance = region_based_cross_modal_meanshift_merging(exsiting_offsets, pred_embedding, pred_semantic, pred_displacements, xyz, pred_bw, regions,pred_occupancy_size)
            else:
                pred_instance = region_based_cross_modal_meanshift_clustering(exsiting_offsets, pred_embedding, pred_semantic, pred_displacements, xyz, pred_bw,regions)
#                pred_instance = cross_modal_meanshift_clustering_simple(exsiting_offsets, pred_embedding, pred_semantic, pred_displacements, xyz, pred_bw)
        else:
                pred_instance = cross_modal_meanshift_clustering(exsiting_offsets, pred_embedding, pred_semantic, pred_displacements, xyz, pred_bw)

        shiftX = np.zeros((xyz.shape[0],3), dtype = np.float32)
        shiftX[:,0] += 6
        shiftY = np.zeros((xyz.shape[0],3), dtype = np.float32)
        shiftY[:,1] += 6



        #calculate tp here.
        proposals = [[] for i in range(class_num )]
        for gid in np.unique(pred_instance):
            if gid >= 0:
                indices = (pred_instance == gid)
                cls = int(stats.mode(pred_semantic[indices])[0])
                outlier_ratio = 0.25
                if(cls == 19):
                    outlier_ratio = 0.1
                size = indices.sum(0)
                #size = (pred_semantic[indices] == cls).sum(0)
                min_pts_threshold = outlier_ratio * volumetric_sizes_per_category[cls]

                occupancy_ratio = indices.sum(0) / np.exp(np.mean(pred_occupancy_size[indices]))
                if (size > min_pts_threshold) and (occupancy_ratio > config['occupancy_ratio_threshold']) and (occupancy_ratio < config['occupancy_ratio_threshold_up']): # remove small instances
#                if (occupancy_ratio > config['occupancy_ratio_threshold']) and (occupancy_ratio < config['occupancy_ratio_threshold_up']):
                    proposals[cls] += [indices]
                """
                if config['occupancy_ratio_threshold'] > 0:
                    if occupancy_ratio > config['occupancy_ratio_threshold']:
                        proposals[cls] += [indices]
                else:
                """

        instance_segmentation_info = {}
        instance_segmentation_info['file'] = file
        instance_segmentation_info['proposals'] = proposals
        instance_segmentation_info['pred_semantic'] = pred_semantic
        instance_segmentation_info['volumetric_sizes_per_category'] = volumetric_sizes_per_category
        scene_instance_segmentation_info.append(instance_segmentation_info)
        instances = [[] for i in range(class_num )]
        for gid in np.unique(true_instance):
            indices = (true_instance == gid)
            cls = int(stats.mode(true_semantic[indices])[0])
            if(cls >= 0):
                instances[cls] += [indices]

        for i in range(class_num):
            for iid,v in enumerate(instances[i]):
                pts_per_class[file_index, i,iid] = np.sum(v)
            for iid,u in enumerate(proposals[i]):
                predict_pts_per_class[file_index,i,iid] = np.sum(u)

        #merge point clouds that are too close to each other, to reduce the number of outliers

        for i in range(class_num):
            if i not in [5,9,10,11,14]:
                continue
            remove_point_cloud = []
            new_proposals = []
            for iid,u in enumerate(proposals[i]):
                if(iid in remove_point_cloud):
                    continue
                else:
                    pos1 = (xyz - pred_displacements)[u,:]
                    for candidate in range(iid+1, len(proposals[i])):
                        v = proposals[i][candidate]
                        pos2 = (xyz - pred_displacements)[v,:]
                        dists_a_to_b, corrs_a_to_b = pcu.point_cloud_distance(pos1, pos2)
                        inlier_num = (dists_a_to_b < 0.05).sum(0)
                        if(inlier_num > 0.2 * u.sum(0)):
                            u = u + v
                            remove_point_cloud.append(candidate)
                    new_proposals.append(u)
            print(len(proposals[i]), len(new_proposals))
            proposals[i] = new_proposals
    #        proposals[i] = new_proposals
        for i in range(class_num):
            total[i] += len(instances[i])
            tp = np.zeros(len(proposals[i]))
            fp = np.zeros(len(proposals[i]))
            pred_instance_size = np.zeros(len(proposals[i]))
            gt = np.zeros(len(instances[i]))
            for pid, u in enumerate(proposals[i]):
                overlap = 0.0
                detected = 0
                for iid, v in enumerate(instances[i]):
                    iou = np.sum((u & v)) / np.sum((u | v))
                    if iou > overlap:
                        overlap = iou
                        detected = iid
                if overlap >= 0.25:
                    tp[pid] = 1
                else:
                    fp[pid] = 1
                pred_instance_size[pid] = np.sum(u)
            tps[i] += [tp]
            fps[i] += [fp]

            tp = np.zeros(len(proposals[i]))
            fp = np.zeros(len(proposals[i]))
            pred_instance_size = np.zeros(len(proposals[i]))
            pred_instance_occupancy = np.zeros(len(proposals[i]))
            true_instance_occupancy = np.zeros(len(proposals[i]))
            pred_instance_confidence = np.zeros(len(proposals[i]))
            pred_overlap = np.zeros(len(proposals[i]))


            gt = np.zeros(len(instances[i]))
            for pid, u in enumerate(proposals[i]):
                overlap = 0.0
                detected = 0
                for iid, v in enumerate(instances[i]):
                    iou = np.sum((u & v)) / np.sum((u | v))
                    if iou > overlap:
                        overlap = iou
                        detected = iid
                if overlap >= 0.5:
                    tp[pid] = 1
                else:
                    fp[pid] = 1
                pred_instance_size[pid] =  u.sum(0)#(pred_semantic[u] == i).sum(0)
                pred_instance_occupancy[pid] = u.sum(0) / np.exp(np.mean(pred_occupancy_size[u]))
                true_instance_occupancy[pid] = u.sum(0) / np.exp(np.mean(true_occupancy_size[u]))
                pred_instance_confidence[pid] = (pred_semantic[u] == i).sum(0) / u.sum(0)
                pred_overlap[pid] = overlap

            tps_1[i] += [tp]
            fps_1[i] += [fp]
            print("**************class ", i, ": ", CLASS_LABELS[i], " @mAP0.5**************")
            print("true instances ", len(instances[i]))
            print("true positive: ", tp)
            print("false positive: ", fp)
            print("pred_instance_size: ", pred_instance_size)
            print("pred_instance_occupancy: ", pred_instance_occupancy)
            print("true_instance_occupancy: ", true_instance_occupancy)
            print("pred_instance_prob: ", pred_instance_confidence)
            print("pred_instance_miou: ", pred_overlap)
            statistics[file_index,i,0,:len(proposals[i])] = tp
            statistics[file_index,i,1,:len(proposals[i])] = fp
            statistics[file_index,i,2,:len(proposals[i])] = pred_instance_size
            statistics[file_index,i,3,:len(proposals[i])] = pred_instance_confidence
            statistics[file_index,i,4,:len(proposals[i])] = pred_overlap

        ## visualize point cloud     here
        if config['visualize']:
            pcd_gt = open3d.geometry.PointCloud()
            pcd_gt.points = open3d.utility.Vector3dVector(xyz+ shiftX * 2)
            random_colors = np.random.rand(np.unique(true_instance).size,3)
            mask = np.unique(true_instance,False,True)[1]
            gt_color = random_colors[mask,:]
            gt_color[true_semantic < 0, :] = 0
            pcd_gt.colors = open3d.utility.Vector3dVector(gt_color)



            pcd_occupancy_instance = open3d.geometry.PointCloud()
            pcd_occupancy_instance.points = open3d.utility.Vector3dVector(xyz + shiftX * 2 + shiftY)
            labels = np.unique(true_instance ,False,True)[0]
            occupancy_diff = np.zeros(true_instance.shape[0])
            for index in range(labels.shape[0]):
                occupancy_diff[true_instance == index] = np.mean(np.abs(true_occupancy_size[true_instance == index] - pred_occupancy_size[true_instance == index]))
                print(np.mean(np.abs(true_occupancy_size[true_instance == index] - pred_occupancy_size[true_instance == index])))
                print('class: ', stats.mode(true_semantic[true_instance == index])[0])
            gt_color = cm.hot(occupancy_diff)[:,0:3]
            pcd_occupancy_instance.colors = open3d.utility.Vector3dVector(gt_color)

            pcd_leak_instance = open3d.geometry.PointCloud()
            pcd_leak_instance.points = open3d.utility.Vector3dVector(xyz + shiftX + shiftY)
            random_colors = np.random.rand(np.unique(pred_instance).size,3)

            mask = np.unique(pred_instance ,False,True)[1]
            gt_color = random_colors[mask,:]
            gt_color[pred_semantic < 2, :] = 0
            labels = np.unique(pred_instance ,False,True)[0]
            for index in range(labels.shape[0]):
                if((pred_instance == index).sum() > 200):
                    gt_color[pred_instance == index,:] = 0
            pcd_leak_instance.colors = open3d.utility.Vector3dVector(gt_color)


            pcd_pred_instance = open3d.geometry.PointCloud()
            pcd_pred_instance.points = open3d.utility.Vector3dVector(xyz)
            random_colors = np.random.rand(np.unique(pred_instance).size,3)
            mask = np.unique(pred_instance ,False,True)[1]
            gt_color = random_colors[mask,:]
#            gt_color[:, :] = 0
            pcd_pred_instance.colors = open3d.utility.Vector3dVector(gt_color)
            """
            true_instance_num = max(true_instance)
            for i in range(true_instance_num):
                if(true_semantic[true_instance == i][0] <= 1):
                    continue
                instance_flag = (true_instance == i)
                center_embedding = (pred_embedding[instance_flag ,:]).mean(axis=0)
                center_pos = xyz[true_instance == i,:].mean(axis = 0)
                embedding_offset = pred_embedding - center_embedding
                spatial_offset = xyz - pred_displacements - center_pos
                delta_1 = pred_bw[instance_flag,0].mean(axis = 0) * config['bw_relax_factor']
                delta_2 = pred_bw[instance_flag,1].mean(axis = 0) * config['bw_relax_factor']
                d1 = np.linalg.norm(embedding_offset,axis =1 )
                d2 = np.linalg.norm(spatial_offset, axis = 1)
                weight = np.exp(-((d1/delta_1))**2 - ((d2/delta_2))**2)
                gt_color = (cm.hot(weight)[:,0:3])
                gt_color[true_semantic < 0, :] = 0
                threshold = 0.6
                gt_color[weight < threshold , :] = 0
                u = weight >= threshold
                v = instance_flag
                # computing ious between u & v
                iou = np.sum((u & v)) / np.sum((u | v))
                print('predicted iou: ', iou, u.sum(), v.sum(), (u&v).sum(), (u|v).sum())
                gt_color[v,:] = 0
                pcd_pred_instance.colors = open3d.utility.Vector3dVector(gt_color)

                pcd_gt = open3d.geometry.PointCloud()
                pcd_gt.points = open3d.utility.Vector3dVector(xyz+ shiftX * 2)
                random_colors = np.random.rand(np.unique(true_instance).size,3)
                mask = np.unique(true_instance,False,True)[1]
                gt_color = random_colors[mask,:]
                gt_color[true_semantic < 0, :] = 0
                gt_color[true_instance != i , :] = 0
                pcd_gt.colors = open3d.utility.Vector3dVector(gt_color)
                open3d.visualization.draw_geometries([pcd_pred_instance ,pcd_gt])

            #open3d.utility.Vector3dVector(pred_displacements + 0.5)
            #open3d.utility.Vector3dVector(cm.hot(pred_offsets)[:,0,0:3])#
            #open3d.utility.Vector3dVector(gt_color)#

            """
            pcd_color = open3d.geometry.PointCloud()
            pcd_color.points = open3d.utility.Vector3dVector(xyz+ shiftX * 3)
            random_colors = np.random.rand(np.unique(pred_instance).size,3)
            mask = np.unique(pred_instance ,False,True)[1]
            gt_color = random_colors[mask,:]
            pcd_color.colors = open3d.utility.Vector3dVector((1 + features[:,0:3]) / 2)

            open3d.visualization.draw_geometries([pcd_pred_instance,pcd_gt,pcd_color,pcd_leak_instance,pcd_occupancy_instance])
        """
        if True:
            pcd_pred_instance = open3d.geometry.PointCloud()
            pcd_pred_instance .points = open3d.utility.Vector3dVector(xyz + shiftX)

            random_colors = np.random.rand(np.unique(pred_instance).size,3)
            mask = np.unique(pred_instance,False,True)[1]
            gt_color = random_colors[mask,:]
            gt_color[pred_instance == -1, :] = 0
            pcd_pred_instance.colors = open3d.utility.Vector3dVector(gt_color)

            pcd_gt = open3d.geometry.PointCloud()
            pcd_gt.points = open3d.utility.Vector3dVector(xyz)
            random_colors = np.random.rand(np.unique(true_instance).size,3)
            mask = np.unique(true_instance,False,True)[1]
            gt_color = random_colors[mask,:]
            pcd_gt.colors = open3d.utility.Vector3dVector(gt_color)


            pcd_ori = open3d.geometry.PointCloud()
            pcd_ori.points = open3d.utility.Vector3dVector(xyz + shiftY)
            pcd_ori.colors = open3d.utility.Vector3dVector(cm.hot(pred_offsets)[:,0,0:3])


            pcd_gt_hot = open3d.geometry.PointCloud()
            pcd_gt_hot.points = open3d.utility.Vector3dVector(xyz + shiftX + shiftY)
            pcd_gt_hot.colors = open3d.utility.Vector3dVector(cm.hot(true_offsets)[:,0,0:3])

#            pcd_ori.colors = open3d.utility.Vector3dVector(cm.hot(pred_offsets)[:,0,0:3])

        else:
            pcd_gt = open3d.geometry.PointCloud()
            pcd_gt.points = open3d.utility.Vector3dVector(xyz + shift)

            random_colors = np.random.rand(np.unique(true_instance).size,3)
            mask = np.unique(true_instance,False,True)[1]
            gt_color = random_colors[mask,:]
            pcd_gt.colors = open3d.utility.Vector3dVector(gt_color)



            random_projection = np.random.rand(32,3)
            visualization = np.matmul(pred_embedding ,random_projection)
            visualization  = (visualization - np.min(visualization))/np.max(visualization- np.min(visualization))
            pcd_ori = open3d.geometry.PointCloud()
            pcd_ori.points = open3d.utility.Vector3dVector(xyz)
            pcd_ori.colors = open3d.utility.Vector3dVector(visualization)
            print(xyz.shape, visualization.shape)

            pcd_color = open3d.geometry.PointCloud()
            pcd_color.points = open3d.utility.Vector3dVector(xyz + shift*4)
            pcd_color.colors = open3d.utility.Vector3dVector((1 + features[:,0:3]) / 2)

#        open3d.visualization.draw_geometries([pcd_pred_instance, pcd_gt, pcd_gt_hot, pcd_ori])
        """

    p = np.zeros(class_num)
    r = np.zeros(class_num)
    for i in range(class_num):
        tp = np.concatenate(tps[i], axis=0)
        fp = np.concatenate(fps[i], axis=0)
        tp = np.sum(tp)
        fp = np.sum(fp)
        p[i] = tp / (tp + fp)
        r[i] = tp / total[i]
        print(i, tp,total[i],fp,p[i] ,r[i])
        print("precision")
    print(p)
    print("recall")
    print(r)
    print("mAP@0.25: ", p[np.isfinite(p)].mean())
    print("mRecall@0.25: ", r[np.isfinite(r)].mean())


    scipy.io.savemat('instance_stat.mat',{'stats':statistics,'pts_per_class':pts_per_class,'predict_pts_per_class':predict_pts_per_class})
    p = np.zeros(class_num)
    r = np.zeros(class_num)
    for i in range(class_num):
        tp = np.concatenate(tps_1[i], axis=0)
        fp = np.concatenate(fps_1[i], axis=0)
        tp = np.sum(tp)
        fp = np.sum(fp)
        p[i] = tp / (tp + fp)
        r[i] = tp / total[i]
        print(i, tp,total[i],fp,p[i] ,r[i])
        print("precision")
    print(p)
    print("recall")
    print(r)
    print("mAP@0.5: ", p[np.isfinite(p)].mean())
    print("mRecall@0.5: ", r[np.isfinite(r)].mean())

    p = mp.Pool(processes=mp.cpu_count()-1)
    p.map(write_results, scene_instance_segmentation_info)
    p.close()
    p.join()