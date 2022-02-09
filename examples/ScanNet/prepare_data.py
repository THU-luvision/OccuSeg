# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import open3d as o3d
import glob, plyfile, numpy as np, multiprocessing as mp, torch
import copy
import numpy as np
import json
import pdb
import os


#CLASS_LABELS = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
CLASS_LABELS = ['wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink', 'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window', 'shower curtain', 'refrigerator', 'picture', 'cabinet', 'otherfurniture']
VALID_CLASS_IDS = np.array([1,2,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
NYUID_TO_LABEL = {}
LABEL_TO_NYUID = {}
NYUID_TO_SEGID = {}
for i in range(len(VALID_CLASS_IDS)):
    LABEL_TO_NYUID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
    NYUID_TO_LABEL [VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
    NYUID_TO_SEGID[VALID_CLASS_IDS[i]] = i

SELECTED_LABEL_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]

LABEL_ID_TO_CLASS_ID = {
#    9: 3 #force 'desk' to be the same class_id as 'table'
}
for i, label_id in enumerate(SELECTED_LABEL_IDS):
    if label_id not in LABEL_ID_TO_CLASS_ID:
        LABEL_ID_TO_CLASS_ID[label_id] = i

UNKNOWN_ID = -100

g_label_names = ['unannotated', 'wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink', 'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window', 'shower curtain', 'refrigerator', 'picture', 'cabinet', 'otherfurniture']
def get_raw2scannet_label_map():
    lines = [line.rstrip() for line in open('scannet-labels.combined.tsv')]
    lines = lines[1:]
    raw2scannet = {}
    for i in range(len(lines)):
        label_classes_set = set(g_label_names)
        elements = lines[i].split('\t')
        raw_name = elements[0]
        nyu40_name = elements[6]
        if nyu40_name not in label_classes_set:
            raw2scannet[raw_name] = 'unannotated'
        else:
            raw2scannet[raw_name] = nyu40_name
    return raw2scannet

raw2scannet = get_raw2scannet_label_map()

def f_from_test(fn):
    fn2 = fn[:-3] + 'labels.ply'

    mesh = o3d.io.read_triangle_mesh(fn)
    coords = np.ascontiguousarray(mesh.vertices)
    colors = np.ascontiguousarray(mesh.vertex_colors) - 0.5
    a = plyfile.PlyData().read(fn2).elements[0]['label']
#    class_label = np.array(list(
#        map(lambda label_id: LABEL_ID_TO_CLASS_ID[label_id] if label_id in LABEL_ID_TO_CLASS_ID else UNKNOWN_ID,
#            a.elements[0]['label'])))

    w = np.zeros((len(coords),2),  dtype = np.int32)
    w[:,:] = UNKNOWN_ID

    """
    json_file_name = fn[:-6] + '.aggregation.json'
    json_file = open(json_file_name)
    objects = json.load(json_file)['segGroups']
 

    for object in objects:
        segments = object['segments']
        label = object['label']
        segment_id = object['id']
        if label in CLASS_LABELS:
            class_id = NYUID_TO_SEGID[LABEL_TO_NYUID[label]]
            print(label,class_id)
        else:
            class_id = UNKNOWN_ID
            print(label,class_id)
        for segment in segments:
            indices = (instance_label == segment)
            w[indices,0] = class_id
            w[indices,1] = segment_id
    """
    json_file = open(fn[:-3]+'regions.json')
    region = np.asarray(json.load(json_file)['segIndices'])
    all={
        "coords":coords,
        "colors":colors,
        "w":w,
        'region': region,
    }


    if(fn.find('scans_test') < 0 and fn.find('scans_val') < 0):
        dst = './datasets/scannet_data/instance/train/'
    elif(fn.find('scans_val') >= 0):
        dst = './datasets/scannet_data/instance/val/'
    elif(fn.find('scans_test') >= 0):
        dst = './datasets/scannet_data/instance/test/'
    else:
        print('error in input data organization! expected three kind of inputs: scans/ scans/val scans/test as the original scannet')
        raise NotImplementedError

    fileName = dst  +  fn[len(fn)-27:len(fn)-4] + '_instance.pth'
    torch.save(all, fileName)
    print(fileName)
def f(fn):
    fn2 = fn[:-3] + 'labels.ply'
    convert = 'scannet-labels.combined.tsv'
    mesh = o3d.io.read_triangle_mesh(fn)
    coords = np.ascontiguousarray(mesh.vertices)
    colors = (np.ascontiguousarray(mesh.vertex_colors) - 0.5) * 2

    w = np.zeros((len(coords),2),  dtype = np.int32)
    w[:,:] = UNKNOWN_ID

    json_file_name = fn[:-6] + '.aggregation.json'
    if os.path.exists(json_file_name):
        json_file = open(json_file_name)
        labels = plyfile.PlyData().read(fn2).elements[0]['label']
        class_label = np.array(list(
            map(lambda label_id: LABEL_ID_TO_CLASS_ID[label_id] if label_id in LABEL_ID_TO_CLASS_ID else UNKNOWN_ID,
                labels)))

        objects = json.load(json_file)['segGroups']
        json_file_name = fn[:-3] + '0.010000.segs.json'
        json_file = open(json_file_name)
        data = json.load(json_file)
        instance_label = np.asarray(data['segIndices'])


        for object in objects:
            segments = object['segments']
            label = object['label']
            segment_id = object['id']
            if label not in raw2scannet:
                label = 'unannotated'
            else:
                label = raw2scannet[label]
            if label in CLASS_LABELS:
                class_id = NYUID_TO_SEGID[LABEL_TO_NYUID[label]]
            else:
                class_id = UNKNOWN_ID
            for segment in segments:
                indices = (instance_label == segment)
                w[indices,0] = class_label[indices]
                w[indices,1] = segment_id

    json_file = open(fn[:-3]+'regions.json')
    region = np.asarray(json.load(json_file)['segIndices'])
    all={
        "coords":coords,
        "colors":colors,
        "w":w,
        'region': region,
    }

    if(fn.find('train') >= 0):
        dst = './datasets/scannet_data/instance/train/'
    elif(fn.find('val') >= 0):
        dst = './datasets/scannet_data/instance/val/'
    elif(fn.find('test') >= 0):
        dst = './datasets/scannet_data/instance/test/'
    else:
        print('error in input data organization! expected three kind of inputs: scans/ scans/val scans/test as the original scannet')
        raise NotImplementedError

    fileName = dst  +  fn[len(fn)-27:len(fn)-4] + '_instance.pth'
    torch.save(all, fileName)
    print(fileName)

#f_from_test('/fileserver/lhanaf/data/scannet/scans/scene0001_00/scene0001_00_vh_clean_2.ply')

# Create folders if not exist
if not os.path.exists('./datasets/scannet_data'):
    raise ValueError('Dataset not found. Make a symlink to ScanNet dataset as \"dataset/scannet_data\" ')

if not os.path.exists('./datasets/scannet_data/instance/'):
    os.makedirs('./datasets/scannet_data/instance/')

if not os.path.exists('./datasets/scannet_data/instance/train/'):
    os.makedirs('./datasets/scannet_data/instance/train/')

if not os.path.exists('./datasets/scannet_data/instance/val/'):
    os.makedirs('./datasets/scannet_data/instance/val/')

if not os.path.exists('./datasets/scannet_data/instance/test/'):
    os.makedirs('./datasets/scannet_data/instance/test/')


print("avilable cpus: ", mp.cpu_count())

files = sorted(glob.glob('./datasets/scannet_data/test/*/*_vh_clean_2.ply'))
p = mp.Pool(processes=mp.cpu_count() - 4)
p.map(f, files)
p.close()
p.join()

files = sorted(glob.glob('./datasets/scannet_data/train/*/*_vh_clean_2.ply'))
p = mp.Pool(processes=mp.cpu_count() - 4)
p.map(f, files)
p.close()
p.join()

files = sorted(glob.glob('./datasets/scannet_data/val/*/*_vh_clean_2.ply'))
p = mp.Pool(processes=mp.cpu_count() - 4)
p.map(f, files)
p.close()
p.join()
"""
#files = sorted(glob.glob('/fileserver/lhanaf/data/scannet/scans_test/*/*_vh_clean_2.ply'))
#files2 = sorted(glob.glob('/fileserver/lhanaf/data/scannet/scans_test/*/*_vh_clean_2.labels.ply'))
#assert len(files) == len(files2)
#p = mp.Pool(processes=mp.cpu_count() - 4)
#p.map(f, files)
#p.close()
#p.join()
"""