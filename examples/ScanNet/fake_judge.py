import glob, plyfile, numpy as np, multiprocessing as mp, torch
import copy
import numpy as np
from open3d import *
import sys
import os
SELECTED_LABEL_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
def gen_fake_gt(gt_path,test_result_path):
    sample_list = sorted(glob.glob(os.path.join(gt_path, '*labels.ply')))
    for sample in sample_list:
        basename=os.path.basename(sample)
        latter='_vh_clean_2.labels.ply'
        with open(os.path.join(test_result_path,basename[0:-len(latter)]+'.gt'),"w") as f:
            gt = plyfile.PlyData().read(sample).elements[0]['label']
            for i in range(len(gt)):
                if i==len(gt)-1:
                    print(gt[i],end='',file=f)
                else:
                    print(gt[i],file=f)

SELECTED_LABEL_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]

#Predictions will all be in the set {0,1,...,19}
VALID_CLASS_IDS = range(0, len(SELECTED_LABEL_IDS))

def confusion_matrix(pred_ids, gt_ids):
    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
    idxs= gt_ids>=0
    return np.bincount(pred_ids[idxs]*20+gt_ids[idxs],minlength=400).reshape((20,20)).astype(np.ulonglong)

def get_iou(label_id, confusion):
    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false negatives
    fn = np.longlong(confusion[label_id, :].sum()) - tp
    # false positives
    not_ignored = [l for l in VALID_CLASS_IDS if not l == label_id]
    fp = np.longlong(confusion[not_ignored, label_id].sum())

    denom = (tp + fp + fn)
    if denom == 0:
        return False
    return (float(tp) / denom, tp, denom, fp, fn)
CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']

def evaluate(predLabels, valLabels):
    print('evaluating', valLabels.size, 'points...')
    confusion=confusion_matrix(predLabels, valLabels)
    class_ious = {}
    for i in range(len(VALID_CLASS_IDS)):
        label_name = CLASS_LABELS[i]
        label_id = VALID_CLASS_IDS[i]
        class_iou = get_iou(label_id, confusion)
        if class_iou is not False:
            class_ious[label_name] = get_iou(label_id, confusion)
            print(label_name,":",class_ious[label_name])

    sum_iou = 0
    for label_name in class_ious:
        sum_iou+=class_ious[label_name][0]
    mean_iou = sum_iou/len(class_ious)
    print('mean_iou:',mean_iou)

def judge(test_result_path):
    SELECTED_LABEL_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
    Label2class={}
    for i in range(len(SELECTED_LABEL_IDS)):
        Label2class[SELECTED_LABEL_IDS[i]]=i
    sample_list=sorted(glob.glob(os.path.join(test_result_path, '*.txt')))
    gt_list=sorted(glob.glob(os.path.join(test_result_path, '*.gt')))
    assert len(sample_list)==len(gt_list)
    pd=[]
    gt=[]
    for i in range(len(sample_list)):
        print(i)
        with open(gt_list[i],"r") as f1:
            gts=f1.readlines()
        with open(sample_list[i],"r") as f2:
            samples=f2.readlines()
        assert len(samples)==len(gts)
        for j in range(len(samples)):
            if int(gts[j].replace('\n','')) in SELECTED_LABEL_IDS:
                # print(sample_list[i])
                pd.append(Label2class[int(samples[j])])
                gt.append(Label2class[int(gts[j])])
    pd=np.array(pd)
    gt=np.array(gt)
    evaluate(pd,gt)

if __name__ == '__main__':
    # gen_fake_gt('./datasets/scannet/val','./result/fake_judge')
    judge('./result/fake_judge')