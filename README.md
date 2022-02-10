# OccuSeg: Occupancy-aware 3D Instance Segmentation

## Introduction
This is the official code repository for OccuSeg, a state-of-the-art method for accurate joint 3D semantic and instance segmentation.

## License
This project is released under the [GPLv3 license](LICENSE). We only allow free use for academic use. For commercial use, please contact us to negotiate a different license at luvisionsigma@outlook.com.

## Citing

If you find our code useful, please kindly cite our paper:

```bibtex
@article{Han2020OccuSegO3,
  title={OccuSeg: Occupancy-Aware 3D Instance Segmentation},
  author={Lei Han and Tian Zheng and Lan Xu and Lu Fang},
  journal={2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020},
  pages={2937-2946}
}
```

## Quickstart with docker
0. Install docker and nvidia runtime following [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
1. Download the preprocessed ScanNet dataset from http://www.example.com
2. Modify the correct paths to the dataset and output directory in `docker_run.sh`
3. Train the model with `bash docker_run.sh`. Replace `train_instance.sh` with `evaluate_instance.sh` to perform evaluation. A pertrained model is available at http://www.example.com, which is trained on the ScanNet training set.

If you prefer to setup the environment or prepare the data manually, following the below instructions or checkout the `docker_build.sh`.

## Environment setup

### Preliminary Requirements:
* Ubuntu 16.04
* CUDA 9.0

We trained using NVidia GV100.

### Conda environment
Create the conda environment using:
```bash
conda env create -f p1.yml
```
and activate it.

### Install dependencies and SparseConvNet
```bash
sh all_build.sh
```

### Data preparation

1. ScanNet data split

Put the scannet data folder at `examples/ScanNet/datasets/scannet_data/`,
organized as 
```
scannet_data
|---- train
        |--- scenexxxx_xx
|---- val
        |--- scenexxxx_xx
|---- test
        |--- scenexxxx_xx
```

2. Perform super-voxel clustering

Use the Segmentator at:

https://github.com/ScanNet/ScanNet/tree/master/Segmentator

For each `scenexxxx_xx_vh_clean_2.ply`, we generate a `scenexxxx_xx_vh_clean_2.regions.json` using the Segmentator with default parameters.

3. Generate pth files

Under `examples/ScanNet/`, run 
```
python prepare_data.py
```

This will generate pths at `scannet_data/instance/`. Then divide into train/val/test.

We use a `full_train` data split to train the final model for benchmark, which is described in `examples/ScanNet/datasets/full_train.txt` and `examples/ScanNet/datasets/full_val.txt`, showing the files in folder `scannet_data/instance/full_train` and `scannet_data/instance/full_val` respectively.

After this step, the `scannet_data` folder should look like this:
```
scannet_data
|---- instance
        |---- train
                |--- scenexxxx_xx_vh_clean_2_instance.pth
        |---- val
                |--- scenexxxx_xx_vh_clean_2_instance.pth
        |---- test
                |--- scenexxxx_xx_vh_clean_2_instance.pth
        |---- full_train
                |--- scenexxxx_xx_vh_clean_2_instance.pth
        |---- full_val
                |--- scenexxxx_xx_vh_clean_2_instance.pth
```

### Training and testing

To train the model, go to ScanNet folder:

```
cd example/ScanNet
sh training_script/train_instance.sh
```

Training checkpoints are saved at `ckpts`.

To evaluate the results on test set:
```
sh training_script/evaluate_instance.sh
```
Make sure to modify the `evaluate_instance.sh` to select a checkpoint.

To evaluate on the val set, run:
```
sh training_script/evaluate_instance.sh
```