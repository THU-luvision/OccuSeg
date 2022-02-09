# OccuSeg: Occupancy-aware 3D Instance Segmentation

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

For each `scenexxxx_xx_vh_clean_2.ply`, generate a `scenexxxx_xx_vh_clean_2.regions.json` using the Segmentator with default parameters.

3. Generate pth files

Under `examples/ScanNet/`, run 
```
python prepare_data.py
```

This will generate pths at `scannet_data/instance/`, divided into train/val/test.

We use a `full_train` data split to train the final model, which is described in `examples/ScanNet/datasets/full_train.txt` and `examples/ScanNet/datasets/full_val.txt`, showing the files in folder `scannet_data/instance/full_train` and `scannet_data/instance/full_val` respectively.

After this step, the `scannet_data` should look like this:
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
sh training_script/lhanaf_instance.sh
```

Training checkpoints are saved at ckpts.

To evaluate the results on test set:
```
sh training_script/lhanaf_evaluate_instance.sh
```
Make sure to modify the `lhanaf_evaluate_instance.sh` to select a checkpoint. Our pretrained model is saved at `ckpts/lhanaf_instance_s50_val_rep1_withElastic/Epoch240.pth`

To evaluate on the val set:
Modify this line in `evaluate_instance.py`
```
candidate_path = './datasets/scannet_data/instance/test/*.npz'
```

to 

```
candidate_path = './datasets/scannet_data/instance/val/*.npz'
```

and run

```
sh training_script/lhanaf_evaluate_instance.sh
```