#!/bin/bash

#block(name=zyhseg006_00_test, threads=10, memory=100000, subtasks=1, gpu=true, hours=200)
   source activate SparseConvNet
   TASK_NAME=zyhseg006_00_test
   FROM_TASK_NAME=zyhseg006_00
   TEST_PATH=./datasets/scannet/scans_test_light/*test.pth
   TEST_RESULT_PATH=./result/$FROM_TASK_NAME/
   python -u test.py  \
   --test_path $TEST_PATH \
   --taskname $TASK_NAME \
   --test_result_path $TEST_RESULT_PATH \
   --dataset scannet \
   --batch_size 1  \
   --loss cross_entropy \
   --optim Adam \
   --lr 0.001 \
   --gamma 0.2 \
   --step_size 1000 \
   --checkpoint_file ./ckpts/$FROM_TASK_NAME/Epoch245.pth \
   --checkpoint 0 \
   --snapshot 5 \
   --m 16 \
   --block_reps 1 \
   --scale 50 \
   --residual_blocks \
   --kernel_size 3 \
    echo "Done"

# if you want to schedule multiple gpu jobs on a server, better to use this tool.
# run: `bash ./qsub-SurfaceNet_inference.sh`
# for installation & usage, please refer to the author's github: https://github.com/alexanderrichard/queueing-tool
