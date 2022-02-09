#!/bin/bash

#block(name=occuseg_training_0, threads=10, memory=100000, subtasks=1, gpu=true, hours=200)
   TASK_NAME=occuseg_training_0
   python evaluate_instance.py  \
   --taskname $TASK_NAME \
   --dataset scannet\
   --evaluate \

    echo "Done"

# if you want to schedule multiple gpu jobs on a server, better to use this tool.
# run: `bash ./qsub-SurfaceNet_inference.sh`
# for installation & usage, please refer to the author's github: https://github.com/alexanderrichard/queueing-tool
