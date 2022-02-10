#!/bin/bash

nvidia-docker run --rm \
    --ipc=host \
    -w /workspace/examples/ScanNet \
    -v <path-to-scannet_data>:/workspace/examples/ScanNet/datasets/scannet_data \
    -v <path-to-model-checkpoints-output-dir>:/workspace/examples/ScanNet/ckpts \
    luvisionsigma/occuseg:dev \
    bash training_script/train_instance.sh
    # bash training_script/evaluate_instance.sh
