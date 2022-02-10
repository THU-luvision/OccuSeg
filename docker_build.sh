#!/bin/bash

docker build -t luvisionsigma/occuseg:dev .
nvidia-docker run -w /workspace luvisionsigma/occuseg:dev bash all_build.sh
docker commit `docker ps -q -l` luvisionsigma/occuseg:dev
docker rm `docker ps -q -l`
