#!/bin/bash
# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
source activate p1
#
## make cudpp
cd extra/cudpp
rm -rf build/
mkdir build
cd build
cmake ..
make -j32
sudo make install
cd ../../..
#
## make easy profile
#
cd extra/easy_profiler
rm -rf build/
mkdir build
cd build
cmake ..
make -j32
sudo make install
cd ../../..
#
#

rm -rf build/ dist/ sparseconvnet.egg-info sparseconvnet/*.so
python setup.py develop
cd examples/ScanNet


