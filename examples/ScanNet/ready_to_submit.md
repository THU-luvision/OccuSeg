source activate SparseConvNet
cd examples/ScanNet
python prepare_data_test.py ./datasets/scannet/scans_test_light
zsh submit_script/zyhseg006_01.sh
zip xxx

------------
# fake judge on valid set

source activate SparseConvNet
cd examples/ScanNet
python prepare_data_test.py ./datasets/scannet/val
zsh submit_script/fake_judge.sh
python fake_judge.py
