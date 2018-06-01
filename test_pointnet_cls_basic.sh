caffe/build/tools/caffe test -model=models/pointnet_cls_basic.prototxt -weights=snapshots/pointnet_cls_basic_iter_80000.caffemodel -gpu 0  2>&1 | tee logs/log_test_pointnet_cls_basic.txt
