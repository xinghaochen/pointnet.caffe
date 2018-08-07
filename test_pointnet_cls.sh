caffe/build/tools/caffe test -model=models/pointnet_cls.prototxt -weights=snapshots/pointnet_cls_iter_80000.caffemodel -gpu 0 2>&1 -iterations 77 | tee logs/log_test_pointnet_cls.txt
