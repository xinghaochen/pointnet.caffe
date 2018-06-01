caffe/build/tools/caffe train --solver=models/solver_pointnet_cls_basic.prototxt -gpu 0 2>&1 | tee logs/log_train_point_cls_basic.txt
