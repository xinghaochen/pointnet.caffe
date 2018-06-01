caffe/build/tools/caffe train --solver=models/solver_pointnet_cls.prototxt -gpu 0 2>&1 | tee logs/log_train_pointnet_cls.txt
