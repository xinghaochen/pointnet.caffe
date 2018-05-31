# Caffe Re-implementation of [PointNet [Qi et al., CVPR 2017]](https://github.com/charlesq34/pointnet)

This repository contains caffe implementation of PointNet, which is described in:

PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation\
Oral Presentation, CVPR 2017\
Charles R. Qi*, Hao Su*, Kaichun Mo, and Leonidas J. Guibas (*: equal contribution)

## Dependencies

- Caffe (with python layer, hdf5 support)
- unzip


## Usage

### Preparation

Download and unzip the data:
```
cd data
sh get_data.sh
```

### Training
For basic classification netwoek without data augmentation:
```
sh train.sh 2>&1 | tee logs/log_train.txt
```
For basic classification netwoek with data augmentation, add `path/to/libs` to PYTHONPATH and run
```
sh train_aug.sh 2>&1 | tee logs/log_train_aug.txt
```

### Testing
```
sh test.sh 2>&1 | tee logs/log_test.txt
```
or 
```
sh test_aug.sh 2>&1 | tee logs/log_test_aug.txt
```

## Performance
Classification results on ModelNet40:

| Network | Original Paper | Ours w/o aug | Ours w/ aug|
|---------- | ----------- |---------- | ----------- |
| PointNet basic cls | 87.1% | 88.25% | 85.255|

## TODOs

- [x] Basic PointNet for Classification
- [ ] Basic PointNet for Part/Semantic Segmentation
- [ ] Better data augmentation
- [ ] T-Nets
