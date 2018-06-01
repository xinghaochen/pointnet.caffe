# Caffe Re-implementation of [PointNet [Qi et al., CVPR 2017]](https://github.com/charlesq34/pointnet)

This repository contains caffe implementation of PointNet, which is described in:

PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation\
Oral Presentation, CVPR 2017\
Charles R. Qi*, Hao Su*, Kaichun Mo, and Leonidas J. Guibas (*: equal contribution)

## Dependencies

- [simbaforrest's Caffe](https://github.com/simbaforrest/caffe/tree/3d42ec54eade6243fe84b068ba049340acd3b687) that supports Matrix Multiplication Layer, Permute Layer
- unzip


## Usage

### Installation

Clone the repository recursively:
```
git clone --recursive https://github.com/xinghaochen/pointnet.caffe.git
```

Install caffe:
```
cd caffe
cp Makefile.config.example Makefile.config
# uncomment WITH_PYTHON_LAYER := 1
# change other settings accordingly
make -j16
make pycaffe -j16
```
Add `path/to/pointnet.caffe/libs` and `path/to/pointnet.caffe/caffe/python` to PYTHONPATH

### Training

Download and unzip the data:
```
cd data
sh download_data_modelnet40.sh
```

For basic classification netwoek without data augmentation:
```
sh train_pointnet_cls_basic.sh
```
For basic classification netwoek with data augmentation:
```
sh train_pointnet_cls_basic_aug.sh
```
For classification netwoek with input transform (3x3):
```
sh train_pointnet_cls_input_tnet.sh
```


### Testing
```
sh test_pointnet_cls_basic.sh
```
or
```
sh test_pointnet_cls_basic_aug.sh
```
or
```
sh test_pointnet_cls_input_tnet.sh
```

## Performance
Classification results on ModelNet40:

| Network | Original Paper | Ours w/o aug | Ours w/ aug|
|---------- | ----------- |---------- | ----------- |
| PointNet basic cls | 87.1% | 88.25% | 85.255|
| PointNet w/ input T-Net | 87.9% | 89.19% | -|

## TODOs

- [x] Basic PointNet for Classification
- [ ] Basic PointNet for Part/Semantic Segmentation
- [ ] Better data augmentation
- [x] T-Nets

## Acknowledges
- [simbaforrest](https://github.com/simbaforrest) for his fork of [Caffe](https://github.com/simbaforrest/caffe/tree/3d42ec54eade6243fe84b068ba049340acd3b687)
- [Charles R. Qi](https://github.com/charlesq34/) for his awesome work of PointNet family.
