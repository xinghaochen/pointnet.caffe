# Caffe Re-implementation of [PointNet](https://github.com/charlesq34/pointnet) [Qi et al., CVPR 2017]

This repository contains caffe implementation of PointNet, which is described in:

[PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf), CVPR 2017, Oral,\
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

For basic classification network without data augmentation:
```
sh train_pointnet_cls_basic.sh
```
For basic classification network with data augmentation:
```
sh train_pointnet_cls_basic_aug.sh
```
For classification network with input transform (3x3):
```
sh train_pointnet_cls_input_tnet.sh
```
For classification network with input transform (3x3) and feature transform (64x64):
```
sh train_pointnet_cls.sh
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
or 
```
sh test_pointnet_cls.sh
```

## Performance
Classification results on ModelNet40:

| Network | Original Paper | Ours w/o aug | Ours w/ aug| model name |
|---------- | ----------- |---------- | ----------- | ----------- |
| PointNet basic cls | 87.1% | 88.25% | 85.255| pointnet_cls_basic |
| PointNet w/ input T-Net | 87.9% | 89.19% | -| pointnet_cls_input_tnet|
| PointNet w/ input & feat T-Net | 89.2% | 88.81% | -| pointnet_cls|

## TODOs

- [x] Basic PointNet for Classification
- [ ] Basic PointNet for Part/Semantic Segmentation
- [ ] Better data augmentation
- [x] T-Nets

## Acknowledges
- [simbaforrest](https://github.com/simbaforrest) for his fork of [Caffe](https://github.com/simbaforrest/caffe/tree/3d42ec54eade6243fe84b068ba049340acd3b687)
- [Charles R. Qi](https://github.com/charlesq34/) for his awesome work of PointNet family.
