import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYCAFFE_DIR = os.path.join(BASE_DIR, '../../caffe-pose/python')
sys.path.append(BASE_DIR)
sys.path.append(PYCAFFE_DIR)
import caffe
import numpy as np
import provider

class PointsAugLayer(caffe.Layer):
    def setup(self, bottom, top):
        # check input
        if len(bottom) != 2 and len(bottom) != 3:
            raise Exception('PointsAugLayer only takes two (data, label) or three (data, label, seg) blobs as input. Got {} bottom blobs.'.format(len(bottom)))
        # bottom 0: data, bottom 1: label
        self.batch_size = bottom[0].data.shape[0]
        self.point_num = bottom[0].data.shape[1]
        self.point_dim = bottom[0].data.shape[2]
        if len(bottom) == 2:
            self.is_seg = False
        else:
            self.is_seg = True
        # config
        # params = eval(self.param_str)
        # self.jitter_sigma = int(params['jitter_sigma'])
        # self.jitter_clip = params['jitter_clip']
        if self.point_dim !=3:
            raise Exception('PointsAugLayer currently only support points with 3 dims.')


    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].data.shape[0], bottom[0].data.shape[1], bottom[0].data.shape[2])
        top[1].reshape(bottom[1].data.shape[0])
        if self.is_seg:
            top[2].reshape(bottom[2].data.shape[0], bottom[2].data.shape[1])


    def forward(self, bottom, top):
        data = np.array(bottom[0].data)
        label = np.array(bottom[1].data)
        if self.is_seg:
            seg = np.array(bottom[2].data)
        # shuffle data
        aug_data, aug_label, order = provider.shuffle_data(data, np.squeeze(label))
        if self.is_seg:
            aug_seg = seg[order, ...]
        # Augment batched point clouds by rotation and jittering
        rotated_data = provider.rotate_point_cloud(aug_data)
        jittered_data = provider.jitter_point_cloud(rotated_data)
        # assign top data
        top[0].data[...] = jittered_data
        top[1].data[...] = aug_label
        if self.is_seg:
            top[2].data[...] = aug_seg


    def backward(self, top, propatate_down, bottom):
        pass