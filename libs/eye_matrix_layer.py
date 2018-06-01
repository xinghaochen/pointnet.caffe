import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYCAFFE_DIR = os.path.join(BASE_DIR, '../caffe/python')
sys.path.append(BASE_DIR)
sys.path.append(PYCAFFE_DIR)
import caffe
import numpy as np

'''
Output a KxK identity matrix
'''
class EyeMatrixLayer(caffe.Layer):
    def setup(self, bottom, top):
        # bottom 0: Bx3x3
        self.batch_size = bottom[0].data.shape[0]
        # config
        params = eval(self.param_str)
        self.K = int(params['K'])
        if self.K <= 0:
            raise Exception('EyeMatrixLayer: K has to be greater than zero.')


    def reshape(self, bottom, top):
        top[0].reshape(self.batch_size, self.K, self.K)


    def forward(self, bottom, top):
        # assign top data
        eye_arrays = [np.eye(self.K) for _ in range(self.batch_size)]        
        top[0].data[...] = np.stack(eye_arrays, axis=0)

    def backward(self, top, propatate_down, bottom):
        pass