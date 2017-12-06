import sys
import matplotlib.pyplot as plt
# sys.path.append("/Users/wszzn/develope/caffe/python")
# sys.path.append("/home/cxy/software/caffe-cvprw15-master/python")
# sys.path.append("/home/cxy/software/caffe-cvprw15-master/python/caffe")

sys.path.append("/home/cq/caffe/python")
sys.path.append("/home/cq/caffe/python/caffe")

import caffe

# chenxinyaun
caffe.set_mode_gpu()
# caffe._caffe.set_mode_cpu()

import numpy as np
import pandas as pd
import time


class CaffeNet(object):
    def __init__(self):
        self.netinit()

    def netinit(self):
        deffile = '/home/cq/caffe-master/models/bvlc_reference_caffenet/deploy.prototxt'
        modfile = '/home/cq/caffe-master/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
        self.net = caffe.Net(deffile, modfile, caffe.TEST)

    def feature_exact(self, img_name):
        tic = time.time()
        transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        transformer.set_mean('data', np.load('/home/cq/caffe-master/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_channel_swap('data', (2, 1, 0))
        transformer.set_raw_scale('data', 255)

        # note we can change the batch size on-the-fly
        # since we classify only one image, we change batch size from 10 to 1
        #data_blob_shape = self.net.blobs['data'].data.shape
        #print data_blob_shape

        self.net.blobs['data'].reshape(1, 3, 227, 227)
        # print 'reshaped'

        # load the image in the data layer
        # print img_name
        im = caffe.io.load_image(img_name)
        # transformered_image = transformer.preprocess('data', im)
        # plt.imshow(im)

        self.net.blobs['data'].data[...] = transformer.preprocess('data', im)
        out =self.net.forward()
        fc8 = self.net.blobs["fc8"].data[:]
        print fc8
        # print out

        # compute
        # out = self.net.forward()
        # out = self.net.forward_all(data=np.asarray([transformer.preprocess('data', im)]))

        # other possibility : out = self.net.forward_all(data=np.asarray([transformer.preprocess('data', im)]))

        # predicted predicted class
        # ret = self.posthandler(out["fc7"])
        # print ret, ret.shape
        pd.DataFrame(fc8).to_csv('%s.csv' % img_name, index=False, header=False)

        print '---- Feature exacted for %f s ----' % (time.time() - tic)
        return True

    def posthandler(self, arr):
        return arr
