caffe_root = '/media/lin/Disk3/caffe-a/python'
import sys
sys.path.insert(0, caffe_root)
import pdb
import caffe
import numpy as np
from PIL import Image
import random
import math
import skimage
import matplotlib.pyplot as plt
import os
import math


def test_forward():
    roots = '../../data/faces/'
    test_file = '../../data/1/test_1.txt'
    mean_file = '../../data/1/256_train_mean.binaryproto'
    
    network_file = './alexnet-deploy.prototxt'
    pretrained_model = './snapshot/1/alex_iter_20000.caffemodel'

    batch_shape = [1, 3, 224, 224]
    means = get_mean_npy(mean_file, crop_size = batch_shape[2:])
    
    caffe.set_mode_gpu()
    net = caffe.Net(network_file, pretrained_model, caffe.TEST)  #set caffe model

    file = open(test_file,'r')
    lines = file.readlines()
    linesize = len(lines)
    file.close()
    ground_label = np.zeros((linesize, 1), dtype=np.float32)
    predict_label = np.zeros((linesize, 1), dtype=np.float32)
    mae = 0.0
    rmse = 0.0

    for i in range(linesize):
        linesplit = lines[i].split(' ')
        filename = linesplit[0]
        ground = float(linesplit[1].split('\n')[0])
        ground_label[i] = ground
        imgdir = roots + filename
        _load_img = load_img(imgdir, resize = (256, 256), isColor = True, crop_size = 224, crop_type = 'center_crop', raw_scale = 255, means = means)

        if net.blobs.has_key('extra'):
            if filename.find('f') != -1:
                gender = 1
            elif filename.find('m') != -1:
                gender = -1
            else:
                print 'filename wrong!'

            if filename.find('y') != -1:
                eth = 1
            elif filename.find('w') != -1:
                eth = -1
            else:
                print 'filename wrong!'

            net.blobs['extra'].data[0][0] = gender
            net.blobs['extra'].data[0][1] = eth

        net.blobs['data'].data[...] = _load_img  
        out = net.forward()
        predict = net.blobs['feat1'].data[...][0][0]
        diff = ground - predict
        mae += abs(diff.item())
        rmse += diff * diff
        predict_label[i] = predict

        print filename

    pearson_correlation = cal_correlation(ground_label, predict_label)
    mae = mae / linesize
    rmse = math.sqrt(rmse / linesize)
    print pearson_correlation, mae, rmse


def get_mean_npy(mean_bin_file, crop_size=None):
    mean_blob = caffe.proto.caffe_pb2.BlobProto()
    mean_blob.ParseFromString(open(mean_bin_file, 'rb').read())
    mean_npy = caffe.io.blobproto_to_array(mean_blob)
    _shape = mean_npy.shape
    mean_npy = mean_npy.reshape(_shape[1], _shape[2], _shape[3])

    if crop_size:
        mean_npy = mean_npy[
            :, (_shape[2] - crop_size[0]) / 2:(_shape[2] + crop_size[0]) / 2, 
            (_shape[3] - crop_size[1]) / 2:(_shape[3] + crop_size[1]) / 2]
    return mean_npy

def crop_img(img, crop_size, crop_type='center_crop'):
    '''
        crop_type is one of 'center_crop',
                            'random_crop', 'random_size_crop'
    '''
    if crop_type == 'center_crop':
        sh = crop_size 
        sw = crop_size
        hh = (img.shape[0] - sh) / 2
        ww = (img.shape[1] - sw) / 2
    elif crop_type == 'random_crop':
        sh = crop_size
        sw = crop_size
        hh = random.randint(0, img.shape[0] - sh)
        ww = random.randint(0, img.shape[1] - sw)
    elif crop_type == 'random_size_crop':
        sh = random.randint(crop_size[0], img.shape[0])
        sw = random.randint(crop_size[1], img.shape[1])
        hh = random.randint(0, img.shape[0] - sh)
        ww = random.randint(0, img.shape[1] - sw)
    img = img[hh:hh + sh, ww:ww + sw]
    if crop_type == 'random_size_crop':
        img = skimage.transform.resize(img, crop_size, mode='reflect')    
   
    return img


def load_img(path, resize=128, isColor=True,
             crop_size=112, crop_type='center_crop',
             raw_scale=1, means=None):
    '''
        crop_type is one of None, 'center_crop',
                            'random_crop', 'random_size_crop'
    '''
    img = skimage.io.imread(path)
    # pdb.set_trace()

    if resize is not None and img.shape != resize:
        img = skimage.transform.resize(img, resize, mode='reflect')
    if crop_size and crop_type:
        img = crop_img(img, crop_size, crop_type)
    if isColor:
        img = skimage.color.gray2rgb(img)
        img = img.transpose((2, 0, 1))
        img = img[(2, 1, 0), :, :]
    else:
        img = skimage.color.rgb2gray(img)
        img = img[np.newaxis, :, :]
    img = skimage.img_as_float(img).astype(np.float32) * 255 #skimage float32 is between[0,1]

    if means is not None:
        if means.ndim == 1 and isColor:
            means = means[:, np.newaxis, np.newaxis]
        img -= means

    img = img / raw_scale 
    return img


def cal_correlation(ground_label, predict_label):
    num = len(ground_label)
    ground_sum = np.sum(ground_label)
    predict_sum = np.sum(predict_label)
    ground_power_sum = np.sum(ground_label * ground_label)
    predict_power_sum = np.sum(predict_label * predict_label)
    xy_sum = np.sum(predict_label * ground_label)
    ground_sub = num * ground_power_sum - ground_sum * ground_sum
    predict_sub = num * predict_power_sum - predict_sum * predict_sum


    if ground_sub <= 0 or predict_sub <= 0:
        correlation = None
    else:
        correlation = (num * xy_sum - ground_sum * predict_sum) / (math.sqrt(ground_sub) * math.sqrt(predict_sub))
    return correlation

if __name__ == '__main__':
    test_forward()
