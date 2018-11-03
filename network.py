import tensorflow as tf
import cv2 as cv
import numpy as np
import random
from PIL import Image
import os
import sys
import cfg


class network():
    def __init__(self):
        self.inputImageSize = cfg.imgSize
        self.inputImageNChanel = cfg.imgChanel
        self.input = tf.placeholder(tf.float32, [None, self.inputImageSize, self.inputImageSize, self.inputImageNChanel], name='input')

    def conv2d(self, tensor, outPutFilters, kernelSize, stride, name):
        with tf.name_scope(name):
            inputFilters = int(tensor.get_shape()[3])
            w = tf.Variable(tf.truncated_normal([kernelSize, kernelSize, inputFilters, outPutFilters], stddev=0.01), name ="w")
            b = tf.Variable(tf.ones([outPutFilters]) / 100, name ="b")
            stride = [1, stride, stride, 1]
            tensor = tf.nn.conv2d(tensor, w, strides=stride, padding='SAME')
            tensor = tf.nn.bias_add(tensor, b)
            tf.summary.histogram("w", w)
            tf.summary.histogram("b", b)
            tensor = tf.nn.relu(tensor)
        return tensor

    def flatten(self, tensor):
        tensorSize = tensor.get_shape().as_list()
        batchSize = tensorSize[0]
        newTensorSize = tensorSize[1] * tensorSize[2] * tensorSize[3]
        return tf.contrib.layers.flatten(tensor, [batchSize, newTensorSize])

    def fullyConnect(self, tensor, numNurun,name):
        with tf.name_scope(name):
            _input = int(tensor.get_shape()[1])
            w = tf.Variable(tf.truncated_normal([_input, numNurun], stddev=0.01), name = "w")
            b = tf.Variable(tf.ones(numNurun) / 100,name = "b")
            tensor = tf.add(tf.matmul(tensor, w), b)
            tf.summary.histogram("w", w)
            tf.summary.histogram("b", b)
            tensor = tf.nn.relu(tensor)
        return tensor

    def outPutRegression(self, tensor, numClass,name):
        with tf.name_scope(name):
            numInPut = int(tensor.get_shape()[1])
            w = tf.Variable(tf.truncated_normal([numInPut, numClass], stddev=0.01), name = "w")
            b = tf.Variable(tf.ones(numClass) / 100, name = "b")
            tensor = tf.add(tf.matmul(tensor, w), b)
            tf.summary.histogram("w", w)
            tf.summary.histogram("b", b)
        return tensor

    def outPutClassification(self, tensor, numClass,name):
        with tf.name_scope(name):
            numInPut = int(tensor.get_shape()[1])
            w = tf.Variable(tf.truncated_normal([numInPut, numClass], stddev=0.01),name = "w")
            b = tf.Variable(tf.ones(numClass) / 100, name = "b")
            tensor = tf.add(tf.matmul(tensor, w), b)
            tf.summary.histogram("w", w)
            tf.summary.histogram("b", b)
            tensor = tf.nn.relu(tensor)
        return tensor

    def maxPooling(self,tensor, kernelSize, strides):
        return tf.nn.max_pool(tensor, [1, kernelSize, kernelSize, 1], [1, strides, strides, 1], 'SAME')

    def model(self):
        tensor = self.conv2d(self.input, 16, 5, 2,"conv1")
        tensor = self.maxPooling(tensor, 2, 2)
        tensor = self.conv2d(tensor, 16, 3, 1, "conv2")
        tensor = self.maxPooling(tensor, 2, 2)
        tensor = self.conv2d(tensor, 24, 3, 1, "conv3")
        tensor = self.maxPooling(tensor, 2, 2)
        tensor = self.conv2d(tensor, 32, 3, 1, "conv4")
        # tensor = self.maxPooling(tensor, 2, 2)
        tensor = self.flatten(tensor)

        tensor = self.fullyConnect(tensor, 42, "fc1")
        tf.summary.histogram("fcOut", tensor)
        tensor = self.outPutRegression(tensor, 10, "out")
        tf.summary.histogram("outRes", tensor)

        # tf.summary.histogram("usedSec", Chead)
        return tensor

nn = network()
outPut= nn.model()
outPut = tf.identity(outPut, name = "outPut")
# outPut = tf.identity(primeryPoints, name='primeryPoints')
# secnderyPoint = tf.identity(secnderyPoint, name='secnderyPoint')
# secnderyPointUsed = tf.identity(secnderyPointUsed, name='secnderyPointUsedOutPut')

if __name__=='__main__':
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./graphs', sess.graph)