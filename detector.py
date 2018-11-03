import cfg
import tensorflow as tf
import cv2
import numpy as np
import random
import os
import sys
import dataManager

class detector:
    def __init__(self,address,fileName):
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(address+"/"+fileName)
        saver.restore(self.sess, tf.train.latest_checkpoint(address))
        self.graph = tf.get_default_graph()
        self.input = self.graph.get_tensor_by_name("input:0")

        self.model =self.graph.get_tensor_by_name("outPut:0")



        self.data = dataManager.dataManager()

    def detect(self,inputData):
        outPut = self.sess.run(self.model, feed_dict={self.input: [inputData]})
        return outPut[:,0:8], outPut[:,8:10]

    def detectFolder(self,address):
        inputList = self.data.loadFromFolder(address)
        for data in inputList:
            d = data
            data = cv2.cvtColor(data, cv2.COLOR_HSV2BGR)
            d=d/255.0
            prim, secon = self.detect(d)
            print("pred: ",prim, secon)
            cv2.line(data, (int(cfg.imgSize * float(prim[0][0])), int(cfg.imgSize* float(prim[0][1]))),(int(cfg.imgSize * float(prim[0][2])), int(cfg.imgSize* float(prim[0][3]))), (255, 0, 0), 2)
            cv2.line(data, (int(cfg.imgSize * float(prim[0][2])), int(cfg.imgSize * float(prim[0][3]))),(int(cfg.imgSize * float(secon[0][0])), int(cfg.imgSize * float(secon[0][1]))), (0, 0, 255), 2)
            cv2.line(data, (int(cfg.imgSize * float(secon[0][0])), int(cfg.imgSize * float(secon[0][1]))),(int(cfg.imgSize * float(prim[0][4])), int(cfg.imgSize * float(prim[0][5]))), (0, 0, 255), 2)
            cv2.line(data, (int(cfg.imgSize * float(prim[0][4])), int(cfg.imgSize * float(prim[0][5]))),(int(cfg.imgSize * float(prim[0][6])), int(cfg.imgSize * float(prim[0][7]))), (255, 0, 0), 2)
            cv2.line(data, (int(cfg.imgSize * float(prim[0][0])), int(cfg.imgSize * float(prim[0][1]))),(int(cfg.imgSize * float(prim[0][6])), int(cfg.imgSize * float(prim[0][7]))), (255, 0, 0), 2)
            cv2.imshow("test", data)
            cv2.waitKey(0)

test = detector("./modelsDay6/","10.meta")
test.detectFolder("/home/hsl/deli/dataSet/arashRahmaniTest/")




