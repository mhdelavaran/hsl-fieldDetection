import cfg
import tensorflow as tf
import cv2
import numpy as np
import random
from PIL import Image
import os
import sys
seed = 8
tf.set_random_seed(seed)
np.random.seed(seed)

class dataManager():
    def __init__(self):
        self.address = cfg.address
        self.trainValRatio = cfg.trainValRatio
        self.batchSize = cfg.batchSize
        self.dataSet = []
        self.trainSet = []
        self.valSet = []
        self.trainSetPrimePointLabel = []
        self.trainSetSecondPointLabel = []
        self.trainSetSecondPointUsedLabel = []
        self.valSetPrimePointLabel = []
        self.valSetSecondPointLabel = []
        self.valSetSecondPointUsedLabel = []
        self.currentBatch = 0
        self.nBatch = 0
        self.cap = 0

    def loadDataSet(self):
        with open(self.address) as f:
            for line in f:
                if len(line.strip().split(" "))>1:
                    self.dataSet.append(line.strip().split(" "))
        random.shuffle(self.dataSet)
        self.nBatch = int(len(self.dataSet)/self.batchSize)
        i = 0
        for data in self.dataSet:
            i = i+1
            # im = Image.open("./img/" + str(data[0]))
            # resizedImg = im.resize((cfg.imgSize, cfg.imgSize), Image.ANTIALIAS)
            im = cv2.imread("./img/" + str(data[0]))
            resizedImg = cv2.resize(im, (cfg.imgSize, cfg.imgSize))

            imgNum = cv2.cvtColor(resizedImg, cv2.COLOR_BGR2HSV)
            imgNum = imgNum/255.0
            if(float(i / len(self.dataSet)) < self.trainValRatio):
                self.trainSet.append(imgNum)
                if(len(data)<10):
                    self.trainSetPrimePointLabel.append(
                        [float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5]), float(data[6]),
                         float(data[7]), float(data[8])])
                    self.trainSetSecondPointUsedLabel.append([0])
                    self.trainSetSecondPointLabel.append([(float(data[3])+float(data[5]))/2.0, (float(data[4])+float(data[6]))/2.0])
                else:
                    self.trainSetPrimePointLabel.append(
                        [float(data[1]), float(data[2]), float(data[3]), float(data[4]),
                         float(data[7]), float(data[8]), float(data[9]),float(data[10])])
                    if float(data[5]) == 0.0 and float(data[6]) == 0.0:
                        self.trainSetSecondPointUsedLabel.append([0])
                        self.trainSetSecondPointLabel.append([(float(data[3])+float(data[5]))/2.0, (float(data[4])+float(data[6]))/2.0])
                    else:
                        self.trainSetSecondPointUsedLabel.append([1])
                        self.trainSetSecondPointLabel.append([float(data[5]), float(data[6])])
            else:
                self.valSet.append(imgNum)
                if (len(data) < 10):
                    self.valSetPrimePointLabel.append(
                        [float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5]), float(data[6]),
                         float(data[7]), float(data[8])])
                    self.valSetSecondPointUsedLabel.append([0])
                    self.valSetSecondPointLabel.append([(float(data[3])+float(data[5]))/2.0, (float(data[4])+float(data[6]))/2.0])
                else:
                    self.valSetPrimePointLabel.append(
                        [float(data[1]), float(data[2]), float(data[3]), float(data[4]),
                         float(data[7]), float(data[8]), float(data[9]), float(data[10])])
                    if float(data[5]) == 0.0 and float(data[6]) == 0.0:
                        self.valSetSecondPointUsedLabel.append([0])
                        self.valSetSecondPointLabel.append([(float(data[3])+float(data[5]))/2.0, (float(data[4])+float(data[6]))/2.0])
                    else:
                        self.valSetSecondPointUsedLabel.append([1])
                        self.valSetSecondPointLabel.append([float(data[5]), float(data[6])])



    def getBatch(self):
        _next = int((self.currentBatch + 1) * self.batchSize)
        _prv = int(self.currentBatch * self.batchSize)
        # print("currentBatch = ",self.currentBatch)
        self.currentBatch = self.currentBatch + 1
        if self.currentBatch > len(self.trainSet)/self.batchSize:
            self.currentBatch = 0
        t1 = np.concatenate((self.trainSetPrimePointLabel[_prv:_next], self.trainSetSecondPointLabel[_prv:_next]),axis = 1)
        #t1 = np.concatenate((t1,self.trainSetSecondPointUsedLabel[_prv:_next]), axis = 1 )

        return self.trainSet[_prv:_next], t1, self.trainSetSecondPointUsedLabel[_prv:_next]

    def getValSet(self):
        t1 = np.concatenate((self.valSetPrimePointLabel, self.valSetSecondPointLabel), axis=1)
        #t1 = np.concatenate((t1, self.valSetSecondPointUsedLabel), axis=1)

        return self.valSet, t1, self.valSetSecondPointUsedLabel
    def initCamera(self,id):
        self.cap = cv2.VideoCapture(id)
    def loadFromCamera(self):
        ret, im = cap.read()
        resizedImg = cv2.resize(im, (128, 128))
        return [resizedImg]
    def loadFromFolder(self,address):
        inputList = []
        for F in os.listdir(address):
            filename = os.fsdecode(F)
            print(address +"/"+ filename)
            im = cv2.imread(address +"/"+ filename)
            resizedImg = cv2.resize(im, (cfg.imgSize, cfg.imgSize))
            resizedImg = cv2.cvtColor(resizedImg, cv2.COLOR_BGR2HSV)
            #resizedImg = resizedImg/255.0
            inputList.append(resizedImg)
        return inputList

    def getLabelFromTrainSet(self, i):
        return self.trainSetPrimePointLabel[i], self.trainSetSecondPointLabel[i], self.trainSetSecondPointUsedLabel[i]

    def validatDataSet(self):
        if len(self.trainSet) == 0:
            return
        for i in range(len(self.trainSet)):
            data = self.trainSet[i]
            prim, secon, used = self.getLabelFromTrainSet(i)
            print("pred: ",prim, secon, used)
            if used[0] > 0.0:
                cv2.line(data, (int(cfg.imgSize * float(prim[0])), int(cfg.imgSize* float(prim[1]))),(int(cfg.imgSize * float(prim[2])), int(cfg.imgSize* float(prim[3]))), (255, 0, 0), 2)
                cv2.line(data, (int(cfg.imgSize * float(prim[2])), int(cfg.imgSize * float(prim[3]))),(int(cfg.imgSize * float(secon[0])), int(cfg.imgSize * float(secon[1]))), (255, 0, 0), 2)
                cv2.line(data, (int(cfg.imgSize * float(secon[0])), int(cfg.imgSize * float(secon[1]))),(int(cfg.imgSize * float(prim[4])), int(cfg.imgSize * float(prim[5]))), (255, 0, 0), 2)
                cv2.line(data, (int(cfg.imgSize * float(prim[4])), int(cfg.imgSize * float(prim[5]))),(int(cfg.imgSize * float(prim[6])), int(cfg.imgSize * float(prim[7]))), (255, 0, 0), 2)
                cv2.line(data, (int(cfg.imgSize * float(prim[0])), int(cfg.imgSize * float(prim[1]))),(int(cfg.imgSize * float(prim[6])), int(cfg.imgSize * float(prim[7]))), (255, 0, 0), 2)

            else:
                cv2.line(data, (int(cfg.imgSize * float(prim[0])), int(cfg.imgSize * float(prim[1]))),(int(cfg.imgSize * float(prim[2])), int(cfg.imgSize * float(prim[3]))), (255, 0, 0), 2)
                cv2.line(data, (int(cfg.imgSize * float(prim[0])), int(cfg.imgSize * float(prim[1]))),(int(cfg.imgSize * float(prim[6])), int(cfg.imgSize * float(prim[7]))), (255, 0, 0), 2)
                cv2.line(data, (int(cfg.imgSize * float(prim[2])), int(cfg.imgSize * float(prim[3]))),(int(cfg.imgSize * float(prim[4])), int(cfg.imgSize * float(prim[5]))), (255, 0, 0), 2)
                cv2.line(data, (int(cfg.imgSize * float(prim[4])), int(cfg.imgSize * float(prim[5]))),(int(cfg.imgSize * float(prim[6])), int(cfg.imgSize * float(prim[7]))), (255, 0, 0), 2)
            cv2.imshow("test", data)
            cv2.waitKey(0)