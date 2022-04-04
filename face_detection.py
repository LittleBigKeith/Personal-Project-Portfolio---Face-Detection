# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 17:39:31 2022

@author: keith
"""

from os import listdir
import imghdr
import cv2
import matplotlib.pyplot as plt
import face_recognition

#Constant image folder name
imfolder = "5_celebrity_dataset/data/train/ben_afflek"
    
def detection(images):
    #read images from working directory and plot them
    for e in images:
        #draw bounding boxes around faces
        faces = face_recognition.face_locations(e)
        for face in faces:
            cv2.rectangle(e,(face[3], face[2]), (face[1],face[0]), \
                      (255, 0, 255),3)
        #cv2.imshow('rgb', rgb_image)
        #cv2.waitkey(0)
        plt.imshow(e)
        plt.show()

def encoding(images):
    #encode all the train images
    ben_encode_list = []
    for e in images:
        train_ben_encodings = face_recognition.face_encodings(e)
        for encoding in train_ben_encodings:
            ben_encode_list.append(train_ben_encodings)
    return ben_encode_list

def recognition(images):
    pass
            
#Get list of all images in folder
imagePath = list(e for e in listdir(imfolder) \
                 if imghdr.what(f"{imfolder}/{e}") != None)
    
#read images from working directory and plot 
images = []
for e in imagePath:
    bgr_image = cv2.imread(f"{imfolder}/{e}" , cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    images.append(rgb_image)
    detection(images)
    training_result = encoding(images)
    recognition(images)