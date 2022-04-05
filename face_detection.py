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

person = "ben_afflek"

#Constant image folder name
train_imfolder = "5_celebrity_dataset/data/train/" + person
vdofolder = "videos/"
    
def read_images(imfolder):
    #Get list of all images in folder
    imagePath = list(e for e in listdir(imfolder) \
                     if imghdr.what(f"{imfolder}/{e}") != None)
    #read images from working directory and plot 
    images = []
    for e in imagePath:
        bgr_image = cv2.imread(f"{imfolder}/{e}" , cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        images.append(rgb_image)
    return images

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

def recognition_video(vdofolder):
    vdoname = vdofolder + person + ".mp4"
    print(vdoname)
    dim = (1024, 576)
    cap = cv2.VideoCapture(vdoname)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting...")
            break
        #rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow('frame', frame)  
        try:
            while cv2.getWindowProperty('frame', 0) >= 0:
                ret, frame = cap.read()
                frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                
                #Face recognition
                faces_in_frame = face_recognition.face_locations(frame)
                encoded_faces = face_recognition.face_encodings(frame, faces_in_frame)
                for face, faceloc in zip(encoded_faces, faces_in_frame):
                    # TO BE IMPLEMENTED
                    pass
                cv2.imshow('frame', frame)
                if cv2.waitKey(10) == ord('q'):
                    break
        except:
            break

    cap.release()
    cv2.destroyAllWindows()

train_images = read_images(train_imfolder)
detection(train_images)
ben_encoding = encoding(train_images)
recognition_video(vdofolder)
