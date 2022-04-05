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


#Settings

person = "ben_afflek"

#Constant image folder name
train_imfolder = "5_celebrity_dataset/data/train/" + person
vdofolder = "5_celebrity_dataset/videos/"
vdopath = vdofolder + person + ".mp4"

threshold = 0.7


    
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
        #face_encodings return a list of an array
        train_ben_encodings = face_recognition.face_encodings(e, model = "small")
        #for encoding in train_ben_encodings:
        if(len(train_ben_encodings) > 0):
            #for each face detected in face_encodings (usually 1)
            for e in train_ben_encodings:
                ben_encode_list.append(e)
    return ben_encode_list

def recognition_video(vdopath, ben_encoding):
    print(vdopath)
    dim = (512,288)
    cap = cv2.VideoCapture(vdopath)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting...")
            break
        #rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('frame', frame)  
        try:
            while cv2.getWindowProperty('frame', 0) >= 0:
                ret, frame = cap.read()
                frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                
                #Face recognition
                faces_in_frame = face_recognition.face_locations(frame)
                encoded_faces = face_recognition.face_encodings(frame, faces_in_frame, model = "small")
                for face, faceloc in zip(encoded_faces, faces_in_frame):
                    #compare_faces return a list True/False values
                    matches = face_recognition.compare_faces(ben_encoding, face)
                    #face_distance returns Euclidean distance for each face encoding
                    face_dist = face_recognition.face_distance(ben_encoding, face)
                    print(face_dist)
                    likelihood = 1 - sum(pow(face_dist,2)) / len(face_dist)
                    #if match 70% of training images or above, found a face of this person
                    if sum(matches)/len(matches) >= threshold:
                         y1,x2,y2,x1 = faceloc
                         cv2.rectangle(frame, (x1, y1), (x2, y2), (64, 64, 255),3)
                         cv2.rectangle(frame, (x1-2, y2+35), (x2+2, y2), (64, 64, 255), cv2.FILLED)
                         cv2.putText(frame, person + f"{likelihood:.2f}", (x1+3, y2+16), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) == ord('q'):
                    break
        except:
            print("exception!")
            break
    cap.release()
    cv2.destroyAllWindows()

train_images = read_images(train_imfolder)
detection(train_images)
ben_encoding = encoding(train_images)
recognition_video(vdopath, ben_encoding)
