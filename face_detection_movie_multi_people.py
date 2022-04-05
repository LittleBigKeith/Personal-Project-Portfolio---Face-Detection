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
from collections import defaultdict

name_list = ["Chang", "Harry", "Hermione", "Malfoy", "Ron", "Snape"]

#Constant image folder name
train_imfolder = "openCV_face_detection_harry_potter-master/data/test/"
vdofolder = "openCV_face_detection_harry_potter-master/videos/"
vdoname = "Harry_Potter.mp4"
vdopath = vdofolder + vdoname

threshold = 0.33
    
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

def recognition_video(vdopath, encode_dict):
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
                    matching_list = []
                    for name in name_list:
                        #compare_faces return a list True/False values
                        matches = face_recognition.compare_faces(encode_dict[name], face)
                        #face_distance returns Euclidean distance for each face encoding
                        face_dist = face_recognition.face_distance(encode_dict[name], face)
                        likelihood = 1 - sum(pow(face_dist,2)) / len(face_dist)
                        #if match 70% of training images or above, found a face of this person
                        if sum(matches)/len(matches) >= threshold:
                            matching_list.append((name, likelihood))
                        
                    # if can identify which person with confidence
                    if len(matching_list) == 1:
                         y1,x2,y2,x1 = faceloc
                         cv2.rectangle(frame, (x1, y1), (x2, y2), (64, 64, 255),3)
                         cv2.rectangle(frame, (x1-2, y2+35), (x2+2, y2), (64, 64, 255), cv2.FILLED)
                         cv2.putText(frame, matching_list[0][0] + f"{matching_list[0][1]:.2f}", (x1+3, y2+16), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
                    
                    # if conflicting results (2 persons or more with high confidence)
                    elif len(matching_list) > 1:
                        y1,x2,y2,x1 = faceloc
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (64, 64, 255),3)
                        cv2.rectangle(frame, (x1-2, y2+35), (x2+2, y2), (64, 64, 255), cv2.FILLED)
                        cv2.putText(frame, "unknown", (x1+3, y2+16), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                    
                    elif len(matching_list) == 0 and len(faces_in_frame) > 0:
                        y1,x2,y2,x1 = faceloc
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (64, 64, 255),3)
                        cv2.rectangle(frame, (x1-2, y2+35), (x2+2, y2), (64, 64, 255), cv2.FILLED)
                        cv2.putText(frame, "unknown", (x1+3, y2+16), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                    
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) == ord('q'):
                     break
        except:
            print("exception!")
            break
    cap.release()
    cv2.destroyAllWindows()

encode_dict = defaultdict(list)
for name in name_list:
    train_images = read_images(train_imfolder + name)
    detection(train_images)
    encode_dict[name] = encoding(train_images)
print(encode_dict)
recognition_video(vdopath, encode_dict)
