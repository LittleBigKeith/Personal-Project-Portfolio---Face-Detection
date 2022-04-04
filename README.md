# Personal Project Portfolio - Face Detection
## Developing a face detection app that can recognize (and hopefully remember) faces in images, videos and real-time cameras.

Difference between **face detection** and **facial recognition**

**Face detection**: detect the presence of human face in an image

**Facial recognition**: detect the presence of human faces *and* determine their identity
 
### To begin

**Required libraries**
1. OpenCV (Open Source Computer Vision) Library

cv2 is the module we need to detect faces

```
pip install opencv-python
```
2. cMake

for compiling dlib, a modern C++ toolkit containing machine learning algorithms and
tools for creating complex software in C++ to solve real-world problems.

```
pip install cmake
```
3. dlib

dlib is a modern C++ toolkit containing machine learning algorithms and
tools for creating complex software in C++ to solve real-world problems.

```
conda install -c conda-forge dlib
```
4. Face Recognition

face recognition contains facial recognition functionalities that wrap around dlib.

```
conda install -c conda-forge face_recognition
```
**Optional libraries:**
1. Matplotlib

A helper library to plot images with coordinates

```
pip install matplotlib
```


### Datasets used in this project
1. Google Facial Expression Comparison Dataset

   R Vemulapalli, A Agarwala, “A Compact Embedding for Facial Expression Similarity”, CoRR, abs/1811.11283, 2018.
 
 2 5 Celebrity Faces Dataset
 
   .https://www.kaggle.com/datasets/dansbecker/5-celebrity-faces-dataset
