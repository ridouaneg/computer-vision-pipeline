# computer-vision-pipeline
A highly modulable computer vision pipeline.


Human detection
Source : GluonCV

Human pose estimation
Source : GluonCV

Pose-based action recognition
Source : EHPI

Face detection
Source : GluonCV

Facial landmarks estimation
Source : http://dlib.net/
Weights : http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
Example : http://dlib.net/face_landmark_detection.py.html

Facial emotion classification
Source : https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection
Weights : https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection/tree/master/model

Object tracking and matching
Source : https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/


Requirements :
basics : numpy, cv2
human detection : mxnet, gluoncv
face detection : mxnet
mtcnn face detection : mtcnn
pose estimation : mxnet, gluoncv
facial landmarks : dlib
action recognition : mxnet, gluoncv
