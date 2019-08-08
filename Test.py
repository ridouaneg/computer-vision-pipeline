# ---- ---- ---- IMPORTATIONS ---- ---- ----
import cv2
import sys
import time
import numpy as np

from Pipeline import Pipeline
from ObjectDetector import HumanDetector, FaceDetector, MtcnnFaceDetector
from HumanPoseEstimator import HumanPoseEstimator
from FacialLandmarksEstimator import FacialLandmarksEstimator
from ObjectTracker import MultiObjectTracker
from FacialEmotionClassifier import FacialEmotionClassifier
from ActionRecognizer import ActionRecognizer


# ---- ---- ---- VIDEO FILE ---- ---- ----

# Video files
#video_path = './videos/test6.avi'
# Webcam
video_path = 0

cap = cv2.VideoCapture(video_path)

# Check if video/camera opened successfully
if (cap.isOpened() == False):
    print('Error opening video stream or file')
    sys.exit()

# Information of the video are obtained
video_frame_width = int(cap.get(3))
video_frame_height = int(cap.get(4))
video_fps = int(cap.get(5))
print('Video information :')
if video_path == 0:
    print('    Video name : webcam', str(video_path))
else:
    print('    Video name :', video_path)
print('    Video resolution :', video_frame_width, video_frame_height)
print('    Frame per second (fps) :', video_fps)

input_resolution = (1024, 512)

# Define the output video path and the fps and resolution
#out = cv2.VideoWriter('./output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), video_fps, input_resolution)


# ---- ---- ---- IMPORT MODEL ---- ---- ----

# Human detection - models available :
#   yolo3_mobilenet1.0_coco
#   yolo3_darknet53_coco
#   faster_rcnn_fpn_resnet101_v1d_coco

# Object tracking - models available :
#   CSRT

# Pose estimation - models available :
#   simple_pose_resnet18_v1b
#   simple_pose_resnet50_v1d
#   simple_pose_resnet101_v1d
#   simple_pose_resnet152_v1d

# Action recognition - models available :
#   ehpi

# Face detection - models available :
#   MTCNN

human_detection = True
pose_estimation = True

face_detection = True
facial_landmarks_estimation = True
facial_emotion_recognition = True

action_recognition = False

pipeline = Pipeline(regime='detection')

if human_detection:
    human_detector = HumanDetector(model_name='yolo3_mobilenet1.0_coco', threshold=0.25, input_size=(512, 1024, 3), do_timing=True)
    multi_human_tracker = MultiObjectTracker(model_name='CSRT', do_timing=True)
if pose_estimation:
    human_pose_estimator = HumanPoseEstimator(model_name='simple_pose_resnet18_v1b', threshold=0.20, do_timing=True)

if face_detection:
    face_detector = FaceDetector(model_name='MTCNN', threshold=0.25, input_size=(256, 512, 3), do_timing=True)
    multi_faces_tracker = MultiObjectTracker(model_name='CSRT', do_timing=True)
if facial_landmarks_estimation:
    facial_landmarks_estimator = FacialLandmarksEstimator(model_name='dlib', input_size=(256, 256, 3), threshold=0.20, do_timing=True)
if facial_emotion_recognition:
    facial_emotion_classifier = FacialEmotionClassifier(model_name='CNN', input_size=(64, 64, 3), threshold=0.20, do_timing=True)

if action_recognition:
    action_classifier = ActionRecognizer(model_name='vgg16_ucf101', threshold=0.10, do_timing=True)


# ---- ---- ---- VIDEO CAPTURE AND PROCESSING ---- ---- ----

frameNr = 0

# Read until video is completed
while(cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    frameNr += 1
    pipeline.frameNr += 1
    print('---- ---- Frame number :', frameNr, '---- ----')


    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, input_resolution)


    # Process frame

    if pipeline.regime == 'detection':

        print('     ---- Detection stage ----     ')

        if human_detection:
            print('Human detection')
            human_bboxes, human_bboxes_confidences = human_detector.predict(image)
            res_human_detection = human_detector.get_result()
            multi_human_tracker.initialize(image, human_bboxes)
            #human_detector.visualize(image, human_bboxes, human_bboxes_confidences)

        if pose_estimation:
            print('Human pose estimation')
            poses, poses_confidences = human_pose_estimator.predict(image, human_bboxes, human_bboxes_confidences)
            res_poses_estimation = human_pose_estimator.get_result()
            #human_pose_estimator.visualize(image, poses, poses_confidences)

        if face_detection:
            print('Face detection')
            face_bboxes, face_bboxes_confidences = face_detector.predict(image)
            res_face_detection = face_detector.get_result()
            multi_faces_tracker.initialize(image, face_bboxes)
            #face_detector.visualize(image, face_bboxes, face_bboxes_confidences)

        if facial_landmarks_estimation:
            print('Facial landmarks estimation')
            facial_landmarks, facial_landmarks_confidences = facial_landmarks_estimator.predict(image, face_bboxes, face_bboxes_confidences)
            res_facial_landmarks_estimation = facial_landmarks_estimator.get_result()
            #facial_landmarks_estimator.visualize(image, facial_landmarks, facial_landmarks_confidences)

        if facial_emotion_recognition:
            print('Facial emotion recognition')
            facial_emotions, facial_emotions_confidences = facial_emotion_classifier.predict(image, face_bboxes, face_bboxes_confidences)
            res_facial_emotion_classification = facial_emotion_classifier.get_result()
            #facial_emotion_classifier.visualize(image, facial_emotions, facial_emotions_confidences)

        if action_recognition:
            print('Action classification')
            action_classifier.predict(image)
            #action_classifier.predict(res_humans)
            #res_action_recognition = action_classifier.get_result()
            #actions, actions_confidences = res_action_recognition.convert_to_list()
            #action_classifier.visualize(image)

        pipeline.match(res_human_detection.bounding_boxes, \
                       res_poses_estimation.poses, \
                       res_face_detection.bounding_boxes, \
                       res_facial_landmarks_estimation.facial_landmarks, \
                       res_facial_emotion_classification.facial_emotions)

        pipeline.visualize(image)
        cv2.putText(image, 'Detection stage', (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

        pipeline.update_regime()


    elif pipeline.regime == 'tracking':

        print('     ---- Tracking stage ----     ')

        #if len(multi_human_tracker.trackers) == 0:
        #    pass
        #if len(multi_faces_tracker.trackers) == 0:
        #    pass

        if human_detection:
            print('Human tracking')
            human_bboxes, human_bboxes_confidences = multi_human_tracker.update(image)
            res_human_detection = multi_human_tracker.get_result()
            #human_detector.visualize(image, human_bboxes, human_bboxes_confidences, color=(0, 255, 0))

        if pose_estimation:
            print('Human pose estimation')
            poses, poses_confidences = human_pose_estimator.predict(image, human_bboxes, human_bboxes_confidences)
            res_poses_estimation = human_pose_estimator.get_result()
            #human_pose_estimator.visualize(image, poses, poses_confidences)

        if face_detection:
            print('Face tracking')
            face_bboxes, face_bboxes_confidences = multi_faces_tracker.update(image)
            res_face_detection = multi_faces_tracker.get_result()
            #face_detector.visualize(image, face_bboxes, face_bboxes_confidences, color=(0, 255, 0))

        if facial_landmarks_estimation:
            print('Facial landmarks estimation')
            facial_landmarks, facial_landmarks_confidences = facial_landmarks_estimator.predict(image, face_bboxes, face_bboxes_confidences)
            res_facial_landmarks_estimation = facial_landmarks_estimator.get_result()
            #facial_landmarks_estimator.visualize(image, facial_landmarks, facial_landmarks_confidences, color=(0, 255, 0))

        if facial_emotion_recognition:
            print('Facial emotion recognition')
            facial_emotions, facial_emotions_confidences = facial_emotion_classifier.predict(image, face_bboxes, face_bboxes_confidences)
            res_facial_emotion_classification = facial_emotion_classifier.get_result()
            #facial_emotion_classifier.visualize(image, facial_emotions, facial_emotions_confidences)

        if action_recognition:
            print('Action classification')
            action_classifier.predict(image)
            #action_recognizer.visualize(image)

        pipeline.match(res_human_detection.bounding_boxes, \
                       res_poses_estimation.poses, \
                       res_face_detection.bounding_boxes, \
                       res_facial_landmarks_estimation.facial_landmarks, \
                       res_facial_emotion_classification.facial_emotions)

        pipeline.visualize(image)
        cv2.putText(image, 'Tracking stage', (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

        pipeline.update_regime()


    # Show and/or write frame

    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Write the frame into the file 'output.avi'
    #out.write(frame)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()
