import numpy as np
import cv2
import time

from Util import Util

import dlib
# install mxnet with gpu support and gluoncv with :
# pip install --upgrade mxnet-cu100 gluoncv
import mxnet as mx
from gluoncv.model_zoo import get_model

class FacialLandmarks:

    def __init__(self, landmarks, confidence):
        # the person keypoints in the frame as a numpy array
        self.landmarks = landmarks
        # the keypoints confidences as a numpy array
        self.confidence = confidence
        # the person keypoints in the frame as a dict
        #self.facial_landmarks_dict =


class FacialLandmarksEstimatorResult:

    def __init__(self, facial_landmarks=[]):
        self.facial_landmarks = facial_landmarks

    def convert_to_list(self):
        landmarks = [fl.landmarks for fl in self.facial_landmarks]
        confidences = [fl.confidence for fl in self.facial_landmarks]
        return landmarks, confidences


class FacialLandmarksEstimator:
    """
    This class contains the facial landmarks estimation model.


    Attributes
    ----------
    model_ : str, default : 'simple_pose_resnet50_v1d'
        the name of the model in GluonCV model zoo (https://gluon-cv.mxnet.io/model_zoo/pose.html),
        models available :
            'simple_pose_resnet18_v1b'  -> the fastest one
            'simple_pose_resnet50_v1d'  -> good compromise
            'simple_pose_resnet152_v1d' -> the most accurate
    gpu_device_ :
        the gpu to be used in mxnet format (e.g. mx.gpu(0))
    input_size_ : (height, width, channels), 3-tuple of int
        the input size of the model
    threshold_ : float
        the pose estimation threshold
    do_timing_ : bool
        if True, display the runtime of the pose estimation pipeline

    Methods
    -------
    predict(image, bounding_boxes, frameNr)
        Predict human poses from a RGB image and the previously detected humans
    preprocess(image, bounding_boxes)
        Pre-process the frame before feeding it into the model
    postprocess(predicted_heatmap, bbox)
        Postprocess the model output
    visualize(image)
        Draw bounding boxes on the frame
    """

    def __init__(self, model_name='dlib', input_size=(256, 256, 3), threshold=0.20, do_timing=False):
        self.threshold = threshold
        self.input_size = input_size
        self.image_size = None
        self.do_timing = do_timing
        self.result = None

        self.set_model(model_name)

    def get_result(self):
        return self.result

    def predict(self, image, bounding_boxes, confidences):

        # pre-process
        pre_process_runtime_start = time.time()
        model_input = self.pre_process(image, bounding_boxes, confidences)
        pre_process_runtime_end = time.time()

        # model prediction
        model_predict_runtime_start = time.time()
        model_output = self.model_predict(model_input)
        model_predict_runtime_end = time.time()

        # postprocess
        post_process_runtime_start = time.time()
        poses, confidences = self.post_process(model_output)
        post_process_runtime_end = time.time()

        self.result = FacialLandmarksEstimatorResult([FacialLandmarks(landmarks, confidence) for landmarks, confidence in zip(facial_landmarks, confidences)])

        if self.do_timing:
            print('facial landmarks estimator preprocessing time (ms):', (pre_process_runtime_end - pre_process_runtime_start) * 1e3)
            print('facial landmarks estimator prediction time (ms):', (model_predict_runtime_end - model_predict_runtime_start) * 1e3)
            print('facial landmarks estimator post-processing time (ms):', (post_process_runtime_end - post_process_runtime_start) * 1e3)

        return poses, confidences

    def set_model(self, model_name):
        from dlib import shape_predictor
        self.model = shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

    def pre_process(self, image, bounding_boxes, confidences):
        self.image_size = np.shape(image)
        model_input = [dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3]) for bbox in bounding_boxes]
        return model_input

    def predict(self, model_input):
        if model_input is not None:
            model_output = []
            for d in model_input:
                model_output.append(self.model(image, d))
        else:
            model_output = np.empty((0, 0, 0, 0))
        return model_output

    def post_process(self, model_output):
        poses = [np.array([[model_output[i].part(j).x, model_output[i].part(j).y] for j in range(68)]) for i in range(len(model_output))]
        confidences = [np.ones((68, 1)) for i in range(len(model_output))]
        return poses, confidences

    def visualize(self, image, facial_landmarks=None, confidences=None, color=(255, 0, 0)):
        """Draw facial landmarks on the image

        Parameters
        ----------
        image : numpy array of shape (height, width, channels)
            The image
        """

        if facial_landmarks is None:
            facial_landmarks, facial_landmarks_confidences = self.result.convert_to_list()
        else:
            facial_landmarks, facial_landmarks_confidences = facial_landmarks, confidences

        for i in range(len(facial_landmarks)):

            landmarks = facial_landmarks[i]
            confidences = facial_landmarks_confidences[i]

            for j in range(68):

                x, y = landmarks[j]

                if j in [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]:
                    cv2.circle(image, (int(x), int(y)), 1, (0, 0, 255), -1)
                else:
                    cv2.circle(image, (int(x), int(y)), 1, color, -1)

        for i in range(37, 43):
            cv2.circle(image, (int(x), int(y)), 1, color, -1)
