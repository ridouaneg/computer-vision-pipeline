# code for avoiding Tensorflow to use the GPU
import tensorflow as tf
from keras import backend as K
num_cores = 4
num_CPU = 1
num_GPU = 0
config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)

import numpy as np
import cv2
import time

from Util import Util

class FacialEmotion:

    def __init__(self, facial_emotion, confidence):
        self.facial_emotion = facial_emotion
        self.confidence = confidence


class FacialEmotionClassifierResult:

    def __init__(self, facial_emotions=[]):
        self.facial_emotions = facial_emotions

    def convert_to_list(self):
        emotions = [emotion.facial_emotion for emotion in self.facial_emotions]
        confidences = [emotion.confidence for emotion in self.facial_emotions]
        return emotions, confidences


class FacialEmotionClassifier:

    def __init__(self, model_name='CNN', input_size=(64, 64, 3), threshold=0.20, do_timing=False):
        self.threshold = threshold
        self.input_size = input_size
        self.image_size = None
        self.do_timing = do_timing
        self.result = None

        self.emotion_labels = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'sad', 5:'surprise', 6:'neutral'}

        self.set_model(model_name)

    def get_result(self):
        return self.result

    def predict(self, image, face_bounding_boxes, confidences):

        # pre-process
        pre_process_runtime_start = time.time()
        model_input = self.pre_process(image, face_bounding_boxes)
        pre_process_runtime_end = time.time()

        # model prediction
        model_predict_runtime_start = time.time()
        model_output = self.model_predict(model_input)
        model_predict_runtime_end = time.time()

        # postprocess
        post_process_runtime_start = time.time()
        facial_emotions, confidences = self.post_process(model_output)
        post_process_runtime_end = time.time()

        self.result = FacialEmotionClassifierResult([FacialEmotion(facial_emotion, confidence) for facial_emotion, confidence in zip(facial_emotions, confidences)])

        if self.do_timing:
            print('facial emotion classifier preprocessing time (ms):', (pre_process_runtime_end - pre_process_runtime_start) * 1e3)
            print('facial emotion classifier prediction time (ms):', (model_predict_runtime_end - model_predict_runtime_start) * 1e3)
            print('facial emotion classifier post-processing time (ms):', (post_process_runtime_end - post_process_runtime_start) * 1e3)

        return facial_emotions, confidences

    def set_model(self, model_name):
        if model_name == 'CNN':
            from keras.models import load_model
            emotion_model_path = './models/emotion_classification/cnn_mxnet/emotion_model.hdf5'
            self.model = load_model(emotion_model_path)

    def pre_process(self, image, face_bounding_boxes):

        self.image_size = np.shape(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        emotion_offsets = (20, 40)

        if len(face_bounding_boxes) > 0:

            # upscale bounding boxes
            upscale_bounding_boxes = Util.upscale(face_bounding_boxes, self.image_size, offsets=emotion_offsets)

            # crop and resize each previously detected humans
            model_input = Util.crop_resize(gray_image, upscale_bounding_boxes, self.input_size)

            # normalize the resulting images
            model_input = np.array(model_input)
            model_input = model_input.astype('float32')
            model_input /= 255.
            model_input = (model_input - 0.5) * 2.

            # expand
            model_input = np.expand_dims(model_input, -1)

        else:

            model_input = None
            upscale_bounding_boxes = None

        return model_input

    def model_predict(self, model_input):
        if model_input is not None:
            model_output = self.model.predict(model_input)
        else:
            model_output = np.empty((0, 0))
        return model_output

    def post_process(self, model_output):

        nb_humans = model_output.shape[0]
        nb_emotions = model_output.shape[1]

        #emotion_texts = np.zeros((nb_humans, 1))
        #emotion_probabilities = np.zeros((nb_humans, 1))

        emotion_probabilities = np.max(model_output, axis=1)
        emotion_label_args = np.argmax(model_output, axis=1)
        emotion_texts = [self.emotion_labels[emotion_label_args[i]] for i in range(len(emotion_label_args))]

        return emotion_texts, emotion_probabilities

    def visualize(self, image, emotions=None, confidences=None):

        if emotions is None:
            emotions, confidences = self.result.convert_to_list()
        else:
            emotions, confidences = emotions, confidences

        cv2.putText(image, emotions[0], (5, 75), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 0, 255), 1, cv2.LINE_AA)
