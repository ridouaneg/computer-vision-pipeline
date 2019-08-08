import numpy as np
import cv2
import time

# install mxnet and gluoncv on windows via 'pip install mxnet-cu90 gluoncv'. Note you must have installed also Cuda Toolkit 9.0
import mxnet as mx
from gluoncv import data

from Util import Util

class BoundingBox:
    """
    This class contains information of an object bounding box.

    For each detected objects, we store the corresponding bounding box into this
    class. We can then access to the bounding box coordinates, confidence and a
    dictionnary which contains


    Attributes
    ----------
    boundingBox : numpy array of shape (4, 1) and type float, [xmin, ymin, xmax, ymax]
    confidence : float
    boundingBoxDict : dict
    """

    def __init__(self, bounding_box, confidence):
        self.bounding_box = bounding_box
        self.confidence = confidence
        self.bounding_box_dict = {
            'xmin':bounding_box[0],
            'ymin':bounding_box[1],
            'xmax':bounding_box[2],
            'ymax':bounding_box[3],
            'w':bounding_box[2] - bounding_box[0],
            'h':bounding_box[3] - bounding_box[1],
            'confidence':confidence
        }

class ObjectDetectorResult:
    """
    This class contains information of the result of an object detector.

    When we have decteted several objects, we store them into a list of elements
    of type 'BoungingBox'


    Attributes
    ----------
    bounding_boxes : list of 'BoundingBox' objects

    Methods
    -------
    convert_to_list()
        Extract bounding boxes coordinates and confidences from the list of
        'BoundingBox' objects
    """

    def __init__(self, bounding_boxes=[]):
        self.bounding_boxes = bounding_boxes

    def convert_to_list(self):
        """Extract bounding boxes coordinates and confidences from the list of
        'BoundingBox' objects

        Returns
        ------
        bounding_boxes : list of numpy array of shape (4, 1) and type float
        scores : list of float
        """
        bounding_boxes = [bbox.bounding_box for bbox in self.bounding_boxes]
        confidences = [bbox.confidence for bbox in self.bounding_boxes]
        return bounding_boxes, confidences

class ObjectDetector:
    """
    This class contains the object detection model.

    When we have decteted several objects, we store them into a list of elements
    of type 'BoungingBox'


    Attributes
    ----------
    bounding_boxes : list of 'BoundingBox' objects

    Methods
    -------
    convert_to_list()
        Extract bounding boxes coordinates and confidences from the list of
        'BoundingBox' objects
    """

    def __init__(self, threshold=0.25, input_size=(512, 1024, 3), do_timing=False):
        self.threshold = threshold
        self.input_size = input_size
        self.image_size = None
        self.do_timing = do_timing
        self.result = None
        self.model_name = None

    def get_result(self):
        return self.result

    def predict(self, image):

        # pre-process
        pre_process_runtime_start = time.time()
        model_input = self.pre_process(image)
        pre_process_runtime_end = time.time()

        # model prediction
        model_predict_runtime_start = time.time()
        model_output = self.model_predict(model_input)
        model_predict_runtime_end = time.time()

        # postprocess
        post_process_runtime_start = time.time()
        bounding_boxes, confidences = self.post_process(model_output)
        post_process_runtime_end = time.time()

        self.result = ObjectDetectorResult([BoundingBox(bounding_box, confidence) for bounding_box, confidence in zip(bounding_boxes, confidences)])

        if self.do_timing:
            print('object detector preprocessing time (ms):', (pre_process_runtime_end - pre_process_runtime_start) * 1e3)
            print('object detector prediction time (ms):', (model_predict_runtime_end - model_predict_runtime_start) * 1e3)
            print('object detector post-processing time (ms):', (post_process_runtime_end - post_process_runtime_start) * 1e3)

        return bounding_boxes, confidences


    def visualize(self, image, bounding_boxes=None, confidences=None, color=(255, 0, 0)):
        """Draw detected bounding boxes on the image

        Parameters
        ----------
        image : numpy array of shape (height, width, channels)
            The image
        """
        if bounding_boxes is None:
            bounding_boxes, confidences = self.result.convert_to_list()
        else:
            bounding_boxes, confidences = bounding_boxes, confidences

        for k in range(len(bounding_boxes)):

            xmin, ymin, xmax, ymax = bounding_boxes[k]
            confidence = round(confidences[k], 2)

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.putText(image, str(confidence), (xmax, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, color)


class HumanDetector(ObjectDetector):
    """
    This class contains the human detection model.


    Attributes
    ----------
    model_name : str
        the name of the model in GluonCV model zoo (https://gluon-cv.mxnet.io/model_zoo/detection.html),
        models available :
            'yolo3_mobilenet1.0_coco' -> the fastest one
            'yolo3_darknet53_coco' -> good compromise
            'faster_rcnn_fpn_resnet101_v1d_coco' -> the most accurate
    input_size : (height, width, channels), 3-tuple of int
        the input size of the model
    threshold : float
        the detection threshold
    do_timing : bool
        if True, display the runtime of the detection pipeline

    Methods
    -------
    set_model(model_name)
    preprocess(image)
        Pre-process the frame before feeding it into the model
    model_predict(image, frameNr)
        Predict human bounding boxes from a RGB image
    postprocess(output_bounding_boxes, output_scores)
        Postprocess the model output
    visualize(image)
        Draw bounding boxes on the frame
    """

    def __init__(self, model_name='yolo3_darknet53_coco', threshold=0.25, input_size=(512, 1024, 3), do_timing=False):
        super().__init__(threshold, input_size, do_timing)
        self.set_model(model_name)

    def set_model(self, model_name):
        from gluoncv.model_zoo import get_model
        self.model = get_model(model_name, pretrained=True, ctx=mx.gpu(0))
        self.model.reset_class(['person'], reuse_weights=['person'])

    def pre_process(self, image):
        """Pre-process the image before feeding it into the model :
        1 - save image size
        2 - resize the image
        3 - normalize the image
        4 - convert the image to mxnet ndarray, swap axes and put on gpu

        Parameters
        ----------
        image : numpy array of shape (height, width, channels)
            The input image

        Returns
        ------
        model_input : mxnet ndarray of shape (1, 3, input size width, input size height)
            The input to the model
        """

        # save image size
        self.image_size = np.shape(image)

        # resize, normalize, swap axes, convert to mx.ndarray and put on gpu
        model_input, _ = data.transforms.presets.yolo.transform_test(mx.nd.array(image), short=self.input_size[0])
        model_input = model_input.as_in_context(mx.gpu(0))

        return model_input

    def model_predict(self, model_input):
        model_output = self.model(model_input)
        return model_output

    def post_process(self, model_output):
        """Postprocess the model output :
        - ignore bounding boxes below the confidence threshold
        - upscale bounding boxes to the original image size

        Parameters
        ----------
        output_bounding_boxes : numpy array of shape (d, 4, 1)
            The bounding boxes output by the model
        output_scores : numpy array of shape (d, 1, 1)
            The scores output by the model

        Returns
        ------
        bounding_boxes : list of numpy array of shape (4, 1)
            The bounding boxes coordinates corresponding to the detected humans
        scores : list of numpy array of shape (1, 1)
            The corresponding confidences for each bounding box
        """

        _, output_confidences, output_bounding_boxes = model_output
        output_confidences = output_confidences[0].asnumpy() # np array of shape (d, 1, 1)
        output_bounding_boxes = output_bounding_boxes[0].asnumpy() # np array of shape (d, 4, 1)

        bounding_boxes = []
        confidences = []

        nb_dets = output_bounding_boxes.shape[0]

        width_ratio = self.image_size[0] / self.input_size[0]
        height_ratio = self.image_size[1] / self.input_size[1]

        for i in range(nb_dets):

            confidence = output_confidences[i][0]

            if confidence < self.threshold:
                continue

            xmin, ymin, xmax, ymax = output_bounding_boxes[i]

            xmin *= width_ratio
            ymin *= height_ratio
            xmax *= width_ratio
            ymax *= height_ratio

            bounding_boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
            confidences.append(confidence)

        return bounding_boxes, confidences


class MtcnnFaceDetector(ObjectDetector):

    def __init__(self, model_name='MTCNN', threshold=0.25, input_size=(512, 1024, 3), do_timing=False):
        super().__init__(threshold, input_size, do_timing)
        self.set_model(model_name)

    def set_model(self, model_name):
        if model_name == 'MTCNN':
            from mtcnn.mtcnn import MTCNN
            self.model = MTCNN()
        #elif model_name == 'cv2_Haar':
        #elif model_name == 'dlib_HOG':
        #    from dlib import get_frontal_face_detector
        #    self.model = get_frontal_face_detector()
        #elif model_name == 'dlib_CNN':
        #    from dlib import get_frontal_face_detector
        #    dlib.cnn_face_detection_model_v1('./models/dlib_cnn_face_detector.dat')

    def pre_process(self, image):
        self.image_size = np.shape(image)
        model_input = cv2.resize(image, (self.input_size[1], self.input_size[0]))
        return model_input

    def model_predict(self, model_input):
        model_output = self.model.detect_faces(model_input)
        return model_output

    def post_process(self, model_output):

        output_bounding_boxes = [model_output[i]['box'] for i in range(len(model_output))]
        output_confidences = [[model_output[i]['confidence']] for i in range(len(model_output))]

        bounding_boxes = []
        confidences = []

        nb_dets = len(output_bounding_boxes)

        width_ratio = self.image_size[0] / self.input_size[0]
        height_ratio = self.image_size[1] / self.input_size[1]

        for i in range(nb_dets):

            confidence = output_confidences[i][0]

            if confidence < self.threshold:
                continue

            xmin, ymin, w, h = output_bounding_boxes[i]
            xmax, ymax = xmin + w, ymin + h

            xmin *= width_ratio
            ymin *= height_ratio
            xmax *= width_ratio
            ymax *= height_ratio

            bounding_boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
            confidences.append(confidence)

        return bounding_boxes, confidences


class FaceDetector(ObjectDetector):

    def __init__(self, model_name='MTCNN', threshold=0.25, input_size=(512, 1024, 3), do_timing=False):
        super().__init__(threshold, input_size, do_timing)
        self.set_model(model_name)

    def set_model(self, model_name):
        from mtcnn_detector import MtcnnDetector
        self.model = MtcnnDetector(model_folder='./models/mtcnn_weights')

    def pre_process(self, image):
        self.image_size = np.shape(image)
        model_input = cv2.resize(image, (self.input_size[1], self.input_size[0]))
        model_input = cv2.cvtColor(model_input, cv2.COLOR_RGB2BGR)
        return model_input

    def model_predict(self, model_input):
        model_output = self.model.detect_face(model_input)
        if model_output is not None:
            model_output = model_output[0]
        else:
            model_output = []
        return model_output

    def post_process(self, model_output):

        output_bounding_boxes = [[model_output[i][0], model_output[i][1], model_output[i][2] - model_output[i][0], model_output[i][3] - model_output[i][1]] for i in range(len(model_output))]
        output_confidences = [[model_output[i][4]] for i in range(len(model_output))]

        bounding_boxes = []
        confidences = []

        nb_dets = len(output_bounding_boxes)

        width_ratio = self.image_size[0] / self.input_size[0]
        height_ratio = self.image_size[1] / self.input_size[1]

        for i in range(nb_dets):

            confidence = output_confidences[i][0]

            if confidence < self.threshold:
                continue

            xmin, ymin, w, h = output_bounding_boxes[i]
            xmax, ymax = xmin + w, ymin + h

            xmin *= width_ratio
            ymin *= height_ratio
            xmax *= width_ratio
            ymax *= height_ratio

            bounding_boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
            confidences.append(confidence)

        return bounding_boxes, confidences
