import numpy as np
import cv2
import time

from Util import Util

# install mxnet with gpu support and gluoncv with :
# pip install --upgrade mxnet-cu100 gluoncv
import mxnet as mx

class HumanPose:

    def __init__(self, keypoints, confidence):
        # the person keypoints in the frame as a numpy array
        self.keypoints = keypoints
        # the keypoints confidences as a numpy array
        self.confidence = confidence
        # the person keypoints in the frame as a dict
        self.human_pose_dict = {
            'nose' : keypoints[0],
            'left eye' : keypoints[1], 'right eye' : keypoints[2],
            'left ear' : keypoints[3], 'right ear' : keypoints[4],

            'left shoulder' : keypoints[5], 'right shoulder' : keypoints[6],
            'left elbow' : keypoints[7], 'right elbow' : keypoints[8],
            'left wrist' : keypoints[9], 'right wrist' : keypoints[10],

            'left hip' : keypoints[11], 'right hip' : keypoints[12],
            'left knee' : keypoints[13], 'right knee' : keypoints[14],
            'left ankle' : keypoints[15], 'right ankle' : keypoints[16]
        }


class HumanPoseEstimatorResult:

    def __init__(self, poses=[]):
        self.poses = poses

    def convert_to_list(self):
        keypoints = [pose.keypoints for pose in self.poses]
        confidences = [pose.confidence for pose in self.poses]
        return keypoints, confidences


# 'PoseEstimator' invokes the mxnet implementation (in gluoncv) of the paper 'Simple Baselines for Human Pose estimation and Tracking'
# (https://arxiv.org/abs/1804.06208). Actually, the method is quite good. See https://gluon-cv.mxnet.io/model_zoo/pose.html
class HumanPoseEstimator:
    """
    This class contains the pose estimation model.


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

    def __init__(self, model_name='simple_pose_resnet50_v1d', input_size=(256, 192, 3), threshold=0.20, do_timing=False):
        self.threshold = threshold
        self.input_size = input_size
        self.image_size = None
        self.do_timing = do_timing
        self.result = None

        self.set_model(model_name)

    def set_model(self, model_name):
        from gluoncv.model_zoo import get_model
        self.model = get_model(model_name, pretrained=True, ctx=mx.gpu(0))

    def get_result(self):
        return self.result

    def predict(self, image, bounding_boxes, bounding_boxes_confidences):

        # pre-process
        pre_process_runtime_start = time.time()
        model_input = self.pre_process(image, bounding_boxes, bounding_boxes_confidences)
        pre_process_runtime_end = time.time()

        # model prediction
        model_predict_runtime_start = time.time()
        model_output = self.model_predict(model_input)
        model_predict_runtime_end = time.time()

        # postprocess
        post_process_runtime_start = time.time()
        poses, confidences = self.post_process(model_output)
        post_process_runtime_end = time.time()

        self.result = HumanPoseEstimatorResult([HumanPose(keypoints, confidence) for keypoints, confidence in zip(poses, confidences)])

        if self.do_timing:
            print('human pose estimator preprocessing time (ms):', (pre_process_runtime_end - pre_process_runtime_start) * 1e3)
            print('human pose estimator prediction time (ms):', (model_predict_runtime_end - model_predict_runtime_start) * 1e3)
            print('human pose estimator post-processing time (ms):', (post_process_runtime_end - post_process_runtime_start) * 1e3)

        return poses, confidences

    def pre_process(self, image, bounding_boxes, bounding_boxes_confidences):
        """Pre-process the frame before feeding it into the model :
        1 - save image size
        2 - upscale bounding boxes to be sure to have the full human body
        3 - crop and resize each previously detected humans
        4 - normalize the resulting images
        5 - convert the images to mxnet ndarray, swap axes and put on gpu

        Parameters
        ----------
        image : numpy array of shape (height, width, 3)
            The input image
        bounding_boxes : [(xmin, ymin, xmax, ymax), ... ], list of 4-tuple of floats
            The bounding boxes coordinates of detected humans on the image

        Returns
        ------
        pose_input : mxnet array of shape ()
            The input to the model
        upscale_bounding_boxes : [(xmin, ymin, xmax, ymax), ... ], list of 4-tuple of floats
            The upscaled bounding boxes
        """

        self.image_size = np.shape(image)

        if len(bounding_boxes) > 0:

            # upscale bounding boxes
            upscale_bounding_boxes = Util.upscale(bounding_boxes, self.image_size)

            # crop and resize each previously detected humans
            model_input = Util.crop_resize(image, upscale_bounding_boxes, self.input_size)

            # swap axes
            model_input = np.swapaxes(model_input, 1, 3)
            model_input = np.swapaxes(model_input, 2, 3)

            # normalize the resulting images
            model_input = Util.normalize(model_input)

            # convert the images to mxnet ndarray and put on gpu
            model_input = mx.nd.array(model_input, ctx=mx.gpu(0))

        else:

            model_input = None
            upscale_bounding_boxes = None

        self.upscale_bounding_boxes = upscale_bounding_boxes

        return model_input

    def model_predict(self, model_input):

        if model_input is not None:
            model_output = self.model(model_input)
            model_output = model_output.asnumpy() # np array of shape (d, 17, hm_h, hm_w)
        else:
            model_output = np.empty((0, 0, 0, 0))

        return model_output

    def post_process(self, model_output):
        """Post-process the heatmaps output by the model :
        - ignore bounding boxes below the confidence threshold
        - extract keypoints coordinates and scores from predicted heatmaps

        Parameters
        ----------
        predicted_heatmap : numpy array of shape (nb of detections, nb of
        keypoints, heatmap height, heatmap width)
            The output of the model
        bbox : [(xmin, ymin, xmax, ymax), ... ], list of 4-tuple of floats
            The bounding boxes coordinates of detected humans on the image
        threshold : float, default : 0.20
            Keypoints with a lower confidence than this value are set to (-1, -1)

        Returns
        ------
        pred_coords : numpy array of shape (nb of detections, nb of keypoints, 2)
            The keypoint coordinates for each detected humans
        confidence : numpy array of shape (nb of detections, nb of keypoints, 1)
            The corresponding confidences for each keypoint
        """

        # extract keypoints coordinates from predicted heatmaps
        poses, confidences = Util.heatmap_to_coord(model_output, self.upscale_bounding_boxes)

        # remove keypoints with a low confidence
        for k in range(poses.shape[0]):
            for i in range(poses.shape[1]):
                if confidences[k][i][0] < self.threshold:
                    poses[k][i][0] = -1
                    poses[k][i][1] = -1
                    confidences[k][i][0] = -1

        return poses, confidences

    def visualize(self, image, poses=None, confidences=None):
        """Draw poses on the frame

        Parameters
        ----------
        image : numpy array of shape (height, width, channels)
            The image
        """

        color_dict = {
            # head
            '0':(255, 255, 255), # center
            '1':(255, 255, 0), '3':(255, 255, 0), # left part
            '2':(255, 0, 0), '4':(255, 0, 0), # right part

            # arms
            '5':(255, 255, 255), '7':(0, 255, 255), '9':(0, 255, 255), # left arm
            '6':(255, 255, 255), '8':(0, 255, 0), '10':(0, 255, 0), # right arm

            # legs
            '11':(255, 255, 255), '13':(255, 0, 255), '15':(255, 0, 255), #left leg
            '12':(255, 255, 255), '14':(0, 0, 255), '16':(0, 0, 255) # right leg
        }

        joint_pairs = [
            [1,0], [1,3], [2,0], [2,4],
            [5,6], [11,12], [5,11], [6,12],
            [7,5], [7,9], [8,6], [8,10],
            [13,11], [13,15], [14,12], [14,16]
        ]

        if poses is None:
            poses, scores = self.result.convert_to_list()
        else:
            poses, scores = poses, confidences


        for i in range(len(poses)):

            keypoints = poses[i]
            score = scores[i]

            for j in range(keypoints.shape[0]):

                if score[j] == -1:
                    continue

                x, y = keypoints[j]
                color = color_dict[str(j)]

                # draw circle
                cv2.circle(image, (int(x), int(y)), 1, color, -1)

                # write text
                txt = str(round(score[j][0], 2))
                cv2.putText(image, txt, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                # draw lines
                for joint in joint_pairs:
                    if joint[0] == j and score[joint[1]][0] != -1:
                        tmp_x, tmp_y = keypoints[joint[1]]
                        cv2.line(image, (int(x), int(y)), (int(tmp_x), int(tmp_y)), color, 1)
