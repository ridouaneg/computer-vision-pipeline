import cv2
import time
import numpy as np

class Util:

    @staticmethod
    def upscale_one(bounding_box, image_size, scale=1.25, offsets=None):
        """Increase the size of a bounding box by a given scale, taking into account
        the boundaries of the image.

        Parameters
        ----------
        bounding_box : (xmin, ymin, xmax, ymax), 4-tuple of floats
            The bounding box we want to upscale
        image_size : (height, width), 2-tuple of ints
            The size of the image
        scale : float, default : 1.25
            The scale by which we want to increase the bounding box

        Returns
        ------
        upscale_bounding_box : (xmin, ymin, xmax, ymax), 4-tuple of floats
            The upscaled bounding box
        """
        xmin, ymin, xmax, ymax = bounding_box

        if offsets is None:

            width = xmax - xmin
            height = ymax - ymin
            center = [xmin + (width / 2), ymin + (height / 2)]

            dx = scale * width / 2
            dy = scale * height / 2

            new_xmin = max(center[0] - dx, 0)
            new_ymin = max(center[1] - dy, 0)
            new_xmax = min(center[0] + dx, image_size[1])
            new_ymax = min(center[1] + dy, image_size[0])

        else:

            xoff, yoff = offsets

            new_xmin = max(xmin - xoff, 0)
            new_ymin = max(ymin - yoff, 0)
            new_xmax = min(xmax + xoff, image_size[1])
            new_ymax = min(ymax + yoff, image_size[0])

        upscale_bounding_box = [new_xmin, new_ymin, new_xmax, new_ymax]
        return upscale_bounding_box

    @staticmethod
    def upscale(bounding_boxes, image_size, scale=1.25, offsets=None):
        """Upscale a batch of bounding boxes of the same image."""
        upscale_bounding_boxes = [Util.upscale_one(bounding_box, image_size, scale, offsets) for bounding_box in bounding_boxes]
        return upscale_bounding_boxes

    @staticmethod
    def crop_resize_one(image, bounding_box, output_size=(256, 192)):
        """Crop a part of the image according to bounding box coordinates and resize
         it to the desired size.

        Parameters
        ----------
        image : numpy array of shape (height, width, 3)
            The input image
        bounding_box : (xmin, ymin, xmax, ymax), 4-tuple of floats
            The bounding box of a human detected on the image
        output_size : (height, width), 2-tuple of ints, default : (256, 192)
            The desired size of each detected humans

        Returns
        ------
        cropped_resized_image : numpy array of size (output height, output width, 3)
            The cropped and resized image
        """
        x0, y0, x1, y1 = bounding_box
        x0 = max(int(x0), 0)
        y0 = max(int(y0), 0)
        x1 = min(int(x1), int(image.shape[1]))
        y1 = min(int(y1), int(image.shape[0]))

        # Crop the original image
        tmp = np.array(image, dtype=np.float32)
        cropped_image = tmp[y0:y1, x0:x1]

        # Resize the cropped image
        cropped_resized_image = cv2.resize(cropped_image, (output_size[1], output_size[0]))

        return cropped_resized_image

    @staticmethod
    def crop_resize(image, bounding_boxes, output_size=(256, 192)):
        """Crop and resize a batch of bounding boxes of the same image."""
        cropped_resized_images = [Util.crop_resize_one(image, bounding_box, output_size) for bounding_box in bounding_boxes]
        return cropped_resized_images

    @staticmethod
    def normalize(batch_images, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """Normalize the input to be fed into a mxnet model

        Parameters
        ----------
        batch_images : list of numpy array of same shape (height, width, 3)
            The batch of images we want to normalize
        mean : 3-tuple of floats, default : (0.485, 0.456, 0.406)
        std : 3-tuple of floats, default : (0.229, 0.224, 0.225)

        Returns
        ------
        model_input : numpy array of shape (nb of detections, 3, height, width)
            The cropped and resized image
        """

        # Scale pixel values from [0, 255] to [0, 1]
        model_input = batch_images / 255.

        # Normalize mean and std
        mean, std = np.reshape(mean, (1, 3, 1, 1)), np.reshape(std, (1, 3, 1, 1))
        res_img = (model_input - mean) / std

        return model_input

    @staticmethod
    def heatmap_to_coord(heatmaps, bounding_boxes):
        """Extract keypoints coordinates and confidences from a batch of predicted
        heatmaps from a pose estimation model. For each heatmap, we need to map
        between the point coordinates of the maximum value within the heatmap and
        its coordinates in the original image.

        Parameters
        ----------
        heatmaps : numpy array of shape (nb of detections, nb of keypoints,
        heatmap height, heatmap width)
            The predicted heatmaps output by the pose estimation model
        bounding_boxes : numpy array of shape (nb of detections, 4)
            The bounding boxes coordinates of detected humans on the image

        Returns
        ------
        keypoints : numpy array of shape (nb of detections, nb of keypoints, 2)
            The keypoint coordinates for each detected humans
        confidences : numpy array of shape (nb of detections, nb of keypoints, 1)
            The corresponding confidences for each keypoint
        """
        nb_humans = heatmaps.shape[0]
        nb_keypoints = heatmaps.shape[1]
        heatmap_height, heatmap_width = heatmaps.shape[2], heatmaps.shape[3]

        keypoints = np.zeros((nb_humans, nb_keypoints, 2))
        confidences = np.zeros((nb_humans, nb_keypoints, 1))

        for i in range(nb_humans):
            xmin, ymin, xmax, ymax = bounding_boxes[i]
            width_ratio = (xmax - xmin) / heatmap_width
            height_ratio = (ymax - ymin) / heatmap_height
            for j in range(nb_keypoints):
                heatmap = heatmaps[i][j]
                y0, x0 = np.unravel_index(heatmap.argmax(), heatmap.shape)
                confidences[i][j][0] = heatmap[y0, x0]
                x, y = xmin + width_ratio * x0, ymin + height_ratio * y0
                keypoints[i][j][0], keypoints[i][j][1] = x, y
        return keypoints, confidences

    @staticmethod
    def bbox_distance_euclidean(bbox1, bbox2):
        xmin1, ymin1, xmax1, ymax1 = bbox1
        xmin2, ymin2, xmax2, ymax2 = bbox2
        center1 = np.array([(xmax1 - xmin1) / 2, (ymax1 - ymin1) / 2])
        center2 = np.array([(xmax2 - xmin2) / 2, (ymax2 - ymin2) / 2])
        dist = np.linalg.norm(center2 - center1)
        return dist

    @staticmethod
    def bbox_distance_iou(bbox1, bbox2):
        xmin1, ymin1, xmax1, ymax1 = bbox1
        xmin2, ymin2, xmax2, ymax2 = bbox2

        xA = max(xmin1, xmin2)
        yA = max(ymin1, ymin2)
        xB = min(xmax1, xmax2)
        yB = min(ymax1, ymax2)

        intersection = max(0, xB - xA) * max(0, yB - yA)

        bbox1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
        bbox2_area = (xmax2 - xmin2) * (ymax2 - ymin2)

        union = bbox1_area + bbox2_area - intersection

        IoU = intersection / union

        return IoU

    @staticmethod
    def normalize_ehpi(unnormalized_ehpi):
        """Normalize an unnormalized Encoded Human Pose Image (ehpi)
        We don't normalize according the image size nor to the bounding box
        size BUT according to the amplitude of the movement, i.e. we consider
        the min and max coordinates of the keypoints during the whole action
        and normalize each keypoint according to these values.

        Parameters
        ----------
        unnormalized_ehpi : numpy array of shape (m, n, 2)
        with m, number of frames, and n, number of keypoints.
            The unnormalized Encoded Human Pose Image (ehpi)

        Returns
        ------
        normalized_ehpi : numpy array of shape (m, n, 2)
        with m, number of frames, and n, number of keypoints.
            The normalized Encoded Human Pose Image (ehpi)
        """

        normalized_ehpi = np.copy(unnormalized_ehpi)

        normalized_ehpi = np.transpose(normalized_ehpi, (2, 0, 1))

        xmin, ymin = np.min(normalized_ehpi[0, :, :]), np.min(normalized_ehpi[1, :, :])
        xmax, ymax = np.max(normalized_ehpi[0, :, :]), np.max(normalized_ehpi[1, :, :])

        normalized_ehpi[0, :, :] = (normalized_ehpi[0, :, :] - xmin) / (xmax - xmin)
        normalized_ehpi[1, :, :] = (normalized_ehpi[1, :, :] - ymin) / (ymax - ymin)

        normalized_ehpi = np.transpose(normalized_ehpi, (2, 1, 0))

        return normalized_ehpi

    @staticmethod
    def convert_to_ehpi(temporal_poses, m=32, n=15):
        """Convert poses through time to an Encoded Human Pose Image

        Parameters
        ----------
        pose_list: numpy array of shape (f, n, 2)
            The keypoints detected for one person during a video, f is the number of frames and n the number of keypoints

        Returns
        ------
        ehpi : numpy array of shape (m, n, 2) with m, number of considered frames, and n, number of keypoints
            The Encoded Human Pose Image
        """
        f = temporal_poses.shape[0]

        if f > m:
            ehpi = temporal_poses[f-m:f, :, :]
        else:
            ehpi = np.zeros((m, n, 2))
            ehpi[m-f:m, :, :] = temporal_poses

        ehpi = Util.normalize_ehpi(ehpi)

        return ehpi

    @staticmethod
    def convert_pose_coco_to_mpii(pose_coco):
        pose_mpii = np.zeros((15, 2))

        pose_mpii[4] = pose_coco[5]
        pose_mpii[8] = pose_coco[7]
        pose_mpii[12] = pose_coco[9]

        pose_mpii[3] = pose_coco[6]
        pose_mpii[7] = pose_coco[8]
        pose_mpii[11] = pose_coco[10]

        pose_mpii[6] = pose_coco[11]
        pose_mpii[10] = pose_coco[13]
        pose_mpii[14] = pose_coco[15]

        pose_mpii[5] = pose_coco[12]
        pose_mpii[9] = pose_coco[14]
        pose_mpii[13] = pose_coco[16]

        pose_mpii[0] = (pose_coco[6] + pose_coco[5]) / 2
        pose_mpii[1] = (pose_coco[12] + pose_coco[11]) / 2
        pose_mpii[2] = pose_coco[0]

        return pose_mpii

    @staticmethod
    def visualize_ehpi(ehpi):
        tmp = np.zeros((ehpi.shape[0], ehpi.shape[1], 3))
        for i in range(ehpi.shape[0]):
            for j in range(ehpi.shape[1]):
                tmp[i, j, :][0] = ehpi[i, j, 0]
                tmp[i, j, :][1] = ehpi[i, j, 1]
                tmp[i, j, :][2] = 0.
        img_ehpi = (tmp * 255.).astype(np.uint8)
        return img_ehpi
