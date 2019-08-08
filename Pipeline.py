import random
import string
import cv2
import numpy as np
import scipy

from Util import Util
from ObjectDetector import BoundingBox

class Person:

    def __init__(self, human_bounding_box=None, human_pose=None, face_bounding_box=None, facial_landmarks=None, facial_emotion=None, id='random'):
        if id == 'random':
            self.person_id = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(8)])

        self.human_bounding_box_path = [human_bounding_box]
        self.human_pose_path = [human_pose]
        self.face_bounding_box_path = [face_bounding_box]
        self.facial_landmarks_path = [facial_landmarks]
        self.facial_emotion_path = [facial_emotion]

class Pipeline:

    def __init__(self, humans=[], regime='detection'):
        self.humans = humans
        self.regime = regime
        self.frameNr = 0

    def update_regime(self):
        if self.frameNr % 12 == 1:
            self.regime = 'detection'
        else:
            self.regime = 'tracking'

    def match_faces_with_body(self, human_bboxes, face_bboxes):
        face_indices = [0] * len(human_bboxes)
        for i in range(len(face_bboxes)):

            xmin1, ymin1, xmax1, ymax1 = face_bboxes[i].bounding_box
            face_area = (xmax1 - xmin1) * (ymax1 - ymin1)

            for j in range(len(human_bboxes)):

                xmin2, ymin2, xmax2, ymax2 = human_bboxes[j].bounding_box

                xA = max(xmin1, xmin2)
                yA = max(ymin1, ymin2)
                xB = min(xmax1, xmax2)
                yB = min(ymax1, ymax2)

                intersection = max(0, xB - xA) * max(0, yB - yA)


                ### to change : a face corresponds to a body no just because the bounding box are inside one another
                if intersection == face_area:
                    face_indices[j] = i

        return face_indices

    def get_last_human_bboxes(self):
        prev_human_bboxes = [self.humans[i].human_bounding_box_path[-1].bounding_box for i in range(len(self.humans))]
        prev_human_bboxes_confidences = [self.humans[i].human_bounding_box_path[-1].confidence for i in range(len(self.humans))]
        return prev_human_bboxes, prev_human_bboxes_confidences

    def match(self, new_human_detections, new_pose_estimations, new_face_detections, new_facial_landmarks, new_facial_emotions):

        face_indices = self.match_faces_with_body(new_human_detections, new_face_detections)

        distances = np.array([[Util.bbox_distance_iou(new_human_detections[i].bounding_box, self.humans[j].human_bounding_box_path[-1].bounding_box) \
                                                    for j in range(len(self.humans))]
                                                    for i in range(len(new_human_detections))])
        if distances.shape[0] == 0:
            row_ind, col_ind = [], []
        else:
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(- distances)

        unmatched_tracking_indices = []
        for j in range(len(self.humans)):
            # unmatched tracking
            if j not in col_ind:
                unmatched_tracking_indices.append(j)

        matched_detection_indices, unmatched_detection_indices = [], []
        for i in range(len(new_human_detections)):
            # matched detection
            if i in row_ind:
                j = col_ind[np.where(row_ind == i)[0][0]]
                matched_detection_indices.append([i, j])

            # unmatched detection
            else:
                unmatched_detection_indices.append(i)

        humans = []

        for inds in matched_detection_indices:
            i, j = inds
            human = self.humans[j]
            human.human_bounding_box_path.append(new_human_detections[i])
            human.human_pose_path.append(new_pose_estimations[i])
            human.face_bounding_box_path.append(new_face_detections[face_indices[i]])
            human.facial_landmarks_path.append(new_facial_landmarks[face_indices[i]])
            human.facial_emotion_path.append(new_facial_emotions[face_indices[i]])
            humans.append(human)

        for ind in unmatched_detection_indices:
            bounding_box = new_human_detections[ind]
            pose = new_pose_estimations[ind]
            face_bounding_box = new_face_detections[face_indices[ind]]
            facial_landmarks = new_facial_landmarks[face_indices[ind]]
            facial_emotion = new_facial_emotions[face_indices[ind]]
            humans.append(Person(bounding_box, pose, face_bounding_box, facial_landmarks, facial_emotion))

        self.humans = humans

    def update(self, new_human_detections, new_pose_estimations):

        for k in range(len(self.humans)):
            bbox = new_detections.boundingBoxes[k]

            pose = new_poses.personPoses[k]
            prev_pose = self.humans[k].posePath[-1]

            for i in range(17):
                if pose.keypoints[i][0] == -1:
                    pose.keypoints[i][0] = prev_pose.keypoints[i][0]
                    pose.keypoints[i][1] = prev_pose.keypoints[i][1]

            self.humans[k].bboxPath.append(bbox)
            self.humans[k].posePath.append(pose)

    def visualize(self, image):

        if self.regime == 'detection':
            col = (255, 0, 0)
        elif self.regime == 'tracking':
            col = (0, 255, 0)

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

        for human in self.humans:

            id = human.person_id

            xmin, ymin, xmax, ymax = human.human_bounding_box_path[-1].bounding_box
            score = human.human_bounding_box_path[-1].confidence

            cv2.putText(image, id, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, col)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), col, 1)
            cv2.putText(image, str(round(score, 2)), (xmax, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, col)

            xmin, ymin, xmax, ymax = human.face_bounding_box_path[-1].bounding_box
            score = human.face_bounding_box_path[-1].confidence

            cv2.putText(image, id, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, col)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), col, 1)
            cv2.putText(image, str(round(score, 2)), (xmax, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, col)

            emotion = human.facial_emotion_path[-1].facial_emotion
            confidence = human.facial_emotion_path[-1].confidence

            cv2.putText(image, emotion, (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, 1, col)
            #cv2.putText(image, confidence, (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, 1, col)

            keypoints = human.human_pose_path[-1].keypoints
            score = human.human_pose_path[-1].confidence

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


            landmarks, confidences = \
                    human.facial_landmarks_path[-1].landmarks, \
                    human.facial_landmarks_path[-1].confidence

            eye1 = [landmarks[j] for j in range(36, 42)]
            eye2 = [landmarks[j] for j in range(42, 48)]
            ear1 = (np.linalg.norm(eye1[1] - eye1[5]) + np.linalg.norm(eye1[2] - eye1[4])) / (2 * np.linalg.norm(eye1[0] - eye1[3]))
            ear2 = (np.linalg.norm(eye2[1] - eye2[5]) + np.linalg.norm(eye2[2] - eye2[4])) / (2 * np.linalg.norm(eye2[0] - eye2[3]))
            print('EAR 1:', ear1)
            print('EAR 2 :', ear2)

            mouth = [landmarks[j] for j in range(48, 60)]
            mar = (np.linalg.norm(mouth[2] - mouth[10]) + np.linalg.norm(mouth[4] - mouth[8])) / (2 * np.linalg.norm(mouth[0] - mouth[6]))
            print('MAR :', mar)

            for j in range(68):

                x, y = landmarks[j]

                if j in [k for k in range(36, 48)]:
                    cv2.circle(image, (int(x), int(y)), 1, (0, 0, 255), -1)
                elif j in [k for k in range(48, 62)]:
                    cv2.circle(image, (int(x), int(y)), 1, (255, 0, 255), -1)
                else:
                    cv2.circle(image, (int(x), int(y)), 1, color, -1)
