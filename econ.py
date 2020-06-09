import socket
import numpy as np
import cv2
from scipy.spatial import distance as dist
from imutils import face_utils
import torchvision.transforms as transforms
import imutils
import dlib
import random
import os
import torch
import math





def recvAll(sock, count):
    buf = b''
    while count:
        # count만큼의 byte를 읽어온다. socket이 이미지 파일의 bytes를 한번에 다 못 읽어오므로
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def modeList(L):
    return max(set(L), key=L.count)


def loadStateDict(state_direction_num,state_Q_num):

    state_memory = dict()

    state_memory['GazeRatioLR'] = np.array([1.0 for i in range(state_direction_num)])
    state_memory['GazeRatioTB'] = np.array([1.0 for i in range(state_direction_num)])
    state_memory['FacePointX'] = np.array([200 for i in range(state_direction_num)])
    state_memory['FacePointY'] = np.array([300 for i in range(state_direction_num)])

    state_memory['Click'] = [0 for i in range(state_Q_num)]
    state_memory['Scroll'] = [0 for i in range(state_Q_num)]
    state_memory['FER'] = [0 for i in range(state_Q_num)]

    return state_memory


def rateToDistance(r_x,r_y,width,height, weight = 1.5) :
    d_x = (width*r_x) * weight
    d_y = (height*r_y) * weight
    return int(d_x),int(d_y)


def loadClassifier(util_path) :

    face_haar_path = util_path + 'haarcascade_frontalface_default.xml'
    faceClassifier = cv2.CascadeClassifier(face_haar_path)
    
    return faceClassifier






def getEAR(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def isBlink(eyeLandmark,blink_th,start_idx,end_idx) :

    eye = eyeLandmark[start_idx:end_idx]
    EAR = getEAR(eye)

    if EAR > blink_th:
        return False
    else :
        return True

def getFaceXY(faceImage) :
    face = faceImage[0]
    x = face[0] + face[2]/2
    y = face[1] + face[3]/2
    return x,y

def getGazePoint(model,image,W,H):
    result = model(image[None, ...])[0]
    x_rate = 0
    y_rate = 0
    for i in [0,1,4,5,8,9,12,13] :
        x_rate += result[i]
    for i in [0,1,2,3,4,5,6,7] :
        y_rate += result[i]
    x = (W/4) + (1 - x_rate ) * W/2 
    y = (H/4) + (1 - y_rate ) * H/2 
    return x,y



def getExpression(faceFrame,gray,FERmodel) :


    for (x, y, w, h) in faceFrame:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(cv2.resize(roi_gray, (224, 224)), -1)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        # cropped_img = transforms.ToTensor()
        cropped_img = torch.from_numpy(cropped_img)
        cropped_img = cropped_img.float()
        output = FERmodel(cropped_img)
        _, prediction = torch.max(output, 1)
        prediction = prediction.data[0]
        prediction = prediction.data[0]
        return int(prediction.data[0]) ,( x, y)
    return 0,(0,0)


def getGazeRatio(gray,faceLandmark,eye_points):
    eye_region = np.array([(faceLandmark.part(eye_points[0]).x, faceLandmark.part(eye_points[0]).y),
                                (faceLandmark.part(eye_points[1]).x, faceLandmark.part(eye_points[1]).y),
                                (faceLandmark.part(eye_points[2]).x, faceLandmark.part(eye_points[2]).y),
                                (faceLandmark.part(eye_points[3]).x, faceLandmark.part(eye_points[3]).y),
                                (faceLandmark.part(eye_points[4]).x, faceLandmark.part(eye_points[4]).y),
                                (faceLandmark.part(eye_points[5]).x, faceLandmark.part(eye_points[5]).y)], np.int32)

    height, width = gray.shape
    mask = np.zeros((height, width), np.uint8)
    # cv2.polylines(image, [eye_region], True, 255, 2)
    # cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)


    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])

    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]

    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY_INV)

    height, width = threshold_eye.shape

    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    upper_side_threshold = threshold_eye[0: int(height / 2), 0: width]
    upper_side_white = cv2.countNonZero(upper_side_threshold)

    down_side_threshold = threshold_eye[int(height / 2): height, 0: width]
    down_side_white = cv2.countNonZero(down_side_threshold)

    if left_side_white == 0:
        horizontal_gaze_ratio = 1
    elif right_side_white == 0:
        horizontal_gaze_ratio = 5
    else:
        horizontal_gaze_ratio = left_side_white / right_side_white

    if upper_side_white == 0:
        vertical_gaze_ratio = 1
    elif down_side_white == 0:
        vertical_gaze_ratio = 5
    else:
        vertical_gaze_ratio = upper_side_white / down_side_white

    return horizontal_gaze_ratio, vertical_gaze_ratio

# Model
# def getGazeXY(image,face,eyes) : 
# 	x = 500 + random.randint(-50, 50)
# 	y = 800 + random.randint(-50, 50)
# 	return x,y