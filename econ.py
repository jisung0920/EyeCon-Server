import socket
import numpy as np
import cv2
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import random
import os
import torch
import eModel
import math


def getCascade(util_path) :

    face_haar_path = util_path + 'haarcascade_frontalface_default.xml'
    eye_haar_path  = util_path + 'haarcascade_eye.xml'
    
    faceCascade = cv2.CascadeClassifier(face_haar_path)
    eyeCascade = cv2.CascadeClassifier(eye_haar_path)
    
    return faceCascade,eyeCascade  




def getFaceEyePoint(frame,faceCascade, eyeCascade) :
    
    faces = faceCascade.detectMultiScale(frame,1.2,cv2.COLOR_BGR2GRAY)
    faceFrame = frame
    for (x,y,w,h) in faces :
        faceFrame = frame[y:y+h,x:x+w]
    eyes=eyeCascade.detectMultiScale(faceFrame,1.2,cv2.COLOR_BGR2GRAY)
    
    return faces,eyes



def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

        
def frameBlinkChecker(eyeLandmark,blink_th,start_idx,end_idx) :

    eye = eyeLandmark[start_idx:end_idx]
    EAR = eye_aspect_ratio(eye)

    if EAR > blink_th:
        return True
    else :
        return False


def getGazeXY(model,image):
    tmp = model(image[None, ...])[0]
    x = tmp[0].item()
    y = tmp[1].item()
    return x,y

def getFaceXY(faces) :
    face = faces[0]
    x = face[0] + face[2]/2
    y = face[1] + face[3]/2
    return x,y

def getXY(cur_point,next_point,count,diff_TH,freeze_TH,momentum=0.8) :
    
    if(count == -1) :

        return next_point

    point_dff = dist.euclidean(cur_point,next_point)
    
    if(point_dff >diff_TH) :
        count +=1
        if(count>freeze_TH) :

            return next_point
        return cur_point

    return (next_point * momentum) + (cur_point* (1-momentum))
        
def recvAll(sock, count):
    buf = b''
    while count:
        # count만큼의 byte를 읽어온다. socket이 이미지 파일의 bytes를 한번에 다 못 읽어오므로
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def load_checkpoint(filename='./checkpoint.pth.tar'):
    print(filename)
    if not os.path.isfile(filename):
        print('Cant load')
        return None
    
    device = torch.device('cpu')
    state = torch.load(filename,map_location=device)
    return state

def recGenerator(image,faces,eyes) :
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    # for (x,y,w,h) in eyes:
    #     cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

def strechingPoint(x,y,beforeW,beforeH,afterW,afterH,strechFactor=1.5):
    return x*(afterW/beforeW) *strechFactor , y* (afterH/beforeH) *strechFactor

def transferRateToDistance(r_x,r_y,width,height, weight = 1.5) :
    d_x = (width*r_x) * weight
    d_y = (height*r_y) * weight
    return int(d_x),int(d_y)

def getFERModel(filePath = './model/senet50_ferplus_dag.pth'):
    model = eModel.Senet50_ferplus_dag()
    model.load_state_dict(torch.load(filePath))
    return model

def getGazeModel(filePath = './model/senet50_gaze_class.pth.tar') :
    model = eModel.senet50()
    device = torch.device('cpu')
    checkpoint =torch.load(filePath,map_location=device)['state_dict']
    for key in list(checkpoint.keys()):
        if 'module.' in key:
            checkpoint[key.replace('module.', '')] = checkpoint[key]
            del checkpoint[key]
    model.load_state_dict(checkpoint)
    return model


def getDistrictList(W,H,num=4) :
    districtList = [ 0 for i in range(num*num)]
    d_count = 0
    part = num*num*2
    for i in range (0,num) :
        for j in range(0, num) :
            width = (W/(num*2)) * (2*j +1)
            height = (H/(num*2)) * (2*i +1)
            districtList[d_count] = (int(width),int(height))
            d_count += 1
    return np.array(districtList)

def getGazeDistrictIdx(model,image):
    result = model(image[None, ...])[0]
    return torch.argmax(result)



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


def getGazeRatio(image,gray,faceLandmark,eye_points, W=480,H=640):

    eye_region = np.array([(faceLandmark.part(eye_points[0]).x, faceLandmark.part(eye_points[0]).y),
                                (faceLandmark.part(eye_points[1]).x, faceLandmark.part(eye_points[1]).y),
                                (faceLandmark.part(eye_points[2]).x, faceLandmark.part(eye_points[2]).y),
                                (faceLandmark.part(eye_points[3]).x, faceLandmark.part(eye_points[3]).y),
                                (faceLandmark.part(eye_points[4]).x, faceLandmark.part(eye_points[4]).y),
                                (faceLandmark.part(eye_points[5]).x, faceLandmark.part(eye_points[5]).y)], np.int32)

    height, width, _ = image.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(image, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)
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

def modeList(L):
    return max(set(L), key=L.count)
# Model
# def getGazeXY(image,face,eyes) : 
# 	x = 500 + random.randint(-50, 50)
# 	y = 800 + random.randint(-50, 50)
# 	return x,y