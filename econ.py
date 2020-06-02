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

        
def frameBlinkChecker(frame,predictor,detector,blink_th,start_idx,end_idx) :


	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)
	EAR = 0
	for rect in rects:
	    shape = predictor(gray, rect)
	    shape = face_utils.shape_to_np(shape)
	    eye = shape[start_idx:end_idx]
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
    return [x,y]

def getXY(cur_point,next_point,count,diff_TH,freeze_TH,momentum=0.8) :
    
    if(count == -1) :
        count=0
        return next_point

    point_dff = dist.euclidean(cur_point,next_point)
    
    if(point_dff >diff_TH) :
        count +=1
        if(count>freeze_TH) :
            count = 0
            return next_point
        return cur_point
    count  = 0
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
    for (x,y,w,h) in eyes:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

def strechingPoint(x,y,beforeW,beforeH,afterW,afterH,strechFactor=1.5,H_weight=300):
    return x*(afterW/beforeW) *strechFactor , y* (afterH/beforeH) *strechFactor -H_weight


# Model
# def getGazeXY(image,face,eyes) : 
# 	x = 500 + random.randint(-50, 50)
# 	y = 800 + random.randint(-50, 50)
# 	return x,y