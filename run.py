import argparse
import econ
import socket
import numpy as np
import cv2
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import torch
import torchvision.transforms as transforms
from PIL import Image

diff_TH = 300
freeze_TH = 8
WEIGHT_POINT = 0.8

IMG_WIDTH , IMG_HEIGHT= 480, 640
ANRD_WIDTH,ANRD_HEIGHT = 1170,1780

device = torch.device('cpu')

emotion_dict = {0: "neutral", 1: "happiness", 2: "surprise", 3: "sadness", 4: "anger",
5: "disgust", 6: "fear", 7: "comtempt", 8: "unknown", 9: "Not a Face"}


parser = argparse.ArgumentParser()
parser.add_argument('--util_path',default='./utils/',required = False, help='util file path')
parser.add_argument('--blink_th',default=0.25,required=False, help='eye blick value threshold')
parser.add_argument('--port',default=8200,type=int)
parser.add_argument('--ip',default='192.168.0.25')

args = parser.parse_args()

util_path  = args.util_path
IP,PORT = args.ip, args.port
BLINK_TH = args.blink_th


# IP = '192.168.219.101'

faceCascade, eyeCascade = econ.getCascade(util_path)

predictor = dlib.shape_predictor(util_path + 'shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


districtList = econ.getDistrictList(ANRD_WIDTH,ANRD_HEIGHT)

Gaze_model = econ.getGazeModel()
Gaze_model.eval()

FER_model= econ.getFERModel()
FER_model.eval()


transformImg = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])


print(IP,':',PORT)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((IP, PORT))
server_socket.listen()
print('listening...')

cur_point = np.array((IMG_WIDTH/2,IMG_HEIGHT/2))
count = -1
try:
    client_socket, addr = server_socket.accept()
    print('Connected with ', addr)


        
    while True:
        print('Receive IMG',end='\t| ')

        length = econ.recvAll(client_socket, 10)
        frame_bytes = econ.recvAll(client_socket, int(length))
        image = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        

        print('Transform IMG',end='\t| ')

        inputData = Image.fromarray(image) 
        inputData = transformImg(inputData)
        inputData = torch.autograd.Variable(inputData, requires_grad = True)

        print('Estimate district')

        gazePoint = np.array(econ.getGazeDistrictPoint(Gaze_model,inputData,districtList))


        print('Get face point',end='\t| ')

        faceFrame,eyeFrame = econ.getFaceEyePoint(image,faceCascade,eyeCascade)

        if len(faceFrame) > 0 :
            facePoint = np.array(econ.getFaceXY(faceFrame))
        else :
            facePoint = cur_point

        print('Check blink',end='\t| ')
        if(econ.frameBlinkChecker(image,predictor,detector,BLINK_TH,lStart,lEnd)) :
            blink = 0
        else :
            blink = 1
        if(econ.frameBlinkChecker(image,predictor,detector,BLINK_TH,rStart,rEnd)) :
            scroll = 0
        else :
            scroll = 1

        print('Estimate expression')
        if(blink == 1 or scroll == 1):
            emotionNum = 0
            tagPosition = (0,0)
        else :
            emotionNum, tagPosition = econ.getExpression(faceFrame,image,FER_model)
            # emotionNum = 0
            # tagPosition = (0,0)

        x, y  = econ.getXY(cur_point, facePoint,count,diff_TH,freeze_TH)
        x, y  = econ.strechingPoint(x,y,IMG_WIDTH,IMG_HEIGHT,ANRD_WIDTH,ANRD_HEIGHT)
        
        cord = str(int(x)) + '/' + str(int(y)) + '/' +str(blink) + '/' + str(scroll) + '/' + str(emotionNum)
        print('Send to Android :',cord)
        client_socket.sendall(bytes(cord,'utf8'))

        cv2.putText(image,emotion_dict[emotionNum],tagPosition, cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 255, 0), 1, cv2.LINE_AA)
        econ.recGenerator(image,faceFrame,eyeFrame)
        cv2.imshow('Android Screen', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


except Exception as e:
    print(e)
    print('Connecting Error')
    pass
finally:
    print('End of the session')


