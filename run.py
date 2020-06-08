import argparse
import econ
import socket
import numpy as np
import cv2
from imutils import face_utils
import dlib
import torch
from PIL import Image
from eyeconModel import FERModel, GazeModel
from eyeconModel import transformImg

IMG_WIDTH , IMG_HEIGHT= 480, 640
ANRD_WIDTH,ANRD_HEIGHT = 1170,1780

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




device = torch.device('cpu')

faceClassifier = econ.loadClassifier(util_path)
predictor = dlib.shape_predictor(util_path + 'shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


state_direction_num = 5
state_Q_num = 3
state_memory = econ.loadStateDict(state_direction_num,state_Q_num)

Gaze_model, FER_model = GazeModel(),FERModel()
Gaze_model.eval()

prev_x, prev_y = ANRD_WIDTH/2, ANRD_HEIGHT/2
momentum = 0.7

print(IP,':',PORT)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((IP, PORT))
server_socket.listen()
print('listening...')

try:
    client_socket, addr = server_socket.accept()
    print('Connected with ', addr)

    count = 0
        
    while True:
        print('Receive IMG')

        length = econ.recvAll(client_socket, 10)
        frame_bytes = econ.recvAll(client_socket, int(length))
        image = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        print('\tGaze tracking : [Classification]', end='\t ')

        inputData = Image.fromarray(image)
        inputData = transformImg(inputData)
        inputData = torch.autograd.Variable(inputData, requires_grad = True)

        x,y = econ.getGazePoint(Gaze_model,inputData,ANRD_WIDTH,ANRD_HEIGHT)


        print('[Gaze Ratio]', end='\t ')

        rects = detector(gray, 0)
        for rect in rects:
            faceLandmark = predictor(gray, rect)

        if(len(rects) != 0) :
            horizontal_gaze_ratio_left_eye, vertical_gaze_ratio_left_eye =  econ.getGazeRatio(gray, faceLandmark,np.arange(lStart,lEnd+1))
            horizontal_gaze_ratio_right_eye, vertical_gaze_ratio_right_eye =  econ.getGazeRatio(gray, faceLandmark,np.arange(rStart,rEnd+1))
            horizontal_gaze_ratio = (horizontal_gaze_ratio_right_eye + horizontal_gaze_ratio_left_eye) / 2
            vertical_gaze_ratio = (vertical_gaze_ratio_right_eye + vertical_gaze_ratio_left_eye) / 2

            state_memory['GazeRatioLR'][count % state_direction_num] = horizontal_gaze_ratio
            state_memory['GazeRatioTB'][count % state_direction_num] = vertical_gaze_ratio

            gazeRatioLR = horizontal_gaze_ratio - state_memory['GazeRatioLR'].mean()
            gazeRatioTB = vertical_gaze_ratio - state_memory['GazeRatioTB'].mean()
        else :
            state_memory['GazeRatioLR'][(count) % state_direction_num] = 1
            state_memory['GazeRatioTB'][(count) % state_direction_num] = 1

        gazeRatioLR = state_memory['GazeRatioLR'][(count) % state_direction_num] - state_memory['GazeRatioLR'].mean()
        gazeRatioTB = state_memory['GazeRatioTB'][(count) % state_direction_num] - state_memory['GazeRatioTB'].mean()


        print('[Face Point]')

        facePoints = econ.classifyFace(image, faceClassifier)

        if(len(facePoints)!= 0 ) :
            faceX,faceY = econ.getFaceXY(facePoints)
            state_memory['FacePointX'][count % state_direction_num],\
            state_memory['FacePointY'][count % state_direction_num] = faceX,faceY
        else :
            state_memory['FacePointX'][count % state_direction_num] = state_memory['FacePointX'][(count-1) % state_direction_num]
            state_memory['FacePointY'][count % state_direction_num] = state_memory['FacePointY'][(count-1) % state_direction_num]

        faceDirectionX = (state_memory['FacePointX'][count%state_direction_num] - state_memory['FacePointX'].mean())/IMG_WIDTH
        faceDirectionY = (state_memory['FacePointY'][count%state_direction_num] - state_memory['FacePointY'].mean())/IMG_HEIGHT


        print('\tBlink Detection',end='\t| ')

        if(len(rects) != 0 ) :
            eyeLandmark = face_utils.shape_to_np(faceLandmark)

            if(econ.isBlink(eyeLandmark,BLINK_TH,lStart,lEnd)) :
                state_memory['Click'][count%state_Q_num] = 0
            else :
                state_memory['Click'][count % state_Q_num] = 1

            if(econ.isBlink(eyeLandmark,BLINK_TH,rStart,rEnd)) :
                state_memory['Scroll'][count % state_Q_num] = 0
            else :
                state_memory['Scroll'][count % state_Q_num] = 1
        else :
            state_memory['Click'][count % state_Q_num] =0
            state_memory['Scroll'][count % state_Q_num] =0


        print('Estimate expression')

        state_memory['FER'][count%state_Q_num], tagPosition = econ.getExpression(facePoints,image,FER_model)


        print('Integrate')

        blink = econ.modeList(state_memory['Click'])
        scroll = econ.modeList(state_memory['Scroll'])
        if (blink == 1 or scroll == 1):
            FERNum = 0
            tagPosition = (0, 0)
        else:
            FERNum = econ.modeList(state_memory['FER'])
            cv2.putText(image,emotion_dict[FERNum],tagPosition, cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 255, 0), 1, cv2.LINE_AA)

        d1_x, d1_y = econ.rateToDistance(gazeRatioLR,gazeRatioTB,ANRD_WIDTH,ANRD_HEIGHT,weight=0.5)
        d2_x, d2_y = econ.rateToDistance(faceDirectionX, faceDirectionY, ANRD_WIDTH,ANRD_HEIGHT,weight=0.5)

        x += d1_x + d2_x
        y += d1_y + d2_y

        x, y =int(x*momentum + prev_x * (1-momentum)) , int(y*momentum + prev_y * (1-momentum))
        prev_x, prev_y = x, y


        print('\tSent to Client',end='\t')

        cord = str(x) + '/' + str(y) + '/' +str(blink) + '/' + str(scroll) + '/' + str(FERNum)
        print(cord)
        client_socket.sendall(bytes(cord,'utf8'))

        for (x,y,w,h) in facePoints:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imshow('Android Screen', image)
        count += 1


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


except Exception as e:
    print(e)
    print('Connecting Error')
    pass
finally:
    print('End of the session')


