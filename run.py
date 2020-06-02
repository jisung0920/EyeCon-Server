import argparse
import econ
import socket
import numpy as np
import cv2
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import vggModel
import torch
import torchvision.transforms as transforms
from PIL import Image

diff_TH = 300
freeze_TH = 8
WEIGHT_POINT = 0.8

IMG_WIDTH = 480
IMG_HEIGHT = 640
ANRD_WIDTH = 1170
ANRD_HEIGHT = 1780

parser = argparse.ArgumentParser()
parser.add_argument('--util_path',default='./utils/',required = False, help='util file path')
parser.add_argument('--blink_th',default=0.3,required=False, help='eye blick value threshold')
parser.add_argument('--port',default=8100,type=int)
parser.add_argument('--ip',default='192.168.0.25')

args = parser.parse_args()

util_path  = args.util_path
IP,PORT = args.ip, args.port
BLINK_TH = args.blink_th


faceCascade, eyeCascade = econ.getCascade(util_path)

predictor_path = util_path + 'shape_predictor_68_face_landmarks.dat'

predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]


model = vggModel.VGG_16() 


device = torch.device('cpu')
checkpoint =torch.load('./checkpoint.pth.tar',map_location=device)['state_dict']
for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]

model.load_state_dict(checkpoint)


transformImg = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()])


print(IP,':',PORT)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((IP, PORT))
server_socket.listen()
print('listening...')

serverOn = True
count = -1
cur_point = np.array([600,870])
x = 600
y= 860
count = -1
try:
    client_socket, addr = server_socket.accept()
    print('Connected with ', addr)


        
    while serverOn:

        length = econ.recvAll(client_socket, 10)

        frame_bytes = econ.recvAll(client_socket, int(length))

        image = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        print('Get image from Android')
        inputData = Image.fromarray(image) 
        inputData = transformImg(inputData)
        inputData = torch.autograd.Variable(inputData, requires_grad = True)

        faceFrame,eyeFrame = econ.getFaceEyePoint(image,faceCascade,eyeCascade)

        # gazePoint = np.array( econ.getGazeXY(model,inputData) )

        if len(faceFrame) > 0 :
            facePoint = np.array(econ.getFaceXY(faceFrame))
        else :
            facePoint = cur_point


        if(econ.frameBlinkChecker(image,predictor,detector,BLINK_TH,lStart,lEnd)) :
            blink = 0
        else :
            blink = 1

        # blink = 0
        # x = round(x*1200/2  + 1200/2)
        # y = round(y*1740/2  + 1740/2)
        x, y  = econ.getXY(cur_point, facePoint,count,diff_TH,freeze_TH)
        x, y  = econ.strechingPoint(x,y,IMG_WIDTH,IMG_HEIGHT,ANRD_WIDTH,ANRD_HEIGHT)
        cord = str(int(x)) + '/' + str(int(y)) + '/' +str(blink)
        print('Send to Android :',cord)
        client_socket.sendall(bytes(cord,'utf8'))

        econ.recGenerator(image,faceFrame,eyeFrame)
        cv2.imshow('AndroidScreen', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
        	break


except Exception as e:
    print(e)
    print('Connecting Error')
    pass
finally:
    print('End of the session')


