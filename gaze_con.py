import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./utils/shape_predictor_68_face_landmarks.dat")

left_eye_basepoints = [36, 37, 38, 39, 40, 41]
right_eye_basepoints = [42, 43, 44, 45, 46, 47]

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio

def get_gaze_ratio(eye_points, facial_landmarks):
    # Gaze detection
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    # cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(frame, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    print(left_eye_region)
    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])

    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

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


    # cv2.imshow("Down side eye", down_side_threshold)
    # threshold_txt = 'Downside white ratio %s' % upper_side_white
    # cv2.putText(frame, str(threshold_txt), (70, 140), font, 2, (0, 0, 255), 3)

    return horizontal_gaze_ratio, vertical_gaze_ratio

while True:
    _, frame = cap.read()
    new_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        #x, y = face.left(), face.top()
        #x1, y1 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)

        # Detect blinking
        left_eye_ratio = get_blinking_ratio(left_eye_basepoints, landmarks)
        right_eye_ratio = get_blinking_ratio(right_eye_basepoints, landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > 5.7:
            cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0))



        # Gaze detection
        horizontal_gaze_ratio_left_eye, vertical_gaze_ratio_left_eye = get_gaze_ratio(left_eye_basepoints, landmarks)
        horizontal_gaze_ratio_right_eye, vertical_gaze_ratio_right_eye = get_gaze_ratio(right_eye_basepoints, landmarks)
        horizontal_gaze_ratio = (horizontal_gaze_ratio_right_eye + horizontal_gaze_ratio_left_eye) / 2
        vertical_gaze_ratio = (vertical_gaze_ratio_right_eye + vertical_gaze_ratio_left_eye) / 2

        horizontal_view_direction = None
        vertical_view_direction = None
        if horizontal_gaze_ratio < 0.8:
            horizontal_view_direction = '''LEFT %s''' % horizontal_gaze_ratio
            new_frame[:] = (0, 0, 255)
        elif 0.9 < horizontal_gaze_ratio < 1.2:
            horizontal_view_direction = '''CENTER %s''' % horizontal_gaze_ratio
        else:
            new_frame[:] = (255, 0, 0)
            horizontal_view_direction = '''RIGHT %s''' % horizontal_gaze_ratio

        if vertical_gaze_ratio < 1.35:
            vertical_view_direction = '''DOWN %s''' % vertical_gaze_ratio
            new_frame[:] = (0, 0, 255)
        elif 1.4 < vertical_gaze_ratio < 1.65:
            vertical_view_direction = '''CENTER %s''' % vertical_gaze_ratio
        else:
            new_frame[:] = (255, 0, 0)
            vertical_view_direction = '''UP %s''' % vertical_gaze_ratio

        #showing direction

        cv2.putText(frame, str(horizontal_view_direction), (50, 100), font, 2, (0, 0, 255), 3)
        cv2.putText(frame, str(vertical_view_direction), (50, 190), font, 2, (0, 0, 255), 3)



    cv2.imshow("Frame", frame)
    cv2.imshow("New Frame", new_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()



def getDirectionFromEye(frame, predictor,)