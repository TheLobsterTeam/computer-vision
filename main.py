import cv2
import numpy as np
import sys

CAMERA_ID_CLOSE_UP   = 0
CAMERA_ID_WIDE_ANGLE = 2

HOLE_LHSV = (0, 0, 16)
HOLE_HHSV = (179, 255, 255)
WIRE_LHSV = (0, 0, 96)
WIRE_HHSV = (179, 255, 255)
PCB_LHSV = (42, 0, 172)
PCB_HHSV = (179, 255, 255)



# Start capturing video from the webcam
def start_video(camera_id):
    video = cv2.VideoCapture(camera_id)
    if not video.isOpened():
        print("Cannot open the webcam.")
        exit()
    return video



def control_gui():
    cv2.namedWindow('Controls')

    # Create trackbars for hole mask
    cv2.createTrackbar("hole_low_H", "Controls", hole_low_HSV[0], 179, nothing)  # Hue
    cv2.createTrackbar("hole_high_H", "Controls", hole_high_HSV[0], 179)
    cv2.createTrackbar("hole_low_S", "Controls", hole_low_HSV[1], 255)  # Saturation
    cv2.createTrackbar("hole_high_S", "Controls", hole_high_HSV[1], 255)
    cv2.createTrackbar("hole_low_V", "Controls", hole_low_HSV[2], 255)  # Value
    cv2.createTrackbar("hole_high_V", "Controls", hole_high_HSV[2], 255)

    # Create trackbars for wire mask
    cv2.createTrackbar("wire_low_H", "Controls", wire_low_HSV[0], 179)  # Hue
    cv2.createTrackbar("wire_high_H", "Controls", wire_high_HSV[0], 179)
    cv2.createTrackbar("wire_low_S", "Controls", wire_low_HSV[1], 255)  # Saturation
    cv2.createTrackbar("wire_high_S", "Controls", wire_high_HSV[1], 255)
    cv2.createTrackbar("wire_low_V", "Controls", wire_low_HSV[2], 255)  # Value
    cv2.createTrackbar("wire_high_V", "Controls", wire_high_HSV[2], 255)



def mask(img, low, high):
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # maybe dont need this or maybe use RGB2HSV?
    # Smoothing
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    cv2.imshow('blur', mask)
    # Threshhold
    mask = cv2.inRange(mask, low, high)
    cv2.imshow('mask', mask)
    return cv2.Canny(mask, 100, 200)



#def draw_features():



# Cleanup windows and release cameras
def cleanup(video):
    video.release()
    cv2.destroyAllWindows()
    print("DONE.")



# Close up camera flow to detect the wire and hole
def camera_close_up():
    print("Starting close up wire feeding process.")
    video = start_video(CAMERA_ID_CLOSE_UP)
    # control_gui()
    while(True):
        ret, frame = video.read()
        hole = mask(frame, HOLE_LHSV, HOLE_HHSV)
        wire = mask(frame, WIRE_LHSV, WIRE_HHSV)
        # combo = cv2.bitwise_or(wire, hole)
        cv2.imshow('raw', frame)
        cv2.imshow('hole', hole)
        cv2.imshow('wire', wire)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cleanup(video)



# Wide angle camera flow to detect PCB
def camera_wide_angle():
    print("Starting wide angle PCB detection process.")
    video = start_video(CAMERA_ID_WIDE_ANGLE)
    # control_gui()
    pcb = mask(img, PCB_LHSV, PCB_HHSV)
    cleanup()



# Choose which flow to run
def main(args=None):
    if len(sys.argv) <= 1:
        print("Run this script with 'close' or 'wide' as the argument.")
        return

    argument = sys.argv[1]

    if argument.__contains__('c'):
        camera_close_up()
    elif argument.__contains__('w'):
        camera_wide_angle()

if __name__ == '__main__':
    main()

