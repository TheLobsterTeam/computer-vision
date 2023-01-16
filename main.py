import cv2 as cv
import numpy as np
import sys

CAMERA_ID_CLOSE_UP   = 4
CAMERA_ID_WIDE_ANGLE = 2

video = cv.VideoCapture()

# Start capturing video from the webcam
def start_video(camera_id):
    video = cv.VideoCapture(camera_id)
    if not video.isOpened():
        print("Cannot open the webcam.")
        exit()

def control_gui():
    cv.namedWindow('Controls')
    hole_low_HSV = (0, 0, 16)
    hole_high_HSV = (179, 255, 255)
    wire_low_HSV = (0, 0, 96)
    wire_high_HSV = (179, 255, 255)
    pcb_low_HSV = (42, 0, 172)
    pcb_high_HSV = (179, 255, 255)

    # Create trackbars for hole mask
    cv.createTrackbar("hole_low_H", "Controls", hole_low_HSV[0], 179, nothing)  # Hue
    cv.createTrackbar("hole_high_H", "Controls", hole_high_HSV[0], 179)
    cv.createTrackbar("hole_low_S", "Controls", hole_low_HSV[1], 255)  # Saturation
    cv.createTrackbar("hole_high_S", "Controls", hole_high_HSV[1], 255)
    cv.createTrackbar("hole_low_V", "Controls", hole_low_HSV[2], 255)  # Value
    cv.createTrackbar("hole_high_V", "Controls", hole_high_HSV[2], 255)

    # Create trackbars for wire mask
    cv.createTrackbar("wire_low_H", "Controls", wire_low_HSV[0], 179)  # Hue
    cv.createTrackbar("wire_high_H", "Controls", wire_high_HSV[0], 179)
    cv.createTrackbar("wire_low_S", "Controls", wire_low_HSV[1], 255)  # Saturation
    cv.createTrackbar("wire_high_S", "Controls", wire_high_HSV[1], 255)
    cv.createTrackbar("wire_low_V", "Controls", wire_low_HSV[2], 255)  # Value
    cv.createTrackbar("wire_high_V", "Controls", wire_high_HSV[2], 255)


# def mask():
    
# def draw_features():

# Cleanup windows and release cameras
def cleanup():
    video.release()
    cv.destroyAllWindows()
    print("DONE.")

# Close up camera flow to detect the wire and hole
def camera_close_up():
    print("Starting close up wire feeding process.")
    start_video(CAMERA_ID_CLOSE_UP)
    control_gui()
    cleanup()

# Wide angle camera flow to detect PCB
def camera_wide_angle():
    print("Starting wide angle PCB detection process.")
    start_video(CAMERA_ID_WIDE_ANGLE)
    control_gui()
    cleanup()

# Choose which flow to run
def main(args=None):
    argument = sys.argv[1]
    if argument.__contains__('c'):
        camera_close_up()
    elif argument.__contains__('w'):
        camera_wide_angle()
    else:
        print("Run this script with 'close' or 'wide' as the argument.")

if __name__ == '__main__':
    main()

