import cv2
import numpy as np
import sys
import math

import random

CAMERA_ID_CLOSE_UP   = 2
CAMERA_ID_WIDE_ANGLE = 2

HOLE_LHSV = (0, 0, 0)
HOLE_HHSV = (179, 255, 33)
WIRE_LHSV = (42, 0, 172)
WIRE_HHSV = (179, 255, 255)
PCB_LHSV = (42, 0, 172)
PCB_HHSV = (179, 255, 255)

H_CENTER = 1042.5
H_TOLERANCE = 32.5
V_CENTER = 315
V_TOLERANCE = 20
P2MM = 0.0232222        #51/1920, 24/1080



# Start capturing video from the webcam
def start_video(camera_id):
    video = cv2.VideoCapture(camera_id)
    if not video.isOpened():
        print("Cannot open the webcam.")
        exit()
    return video



def init_control_gui():
    cv2.namedWindow('Controls')

    # Create trackbars for hole mask
    cv2.createTrackbar("hole_LH", "Controls", HOLE_LHSV[0], 179, nothing)  # Hue
    cv2.createTrackbar("hole_HH", "Controls", HOLE_HHSV[0], 179, nothing)
    cv2.createTrackbar("hole_LS", "Controls", HOLE_LHSV[1], 255, nothing)  # Saturation
    cv2.createTrackbar("hole_HS", "Controls", HOLE_HHSV[1], 255, nothing)
    cv2.createTrackbar("hole_LV", "Controls", HOLE_LHSV[2], 255, nothing)  # Value
    cv2.createTrackbar("hole_HV", "Controls", HOLE_HHSV[2], 255, nothing)

    # Create trackbars for wire mask
    cv2.createTrackbar("wire_LH", "Controls", WIRE_LHSV[0], 179, nothing)  # Hue
    cv2.createTrackbar("wire_HH", "Controls", WIRE_HHSV[0], 179, nothing)
    cv2.createTrackbar("wire_LS", "Controls", WIRE_LHSV[1], 255, nothing)  # Saturation
    cv2.createTrackbar("wire_HS", "Controls", WIRE_HHSV[1], 255, nothing)
    cv2.createTrackbar("wire_LV", "Controls", WIRE_LHSV[2], 255, nothing)  # Value
    cv2.createTrackbar("wire_HV", "Controls", WIRE_HHSV[2], 255, nothing)

    # Create trackbars for canny threshhold
    cv2.createTrackbar("canny_L", "Controls", 100, 255, nothing)
    cv2.createTrackbar("canny_H", "Controls", 200, 255, nothing)

def nothing(x):
	pass



def update_control_values():
    hole_low = (cv2.getTrackbarPos('hole_LH','Controls'), cv2.getTrackbarPos('hole_LS','Controls'), cv2.getTrackbarPos('hole_LV','Controls'))
    hole_high = (cv2.getTrackbarPos('hole_HH','Controls'), cv2.getTrackbarPos('hole_HS','Controls'), cv2.getTrackbarPos('hole_HV','Controls'))
    wire_low = (cv2.getTrackbarPos('wire_LH','Controls'), cv2.getTrackbarPos('wire_LS','Controls'), cv2.getTrackbarPos('wire_LV','Controls'))
    wire_high = (cv2.getTrackbarPos('wire_HH','Controls'), cv2.getTrackbarPos('wire_HS','Controls'), cv2.getTrackbarPos('wire_HV','Controls'))
    canny_thresh = (cv2.getTrackbarPos('canny_L','Controls'), cv2.getTrackbarPos('canny_H','Controls'))
    return (hole_low, hole_high, wire_low, wire_high, canny_thresh)



def mask(img, low, high, canny):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # maybe dont need this or maybe use RGB2HSV?
    # Smoothing
    blur = cv2.GaussianBlur(hsv, (9, 9), 0)
    # Threshhold
    mask = cv2.inRange(blur, low, high)
    return cv2.Canny(mask, canny[0], canny[1]), mask, blur



def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
   
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image



def draw_line(img, point1, point2):
    line = np.zeros_like(img)
    cv2.line(line, point1, point2, [0,255,0], 50)
    return cv2.addWeighted(img, 0.8, line, 1., 0.)



def draw_circle(img, center, radius):
    circle = np.zeros_like(img)
    cv2.circle(circle, (math.floor(center[0]), math.floor(center[1])), math.floor(radius), [0,0,255], 20)
    return cv2.addWeighted(img, 0.8, circle, 1., 0.)



def bounding_box(img):
    contours, hc = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print("HERE:", contours[0])
    #print("Size: ",len(contours))
    #crop = cv2.drawContours(crop, contours, -1, (0,255,0), 30)

    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    closest = (9999, 9999)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
        #center = (boundRect[i][0]+(boundRect[i][2]/2), boundRect[i][1]+(boundRect[i][3]/2))
        #print("cen:", centers[i], " found: ", center)
        if (math.dist(centers[i], (H_CENTER, V_CENTER)) < math.dist(closest, (H_CENTER, V_CENTER))):
            closest = centers[i]
            closest_index = i
            #print("Closest:",math.dist(closest, (H_CENTER, V_CENTER)))
    
    drawing = np.zeros_like(img)
    
    if (len(contours) == 0):
        return img, (0,0), 10
    else:
        i = closest_index
    #for i in range(len(contours)):
    cv2.drawContours(drawing, contours_poly, i, (255,255,255))
    #cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), 255,255,255, 2)
    cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), (255,255,255), 2)

    return drawing, centers[closest_index], radius[closest_index]



def align(hole_pos):
    adjustment = [0, 0]
    
    # Horizontal alignment
    h_error = H_CENTER - hole_pos[0]
    if (abs(h_error) > H_TOLERANCE):
        print("Horizontal by ", h_error*P2MM, "mm")
        adjustment[0] = h_error*P2MM
    else:
        print("Horizontally centered.")
        adjustment[0] = 0

    # Vertical alignment
    v_error = V_CENTER - hole_pos[1]
    if (abs(v_error) > V_TOLERANCE):
        print("Vertical by ", v_error*P2MM, "mm")
        adjustment[1] = v_error*P2MM
    else:
        print("Vertically centered.")
        adjustment[1] = 0

    return adjustment



# Cleanup windows and release cameras
def cleanup(video):
    video.release()
    cv2.destroyAllWindows()
    print("DONE.")



def display_four(name, topLeft, topRight, botLeft, botRight):
    top = np.hstack((topLeft, np.ones((topLeft.shape[0], 25)), topRight)) # np.concatenate((topLeft, topRight), axis=1)
    bot = np.hstack((botLeft, np.ones((botLeft.shape[0], 25)), botRight)) # np.concatenate((botLeft, botRight), axis=1)
    cv2.imshow(name, np.vstack((top, np.ones((25, top.shape[1])), bot)))#concatenate((top, bot), axis=0))



# Close up camera flow to detect the wire and hole
def camera_close_up():
    print("Starting close up wire feeding process.")
    video = start_video(CAMERA_ID_CLOSE_UP)
    init_control_gui()
    while(True):
        ret, frame = video.read()
        frame = frame[175:1080, 0:1920]

        # Masking
        (hole_low, hole_high, wire_low, wire_high, canny_thresh) = update_control_values()
        hole, hmask, hblur = mask(frame, hole_low, hole_high, canny_thresh)
        wire, wmask, wblur = mask(frame, wire_low, wire_high, canny_thresh)
        
        # ROI - remove snips in hole image and crop wire feeding image
        crop = region_of_interest(frame, [np.array([[1010,660],[1075,660],[1075,120],[1010,120]])])
        wire = region_of_interest(wmask, [np.array([[1010,660],[1075,660],[1075,120],[1010,120]])])
        hole = region_of_interest(hmask, [np.array([[1015,650],[765,75],[0,75],[0,1080],[1920,1080],[1920,0],[1300,75],[1300,590],[1015,100]])])
        
        # Get bounding box around closest hole
        draw, hole_pos, rad = bounding_box(hole)

        frame = draw_circle(frame, hole_pos, rad)

        combo = cv2.bitwise_or(wire, hole)
        crop = draw_line(frame, (1017, 120), (1017, 600))
        cv2.imshow('Controls', crop)
        display_four('wire', hmask, hole, wire, combo)

        # TODO: get hole pos and circle

        align(hole_pos)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cleanup(video)



# Wide angle camera flow to detect PCB
def camera_wide_angle():
    print("Starting wide angle PCB detection process.")
    video = start_video(CAMERA_ID_WIDE_ANGLE)
    control_gui()
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

