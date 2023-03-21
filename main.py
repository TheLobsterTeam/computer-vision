import cv2
import numpy as np
import sys
import math
import time

CAMERA_ID_CLOSE_UP   = 0
CAMERA_ID_WIDE_ANGLE = 2

HOLE_LHSV = (0, 0, 0)
HOLE_HHSV = (179, 255, 8)
WIRE_LHSV = (0, 65, 115)
WIRE_HHSV = (179, 255, 255)
PCB_LHSV = (42, 0, 172)
PCB_HHSV = (179, 255, 255)

H_CENTER = 375
TOLERANCE = 0.95
P2MM = 0.03878049    #wire tube diam: 1.59/41=0.03878049, H_screen 27.65/800=0.03456 (not accurate), V_screen 21.35/520=0.041 (not accurate),tube to tip of L cutter: 6.72/



# Start capturing video from the webcam
def start_video(camera_id):
    video = cv2.VideoCapture(camera_id)
    if not video.isOpened():
        cleanup(video)
        print("Cannot open the webcam.")
        return None
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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

def nothing(x):
	pass



def update_control_values():
    hole_low = (cv2.getTrackbarPos('hole_LH','Controls'), cv2.getTrackbarPos('hole_LS','Controls'), cv2.getTrackbarPos('hole_LV','Controls'))
    hole_high = (cv2.getTrackbarPos('hole_HH','Controls'), cv2.getTrackbarPos('hole_HS','Controls'), cv2.getTrackbarPos('hole_HV','Controls'))
    wire_low = (cv2.getTrackbarPos('wire_LH','Controls'), cv2.getTrackbarPos('wire_LS','Controls'), cv2.getTrackbarPos('wire_LV','Controls'))
    wire_high = (cv2.getTrackbarPos('wire_HH','Controls'), cv2.getTrackbarPos('wire_HS','Controls'), cv2.getTrackbarPos('wire_HV','Controls'))
    return (hole_low, hole_high, wire_low, wire_high)



def mask(img, low, high):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Smoothing
    blur = cv2.GaussianBlur(hsv, (9, 9), 0)
    # Threshhold
    mask = cv2.inRange(blur, low, high)
    return mask



def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    # Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    # Filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # Returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image



def bounding_box(img):
    contours, hc = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    drawing = np.zeros_like(img)
    lowest_point = (H_CENTER, 0)
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    closest = (9999, 9999, 0, 0)

    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        if (boundRect[i][1] < closest[1]):
            closest = boundRect[i]
            closest_index = i

    if (len(contours) == 0):
        return img, (0,0,0,0), None
    # Format contour array to array of points
    wire = contours_poly[closest_index]
    if len(wire) > 1:
        wire = np.squeeze(wire)
    else:
        wire = wire[0]
    # Get lowest point on the wire contour, this is the tip of the wire
    for point in wire:
        if point[1] > lowest_point[1]:
            lowest_point = point
    cv2.drawContours(drawing, contours_poly, closest_index, (255,255,255))
    cv2.rectangle(drawing, (int(closest[0]), int(closest[1])), (int(closest[0]+closest[2]), int(closest[1]+closest[3])), (255,255,255), 2)
    return drawing, closest, lowest_point



def bounding_circle(img, wire_pos):
    contours, hc = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    drawing = np.zeros_like(img)
    contours_poly = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    closest = (9999, 9999)

    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
        if (math.hypot(centers[i][0] - wire_pos[0], centers[i][1] - wire_pos[1]) < math.hypot(closest[0] - wire_pos[0], closest[1] - wire_pos[1])):
            closest = centers[i]
            closest_index = i
    # If no holes are found default to top middle to move the PCB down to reveal holes under the cutters
    if (len(contours) == 0):
        return img, (H_CENTER,0), None
    cv2.drawContours(drawing, contours_poly, closest_index, (255,255,255))
    cv2.circle(drawing, (int(centers[closest_index][0]), int(centers[closest_index][1])), int(radius[closest_index]), (255,255,255), 2)
    return drawing, closest, radius[closest_index]



def align(hole_pos, hole_rad, wire_pos):
    adjustment = [0, 0]
    # Horizontal alignment
    h_error = wire_pos[0] - hole_pos[0]
    if (abs(h_error) > hole_rad*TOLERANCE):
        print("Horizontal by ", h_error*P2MM, "mm")
        adjustment[0] = h_error*P2MM
    else:
        print("Horizontally centered.")
        adjustment[0] = 0
    # Vertical alignment
    v_error = wire_pos[1] - hole_pos[1]
    if (abs(v_error) > hole_rad*TOLERANCE):
        print("Vertical by ", v_error*P2MM, "mm")
        adjustment[1] = v_error*P2MM
    else:
        print("Vertically centered.")
        adjustment[1] = 0
    return adjustment



def display_four(name, topLeft, topRight, botLeft, botRight):
    top = np.hstack((topLeft, np.ones((topLeft.shape[0], 25)), topRight))
    bot = np.hstack((botLeft, np.ones((botLeft.shape[0], 25)), botRight))
    cv2.imshow(name, np.vstack((top, np.ones((25, top.shape[1])), bot)))



# Init via camera return None if error with the camera
def init_via():
    return start_video(CAMERA_ID_CLOSE_UP)

# Init flipping camera return None if error with the camera
def init_flip():
    return start_video(CAMERA_ID_WIDE_ANGLE)



def via_detection(video, DEBUG=0):
    ret, frame = video.read()
    frame = frame[200:720, 300:1100]
    # Masking
    if DEBUG:
        (hole_low, hole_high, wire_low, wire_high) = update_control_values()
        hmask = mask(frame, hole_low, hole_high)
        wmask = mask(frame, wire_low, wire_high)
    else:
        hmask = mask(frame, HOLE_LHSV, HOLE_HHSV)
        wmask = mask(frame, WIRE_LHSV, WIRE_HHSV)
    # ROI - remove snips in hole image and crop wire feeding image
    hole = region_of_interest(hmask,\
        [np.array([[340,50],[340,230],[315,230],[295,200],[220,0],\
        [0,0],[0,720],[1080,720],[1080,0],\
        [595,0],[595,200],[480,50]])])  # 1st line: Stationary clipper, 3rd line: Moving clipper
    wire = region_of_interest(wmask, [np.array([[H_CENTER - 40,450],[H_CENTER + 40,450],[H_CENTER + 40,60],[H_CENTER - 40,60]])])
    # Get bounding around closest hole and the wire
    draw, wire_rect, wire_tip = bounding_box(wire)
    draw2, hole_pos, rad = bounding_circle(hole, wire_tip)
    if wire_tip is None or rad is None:
        adjustments = None
    else:
        # Calculate adjustments
        adjustments = align(hole_pos, rad, wire_tip)
    if DEBUG:
        return adjustments, frame, wire_rect, wire_tip, hole_pos, rad, draw, draw2, hmask, hole, wire
    return adjustments



# Cleanup windows and release cameras
def cleanup(video):
    video.release()
    cv2.destroyAllWindows()
    print("CV DONE.")



# Close up camera flow to detect the wire and hole
def camera_close_up():
    print("Starting close up wire feeding process.")
    init_control_gui()
    video = init_via()
    if video == None:
        exit()
    while(True):
        adjustments, frame, wire_rect, wire_tip, hole_pos, rad, draw, draw2, hmask, hole, wire = via_detection(video, 1)
        if adjustments == None:
            print("Wire or hole not found.")
            continue

        # Show detections on raw camera frame
        cv2.rectangle(frame, (int(wire_rect[0]), int(wire_rect[1])), (int(wire_rect[0]+wire_rect[2]), int(wire_rect[1]+wire_rect[3])), (0,255,0), 2)
        cv2.circle(frame, (int(wire_tip[0]), int(wire_tip[1])), 2, [255,0,0], 10)
        cv2.circle(frame, (int(hole_pos[0]), int(hole_pos[1])), int(rad), [0,0,255], 10)
        # Display video
        combo = cv2.bitwise_or(draw, draw2)
        cv2.imshow('crop', frame)
        display_four('wire', hmask, hole, wire, combo)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cleanup(video)



# Wide angle camera flow to detect PCB
def camera_wide_angle():
    print("Starting wide angle PCB detection process.")
    video = start_video(CAMERA_ID_WIDE_ANGLE)
    init_control_gui()
    pcb = mask(img, PCB_LHSV, PCB_HHSV)
    cleanup(video)



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

