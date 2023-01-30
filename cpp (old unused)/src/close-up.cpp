#include <iostream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

const int CAM_CLOSE_UP = 2;
const int CAM_WIDE_ANGLE = 2;

int main(int argc, char **argv) {
  VideoCapture cap(CAM_CLOSE_UP); // capture the video from webcam

  if (!cap.isOpened()) // if not success, exit program
  {
    cout << "Cannot open the web cam" << endl;
    return -1;
  }

  namedWindow("Control"); // create a window called "Control"

  // Mask thresholds
  int hole_low_H = 0;
  int hole_high_H = 179;
  int hole_low_S = 0;
  int hole_high_S = 255;
  int hole_low_V = 16;
  int hole_high_V = 255;

  int wire_low_H = 0;
  int wire_high_H = 179;
  int wire_low_S = 0;
  int wire_high_S = 255;
  int wire_low_V = 96;
  int wire_high_V = 255;

  // Create trackbars for hole mask
  createTrackbar("hole_low_H", "Control", &hole_low_H, 179); // Hue
  createTrackbar("hole_high_H", "Control", &hole_high_H, 179);
  createTrackbar("hole_low_S", "Control", &hole_low_S, 255); // Saturation
  createTrackbar("hole_high_S", "Control", &hole_high_S, 255);
  createTrackbar("hole_low_V", "Control", &hole_low_V, 255); // Value
  createTrackbar("hole_high_V", "Control", &hole_high_V, 255);

  // Create trackbars for wire mask
  createTrackbar("wire_low_H", "Control", &wire_low_H, 179); // Hue
  createTrackbar("wire_high_H", "Control", &wire_high_H, 179);
  createTrackbar("wire_low_S", "Control", &wire_low_S, 255); // Saturation
  createTrackbar("wire_high_S", "Control", &wire_high_S, 255);
  createTrackbar("wire_low_V", "Control", &wire_low_V, 255); // Value
  createTrackbar("wire_high_V", "Control", &wire_high_V, 255);

  int iLastX = -1;
  int iLastY = -1;

  // Capture a temporary image from the camera
  Mat imgTmp;
  cap.read(imgTmp);

  // Create a black image with the size as the camera output
  Mat imgLines = Mat::zeros(imgTmp.size(), CV_8UC3);

  while (true) {
    // Read a new frame from camera
    Mat imgOriginal;
    bool bSuccess = cap.read(imgOriginal);

    if (!bSuccess) // if not success, break loop
    {
      cout << "Cannot read a frame from video stream" << endl;
      break;
    }

    // Convert the captured frame from BGR to HSV
    Mat imgHSV;
    cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV);

    // Threshold the image to mask out the holes
    Mat imgHoleThresholded;
    Mat imgWireThresholded;
    inRange(imgHSV, Scalar(hole_low_H, hole_low_S, hole_low_V),
            Scalar(hole_high_H, hole_high_S, hole_high_V), imgHoleThresholded);
    inRange(imgHSV, Scalar(wire_low_H, wire_low_S, wire_low_V),
            Scalar(wire_high_H, wire_high_S, wire_high_V), imgWireThresholded);

    // morphological opening (removes small objects from the foreground)
    erode(imgHoleThresholded, imgHoleThresholded,
          getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
    dilate(imgHoleThresholded, imgHoleThresholded,
           getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
    // morphological closing (removes small holes from the foreground)
    dilate(imgHoleThresholded, imgHoleThresholded,
           getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
    erode(imgHoleThresholded, imgHoleThresholded,
          getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

    // Invert hole image
    bitwise_not(imgHoleThresholded, imgHoleThresholded);
    bitwise_not(imgWireThresholded, imgWireThresholded);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(imgWireThresholded, contours, hierarchy, RETR_TREE,
                 CHAIN_APPROX_SIMPLE);
    drawContours(imgWireThresholded, contours, -1, (0, 255, 0), 3, LINE_8,
                 hierarchy, 0);

    // Calculate the moments of the thresholded image
    Moments oMoments = moments(imgHoleThresholded);

    double dM01 = oMoments.m01;
    double dM10 = oMoments.m10;
    double dArea = oMoments.m00;

    // if the area <= 10000, I consider that the there are no object in the
    // image and it's because of the noise, the area is not zero
    if (dArea > 10000) {
      // calculate the position of the ball
      int posX = dM10 / dArea;
      int posY = dM01 / dArea;

      if (iLastX >= 0 && iLastY >= 0 && posX >= 0 && posY >= 0) {
        // Draw a red line from the previous point to the current point
        circle(imgLines, Point(posX, posY), 10, // Point(iLastX, iLastY),
               Scalar(0, 0, 255), 2);
      }

      iLastX = posX;
      iLastY = posY;
    }

    // Show the thresholded image
    // imgHoleThresholded = imgHoleThresholded + imgWireThresholded;
    imshow("Thresholded Image", imgHoleThresholded);

    // Add shapes to original image
    imgOriginal = imgOriginal + imgLines;

    // Show the original image
    imshow("Original", imgOriginal);

    if (waitKey(30) == 27) { // If 'esc' key is pressed, break loop
      cout << "esc key is pressed by user" << endl;
      break;
    }
  }
  return 0;
}
