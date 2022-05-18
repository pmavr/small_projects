import cv2
import numpy as np
from tkinter import *


def update_image(val):
    thresh = cv2.getTrackbarPos('Thresh', title_window)
    rho = cv2.getTrackbarPos('Rho', title_window)
    out = np.copy(img)
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLines(gray, rho, np.pi / 180, thresh)

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow(title_window, out)

if __name__ == '__main__':
    initial_thresh = 150
    initial_rho = 1
    title_window = 'Hough Lines Tuning'

    img_file = 'houghlines3.jpg'
    img = cv2.imread(img_file)

    cv2.namedWindow(title_window)
    cv2.createTrackbar('Thresh', title_window, initial_thresh, 300, update_image)
    cv2.createTrackbar('Rho', title_window, initial_rho, 300, update_image)
    update_image(1)

    # Wait until user press some key
    cv2.waitKey()

