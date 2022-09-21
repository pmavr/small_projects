import cv2
import numpy as np

if __name__=='__main__':

    im = cv2.imread('undistorted.jpg')
    distorted_im = cv2.imread('distorted.jpg')
    gray = cv2.cvtColor(distorted_im, cv2.COLOR_BGR2GRAY)

    obj_points = np.array([[
       [52.50028, 68.00393, 0.], # top mid intersection
       [52.50028, 43.14596, 0.], # top circle intersection
       [52.50028, 24.85796, 0.], # bot circle intersection
       [61.64428, 34.00196, 0.], # right-most circle
       [43.35628, 34.00196, 0.]  # left-most circle
    ]]).astype('float32')

    img_points = np.array([[
        [641, 208], # top mid intersection
        [641, 357], # top circle intersection
        [641, 507], # bot circle intersection
        [970, 433], # right-most circle
        [314, 433]  # left-most circle
    ]]).astype('float32')

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        [obj_points], [img_points], gray.shape[::-1], None, None)

    print('gr')
