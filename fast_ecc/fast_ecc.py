import matlab.engine
import cv2
import numpy as np
import time

if __name__=='__main__':

    eng = matlab.engine.start_matlab()
    path = f'/home/panagiotis/dev/projects/python/small_projects/fast_ecc/matlab_code/'
    eng.addpath(path, nargout=0)

    detected_edge_map = cv2.imread('detected_edge_map.jpg')
    estimated_edge_map = cv2.imread('estimated_edge_map.jpg')
    identity = np.eye(3)

    start = time.time()
    identity = matlab.double(identity)
    detected_edge_map = matlab.double(detected_edge_map.astype(float))
    estimated_edge_map = matlab.double(estimated_edge_map.astype(float))
    end = time.time() - start

    homography = eng.ecc(estimated_edge_map, detected_edge_map, 1, 10, 'homography', identity, .75, 150)


    print(end)