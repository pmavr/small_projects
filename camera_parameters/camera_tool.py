import cv2
import numpy as np
from Camera import Camera, edge_map_from_homography
from rotation import RotationUtil
import scipy.io as sio
from datetime import datetime
import math
import sys

bar_range = 500
pan_bar_range = 1000
camera_samples = []
Q = 113
R = 114
W = 119
A = 97
S = 115
D = 100
F = 102
G = 103
H = 104
J = 106
K = 107
L = 108

def normalize_in_range(value, min, max, b_range):
    return (((value - 0) * (max - min)) / (b_range - 0)) + min


def update_image(val):
    record_params = cv2.getTrackbarPos('Record params', title_window)
    fp = cv2.getTrackbarPos('Focal length', title_window)
    tilt_angle = cv2.getTrackbarPos('Tilt angle', title_window)
    pan_angle = cv2.getTrackbarPos('Pan angle', title_window)
    roll_angle = cv2.getTrackbarPos('Roll angle', title_window)
    xloc = cv2.getTrackbarPos('Camera loc x', title_window)
    yzloc = cv2.getTrackbarPos('Camera loc y-z', title_window)
    bleachers_inclination_yz = cv2.getTrackbarPos('Bleachers slope y-z', title_window)

    fp = normalize_in_range(fp, 1000, 6000, bar_range)

    xloc = normalize_in_range(xloc, 46.2, 57.2, bar_range)
    yloc = normalize_in_range(yzloc, -66., -25., bar_range)
    bleachers_inclination_yz = normalize_in_range(bleachers_inclination_yz, .3, .45, bar_range)
    # zloc = normalize_in_range(zloc, 10.1387, 23.01126, bar_range)
    zloc = yloc * (-bleachers_inclination_yz)

    tilt_angle = normalize_in_range(tilt_angle, -25., 0., bar_range)
    pan_angle = normalize_in_range(pan_angle, -60., 60., pan_bar_range)
    roll_angle = normalize_in_range(roll_angle, -1., 1., bar_range)

    params = np.array([
        image_center_x,
        image_center_y,
        fp,
        tilt_angle,
        pan_angle,
        roll_angle,
        xloc,
        yloc,
        zloc
    ])

    if record_params == 1:
        camera_samples.append(params)

    camera_params = np.array([
        image_center_x,
        image_center_y,
        fp,
        tilt_angle,
        pan_angle,
        roll_angle,
        xloc,
        yloc,
        zloc
    ])

    camera = Camera(camera_params)
    homography = camera.homography()
    edge_map = edge_map_from_homography(homography,
                                        binary_court,
                                        image_resolution)
    # im = cv2.imread('court.jpg')
    # edge_map2 = cv2.warpPerspective(im, homography, image_resolution, flags=cv2.INTER_LINEAR)
    text = f"focal length: {round(camera.focal_length, 3)} \n" \
           f"cam_loc_X: {round(camera.camera_center_x, 3)} \n" \
           f"cam_loc_Y: {round(camera.camera_center_y, 3)} \n" \
           f"cam_loc_Z: {round(camera.camera_center_z, 3)} \n" \
           f"YZ slope: {round(bleachers_inclination_yz, 3)} \n" \
           f"tilt: {round(tilt_angle, 3):.3f} \n" \
           f"pan: {round(pan_angle, 3):.3f} \n" \
           f"roll: {round(roll_angle, 3):.3f} \n"
    y0, dy = 30, 20
    for i, line in enumerate(text.split('\n')):
        y = y0 + i * dy
        cv2.putText(edge_map, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
    cv2.putText(edge_map, f'{np.round(homography[0,0], 3):<8} {np.round(homography[0,1], 3):=10} {np.round(homography[0,2], 3):>10}', (900, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
    cv2.putText(edge_map, f'{np.round(homography[1,0], 3):<8} {np.round(homography[1,1], 3):=10} {np.round(homography[1,2], 3):>10}', (900, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
    cv2.putText(edge_map, f'{np.round(homography[2,0], 3):<8} {np.round(homography[2,1], 3):=10} {np.round(homography[2,2], 3):>10}', (900, 70), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
    cv2.circle(edge_map, (int(camera.image_center_x), int(camera.image_center_y)), 2, (0, 255, 0), 3)
    cv2.putText(edge_map, f'Samples:{len(camera_samples)}', (600, y0), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
    cv2.imshow(title_window, edge_map)


def save_camera_samples():
    samples = np.array(camera_samples)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    num_of_samples = len(camera_samples)
    filename = f'saved_camera_param_data/{timestamp}-{num_of_samples}.npy'
    np.save(filename, samples)



if __name__ == '__main__':
    image_resolution = (1280, 720)
    image_center_x = image_resolution[0]/2
    image_center_y = image_resolution[1]/2
    focal_point = int(bar_range / 2)
    tilt_angle = int(bar_range / 2)
    pan_angle = int(pan_bar_range / 2)
    roll_angle = int(bar_range / 2)
    camera_loc_x = int(bar_range * .57)
    camera_loc_yz = int(bar_range / 2)
    bleachers_inclination_yz = int(bar_range / 2)
    record_params = 0

    title_window = 'Camera Tool'

    binary_court = sio.loadmat('worldcup2014.mat')

    cv2.namedWindow(title_window)
    cv2.createTrackbar('Record params', title_window, record_params, 1, update_image)
    cv2.createTrackbar('Focal length', title_window, focal_point, bar_range, update_image)
    cv2.createTrackbar('Tilt angle', title_window, tilt_angle, bar_range, update_image)
    cv2.createTrackbar('Pan angle', title_window, pan_angle, pan_bar_range, update_image)
    cv2.createTrackbar('Roll angle', title_window, roll_angle, bar_range, update_image)
    cv2.createTrackbar('Camera loc x', title_window, camera_loc_x, bar_range, update_image)
    cv2.createTrackbar('Camera loc y-z', title_window, camera_loc_yz, bar_range, update_image)
    cv2.createTrackbar('Bleachers slope y-z', title_window, bleachers_inclination_yz, bar_range, update_image)
    update_image(1)

    while 1:
        key = cv2.waitKey(0)
        if key == Q:    # quit
            break
        elif key == W:
            val = cv2.getTrackbarPos('Tilt angle', title_window)
            cv2.setTrackbarPos('Tilt angle', title_window, val+1)
            update_image(1)
        elif key == S:
            val = cv2.getTrackbarPos('Tilt angle', title_window)
            cv2.setTrackbarPos('Tilt angle', title_window, val - 1)
            update_image(1)
        elif key == A:
            val = cv2.getTrackbarPos('Pan angle', title_window)
            cv2.setTrackbarPos('Pan angle', title_window, val - 1)
            update_image(1)
        elif key == D:
            val = cv2.getTrackbarPos('Pan angle', title_window)
            cv2.setTrackbarPos('Pan angle', title_window, val + 1)
            update_image(1)

    if len(camera_samples) > 0:
        save_camera_samples()
        print('Camera samples saved!')
    sys.exit()
