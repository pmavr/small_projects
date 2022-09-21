import cv2
import numpy as np
from Camera import Camera
from wand.image import Image
import scipy.io as sio
from datetime import datetime
import sys
import utils

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
    dis_1 = cv2.getTrackbarPos('Distortion param. 1', title_window)
    dis_2 = cv2.getTrackbarPos('Distortion param. 2', title_window)
    dis_3 = cv2.getTrackbarPos('Distortion param. 3', title_window)

    fp = normalize_in_range(fp, 1000, 15000, bar_range)

    xloc = normalize_in_range(xloc, 46.2, 57.2, bar_range)
    yloc = normalize_in_range(yzloc, -156., -25., bar_range)
    bleachers_inclination_yz = normalize_in_range(bleachers_inclination_yz, .27, .35, bar_range)
    # zloc = normalize_in_range(bleachers_inclination_yz, 10.1387, 30.01126, bar_range)
    zloc = yloc * (-bleachers_inclination_yz)

    tilt_angle = normalize_in_range(tilt_angle, -25., 0., bar_range)
    pan_angle = normalize_in_range(pan_angle, -70., 70., pan_bar_range)
    roll_angle = normalize_in_range(roll_angle, -90., 90., bar_range)

    dis_1 = normalize_in_range(dis_1, -.4, .4, pan_bar_range)
    dis_2 = normalize_in_range(dis_2, -.4, .4, pan_bar_range)
    dis_3 = normalize_in_range(dis_3, -.4, .4, pan_bar_range)

    # dis_2 = dis_1
    # dis_3 = dis_1

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
    edge_map = camera.to_edge_map(court_template)
    img = Image.from_array(edge_map)
    img.distort('barrel', (dis_1, dis_2, dis_3, 1.))
    edge_map = np.array(img)

    # im = cv2.imread('images/image.png')
    # im = cv2.resize(im, (1280, 720))
    # edge_map = cv2.addWeighted(src1=im,
    #                                src2=edge_map,
    #                                alpha=.95, beta=1, gamma=0.)

    text = f"focal length: {round(camera.focal_length, 3)} \n" \
           f"cam_loc_X: {round(camera.camera_center_x, 3)} \n" \
           f"cam_loc_Y: {round(camera.camera_center_y, 3)} \n" \
           f"cam_loc_Z: {round(camera.camera_center_z, 3)} \n" \
           f"YZ slope: {round(bleachers_inclination_yz, 3)} \n" \
           f"tilt: {round(tilt_angle, 3):.3f} \n" \
           f"pan: {round(pan_angle, 3):.3f} \n" \
           f"roll: {round(roll_angle, 3):.3f} \n" \
           f"Dist. 1: {round(dis_1, 3):.3f} \n" \
           f"Dist. 2: {round(dis_2, 3):.3f} \n" \
           f"Dist. 3: {round(dis_3, 3):.3f} \n" \
           f"Cam. Orient.: {camera.orientation()} \n"
    y0, dy = 30, 20
    for i, line in enumerate(text.split('\n')):
        y = y0 + i * dy
        cv2.putText(edge_map, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
    cv2.putText(edge_map, f'{np.round(homography[0,0], 3):<8} {np.round(homography[0,1], 3):=10} {np.round(homography[0,2], 3):>10}', (900, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
    cv2.putText(edge_map, f'{np.round(homography[1,0], 3):<8} {np.round(homography[1,1], 3):=10} {np.round(homography[1,2], 3):>10}', (900, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
    cv2.putText(edge_map, f'{np.round(homography[2,0], 3):<8} {np.round(homography[2,1], 3):=10} {np.round(homography[2,2], 3):>10}', (900, 70), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
    cv2.putText(edge_map, f'Samples:{len(camera_samples)}', (600, y0), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))

    cv2.imshow(title_window, edge_map)


def save_camera_samples():
    samples = np.array(camera_samples)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    num_of_samples = len(camera_samples)
    filename = f'saved_camera_param_data/{timestamp}-{num_of_samples}.npy'
    np.save(filename, samples)



if __name__ == '__main__':
    court_template = np.load('binary_court.npy')
    image_resolution = (1280, 720)
    image_center_x = image_resolution[0]/2
    image_center_y = image_resolution[1]/2
    focal_point = int(pan_bar_range / 8)
    tilt_angle = int(bar_range / 2)
    pan_angle = int(pan_bar_range / 2)
    roll_angle = int(bar_range / 2)
    camera_loc_x = int(bar_range * .57) # 208
    camera_loc_yz = int(bar_range / 2) # 456
    bleachers_inclination_yz = 351 # int(bar_range / 2) # 351 | 433
    dis_1 = int(pan_bar_range / 2)
    dis_2 = int(pan_bar_range / 2)
    dis_3 = int(pan_bar_range / 2)
    record_params = 0

    title_window = 'Camera Tool'
    cv2.namedWindow(title_window)
    cv2.createTrackbar('Record params', title_window, record_params, 1, update_image)
    cv2.createTrackbar('Focal length', title_window, focal_point, pan_bar_range, update_image)
    cv2.createTrackbar('Tilt angle', title_window, tilt_angle, bar_range, update_image)
    cv2.createTrackbar('Pan angle', title_window, pan_angle, pan_bar_range, update_image)
    cv2.createTrackbar('Roll angle', title_window, roll_angle, bar_range, update_image)
    cv2.createTrackbar('Camera loc x', title_window, camera_loc_x, bar_range, update_image)
    cv2.createTrackbar('Camera loc y-z', title_window, camera_loc_yz, bar_range, update_image)
    cv2.createTrackbar('Bleachers slope y-z', title_window, bleachers_inclination_yz, bar_range, update_image)
    cv2.createTrackbar('Distortion param. 1', title_window, dis_1, pan_bar_range, update_image)
    cv2.createTrackbar('Distortion param. 2', title_window, dis_2, pan_bar_range, update_image)
    cv2.createTrackbar('Distortion param. 3', title_window, dis_3, pan_bar_range, update_image)
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
