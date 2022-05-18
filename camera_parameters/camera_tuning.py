import cv2
import numpy as np
from Camera import Camera, edge_map_from_homography
import scipy.io as sio


def normalize_in_range(value, min, max):
    return (((value - 0) * (max - min)) / (100 - 0)) + min


def update_image(val):

    focal_point = cv2.getTrackbarPos('focal_point', title_window)
    rotation_0 = cv2.getTrackbarPos('rotation_0', title_window)
    rotation_1 = cv2.getTrackbarPos('rotation_1', title_window)
    rotation_2 = cv2.getTrackbarPos('rotation_2', title_window)
    use_pan_angle = cv2.getTrackbarPos('use_pan_angle', title_window)
    pan_angle = cv2.getTrackbarPos('pan_angle', title_window)
    camera_loc_x = cv2.getTrackbarPos('camera_loc_x', title_window)
    camera_loc_y = cv2.getTrackbarPos('camera_loc_y', title_window)
    camera_loc_z = cv2.getTrackbarPos('camera_loc_z', title_window)

    focal_point = normalize_in_range(focal_point, 1463, 5696)

    rotation_0 = normalize_in_range(rotation_0, 1.60195, 1.83254)
    if use_pan_angle == 1:
        angle_val = normalize_in_range(pan_angle, -0.55999, 0.55999)
        rotation_1 = angle_val
        rotation_2 = angle_val * (-1)
    else:
        rotation_1 = normalize_in_range(rotation_1, -0.55999, 0.55999)
        rotation_2 = normalize_in_range(rotation_2, -0.46343, 0.46286)

    camera_loc_x = normalize_in_range(camera_loc_x, 45.05679, 60.84563)
    camera_loc_y = normalize_in_range(camera_loc_y, -66.07020, -16.74178)
    camera_loc_z = normalize_in_range(camera_loc_z, 10.13871, 23.01126)

    camera_params = np.array([
        image_center_x,
        image_center_y,
        focal_point,
        rotation_0,
        rotation_1,
        rotation_2,
        camera_loc_x,
        camera_loc_y,
        camera_loc_z
    ])
    camera = Camera(camera_params)
    homography = camera.get_homography()
    edge_map = edge_map_from_homography(homography,
                                        binary_court,
                                        image_resolution)
    cv2.imshow(title_window, edge_map)

if __name__ == '__main__':
    image_resolution = (1280, 720)
    image_center_x = image_resolution[0]/2
    image_center_y = image_resolution[1]/2
    focal_point = 0
    rotation_0 = 0
    rotation_1 = 50
    rotation_2 = 50
    pan_angle = 50
    camera_loc_x = 50
    camera_loc_y = 50
    camera_loc_z = 0

    title_window = 'Camera Parameters Tuning'

    binary_court = sio.loadmat('worldcup2014.mat')

    cv2.namedWindow(title_window)
    cv2.createTrackbar('focal_point', title_window, focal_point, 100, update_image)
    cv2.createTrackbar('rotation_0', title_window, rotation_0, 100, update_image)
    cv2.createTrackbar('rotation_1', title_window, rotation_1, 100, update_image)
    cv2.createTrackbar('rotation_2', title_window, rotation_2, 100, update_image)
    cv2.createTrackbar('use_pan_angle', title_window, pan_angle, 1, update_image)
    cv2.createTrackbar('pan_angle', title_window, pan_angle, 100, update_image)
    cv2.createTrackbar('camera_loc_x', title_window, camera_loc_x, 100, update_image)
    cv2.createTrackbar('camera_loc_y', title_window, camera_loc_y, 100, update_image)
    cv2.createTrackbar('camera_loc_z', title_window, camera_loc_z, 100, update_image)
    update_image(1)

    # Wait until user press some key
    cv2.waitKey()

