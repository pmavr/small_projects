import cv2
import numpy as np
from Camera import Camera, edge_map_from_homography
import scipy.io as sio


def normalize_in_range(value, min, max):
    return (((value - 0) * (max - min)) / (100 - 0)) + min


def update_image(val):

    focal_point = cv2.getTrackbarPos('Focal length', title_window)
    rotation_0 = cv2.getTrackbarPos('Tilt angle', title_window)
    rotation_1 = cv2.getTrackbarPos('Pan angle', title_window)
    rotation_2 = cv2.getTrackbarPos('Roll angle', title_window)
    use_pan_angle = cv2.getTrackbarPos('use pan-roll movement', title_window)
    pan_angle = cv2.getTrackbarPos('Pan-Roll angle', title_window)
    camera_loc_x = cv2.getTrackbarPos('Camera loc x', title_window)
    camera_loc_y = cv2.getTrackbarPos('Camera loc y', title_window)
    camera_loc_z = cv2.getTrackbarPos('Camera loc z', title_window)

    focal_point = normalize_in_range(focal_point, 1463, 5696)

    rotation_0 = normalize_in_range(rotation_0, 1.60195, 1.83254)
    if use_pan_angle == 1:
        angle_val = normalize_in_range(pan_angle, -0.60999, 0.60999)
        rotation_1 = angle_val
        rotation_2 = angle_val * (-1)
    else:
        rotation_1 = normalize_in_range(rotation_1, -0.65999, 0.65999)
        rotation_2 = normalize_in_range(rotation_2, -0.51343, 0.51286)

    camera_loc_x = normalize_in_range(camera_loc_x, 10., 90.)
    camera_loc_y = normalize_in_range(camera_loc_y, -66.07020, -16.74178)
    camera_loc_z = normalize_in_range(camera_loc_z, 0., 23.01126)

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
           f"tilt: {round(camera.tilt_angle * 180 / np.pi, 3)} \n" \
           f"pan: {round(camera.pan_angle, 6)} \n" \
           f"roll: {round(camera.roll_angle, 6)} \n"
    y0, dy = 30, 20
    for i, line in enumerate(text.split('\n')):
        y = y0 + i * dy
        cv2.putText(edge_map, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
    cv2.putText(edge_map, f'{np.round(homography[0,0], 3):<8} {np.round(homography[0,1], 3):=10} {np.round(homography[0,2], 3):>10}', (900, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
    cv2.putText(edge_map, f'{np.round(homography[1,0], 3):<8} {np.round(homography[1,1], 3):=10} {np.round(homography[1,2], 3):>10}', (900, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
    cv2.putText(edge_map, f'{np.round(homography[2,0], 3):<8} {np.round(homography[2,1], 3):=10} {np.round(homography[2,2], 3):>10}', (900, 70), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
    cv2.circle(edge_map, (int(camera.image_center_x), int(camera.image_center_y)), 2, (0, 255, 0), 3)
    cv2.imshow(title_window, edge_map)

if __name__ == '__main__':
    image_resolution = (1280, 720)
    image_center_x = image_resolution[0]/2
    image_center_y = image_resolution[1]/2
    focal_point = 0
    tilt_angle = 0
    pan_angle = 50
    roll_angle = 50
    pan_roll_angle = 50
    use_pan_roll_angle = 0
    camera_loc_x = 53
    camera_loc_y = 56
    camera_loc_z = 60

    title_window = 'Camera Parameters Tuning'

    binary_court = sio.loadmat('worldcup2014.mat')

    cv2.namedWindow(title_window)
    cv2.createTrackbar('Focal length', title_window, focal_point, 100, update_image)
    cv2.createTrackbar('Tilt angle', title_window, tilt_angle, 100, update_image)
    cv2.createTrackbar('Pan angle', title_window, pan_angle, 100, update_image)
    cv2.createTrackbar('Roll angle', title_window, roll_angle, 100, update_image)
    cv2.createTrackbar('use pan-roll movement', title_window, use_pan_roll_angle, 1, update_image)
    cv2.createTrackbar('Pan-Roll angle', title_window, pan_roll_angle, 100, update_image)
    cv2.createTrackbar('Camera loc x', title_window, camera_loc_x, 100, update_image)
    cv2.createTrackbar('Camera loc y', title_window, camera_loc_y, 100, update_image)
    cv2.createTrackbar('Camera loc z', title_window, camera_loc_z, 100, update_image)
    update_image(1)

    # Wait until user press some key
    cv2.waitKey()

