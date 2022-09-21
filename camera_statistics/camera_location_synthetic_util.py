import numpy as np
import cv2 as cv
import math

import utils
from Camera import Camera


def generate_ptz_cameras(cc_statistics,
                         fl_statistics,
                         u, v):
    """
    Input: PTZ camera base information
    Output: randomly sampled camera parameters
    :param cc_statistics:
    :param fl_statistics:
    :param roll_statistics:
    :param pan_range:
    :param tilt_range:
    :param u:
    :param v:
    :param camera_num:
    :return: N * 9 cameras
    """

    fl_mean, fl_std, fl_min, fl_max = fl_statistics

    camera_centers = generate_camera_centers(cc_statistics, .005)
    num_of_cameras = camera_centers.shape[0]
    focal_lengths = np.random.normal(fl_mean, fl_std, (num_of_cameras, 1))
    pan_angles = utils.extrapolate_pan_angle(camera_centers[:, 0], camera_centers[:, 1]) * (-1)
    tilt_angles = utils.extrapolate_tilt_angle(camera_centers[:, 2], camera_centers[:, 1]) * (-1)
    roll_angles = np.random.normal(0., 0., (num_of_cameras, 1))

    cameras = np.concatenate([
        np.ones((num_of_cameras, 1)) * u,
        np.ones((num_of_cameras, 1)) * v,
        focal_lengths, tilt_angles, pan_angles, roll_angles, camera_centers],
        axis=1)

    return cameras


def generate_camera_centers(cc_statistics, y_intervals=.2):
    cc_mean, cc_std, cc_min, cc_max = cc_statistics
    z = []
    y = np.arange(cc_min[:, 1], cc_max[:, 1], y_intervals)
    inclinations = np.arange(.3, .45, .01)
    for incl in inclinations:
        z_samples = utils.bleachers_line(y, incl)
        z.append(np.stack([np.ones_like(z_samples) * y, z_samples], axis=1))
    yz_samples = np.concatenate(z, axis=0)

    num_of_samples = yz_samples.shape[0]
    x_samples = np.random.normal(cc_mean[:, 0], cc_std[:, 0], (num_of_samples, 1))
    return np.concatenate([x_samples, yz_samples], axis=1)


def main():
    image_w, image_h = 1280, 720
    cc_mean = np.array([[52.36618474, -45.15650112, 16.82156705]])
    cc_std = np.array([[1.23192608, 9.3825635, 2.94875254]])
    cc_min = np.array([[50.05679141, -66., 10.13871263]])
    cc_max = np.array([[54.84563315, -25., 23.01126126]])
    cc_statistics = [cc_mean, cc_std, cc_min, cc_max]

    fl_mean = np.array([[2500.5139785]])
    fl_std = np.array([[716.06817106]])
    fl_min = np.array([[1463.16468414]])
    fl_max = np.array([[3580.]])
    fl_statistics = [fl_mean, fl_std, fl_min, fl_max]

    cameras = generate_ptz_cameras(cc_statistics,
                                   fl_statistics,
                                   image_w / 2.0, image_h / 2.0)

    binary_court = utils.binary_court()

    for i in range(len(cameras)):
        camera_params = cameras[i]
        camera = Camera(camera_params)
        im = camera.to_edge_map(binary_court)
        text = f"focal length: {round(camera.focal_length, 3)} \n" \
               f"cam_loc_X: {round(camera.camera_center_x, 3)} \n" \
               f"cam_loc_Y: {round(camera.camera_center_y, 3)} \n" \
               f"cam_loc_Z: {round(camera.camera_center_z, 3)} \n" \
               f"tilt: {round(camera.tilt_angle, 3):.3f} \n" \
               f"pan: {round(camera.pan_angle, 3):.3f} \n" \
               f"roll: {round(camera.roll_angle, 3):.3f} \n"
        y0, dy = 30, 20
        for i, line in enumerate(text.split('\n')):
            y = y0 + i * dy
            cv.putText(im, line, (20, y),
                       cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
            cv.circle(im, (int(camera.image_center_x), int(camera.image_center_y)), 2, (0, 255, 0), 3)
        utils.show_image([im])



if __name__ == '__main__':
    main()
