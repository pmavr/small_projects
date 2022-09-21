import numpy as np
import cv2 as cv
import pandas as pd

import utils
from Camera import Camera


def tilt_angle(z, y):
    angle = np.arctan(np.abs(z) / y)
    return np.degrees(angle)


def b_max_for_mid(cmax, theta):
    return abs(cmax / np.sin(np.radians(theta)))


def b_min_for_mid(cmin, theta):
    return abs(cmin / np.sin(np.radians(theta)))


def b_max_for_sides(amax, phi):
    return abs(amax / np.sin(np.radians(phi)))


def b_min_for_sides(cmin, phi):
    return abs(cmin / np.cos(np.radians(phi)))


def angle_extremes_for_camera_location(x, y, z):

    pan_range = np.arange(-65., 65, .5)
    tilt_range = np.zeros((len(pan_range), 2))
    for i, pan_angle in enumerate(pan_range):
        camera_params = np.array([640, 360, 3500.0, -12.5, pan_angle, 0., x, y, z])
        camera = Camera(camera_params)

        orientation = camera.orientation()

        if orientation == -1:  # extremes case
            continue

        elif orientation == 0:  # mid case
            theta = abs(90 - pan_angle)
            cmax = abs(y) + camera.court_width_y
            cmin = abs(y)
            bmax = b_max_for_mid(cmax, theta)
            bmin = b_min_for_mid(cmin, theta)

            max_tilt = tilt_angle(z, bmax) * (-1)
            min_tilt = tilt_angle(z, bmin) * (-1)

        elif orientation > 0:  # side case
            x_camera_shift = x - camera.court_mid_length_x
            amax = camera.court_mid_length_x + x_camera_shift * np.sign(pan_angle) * (-1)
            cmin = abs(y)
            bmax = b_max_for_sides(amax, pan_angle)
            bmin = b_min_for_sides(cmin, pan_angle)

            max_tilt = tilt_angle(z, bmax) * (-1)
            min_tilt = tilt_angle(z, bmin) * (-1)

        tilt_range[i, 0] = min_tilt
        tilt_range[i, 1] = max_tilt

    data = np.concatenate([
        pan_range.reshape(-1, 1),
        tilt_range], axis=1)
    data = pd.DataFrame(data, columns=['pan', 'max_tilt', 'min_tilt'])
    data = data[(data['max_tilt'] < 0.) & (data['min_tilt'] < 0.)]
    return data


def generate_data_within_extremes(extremes, density):
    data = []
    for _, row in extremes.iterrows():
        interpolated_tilt_angles = np.arange(row['max_tilt'], row['min_tilt'], density)
        pan_angle = np.ones_like(interpolated_tilt_angles) * row['pan']
        batch = np.stack([
            interpolated_tilt_angles,
            pan_angle],
            axis=1)
        data.append(batch)
    data = np.concatenate(data, axis=0)
    return data


def generate_ptz_cameras(x, y, z, fl_statistics, u, v):
    fl_min, fl_max = fl_statistics
    extremes = angle_extremes_for_camera_location(x, y, z)
    tilt_pan_angles = generate_data_within_extremes(extremes, density=.05)

    num_of_cameras = tilt_pan_angles.shape[0]
    focal_lengths = np.random.uniform(fl_min, fl_max, (num_of_cameras, 1))
    roll_angles = np.random.normal(0., 0., (num_of_cameras, 1))

    cameras = np.concatenate([
        np.ones((num_of_cameras, 1)) * u,
        np.ones((num_of_cameras, 1)) * v,
        focal_lengths, tilt_pan_angles, roll_angles,
        np.ones((num_of_cameras, 1)) * x,
        np.ones((num_of_cameras, 1)) * y,
        np.ones((num_of_cameras, 1)) * z],
        axis=1)

    return cameras


def main():
    image_w, image_h = 1280, 720
    x = 52.47 # 50.776
    y = -43.864 # -36.664
    slope = .291
    z = utils.bleachers_line(y, slope)

    pan = utils.extrapolate_pan_angle(x, y).item() * (-1) # 1.4
    tilt = utils.extrapolate_tilt_angle(z, y).item() * (-1)  # -9.6
    camera_params = np.array([640, 360, 3500.0, tilt, pan, 0., x, y, z])
    camera = Camera(camera_params)
    fl_min = np.array([[1463.16468414]])
    # fl_max = camera.max_focal_lengths()
    fl_max = np.array([[5937.98454881]])
    fl_statistics = [fl_min, fl_max]

    cameras = generate_ptz_cameras(x, y, z, fl_statistics,
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
