import numpy as np
import cv2 as cv
import math

from rotation import RotationUtil
from Camera import Camera, edge_map_from_homography

court_mid_distance_x = 52.500276
court_mid_distance_y = 34.001964

def camera_to_edge_image(camera_data,
                         model_points, model_line_segment,
                         im_h=720, im_w=1280, line_width=4):
    """
     Project (line) model images using the camera
    :param camera_data: 9 numbers
    :param model_points:
    :param model_line_segment:
    :param im_h: 720
    :param im_w: 1280
    :return: H * W * 3 OpenCV image
    """
    assert camera_data.shape[0] == 9

    # u, v, fl = camera_data[0:3]
    # rod_rot = camera_data[3:6]
    # cc = camera_data[6:9]

    camera = Camera(camera_data)
    im = np.zeros((im_h, im_w, 3), dtype=np.uint8)
    n = model_line_segment.shape[0]
    color = (255, 255, 255)
    for i in range(n):
        idx1, idx2 = model_line_segment[i][0], model_line_segment[i][1]
        p1, p2 = model_points[idx1], model_points[idx2]
        q1 = camera.project_3d(p1[0], p1[1], 0.0, 1.0)
        q2 = camera.project_3d(p2[0], p2[1], 0.0, 1.0)
        q1 = np.rint(q1).astype(np.int)
        q2 = np.rint(q2).astype(np.int)
        cv.line(im, tuple(q1), tuple(q2), color, thickness=line_width)
    return im



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
    pan_angles = extrapolate_pan_angle(camera_centers[:, 0], camera_centers[:, 1]) * (-1)
    tilt_angles = extrapolate_tilt_angle(camera_centers[:, 2], camera_centers[:, 1]) * (-1)
    roll_angles = np.random.normal(0., 0., (num_of_cameras, 1))

    cameras = np.concatenate([
        np.ones((num_of_cameras, 1)) * u,
        np.ones((num_of_cameras, 1)) * v,
        focal_lengths, tilt_angles, pan_angles, roll_angles, camera_centers],
        axis=1)

    return cameras


def extrapolate_tilt_angle(z, y):
    a = np.abs(z)
    b = np.abs(y)+court_mid_distance_y
    angle = np.arctan(a/b)
    return np.degrees(angle).reshape(-1,1)


def extrapolate_pan_angle(x, y):
    a = x - court_mid_distance_x
    b = np.abs(y)+court_mid_distance_y
    angle = np.arctan(a/b)
    return np.degrees(angle).reshape(-1,1)


def generate_camera_centers(cc_statistics, y_intervals=.2):
    cc_mean, cc_std, cc_min, cc_max = cc_statistics
    z = []
    y = np.arange(cc_min[:, 1], cc_max[:, 1], y_intervals)
    inclinations = np.arange(.3, .45, .01)
    for incl in inclinations:
        z_samples = bleachers_line(y, incl)
        z.append(np.stack([np.ones_like(z_samples) * y, z_samples], axis=1))
    yz_samples = np.concatenate(z, axis=0)

    num_of_samples = yz_samples.shape[0]
    x_samples = np.random.normal(cc_mean[:, 0], cc_std[:, 0], (num_of_samples, 1))
    return np.concatenate([x_samples, yz_samples], axis=1)


def bleachers_line(y, slope):
    z = y * (-slope)
    return z



def ut_generate_ptz_cameras():
    """
    Generate PTZ camera demo:  Section 3.1
    """
    import scipy.io as sio
    # data = sio.loadmat('worldcup_dataset_camera_parameter.mat')
    # print(data.keys())
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

    binary_court = sio.loadmat('worldcup2014.mat')

    for i in range(len(cameras)):
        camera_params = cameras[i]
        camera = Camera(camera_params)
        h = camera.homography()
        im = edge_map_from_homography(h, binary_court, (image_w, image_h))
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
        show_image([im])




def show_image(img_list, msg_list=None):
    """
    Display N images. Esc char to close window. For debugging purposes.
    :param img_list: A list with images to be displayed.
    :param msg_list: A list with title for each image to be displayed. If not None, it has to be of equal length to
    the image list.
    :return:
    """
    if not isinstance(img_list, list):
        return 'Input is not a list.'

    if msg_list is None:
        msg_list = [f'{i}' for i in range(len(img_list))]
    else:
        msg_list = [f'{msg}' for msg in msg_list]

    for i in range(len(img_list)):
        cv.imshow(msg_list[i], img_list[i])

    while 1:
        k = cv.waitKey(0)
        if k == 27:
            break
    for msg in msg_list:
        cv.destroyWindow(msg)


if __name__ == '__main__':
    ut_generate_ptz_cameras()

