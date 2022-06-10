import numpy as np
import cv2 as cv
import math

from rotation import RotationUtil
from Camera import Camera, edge_map_from_homography


class SyntheticUtil:
    @staticmethod
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



    @staticmethod
    def generate_ptz_cameras(cc_statistics,
                             fl_statistics,
                             roll_statistics,
                             pan_range, tilt_range,
                             u, v,
                             camera_num):
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
        cc_mean, cc_std, cc_min, cc_max = cc_statistics
        fl_mean, fl_std, fl_min, fl_max = fl_statistics
        roll_mean, roll_std, roll_min, roll_max = roll_statistics
        pan_min, pan_max = pan_range
        tilt_min, tilt_max = tilt_range

        # generate random values from the distribution
        camera_centers = np.random.normal(cc_mean, cc_std, (camera_num, 3))
        focal_lengths = np.random.normal(fl_mean, fl_std, (camera_num, 1))
        rolls = np.random.normal(roll_mean, roll_std, (camera_num, 1))
        pans = np.random.uniform(pan_min, pan_max, camera_num)
        tilts = np.random.uniform(tilt_min, tilt_max, camera_num)

        cameras = np.zeros((camera_num, 9))
        for i in range(camera_num):
            base_rotation = RotationUtil.rotate_y_axis(0) @ RotationUtil.rotate_x_axis(rolls[i]) @ \
                            RotationUtil.rotate_x_axis(-90)
            pan_tilt_rotation = RotationUtil.pan_y_tilt_x(pans[i], tilts[i])
            rotation = pan_tilt_rotation @ base_rotation
            rot_vec, _ = cv.Rodrigues(rotation)

            cameras[i][0], cameras[i][1] = u, v
            cameras[i][2] = focal_lengths[i]
            cameras[i][3], cameras[i][4], cameras[i][5] = rot_vec[0], rot_vec[1], rot_vec[2]
            cameras[i][6], cameras[i][7], cameras[i][8] = camera_centers[i][0], camera_centers[i][1], camera_centers[i][
                2]
        return cameras

    @staticmethod
    def sample_positive_pair(pp, cc, base_roll, pan, tilt, fl,
                             pan_std, tilt_std, fl_std):
        """
        Sample a camera that has similar pan-tilt-zoom with (pan, tilt, fl).
        The pair of the camera will be used a positive-pair in the training
        :param pp: [u, v]
        :param cc: camera center
        :param base_roll: camera base, roll angle
        :param pan:
        :param tilt:
        :param fl:
        :param pan_std:
        :param tilt_std:
        :param fl_std:
        :return:
        """

        def get_nearby_data(d, std):
            assert std > 0
            delta = np.random.uniform(-0.5, 0.5, 1) * std
            return d + delta

        pan = get_nearby_data(pan, pan_std)
        tilt = get_nearby_data(tilt, tilt_std)
        fl = get_nearby_data(fl, fl_std)

        camera = np.zeros(9)
        camera[0] = pp[0]
        camera[1] = pp[1]
        camera[2] = fl

        base_rotation = RotationUtil.rotate_y_axis(0) @ RotationUtil.rotate_y_axis(base_roll) @ \
                        RotationUtil.rotate_x_axis(-90)
        pan_tilt_rotation = RotationUtil.pan_y_tilt_x(pan, tilt)
        rotation = pan_tilt_rotation @ base_rotation
        rot_vec = RotationUtil.rotation_matrix_to_Rodrigues(rotation)
        camera[3: 6] = rot_vec.squeeze()
        camera[6: 9] = cc
        return camera

    @staticmethod
    def generate_database_images(pivot_cameras, positive_cameras,
                                 model_points, model_line_segment):

        """
        Default size 180 x 320 (h x w)
        Generate database image for siamese network training
        :param pivot_cameras:
        :param positive_cameras:
        :return:
        """
        n = pivot_cameras.shape[0]
        assert n == positive_cameras.shape[0]

        # N x 1 x H x W pivot images
        # N x 1 x H x w positive image
        # negative pairs are randomly selected
        im_h, im_w = 180, 320
        pivot_images = np.zeros((n, 1, im_h, im_w), dtype=np.uint8)
        positive_images = np.zeros((n, 1, im_h, im_w), dtype=np.uint8)

        for i in range(n):
            piv_cam = pivot_cameras[i, :]
            pos_cam = positive_cameras[i, :]

            piv_im = SyntheticUtil.camera_to_edge_image(piv_cam, model_points, model_line_segment, 720, 1280, 4)
            pos_im = SyntheticUtil.camera_to_edge_image(pos_cam, model_points, model_line_segment, 720, 1280, 4)

            # to a smaller image
            piv_im = cv.resize(piv_im, (im_w, im_h))
            pos_im = cv.resize(pos_im, (im_w, im_h))

            # to a gray image
            piv_im = cv.cvtColor(piv_im, cv.COLOR_BGR2GRAY)
            pos_im = cv.cvtColor(pos_im, cv.COLOR_RGB2GRAY)

            pivot_images[i, 0, :, :] = piv_im
            positive_images[i, 0, :, :] = pos_im

        return (pivot_images, positive_images)


def ut_generate_ptz_cameras():
    """
    Generate PTZ camera demo:  Section 3.1
    """
    import scipy.io as sio
    # data = sio.loadmat('worldcup_dataset_camera_parameter.mat')
    # print(data.keys())
    image_w, image_h = 640, 360
    cc_mean = np.array([[52.36618474, -45.15650112, 16.82156705]])
    cc_std = np.array([[2.23192608, 9.3825635, 2.94875254]])
    cc_min = np.array([[45.05679141, -66.0702037, 10.13871263]])
    cc_max = np.array([[60.84563315, -16.74178234, 23.01126126]])
    cc_statistics = [cc_mean, cc_std, cc_min, cc_max]

    fl_mean = np.array([[3018.18139785]])
    fl_std = np.array([[716.06817106]])
    fl_min = np.array([[1463.16468414]])
    fl_max = np.array([[5696.98454881]])
    fl_statistics = [fl_mean, fl_std, fl_min, fl_max]
    roll_statistics = [0, 0.2, -1.0, 1.0]

    pan_range = [-35.0, 35.0]
    tilt_range = [-15.0, -5.0]
    num_camera = 100

    cameras = SyntheticUtil.generate_ptz_cameras(cc_statistics,
                                                 fl_statistics,
                                                 roll_statistics,
                                                 pan_range, tilt_range,
                                                 image_w / 2.0, image_h / 2.0,
                                                 num_camera)

    binary_court = sio.loadmat('worldcup2014.mat')

    for i in range(num_camera):
        camera_params = cameras[i]
        camera = Camera(camera_params)
        h = camera.homography()
        im = edge_map_from_homography(h, binary_court, (image_w, image_h))
        text = f"focal length: {round(camera.focal_length, 3)} \n" \
               f"cam_loc_X: {round(camera.camera_center_x, 3)} \n" \
               f"cam_loc_Y: {round(camera.camera_center_y, 3)} \n" \
               f"cam_loc_Z: {round(camera.camera_center_z, 3)} \n" \
               f"tilt: {round(math.degrees(camera.tilt_angle), 3)} \n" \
               f"pan: {round(math.degrees(camera.pan_angle), 3)} \n" \
               f"roll: {round(math.degrees(camera.roll_angle), 3)} \n"
        y0, dy = 30, 20
        for i, line in enumerate(text.split('\n')):
            y = y0 + i * dy
            cv.putText(im, line, (20, y),
                        cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
        show_image([im])



def ut_sample_positive_pair():
    """
    def sample_positive_pair(pp, cc, base_roll, pan, tilt, fl,
                             pan_std, tilt_std, fl_std):
    """
    import scipy.io as sio
    data = sio.loadmat('../../data/worldcup_dataset_camera_parameter.mat')
    cc_mean = data['cc_mean']

    pp = np.asarray([1280.0 / 2, 720.0 / 2])
    cc = cc_mean
    base_roll = np.random.uniform(-0.5, 0.5)
    pan = 10.0
    tilt = -10.0
    fl = 3800
    pan_std = 1.5
    tilt_std = 0.75
    fl_std = 30

    camera = np.zeros(9)
    camera[0] = 640.000000
    camera[1] = 360.000000
    camera[2] = fl

    base_rotation = RotationUtil.rotate_y_axis(0) @ \
                    RotationUtil.rotate_z_axis(base_roll) @ \
                    RotationUtil.rotate_x_axis(-90)
    pan_tilt_rotation = RotationUtil.pan_y_tilt_x(pan, tilt)
    rotation = pan_tilt_rotation @ base_rotation
    rot_vec, _ = cv.Rodrigues(rotation)
    camera[3: 6] = rot_vec.squeeze()
    camera[6: 9] = cc

    pivot = camera

    positive = SyntheticUtil.sample_positive_pair(pp, cc, base_roll, pan, tilt, fl,
                                                  pan_std, tilt_std, fl_std)

    data = sio.loadmat('../../data/worldcup2014.mat')
    print(data.keys())
    model_points = data['points']
    model_line_index = data['line_segment_index']

    im1 = SyntheticUtil.camera_to_edge_image(pivot, model_points, model_line_index, 720, 1280)
    im2 = SyntheticUtil.camera_to_edge_image(positive, model_points, model_line_index, 720, 1280)
    cv.imshow("pivot", im1)
    cv.imshow("positive", im2)
    cv.waitKey(5000)


def ut_generate_database_images():
    import scipy.io as sio
    data = sio.loadmat('../../data/worldcup_sampled_cameras.mat')
    pivot_cameras = data['pivot_cameras']
    positive_cameras = data['positive_cameras']

    n = 10000
    pivot_cameras = pivot_cameras[0:n, :]
    positive_cameras = positive_cameras[0:n, :]

    data = sio.loadmat('../../data/worldcup2014.mat')
    print(data.keys())
    model_points = data['points']
    model_line_index = data['line_segment_index']

    pivot_images, positive_images = SyntheticUtil.generate_database_images(pivot_cameras, positive_cameras,
                                                                           model_points, model_line_index)

    # print('{} {}'.format(pivot_images.shape, positive_images.shape))
    sio.savemat('train_data_10k.mat', {'pivot_images': pivot_images,
                                       'positive_images': positive_images,
                                       'cameras': pivot_cameras})



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
    # ut_sample_positive_pair()
    # ut_generate_database_images()
