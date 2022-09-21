import numpy as np
import cv2

court_mid_distance_x = 52.500276
court_mid_distance_y = 34.001964


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
        cv2.imshow(msg_list[i], img_list[i])

    while 1:
        k = cv2.waitKey(0)
        if k == 27:
            break
    for msg in msg_list:
        cv2.destroyWindow(msg)


def binary_court():
    return np.load('binary_court.npy')


def extrapolate_tilt_angle(z, y):
    a = np.abs(z)
    b = np.abs(y) + court_mid_distance_y
    angle = np.arctan(a / b)
    return np.degrees(angle).reshape(-1, 1)


def extrapolate_pan_angle(x, y):
    a = x - court_mid_distance_x
    b = np.abs(y) + court_mid_distance_y
    angle = np.arctan(a / b)
    return np.degrees(angle).reshape(-1, 1)


def bleachers_line(y, slope):
    z = y * (-slope)
    return z