import numpy as np
import cv2
import math
import utils


class Camera:
    court_mid_length_x = 52.500276  # meters
    court_mid_width_y = 34.001964
    court_length_x = 105.000552
    court_width_y = 68.003928

    def __init__(self, camera_params):
        self.image_center_x = camera_params[0]
        self.image_center_y = camera_params[1]
        self.focal_length = camera_params[2]
        self.tilt_angle = camera_params[3]
        self.pan_angle = camera_params[4]
        self.roll_angle = camera_params[5]
        self.camera_center_x = camera_params[6]
        self.camera_center_y = camera_params[7]
        self.camera_center_z = camera_params[8]
        self.camera_center = camera_params[6:9]
        base_rotation = self.rotate_y_axis(0) @ self.rotate_z_axis(self.roll_angle) @ \
                        self.rotate_x_axis(-90)
        pan_tilt_rotation = self.pan_y_tilt_x(self.pan_angle, self.tilt_angle)
        rotation = pan_tilt_rotation @ base_rotation
        self.rot_vector, _ = cv2.Rodrigues(rotation)
        self.image_width = int(2 * self.image_center_x)
        self.image_height = int(2 * self.image_center_y)

    def calibration_matrix(self):
        return np.array([[self.focal_length, 0, self.image_center_x],
                         [0, self.focal_length, self.image_center_y],
                         [0, 0, 1]])

    def rotation_matrix(self):
        rotation, _ = cv2.Rodrigues(self.rot_vector)
        return rotation

    def homography(self):
        P = self.projection_matrix()
        h = P[:, [0, 1, 3]]
        return h

    def add_barrel_distortion(self, img, dis_1, dis_2, dis_3, dis_4):
        distCoeff = np.zeros((4, 1), np.float64)
        distCoeff[0, 0] = dis_1
        distCoeff[1, 0] = dis_2
        distCoeff[2, 0] = dis_3
        distCoeff[3, 0] = dis_4

        K = self.calibration_matrix()
        output = cv2.undistort(img, K, distCoeff)
        return output

    def projection_matrix(self):
        P = np.eye(3, 4)
        P[:, 3] = -1 * self.camera_center
        K = self.calibration_matrix()
        R = self.rotation_matrix()
        return K @ R @ P

    @staticmethod
    def rotate_x_axis(angle):
        """
        rotate coordinate with X axis
        https://en.wikipedia.org/wiki/Rotation_matrix + transpose
        http://mathworld.wolfram.com/RotationMatrix.html
        :param angle: in degree
        :return:
        """
        angle = math.radians(angle)
        s = math.sin(angle)
        c = math.cos(angle)

        r = np.asarray([[1, 0, 0],
                        [0, c, -s],
                        [0, s, c]])
        r = np.transpose(r)
        return r

    @staticmethod
    def rotate_y_axis(angle):
        """
        rotate coordinate with X axis
        :param angle:
        :return:
        """
        angle = math.radians(angle)
        s = math.sin(angle)
        c = math.cos(angle)

        r = np.asarray([[c, 0, s],
                        [0, 1, 0],
                        [-s, 0, c]])
        r = np.transpose(r)
        return r

    @staticmethod
    def rotate_z_axis(angle):
        """
        :param angle:
        :return:
        """
        angle = math.radians(angle)
        s = math.sin(angle)
        c = math.cos(angle)

        r = np.asarray([[c, -s, 0],
                        [s, c, 0],
                        [0, 0, 1]])
        r = np.transpose(r)
        return r

    def pan_y_tilt_x(self, pan, tilt):
        """
        Rotation matrix of first pan, then tilt
        :param pan:
        :param tilt:
        :return:
        """
        r_tilt = self.rotate_x_axis(tilt)
        r_pan = self.rotate_y_axis(pan)
        m = r_tilt @ r_pan
        return m

    @staticmethod
    def rotation_matrix_to_Rodrigues(m):
        assert m.shape[0] == 3 and m.shape[1] == 3
        rot_vec, _ = cv2.Rodrigues(m)
        return rot_vec

    @staticmethod
    def project_point_on_topview(point, h, s_w=1, s_h=1):
        x, y = point
        w = 1.0
        p = np.zeros(3)
        p[0], p[1], p[2] = x, y, w

        m1 = np.array([[1, 0, 0],
                       [0, -1, 68],
                       [0, 0, 1]])
        scale = np.array([[s_w, 0, 0],
                          [0, s_h, 0],
                          [0, 0, 1]])
        homography_matrix = h @ m1
        homography_matrix = homography_matrix @ scale
        inverted_homography_matrix = np.linalg.inv(homography_matrix)
        q = inverted_homography_matrix @ p

        assert q[2] != 0.0
        projected_x = np.rint(q[0] / q[2]).astype(np.int)
        projected_y = np.rint(q[1] / q[2]).astype(np.int)
        return [projected_x, projected_y]

    @staticmethod
    def project_point_on_frame(x, y, h):
        p = np.array([x, y, 1.])
        q = h @ p

        assert q[2] != 0.0
        projected_x = np.rint(q[0] / q[2]).astype(np.int)
        projected_y = np.rint(q[1] / q[2]).astype(np.int)
        return [projected_x, projected_y]

    @staticmethod
    def scale_point(x, y, s):
        p = np.array([x, y, 1.])
        scale = np.array( [
            [s, 0, 20],
            [0, s, 20],
            [0, 0, 1]])
        q = scale @ p
        projected_x = np.rint(q[0] / q[2]).astype(np.int)
        projected_y = np.rint(q[1] / q[2]).astype(np.int)
        return projected_x, projected_y

    def distance_from_camera(self):
        return self.camera_center_z / np.cos(np.radians(90 - self.tilt_angle)) * (-1)

    def generate_human(self, height=1.8):
        dist_y_from_camera = self.camera_center_z * np.tan(np.radians(90 - self.tilt_angle)) * (-1)
        human_feet_xloc = dist_y_from_camera * np.cos(np.radians(90 - self.pan_angle)) + self.camera_center_x
        human_feet_yloc = dist_y_from_camera * np.sin(np.radians(90 - self.pan_angle)) + self.camera_center_y
        apparent_height = self.focal_length * height / self.distance_from_camera()
        return (human_feet_xloc, human_feet_yloc), apparent_height

    def max_focal_lengths(self):
        court_template = utils.binary_court()
        # homography = self.homography()
        court_x_dist = self.court_length_x
        court_y_dist = self.court_width_y
        x = self.camera_center_x
        x_compl = court_x_dist - x
        y = self.camera_center_y
        y_compl = court_y_dist + abs(y)
        z = self.camera_center_z

        calibration_lines = {
            'left_bot': {
                'line': court_template[32],  # 12340
                'desirable_apparent_width': 257 * 3.25,
                'camera_distance': np.sqrt(np.square(z) + np.square(y) + np.square(x))
            },
            'left_top': {
                'line': court_template[47],  # 12340 fl
                'desirable_apparent_width': 111 * 4,
                'camera_distance': np.sqrt(np.square(z) + np.square(y_compl) + np.square(x))
            },
            'right_bot': {
                'line': court_template[48],  # 12340 fl
                'desirable_apparent_width': 257 * 3.25,
                'camera_distance': np.sqrt(np.square(z) + np.square(y) + np.square(x_compl))
            },
            'right_top': {
                'line': court_template[63],  # 12340 fl
                'desirable_apparent_width': 111 * 4,
                'camera_distance': np.sqrt(np.square(z) + np.square(x_compl) + np.square(y_compl))
            },
            'mid_bot': {
                'line': court_template[64],  # 8420 fl
                'desirable_apparent_width': 107 * 8.5,
                'camera_distance': np.sqrt(np.square(z) + np.square(y))
            },
            'mid_top': {
                'line': court_template[79],  # 10660 fl
                'desirable_apparent_width': 50 * 8.5,
                'camera_distance': np.sqrt(np.square(z) + (abs(y) + np.square(y_compl)))
            }
        }
        # fls = [12340, 12340, 12340, 12340, 8420, 10660]
        # apparent_widths = []
        # for key in calibration_lines:
        #     line_seg = calibration_lines[key]
        #     p1, p2 = line_seg[:2], line_seg[2:]
        #
        #     q1 = Camera.project_point_on_frame(p1[0], p1[1], homography)
        #     q2 = Camera.project_point_on_frame(p2[0], p2[1], homography)
        #     # line = self.get_calibration_lines(q1, q2)
        #     # r1, r2 = line[:2], line[2:]
        #     width = np.linalg.norm(np.array(q1) - np.array(q2))
        #     apparent_widths.append(width)

        actual_width = 4.25025
        max_focal_lengths = []
        for key in calibration_lines:
            d = calibration_lines[key]['camera_distance']
            p = calibration_lines[key]['desirable_apparent_width']
            fl = d * p / actual_width
            max_focal_lengths.append(fl)
        return max_focal_lengths

    @staticmethod
    def get_calibration_lines(p1, p2):
        if p1[1] < p2[1]:
            top_point = p1
            bot_point = p2
        else:
            top_point = p2
            bot_point = p1
        right_vertex = [top_point[0], bot_point[1]]
        return top_point + right_vertex

    def _is_off_image_point(self, point):
        x, y = point
        return x < 0 or y < 0 or x > self.image_width or y > self.image_height

    def to_edge_map(self, court_template):
        edge_map = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        homography = self.homography()
        n_line_segments = court_template.shape[0]
        # focal_calibration_lines = [32, 47, 48, 63, 64, 79]
        focal_calibration_lines = [7, 8, 23, 24]
        # focal_calibration_lines = np.array([
        #     [45.93774, 0., 59.06281, 0.],
        #     [45.93774, 68.00393, 59.06281, 68.00393],
        #     [0., 63.75368, 13.12507, 68.00393],
        #     [0., 12.75074, 4.11480, 13.88516]
        # ])

        for i in range(n_line_segments):
            line_seg = court_template[i]
            p1, p2 = line_seg[:2], line_seg[2:]

            q1 = Camera.project_point_on_frame(p1[0], p1[1], homography)
            q2 = Camera.project_point_on_frame(p2[0], p2[1], homography)

            if self._is_off_image_point(q1) and self._is_off_image_point(q2):
                continue

            cv2.line(edge_map, tuple(q1), tuple(q2), color=(255, 255, 255), thickness=2)
            cv2.circle(edge_map, tuple(q1), radius=1, color=(0, 0, 255), thickness=2)
            cv2.circle(edge_map, tuple(q2), radius=1, color=(0, 0, 255), thickness=2)

        # for line_seg in focal_calibration_lines:
        #     p1, p2 = line_seg[:2], line_seg[2:]
        #     q1 = Camera.project_point_on_frame(p1[0], p1[1], homography)
        #     q2 = Camera.project_point_on_frame(p2[0], p2[1], homography)
        #     cv2.line(edge_map, tuple(q1), tuple(q2), color=(0, 255, 0), thickness=2)
        #     dist = f'{np.linalg.norm(np.array(q1) - np.array(q2)):.2f}'
        #     cv2.putText(edge_map, dist, (q1[0], q2[1] - 15),
        #                 cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))

        # draw points
        points = [
            [self.court_mid_length_x, self.court_mid_width_y],  # court center
            [self.court_length_x, self.court_width_y],  # upper-right corner
            [self.court_length_x, 0.],  # lower-right corner
            [0., 0.],  # lower-left corner
            [0., self.court_width_y]  # upper-left corner
        ]

        for p in points:
            p = Camera.project_point_on_frame(p[0], p[1], homography)
            cv2.circle(edge_map, tuple(p), radius=3, color=(0, 0, 255), thickness=2)

        cv2.circle(edge_map, (int(self.image_center_x), int(self.image_center_y)),
                   radius=2, color=(0, 255, 0), thickness=3)

        # print human model
        human_feet, human_height = self.generate_human()
        human_feet = Camera.project_point_on_frame(human_feet[0], human_feet[1], homography)
        cv2.line(edge_map, tuple(human_feet), (human_feet[0], human_feet[1] - int(human_height)), color=(0, 150, 255),
                 thickness=2)
        cv2.circle(edge_map, (human_feet[0], human_feet[1] - int(human_height)),
                   radius=4, color=(0, 150, 255), thickness=3)
        cv2.putText(edge_map, f'{human_height:.2f}px',
                    (human_feet[0] + 5, human_feet[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 150, 255))

        # print distance from camera
        cv2.putText(edge_map, f'd:{self.distance_from_camera():.2f}m',
                    (int(self.image_center_x + 5), int(self.image_center_y) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0))

        return edge_map

    def orientation(self):
        homography = self.homography()
        upper_right_corner = \
            Camera.project_point_on_frame(self.court_length_x, self.court_width_y, homography)[0]
        upper_left_corner = Camera.project_point_on_frame(0., self.court_width_y, homography)[0]
        lower_right_corner = Camera.project_point_on_frame(self.court_length_x, 0., homography)[0]
        lower_left_corver = Camera.project_point_on_frame(0., 0., homography)[0]

        if self.image_center_x in range(lower_left_corver, upper_left_corner):
            return 1
        elif self.image_center_x in range(upper_left_corner, upper_right_corner):
            return 0
        elif self.image_center_x in range(upper_right_corner, lower_right_corner):
            return 2
        else:
            return -1


def _focalsfromhomography(h):
    h0 = h[0][0]
    h3 = h[1][0]
    return h0 + h3


if __name__ == '__main__':
    K = np.array([[4500., 0., 640.],
                  [0., 4500., 360.],
                  [0., 0., 1.]])

    H = np.array([[4507.262, 566.901, -170863.418],
                  [-8.7, -712.071, 69915.98],
                  [.012, .972, 97.534]])
    f = _focalsfromhomography(H)
    H = H.T
    h1 = H[0]
    h2 = H[1]
    h3 = H[2]
    K_inv = np.linalg.inv(K)
    L = 1 / np.linalg.norm(np.dot(K_inv, h1))
    r1 = L * np.dot(K_inv, h1)
    r2 = L * np.dot(K_inv, h2)
    r3 = np.cross(r1, r2)
    T = L * (K_inv @ h3.reshape(3, 1))
    R = np.array([[r1], [r2], [r3]])
    R = np.reshape(R, (3, 3))

    C = -R.dot(T).flatten()
    print('gr')

