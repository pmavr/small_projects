import numpy as np
import cv2


def edge_map_from_homography(homography, binary_court, edge_map_resolution):
    points = binary_court['points']
    line_segment_indexes = binary_court['line_segment_index']
    edge_map = np.zeros((edge_map_resolution[1], edge_map_resolution[0], 3), dtype=np.uint8)

    n = line_segment_indexes.shape[0]
    for i in range(n):
        idx1, idx2 = line_segment_indexes[i][0], line_segment_indexes[i][1]
        p1, p2 = points[idx1], points[idx2]
        q1 = Camera.project_point_on_frame(p1[0], p1[1], homography)
        q2 = Camera.project_point_on_frame(p2[0], p2[1], homography)
        cv2.line(edge_map, tuple(q1), tuple(q2), color=(255, 255, 255), thickness=4)

    return edge_map


class Camera:
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

    def calibration_matrix(self):
        return np.array([[self.focal_length, 0, self.image_center_x],
                           [0, self.focal_length, self.image_center_y],
                           [0, 0, 1]])

    def camera_center(self):
        camera_center = np.zeros(3)
        camera_center[0] = self.camera_center_x
        camera_center[1] = self.camera_center_y
        camera_center[2] = self.camera_center_z
        return camera_center

    def rotation_matrix(self):
        rotation = np.zeros(3)
        rotation[0] = self.tilt_angle
        rotation[1] = self.pan_angle
        rotation[2] = self.roll_angle
        rotation, _ = cv2.Rodrigues(rotation)
        return rotation

    def homography(self):
        P = self.projection_matrix()
        h = P[:, [0, 1, 3]]
        return h

    def projection_matrix(self):
        P = np.eye(3, 4)
        P[:, 3] = -1*self.camera_center()
        K = self.calibration_matrix()
        R = self.rotation_matrix()
        return K @ R @ P

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
        return projected_x, projected_y

    @staticmethod
    def project_point_on_frame(x, y, h):
        w = 1.0
        p = np.zeros(3)
        p[0], p[1], p[2] = x, y, w
        q = h @ p

        assert q[2] != 0.0
        projected_x = np.rint(q[0] / q[2]).astype(np.int)
        projected_y = np.rint(q[1] / q[2]).astype(np.int)
        return projected_x, projected_y
