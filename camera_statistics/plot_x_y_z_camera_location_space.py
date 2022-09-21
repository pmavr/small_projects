import numpy as np
import pandas as pd
import plotly.graph_objs as go
from torch.utils.data import Dataset
import random

from camera_parameters.original_synthetic_util import ut_generate_ptz_cameras
import utils


def normal_camera_locations():
    cc_mean = np.array([[52.36618474, -45.15650112, 16.82156705]])
    cc_std = np.array([[2.23192608, 9.3825635, 2.94875254]])
    cc_min = np.array([[45.05679141, -66.0702037, 10.13871263]])
    cc_max = np.array([[60.84563315, -16.74178234, 23.01126126]])
    cc_statistics = [cc_mean, cc_std, cc_min, cc_max]
    camera_centers = np.random.normal(cc_mean, cc_std, (camera_num, 3))
    return camera_centers


class SyntheticCameraDataset(Dataset):

    def __init__(self, params, num_of_cameras):
        self.binary_court = utils.binary_court()
        self.court_mid_distance_x = 52.500276
        self.court_mid_distance_y = 34.001964
        self.image_w = 1280
        self.image_h = 720
        self.image_output_dim = params['image_output_dimensions']
        self.params = params

        self.camera_poses = self.generate_ptz_cameras(num_of_cameras)
        self.num_of_cameras = self.camera_poses.shape[0]

    def generate_ptz_cameras(self, num_of_cameras):
        u, v = self.image_w / 2.0, self.image_h / 2.0,
        fl_and_camera_centers = self.generate_camera_centers(num_of_cameras)

        pan_angles = self.extrapolate_pan_angle(fl_and_camera_centers[:, 1], fl_and_camera_centers[:, 2]) * (-1)
        tilt_angles = self.extrapolate_tilt_angle(fl_and_camera_centers[:, 3], fl_and_camera_centers[:, 2]) * (-1)
        roll_angles = np.zeros((num_of_cameras, 1))

        pan_angles = self.get_nearby_data(pan_angles, std=self.params['pan_std'])
        tilt_angles = self.get_nearby_data(tilt_angles, std=self.params['tilt_std'])

        cameras = np.concatenate([
            np.ones((num_of_cameras, 1)) * u,
            np.ones((num_of_cameras, 1)) * v,
            fl_and_camera_centers[:, 0].reshape(-1, 1),
            tilt_angles, pan_angles, roll_angles, fl_and_camera_centers[:, 1:]],
            axis=1)

        return cameras

    def generate_camera_centers(self, num_of_cameras):
        cc_statistics = self.params['camera_param_distributions']['camera_center']
        fl_statistics = self.params['camera_param_distributions']['focal_length']

        fl_min, fl_max = fl_statistics['min'], fl_statistics['max']
        cc_min, cc_max = cc_statistics['min'], cc_statistics['max']
        x_min, x_max = cc_min[0], cc_max[0]
        y_min, y_max = cc_min[1], cc_max[1]

        fl, x, y, slope = np.mgrid[
                          fl_min:fl_max:self.params['fl_density'],
                          x_min:x_max:self.params['xloc_density'],
                          y_min:y_max:self.params['yloc_density'],
                          .3:.45:self.params['slope_density']]

        fl_xy_slope = np.stack([fl.flatten(), x.flatten(), y.flatten(), slope.flatten()], axis=1)
        fl_xy_slope = fl_xy_slope[random.sample(range(len(fl_xy_slope)), num_of_cameras), :]
        z = self.bleachers_line(y=fl_xy_slope[:, 2], slope=fl_xy_slope[:, 3])
        fl_xyz = np.concatenate([fl_xy_slope[:, :3], z.reshape(-1,1)], axis=1)
        return fl_xyz

    @staticmethod
    def bleachers_line(y, slope):
        z = y * (-slope)
        return z

    def extrapolate_tilt_angle(self, z, y):
        a = np.abs(z)
        b = np.abs(y) + self.court_mid_distance_y
        angle = np.arctan(a / b)
        return np.degrees(angle).reshape(-1, 1)

    def extrapolate_pan_angle(self, x, y):
        a = x - self.court_mid_distance_x
        b = np.abs(y) + self.court_mid_distance_y
        angle = np.arctan(a / b)
        return np.degrees(angle).reshape(-1, 1)

    @staticmethod
    def get_nearby_data(d, std):
        sign = np.sign(np.random.uniform(-1, 1, (d.size, 1)))
        std = np.random.uniform(0., std, (d.size, 1))
        std *= sign
        return d + std


if __name__ == '__main__':
    # data = utils.read_data(['forward_shift_elevation_tilt_samples_v2'])

    data_params = {
        'dataset_size': 1000,
        'val_percent_size': .1,
        'test_percent_size': .1,
        'image_w': 1280,
        'image_h': 720,
        'image_output_dimensions': (320, 180),
        'camera_param_distributions': {
            'camera_center': {
                'mean': [52.36618474, -45.15650112, 16.82156705],
                'std': [1.23192608, 9.3825635, 2.94875254],
                'min': [50.05679141, -66., 10.13871263],
                'max': [54.84563315, -25., 23.01126126],
            },
            'focal_length': {
                'mean': 2500.5139785,
                'std': 716.06817106,
                'min': 1463.16468414,
                'max': 3580.
            }
        },
        'fl_density': 150,
        'xloc_density': .1,
        'yloc_density': .1,
        'slope_density': .01,
        'pan_std': .5,
        'tilt_std': .5
    }

    camera_num = 10000
    dataset = SyntheticCameraDataset(data_params, num_of_cameras=camera_num)
    camera_poses = dataset.camera_poses
    camera_poses = pd.DataFrame(camera_poses, columns=['u', 'v', 'fl', 'tilt', 'pan', 'roll', 'x', 'y', 'z'])
    binary_court = utils.binary_court()

    original_poses = normal_camera_locations()

    plots = [
        go.Scatter3d(
            x=camera_poses['x'], y=camera_poses['y'], z=camera_poses['z'],
            name='our synthetic cameras',
            mode='markers',
            marker=dict(size=2, color='red', opacity=1)
        ),
        go.Scatter3d(
            x=original_poses[:, 0], y=original_poses[:, 1], z=original_poses[:, 2],
            name='chen synthetic cameras',
            mode='markers',
            marker=dict(size=2, color='green', opacity=1)
        )
    ]

    for row in binary_court:
        points_in_2d = row.reshape(-1, 2)
        points_in_3d = np.concatenate([points_in_2d, np.zeros((2, 1))], axis=1)
        line = go.Scatter3d(
            x=points_in_3d[:, 0], y=points_in_3d[:, 1], z=points_in_3d[:, 2],
            name='court lines',
            line=dict(color='darkblue', width=3),
            marker=dict(size=1, color='darkblue', opacity=1)
        )
        plots.append(line)

    fig = go.Figure(data=plots)
    fig.update_layout(
        title=f'Synthetic cameras locations',
        scene=dict(
            xaxis=dict(title='X (meters)'),
            yaxis=dict(title='Y (meters)'),
            zaxis=dict(title='Z (meters)'),
            aspectmode='data'
        ),
    )
    fig.show()

    # print(f'Total Loss:{np.sum(losses)}')
    print('gr')
