import numpy as np
import pandas as pd
import plotly.graph_objs as go
from torch.utils.data import Dataset
import random

import utils


def normal_camera_locations():
    cc_mean = np.array([[52.36618474, -45.15650112, 16.82156705]])
    cc_std = np.array([[2.23192608, 9.3825635, 2.94875254]])
    cc_min = np.array([[45.05679141, -66.0702037, 10.13871263]])
    cc_max = np.array([[60.84563315, -16.74178234, 23.01126126]])
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
        fl_and_camera_centers = self.generate_camera_centers_v3(num_of_cameras)

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
        z_min, z_max = cc_min[2], cc_max[2]

        fl, x, origin_y, origin_z = np.mgrid[
                          fl_min:fl_max:self.params['fl_density'],
                          x_min:x_max:self.params['xloc_density'],
                          y_min:y_max:self.params['yloc_density'],
                          z_min:z_max:self.params['zloc_density']]

        fl_xy = np.stack([fl.flatten(), x.flatten(), origin_y.flatten()], axis=1)
        fl_xy = fl_xy[random.sample(range(len(fl_xy)), int(num_of_cameras/2)), :]
        z = self.get_z_from_y(y=fl_xy[:, 2])
        fl_xyz1 = np.concatenate([fl_xy, z.reshape(-1, 1)], axis=1)

        fl_xz = np.stack([fl.flatten(), x.flatten(), origin_z.flatten()], axis=1)
        fl_xz = fl_xz[random.sample(range(len(fl_xz)), int(num_of_cameras/2)), :]
        y = self.get_y_from_z(z=fl_xz[:, 2])
        fl_xyz2 = np.concatenate([fl_xz[:, :2], y.reshape(-1, 1), fl_xz[:, 2].reshape(-1, 1)], axis=1)

        fl_xyz = np.concatenate([fl_xyz1, fl_xyz2], axis=0)
        return fl_xyz

    def generate_camera_centers_v2(self, num_of_cameras):
        cc_statistics = self.params['camera_param_distributions']['camera_center']
        fl_statistics = self.params['camera_param_distributions']['focal_length']

        fl_min, fl_max = fl_statistics['min'], fl_statistics['max']
        cc_min, cc_max = cc_statistics['min'], cc_statistics['max']
        x_min, x_max = cc_min[0], cc_max[0]
        y_min, y_max = cc_min[1], cc_max[1]
        z_min, z_max = cc_min[2], cc_max[2]

        fl, x, y, z = np.mgrid[
                          0:1:1,
                          x_min:x_max:self.params['xloc_density'],
                          y_min:y_max:self.params['yloc_density'],
                          z_min:z_max:self.params['zloc_density']]

        fl_xyz = np.stack([fl.flatten(), x.flatten(), y.flatten(), z.flatten()], axis=1)


        filter = (- fl_xyz[:,3] / fl_xyz[:,2] > .3) & (- fl_xyz[:,3] / fl_xyz[:,2] < .45)
        fl_xyz = fl_xyz[filter]
        fl_xyz = fl_xyz[random.sample(range(len(fl_xyz)), num_of_cameras), :]

        return fl_xyz

    def generate_camera_centers_v3(self, num_of_cameras):
        cc_statistics = self.params['camera_param_distributions']['camera_center']
        fl_statistics = self.params['camera_param_distributions']['focal_length']

        slope_max = self.params['camera_param_distributions']['bleachers_slope']['max']
        slope_min = self.params['camera_param_distributions']['bleachers_slope']['min']
        fl_min, fl_max = fl_statistics['min'], fl_statistics['max']
        cc_min, cc_max = cc_statistics['min'], cc_statistics['max']
        x_min, x_max = cc_min[0], cc_max[0]
        y_min, y_max = cc_min[1], cc_max[1]
        z_min, z_max = cc_min[2], cc_max[2]

        fl, x, y, z = np.mgrid[
                      0:1:1,
                      x_min:x_max:self.params['xloc_density'],
                      y_min:y_max:self.params['yloc_density'],
                      z_min:z_max:self.params['zloc_density']]

        fl_xyz = np.stack([fl.flatten(), x.flatten(), y.flatten(), z.flatten()], axis=1)
        filter = (- fl_xyz[:, 3] / fl_xyz[:, 2] > slope_min) & (- fl_xyz[:, 3] / fl_xyz[:, 2] < slope_max)
        fl_xyz = fl_xyz[filter]
        fl_xyz = fl_xyz[random.sample(range(len(fl_xyz)), num_of_cameras), :]
        return fl_xyz

    def get_z_from_y(self, y):
        slope_range = self.params['camera_param_distributions']['bleachers_slope']
        slope = np.random.uniform(slope_range['min'], slope_range['max'], y.size)
        z = y * (-slope)
        return z

    def get_y_from_z(self, z):
        slope_range = self.params['camera_param_distributions']['bleachers_slope']
        slope = np.random.uniform(slope_range['min'], slope_range['max'], z.size)
        y = z/(-slope)
        return y

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


def view_endpoints(cameras):
    endpoints = []
    for _, c in cameras.iterrows():
        x, y, z = c['x'], c['y'], c['z']
        tilt, pan = c['tilt'], c['pan']
        theta = np.radians(90 - abs(tilt))
        dist_from_camera = abs(z / np.cos(theta))
        horizontal_dist_from_camera = dist_from_camera * np.sin(theta)

        phi = np.radians(pan)
        x_endpoint = horizontal_dist_from_camera * np.sin(phi) + x
        y_endpoint = horizontal_dist_from_camera * np.cos(phi) + y

        endpoints.append([x_endpoint, y_endpoint, 0.])
    return np.array(endpoints)


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
                'min': [45.05679141, -66.0702037, 10.13871263],
                'max': [60.84563315, -16.74178234, 23.01126126],
            },
            'focal_length': {
                'mean': 2500.5139785,
                'std': 716.06817106,
                'min': 1463.16468414,
                'max': 3580.
            },
            'bleachers_slope': {
                'min': .3,
                'max': .45
            }
        },
        'fl_density': 150,
        'xloc_density': .1,
        'yloc_density': .1,
        'zloc_density': .1,
        'slope_density': .01,
        'pan_std': .5,
        'tilt_std': .5
    }

    camera_num = 10000
    dataset = SyntheticCameraDataset(data_params, num_of_cameras=camera_num)
    camera_poses = dataset.camera_poses
    camera_poses = pd.DataFrame(camera_poses, columns=['u', 'v', 'fl', 'tilt', 'pan', 'roll', 'x', 'y', 'z'])
    endpoints = view_endpoints(camera_poses)
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

    # for (_, c), e in zip(camera_poses.iterrows(), endpoints):
    #     line = np.stack([np.array([c['x'], c['y'], c['z']]), e], axis=0)
    #     plots.append(go.Scatter3d(
    #         x=line[:, 0], y=line[:, 1], z=line[:, 2],
    #         line=dict(color='green', width=1),
    #         marker=dict(size=1, color='green', opacity=1)
    #     ))

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
