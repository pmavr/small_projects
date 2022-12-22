import numpy as np
import pandas as pd
import plotly.graph_objs as go
import utils
from torch.utils.data import Dataset
from camera_statistics.Camera import Camera


class SyntheticCameraAngleDataset(Dataset):

    def __init__(self, x, y, z, params, num_of_cameras):
        self.binary_court = utils.binary_court()
        self.court_mid_distance_x = 52.500276
        self.court_mid_distance_y = 34.001964
        self.image_w = 1280
        self.image_h = 720
        self.image_output_dim = params['image_output_dimensions']
        self.x = x
        self.y = y
        self.z = z
        self.pan_density = params['pan_density']
        self.tilt_density = params['tilt_density']
        fl_statistics = params['camera_param_distributions']['focal_length']
        self.fl_min, self.fl_max = fl_statistics['min'], fl_statistics['max']
        self.extra_tilt_threshold = 2

        self.camera_poses = self.generate_ptz_cameras()
        self.num_of_cameras = self.camera_poses.shape[0]

    def generate_ptz_cameras(self):
        u, v = self.image_w / 2.0, self.image_h / 2.0,
        extremes = self.angle_extremes_for_camera_location()
        tilt_pan_angles = self.generate_data_within_extremes(extremes)

        num_of_cameras = tilt_pan_angles.shape[0]
        focal_lengths = np.random.uniform(self.fl_min, self.fl_max, (num_of_cameras, 1))
        roll_angles = np.random.normal(0., 0., (num_of_cameras, 1))

        cameras = np.concatenate([
            np.ones((num_of_cameras, 1)) * u,
            np.ones((num_of_cameras, 1)) * v,
            focal_lengths, tilt_pan_angles, roll_angles,
            np.ones((num_of_cameras, 1)) * self.x,
            np.ones((num_of_cameras, 1)) * self.y,
            np.ones((num_of_cameras, 1)) * self.z],
            axis=1)
        return cameras

    @staticmethod
    def tilt_angle(z, y):
        angle = np.arctan(np.abs(z) / y)
        return np.degrees(angle)

    @staticmethod
    def b_max_for_mid(cmax, theta):
        return abs(cmax / np.sin(np.radians(theta)))

    @staticmethod
    def b_min_for_mid(cmin, theta):
        return abs(cmin / np.sin(np.radians(theta)))

    @staticmethod
    def b_max_for_sides(amax, phi):
        return abs(amax / np.sin(np.radians(phi)))

    @staticmethod
    def b_min_for_sides(cmin, phi):
        return abs(cmin / np.cos(np.radians(phi)))

    def angle_extremes_for_camera_location(self):
        pan_range = np.arange(-65., 65, self.pan_density)
        tilt_range = np.zeros((len(pan_range), 2))
        for i, pan_angle in enumerate(pan_range):
            camera_params = np.array([640, 360, 3500.0, -12.5, pan_angle, 0., self.x, self.y, self.z])
            camera = Camera(camera_params)

            orientation = camera.orientation()

            if orientation == -1:  # extremes case
                continue

            elif orientation == 0:  # mid case
                theta = abs(90 - pan_angle)
                cmax = abs(self.y) + camera.court_width_y
                cmin = abs(self.y)
                bmax = self.b_max_for_mid(cmax, theta)
                bmin = self.b_min_for_mid(cmin, theta)

                max_tilt = self.tilt_angle(self.z, bmax) * (-1)
                min_tilt = self.tilt_angle(self.z, bmin) * (-1)

            elif orientation > 0:  # side case
                x_camera_shift = self.x - camera.court_mid_length_x
                amax = camera.court_mid_length_x + x_camera_shift * np.sign(pan_angle) * (-1)
                cmin = abs(self.y)
                bmax = self.b_max_for_sides(amax, pan_angle)
                bmin = self.b_min_for_sides(cmin, pan_angle)

                max_tilt = self.tilt_angle(self.z, bmax) * (-1) + self.extra_tilt_threshold
                min_tilt = self.tilt_angle(self.z, bmin) * (-1)

            tilt_range[i, 0] = min_tilt
            tilt_range[i, 1] = max_tilt

        data = np.concatenate([
            pan_range.reshape(-1, 1),
            tilt_range], axis=1)
        data = pd.DataFrame(data, columns=['pan', 'max_tilt', 'min_tilt'])
        data = data[(data['max_tilt'] < 0.) & (data['min_tilt'] < 0.)]
        return data

    def generate_data_within_extremes(self, extremes):
        data = []
        for _, row in extremes.iterrows():
            interpolated_tilt_angles = np.arange(row['max_tilt'], row['min_tilt'], self.tilt_density)
            pan_angle = np.ones_like(interpolated_tilt_angles) * row['pan']
            batch = np.stack([
                interpolated_tilt_angles,
                pan_angle],
                axis=1)
            data.append(batch)
        data = np.concatenate(data, axis=0)
        return data

    # @staticmethod
    # def get_nearby_data(d, std):
    #     sign = np.sign(np.random.uniform(-1, 1, (d.size, 1)))
    #     std = np.random.uniform(0., std, (d.size, 1))
    #     std *= sign
    #     return d + std
    #
    # def _prepare_item(self, item):
    #     item = Camera(item).to_edge_map(self.binary_court)
    #     item = cv2.resize(item, self.image_output_dim)
    #     item = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)
    #     item = self.data_transform(item)
    #     return item

    # def __getitem__(self, ndx):
    #     camera = self.camera_poses[ndx]
    #     edge_map = self._prepare_item(camera)
    #
    #     y_true = np.array([camera[2], camera[6], camera[7], camera[8]])
    #     y_true = torch.tensor(y_true, dtype=torch.float)
    #     return edge_map, y_true

    def __len__(self):
        return self.num_of_cameras


def view_endpoints(x, y, z, cameras):
    endpoints = []
    for c in cameras:
        tilt, pan = c[3], c[4]
        theta = np.radians(90 - abs(tilt))
        dist_from_camera = abs(z / np.cos(theta))
        horizontal_dist_from_camera = dist_from_camera * np.sin(theta)

        phi = np.radians(pan)
        x_endpoint = horizontal_dist_from_camera * np.sin(phi) + x
        y_endpoint = horizontal_dist_from_camera * np.cos(phi) + y

        endpoints.append([x_endpoint, y_endpoint, 0.])
    return np.array(endpoints)


if __name__ == '__main__':
    binary_court = utils.binary_court()
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
                'max': 5937.98454881
            }
        },
        'pan_density': 5,
        'tilt_density': .3,

    }
    image_w, image_h = 1280, 720
    x = 50.776
    y = -36.664
    z = 11.912

    dataset = SyntheticCameraAngleDataset(x, y, z, data_params, 10000)
    camera_poses = dataset.camera_poses
    endpoints = view_endpoints(x, y, z, camera_poses)

    plots = []
    plots.append(go.Scatter3d(
        x=[x], y=[y], z=[z],
        name='Camera Location',
        mode='markers',
        marker=dict(size=5, color='red', opacity=1)
    ))

    for e in endpoints:
        line = np.stack([np.array([x, y, z]), e], axis=0)
        plots.append(go.Scatter3d(
            x=line[:, 0], y=line[:, 1], z=line[:, 2],
            line=dict(color='green', width=1),
            marker=dict(size=1, color='green', opacity=1)
        ))

    for row in binary_court:
        points_in_2d = row.reshape(-1, 2)
        points_in_3d = np.concatenate([points_in_2d, np.zeros((2, 1))], axis=1)
        line = go.Scatter3d(
            x=points_in_3d[:, 0], y=points_in_3d[:, 1], z=points_in_3d[:, 2],
            line=dict(color='darkblue', width=3),
            marker=dict(size=1, color='darkblue', opacity=1)
        )
        plots.append(line)

    fig = go.Figure(data=plots)
    fig.update_layout(
        title=f"Camera view directions - Angle step (Pan:{data_params['pan_density']}°, Tilt:{data_params['tilt_density']}°)",
        scene=dict(
            xaxis=dict(title='X (meters)'),
            yaxis=dict(title='Y (meters)'),
            zaxis=dict(title='Z (meters)'),
            aspectmode='data'
        )
    )
    fig.show()
    print('gr')
