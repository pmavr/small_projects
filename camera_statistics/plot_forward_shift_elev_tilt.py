
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from collections import defaultdict
import utils


# def uniform_samples_per_Y(data):
#     uniform_samples = []
#     for forward_shift, batch in data.groupby('y')['tilt', 'z']:
#         generated_samples = interpolate_tilt_ranges_for_given_Y(batch, forward_shift)
#         uniform_samples.append(generated_samples)
#     uniform_samples = np.concatenate(uniform_samples, axis=0)
#     uniform_samples = pd.DataFrame(uniform_samples, columns=['tilt', 'y', 'z'])
#     return uniform_samples

# def interpolate_tilt_ranges_for_given_Y(Y_data, Y, density=.2):
#     tilt_ranges = pd.concat([
#             Y_data.groupby('z')['tilt'].aggregate('max').rename('max'),
#             Y_data.groupby('z')['tilt'].aggregate('min').rename('min')],
#             axis=1)
#
#     generated_samples = []
#     for Z, tilt_maxmin_range in tilt_ranges.iterrows():
#
#
#         n_samples = int((tilt_maxmin_range['max'] - tilt_maxmin_range['min']) / density)
#         samples = np.zeros((n_samples, 3))
#         samples[:, 0] = np.random.uniform(
#             low=tilt_maxmin_range['min'].item(),
#             high=tilt_maxmin_range['max'].item(),
#             size=n_samples
#         )
#         samples[:, 1] = Y
#         samples[:, 2] = Z
#         generated_samples.append(samples)
#     return np.concatenate(generated_samples)


def interpolate_tilt_Y_Z_data(data):
    interpolated_data_across_y = interpolate_data_across_Y_axis(data, n_interpolated_points=20)
    interpolated_data_across_z = interpolate_data_across_Z_axis(interpolated_data_across_y, interpolating_intervals=.2)
    interpolated_z_tilt_planes = interpolate_data_across_tilt_axis(interpolated_data_across_z, interpolating_intervals=.2)

    return interpolated_z_tilt_planes


def interpolate_data_across_Y_axis(data, n_interpolated_points=10):
    # extrapolate line points per z level (6 levels total)
    max_z_levels = defaultdict(list)
    min_z_levels = defaultdict(list)
    for y, batch in data.groupby('y')['tilt', 'z']:
        for i, (z, batch) in enumerate(batch.groupby('z')['tilt']):
            max_point = np.array([y, z, batch.max()])
            min_point = np.array([y, z, batch.min()])
            max_z_levels[i].append(max_point)
            min_z_levels[i].append(min_point)

    max_lines_per_z_level = defaultdict(pd.DataFrame)
    min_lines_per_z_level = defaultdict(pd.DataFrame)
    for i in range(len(max_z_levels)):
        max_line = pd.DataFrame(np.stack(max_z_levels[i], axis=0), columns=['y', 'z', 'tilt'])
        min_line = pd.DataFrame(np.stack(min_z_levels[i], axis=0), columns=['y', 'z', 'tilt'])

        max_lines_per_z_level[i] = max_line
        min_lines_per_z_level[i] = min_line

    # interpolate points along those lines. All lines must have the same number of points
    for i in max_lines_per_z_level:
        max_line_pts = max_lines_per_z_level[i]
        min_line_pts = min_lines_per_z_level[i]

        max_lines_per_z_level[i] = interpolate_line_points_in_y_axis(max_line_pts, n_interpolated_points)
        min_lines_per_z_level[i] = interpolate_line_points_in_y_axis(min_line_pts, n_interpolated_points)

    return max_lines_per_z_level, min_lines_per_z_level


def interpolate_data_across_Z_axis(y_lines, interpolating_intervals=.1):
    max_y_lines, min_y_lines = y_lines
    max_y_lines = np.array([max_y_lines[i] for i in max_y_lines])
    min_y_lines = np.array([min_y_lines[i] for i in min_y_lines])
    z_lines = []
    for i in range(max_y_lines.shape[1]):
        max_tilts_line = pd.DataFrame(max_y_lines[:, i, :], columns=['y', 'z', 'tilt'])
        min_tilts_line = pd.DataFrame(min_y_lines[:, i, :], columns=['y', 'z', 'tilt'])

        z_lines.append([
            pd.DataFrame(interpolate_line_points_in_z_axis(max_tilts_line, interpolating_intervals), columns=['y', 'z', 'tilt']),
            pd.DataFrame(interpolate_line_points_in_z_axis(min_tilts_line, interpolating_intervals), columns=['y', 'z', 'tilt'])
        ])

    return z_lines


def interpolate_data_across_tilt_axis(data, interpolating_intervals=.1):
    interpolated_pts = []
    for line_1, line_2 in data:
        for (_, p1), (_, p2) in zip(line_1.iterrows(), line_2.iterrows()):
            pts = interpolate_between_2_points_in_tilt_axis(p1, p2, interpolating_intervals)
            interpolated_pts.append(pts)

    interpolated_pts = np.concatenate(interpolated_pts, axis=0)
    return pd.DataFrame(interpolated_pts, columns=['y', 'z', 'tilt'])


def interpolate_line_points_in_z_axis(pts, z_intervals):
    datapoints = []
    iterator = pts.iterrows()
    _, p = next(iterator)
    for _, q in iterator:
        interpolated_points = interpolate_between_2_points_in_z_axis(p, q, z_intervals)
        datapoints.append(interpolated_points)
        p = q
    datapoints = np.concatenate(datapoints, axis=0)
    datapoints = np.unique(datapoints, axis=0)
    return datapoints


def interpolate_between_2_points_in_z_axis(p1, p2, z_intervals):
    n_points = len(np.arange(p1['z'].item(), p2['z'].item(), z_intervals))
    y = np.linspace(p1['y'], p2['y'], n_points)
    tilt = np.linspace(p1['tilt'], p2['tilt'], n_points)
    z = np.linspace(p1['z'], p2['z'], n_points)
    interpolated_points = np.stack([y, z, tilt], axis=1)
    return interpolated_points


def interpolate_line_points_in_y_axis(pts, n_points):
    datapoints = []
    iterator = pts.iterrows()
    _, p = next(iterator)
    for _, q in iterator:
        interpolated_points = interpolate_between_2_points_in_y_axis(p, q, n_points)
        datapoints.append(interpolated_points)
        p = q
    datapoints = np.concatenate(datapoints, axis=0)
    datapoints = np.unique(datapoints, axis=0)
    return datapoints


def interpolate_between_2_points_in_y_axis(p1, p2, n_points):
    z = np.linspace(p1['z'], p2['z'], n_points)
    y = np.linspace(p1['y'], p2['y'], n_points)
    tilt = np.linspace(p1['tilt'], p2['tilt'], n_points)
    interpolated_points = np.stack([y, z, tilt], axis=1)
    return interpolated_points


def interpolate_between_2_points_in_tilt_axis(p1, p2, tilt_intervals):
    n_points = len(np.arange(p1['tilt'].item(), p2['tilt'].item(), -tilt_intervals))
    z = np.linspace(p1['z'], p2['z'], n_points)
    y = np.linspace(p1['y'], p2['y'], n_points)
    tilt = np.linspace(p1['tilt'], p2['tilt'], n_points)
    interpolated_points = np.stack([y, z, tilt], axis=1)
    return interpolated_points


if __name__ == '__main__':
    data = utils.read_data(['forward_shift_elevation_tilt_samples_v2'])
    unique_y = data['y'].unique()
    pts = interpolate_tilt_Y_Z_data(data)


    print('gr')

    tilt = data['tilt'].to_numpy()
    y = data['y'].to_numpy()
    z = data['z'].to_numpy()

    # ymax = y.max()
    # ymin = y.min()
    # data_at_maxY = data[data.y == ymax]
    # data_at_minY = data[data.y == ymin]
    #
    # zmin_at_ymin = data_at_minY['z'].min()
    # zmax_at_ymin = data_at_minY['z'].max()
    # data_at_minY_minZ = data_at_minY[data_at_minY.z == zmin_at_ymin]
    # data_at_minY_maxZ = data_at_minY[data_at_minY.z == zmax_at_ymin]
    #
    # zmin_at_ymax = data_at_maxY['z'].min()
    # zmax_at_ymax = data_at_maxY['z'].max()
    # data_at_maxY_minZ = data_at_maxY[data_at_maxY.z == zmin_at_ymax]
    # data_at_maxY_maxZ = data_at_maxY[data_at_maxY.z == zmax_at_ymax]
    #
    # max_tilt_at_maxY_maxZ = data_at_maxY_maxZ['tilt'].min()
    # max_tilt_at_maxY_minZ = data_at_maxY_minZ['tilt'].min()
    # max_tilt_at_minY_maxZ = data_at_minY_maxZ['tilt'].min()
    # max_tilt_at_minY_minZ = data_at_minY_minZ['tilt'].min()
    #
    # min_tilt_at_maxY_maxZ = data_at_maxY_maxZ['tilt'].max()
    # min_tilt_at_maxY_minZ = data_at_maxY_minZ['tilt'].max()
    # min_tilt_at_minY_maxZ = data_at_minY_maxZ['tilt'].max()
    # min_tilt_at_minY_minZ = data_at_minY_minZ['tilt'].max()
    #
    # vertices = np.array([
    #     [min_tilt_at_maxY_minZ, ymax, zmin_at_ymax],
    #     [min_tilt_at_maxY_maxZ, ymax, zmax_at_ymax],
    #     [max_tilt_at_maxY_minZ, ymax, zmin_at_ymax],
    #     [max_tilt_at_maxY_maxZ, ymax, zmax_at_ymax],
    #
    #     [min_tilt_at_minY_minZ, ymin, zmin_at_ymin],
    #     [min_tilt_at_minY_maxZ, ymin, zmax_at_ymin],
    #     [max_tilt_at_minY_minZ, ymin, zmin_at_ymin],
    #     [max_tilt_at_minY_maxZ, ymin, zmax_at_ymin]
    # ])


    plots = [
        # go.Scatter3d(
        #     x=vertices[:,0], y=vertices[:,1], z=vertices[:,2], name='vertices',
        #     mode='markers',
        #     marker=dict(size=5, color='blue', opacity=1)
        # ),
        go.Scatter3d(
            x=tilt, y=y, z=z, name='camera space',
            mode='markers',
            marker=dict(size=3, color='red', opacity=1)
        ),
        go.Scatter3d(
            x=pts['tilt'], y=pts['y'], z=pts['z'], name='interpolated samples',
            mode='markers',
            marker=dict(size=2, color='green', opacity=1)
        ),
    ]


    fig = go.Figure(data=plots)
    fig.update_layout(
        title=f'Tilt - Forw.Shift(Y) - Elev(Z) ',
        scene=dict(
            xaxis=dict(title='Tilt (degrees)'),
            yaxis=dict(title='Y (clicks)'),
            zaxis=dict(title='Z (clicks)')
        ),
    )
    fig.show()