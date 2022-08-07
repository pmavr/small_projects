import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from collections import defaultdict


def read_data(files_list):
    dataset = [np.load(f'{f}.npy') for f in files_list]
    dataset = np.concatenate(dataset, axis=0)
    data = pd.DataFrame(
        dataset, columns=['u', 'v', 'fl', 'tilt', 'pan', 'roll', 'x', 'y', 'z'])
    return data


def uniform_samples_for_given_elevation(elevation_data, elevation):
    tilt_ranges = pd.concat([
        elevation_data.groupby('pan')['tilt'].aggregate('max').rename('max'),
        elevation_data.groupby('pan')['tilt'].aggregate('min').rename('min')],
        axis=1)

    generated_samples = []
    for pan, tilt_maxmin_range in tilt_ranges.iterrows():
        n_samples = int((tilt_maxmin_range['max'] - tilt_maxmin_range['min']) / 2)
        samples = np.zeros((n_samples, 3))
        samples[:, 0] = pan
        samples[:, 1] = np.random.uniform(
            low=tilt_maxmin_range['min'].item(),
            high=tilt_maxmin_range['max'].item(),
            size=n_samples
        )
        samples[:, 2] = elevation
        generated_samples.append(samples)
    return np.concatenate(generated_samples)


def uniform_samples_per_elevation(curves):
    uniform_samples = []
    for elevation in list(curves.keys()):
        data = curves[elevation]
        generated_samples = uniform_samples_for_given_elevation(data, elevation)
        uniform_samples.append(generated_samples)
    uniform_samples = np.concatenate(uniform_samples, axis=0)
    uniform_samples = pd.DataFrame(uniform_samples, columns=['pan', 'tilt', 'z'])
    return uniform_samples


def get_tilt_angles_at_max_pan_angle(data):
    unique_elevations = data['z'].unique()
    groups = [data[data['z'] == unique_elevations[i]] for i in range(len(unique_elevations))]
    tilt_angles_at_max_pan_angle = []
    for i in range(len(unique_elevations)):
        batch = groups[i]
        max_pan_angle_id = batch['pan'].idxmax()
        tilt_angles_at_max_pan_angle.append(batch[['pan','tilt']].loc[max_pan_angle_id].to_numpy())
    return pd.DataFrame(np.array(tilt_angles_at_max_pan_angle), columns=['pan','tilt'], index=unique_elevations)


def split_data_by_tilt_at_max_pan(data):
    unique_elevations = data['z'].unique()
    tilt_at_max_pan = get_tilt_angles_at_max_pan_angle(data)
    lower_data = []
    upper_data = []
    for elevation in unique_elevations:
        tilt = tilt_at_max_pan.loc[elevation]['tilt']
        tmp = data[data['z'] == elevation]
        lower_data.append(tmp[tmp['tilt'] <= tilt][['pan', 'tilt', 'z']])
        upper_data.append(tmp[tmp['tilt'] > tilt][['pan', 'tilt', 'z']])
    lower_data = pd.concat(lower_data)
    upper_data = pd.concat(upper_data)
    return upper_data, lower_data


def tilt_pan_objective(x, p9, p8, p7, p6, p5, p4, p3, p2, p1, p0):
    return p9 * x ** 18 + p8 * x ** 16 + p7 * x ** 14 + p6 * x ** 12 + p5 * x ** 10 + \
           p4 * x ** 8 + p3 * x ** 6 + p2 * x ** 4 + p1 * x ** 2 + p0


def predict_pan_tilt_curves(data):
    upper_data, lower_data = split_data_by_tilt_at_max_pan(data)
    upper_curves = _predict_curves(upper_data)
    lower_curves = _predict_curves(lower_data)

    upper_interpolated_data = interpolate_curved_lines(upper_curves)
    lower_interpolated_data = interpolate_curved_lines(lower_curves)

    upper_interpolated_curves = _predict_curves(upper_interpolated_data)
    lower_interpolated_curves = _predict_curves(lower_interpolated_data)

    u_curves = [upper_interpolated_curves[e] for e in list(upper_interpolated_curves.keys())]
    l_curves = [lower_interpolated_curves[e] for e in list(lower_interpolated_curves.keys())]

    u_curves = pd.DataFrame(np.concatenate(u_curves, axis=0), columns=['pan', 'tilt', 'z'])
    l_curves = pd.DataFrame(np.concatenate(l_curves, axis=0), columns=['pan', 'tilt', 'z'])
    curves = pd.concat([u_curves, l_curves], axis=0)

    pan_tilt_curves = defaultdict(pd.DataFrame)
    unique_elevations = curves['z'].unique()
    for e in unique_elevations:
        pan_tilt_curves[e] = curves[curves['z'] == e]

    return pan_tilt_curves


def _predict_curves(data, pan_intervals=.05):
    unique_elevations = data['z'].unique()
    predicted_curves = defaultdict(pd.DataFrame)
    for elevation in unique_elevations:
        current_elevation_data = data[data['z'] == elevation][['pan', 'tilt', 'z']]
        popt, _ = curve_fit(tilt_pan_objective, current_elevation_data['pan'], current_elevation_data['tilt'])
        a, b, c, d, e, f, g, h, k, l = popt
        pan_angles = np.arange(0, current_elevation_data['pan'].max(), pan_intervals).reshape(-1, 1)
        lower_pred_y = tilt_pan_objective(pan_angles, a, b, c, d, e, f, g, h, k, l)
        predicted_curves[elevation] = pd.concat([
            pd.Series(pan_angles.squeeze(), name='pan'),
            pd.Series(lower_pred_y.squeeze(), name='tilt'),
            pd.Series(np.ones(len(pan_angles))*elevation, name='z'),],
            axis=1
        )
    return predicted_curves


def interpolate_between_2_points(p1, p2, z_intervals):
    n_points = len(np.arange(p1['z'].item(), p2['z'].item(), -z_intervals))
    pan = np.linspace(p1['pan'], p2['pan'], n_points)
    tilt = np.linspace(p1['tilt'], p2['tilt'], n_points)
    z = np.linspace(p1['z'], p2['z'], n_points)
    interpolated_points = np.stack([pan, tilt, z], axis=1)
    return interpolated_points


def interpolate_line_points(pts, z_intervals):
    datapoints = []
    iterator = pts.iterrows()
    _, p = next(iterator)
    for _, q in iterator:
        interpolated_points = interpolate_between_2_points(p, q, z_intervals)
        datapoints.append(interpolated_points)
        p = q
    datapoints = np.concatenate(datapoints, axis=0)
    datapoints = np.unique(datapoints, axis=0)
    return datapoints


def interpolate_curved_lines(curves, z_intervals=.1):
    curved_plane = []
    elevations = list(curves.keys())
    for e in list(curves.keys())[:-1]:
        num_of_points = len(curves[e])
        for _ in range(num_of_points):
            pts = []
            for elev in elevations:
                point = curves[elev].iloc[-1]
                point_idx = point.name
                curves[elev] = curves[elev].drop([point_idx], axis=0)
                pts.append(point)

            pts = pd.concat(pts, axis=1).transpose()
            interpolated_datapoints = interpolate_line_points(pts, z_intervals)
            curved_plane.append(
                pd.DataFrame(interpolated_datapoints, columns=['pan', 'tilt', 'z']))
        del elevations[0]
    curved_plane = np.concatenate(curved_plane, axis=0)
    return pd.DataFrame(curved_plane, columns=['pan', 'tilt', 'z'])
