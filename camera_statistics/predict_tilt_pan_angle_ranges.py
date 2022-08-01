from scipy.stats import multivariate_normal
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from scipy.optimize import curve_fit

import utils


def linear_objective(x, p1, p0):
    return p1 * x + p0


def square_objective(x, p2, p1, p0):
    return p2*x**2 + p1 * x + p0


def calculate_loss(pred, gt):
    return np.sum(np.abs(pred - gt))




if __name__ == '__main__':

    data = utils.read_data()
    unique_elevations = data['z'].unique()

    tilt_at_max_pan = utils.get_tilt_angles_at_max_pan_angle(data, unique_elevations)

    # pan_angle_limits = np.array([round(groups[i]['pan'].to_numpy().max(), 3) for i in range(len(unique_elevations))])
    # max_tilt_angles = np.array([round(groups[i]['tilt'].to_numpy().max(), 3) for i in range(len(unique_elevations))])
    elevations = np.arange(9, 28, .01)

    # fit tilt angles at max pan angles
    params_for_tilt_angles_at_max_pan_angle, _ = curve_fit(square_objective, unique_elevations, tilt_at_max_pan['tilt'])
    pred_tilt_at_max_pan = square_objective(elevations, *params_for_tilt_angles_at_max_pan_angle)

    # fit pan angle limits
    # params_for_pan_angle_limits, _ = curve_fit(linear_objective, unique_elevations, pan_angle_limits)
    # pred_pan_angle_limits = linear_objective(elevations, *params_for_pan_angle_limits)
    # [-1.46784066 72.91603874]

    # fit max tilt angles
    # params_for_max_tilt_angles, _ = curve_fit(linear_objective, unique_elevations, max_tilt_angles)
    # pred_max_tilt_angles = linear_objective(elevations, *params_for_max_tilt_angles)
    # [-0.29414728 -3.19746658]

    fig = go.Figure(data=[
        go.Scatter(x=unique_elevations, y=tilt_at_max_pan['tilt'], name='tilt_angles_at_max_pan_angle'),
        go.Scatter(x=elevations, y=pred_tilt_at_max_pan, name='pred_tilt_angles_at_max_pan_angle'),
        # go.Scatter(x=unique_elevations, y=pan_angle_limits, name='pan_angles'),
        # go.Scatter(x=elevations, y=pred_pan_angle_limits, name='pred_pan_angles'),
        # go.Scatter(x=unique_elevations, y=max_tilt_angles, name='max_tilt'),
        # go.Scatter(x=elevations, y=pred_max_tilt_angles, name='pred_max_tilt'),
    ])
    fig.update_layout(
        title=f'Pan - Tilt - Elev',
        scene=dict(
            xaxis=dict(title='Pan (degrees)'),
            yaxis=dict(title='Tilt (degrees)'),
            zaxis=dict(title='Elevation (clicks)')
        ),
        xaxis_title='Elevation',
        yaxis_title='Tilt at Max Pan',
    )
    fig.show()
    print('gr')