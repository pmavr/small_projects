from scipy.stats import multivariate_normal
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from scipy.optimize import curve_fit

import utils

def tilt_pan_objective(x, p9, p8, p7, p6, p5, p4, p3, p2, p1, p0):
    return p9 * x ** 18 + p8 * x ** 16 + p7 * x ** 14 + p6 * x ** 12 + p5 * x ** 10 + \
           p4 * x ** 8 + p3 * x ** 6 + p2 * x ** 4 + p1 * x ** 2 + p0


def param_objective(x, p3, p2, p1, p0):
    return p3 * x ** 3 + p2 * x ** 2 + p1 * x + p0


def calculate_loss(pred, gt):
    return np.sum(np.abs(pred - gt))


if __name__ == '__main__':
    data = utils.read_data()
    unique_elevations = data['z'].unique()


    l_tiltscaler = StandardScaler()
    u_tiltscaler = StandardScaler()

    groups = [data[data['z'] == unique_elevations[i]]
              for i in range(len(unique_elevations))]
    lower_params = []
    upper_params = []
    losses = []
    for i in range(len(unique_elevations)):
        batch = groups[i]
        max_pan_angle_id = batch['pan'].idxmax()
        tilt_at_max_pan_angle = batch.loc[max_pan_angle_id]['tilt']
        lower_batch = batch[batch['tilt'] <= tilt_at_max_pan_angle]
        upper_batch = batch[batch['tilt'] > tilt_at_max_pan_angle]

        lx = lower_batch['pan'].to_numpy().reshape(-1, 1)
        ly = lower_batch['tilt'].to_numpy().reshape(-1, 1)
        lz = lower_batch['z'].to_numpy().reshape(-1, 1)

        ux = upper_batch['pan'].to_numpy().reshape(-1, 1)
        uy = upper_batch['tilt'].to_numpy().reshape(-1, 1)
        uz = upper_batch['z'].to_numpy().reshape(-1, 1)

        l_tiltscaler = StandardScaler()
        norm_x = l_tiltscaler.fit_transform(lx, ly)

        popt, _ = curve_fit(tilt_pan_objective, norm_x.squeeze(), ly.squeeze())
        a, b, c, d, e, f, g, h, k, l = popt
        lower_pred_y = tilt_pan_objective(norm_x, a, b, c, d, e, f, g, h, k, l)
        lower_params.append([a, b, c, d, e, f, g, h, k, l])

        u_tiltscaler = StandardScaler()
        norm_x = u_tiltscaler.fit_transform(ux, uy)

        popt, _ = curve_fit(tilt_pan_objective, norm_x.squeeze(), uy.squeeze())
        a, b, c, d, e, f, g, h, k, l = popt
        upper_pred_y = tilt_pan_objective(norm_x, a, b, c, d, e, f, g, h, k, l)
        upper_params.append([a, b, c, d, e, f, g, h, k, l])

        loss = calculate_loss(lower_pred_y, ly)
        losses.append(loss)

    lscaler = StandardScaler()
    uscaler = StandardScaler()

    lower_params = np.array(lower_params)
    upper_params = np.array(upper_params)
    pred_lower_params = []
    pred_upper_params = []
    for j in range(len(lower_params)):
        lparams = lower_params[:, j]
        uparams = upper_params[:, j]

        norm_unique_elevations = lscaler.fit_transform(unique_elevations.reshape(-1, 1), lparams)
        popt, _ = curve_fit(param_objective, norm_unique_elevations.squeeze(), lparams)
        p3, p2, p1, p0 = popt
        pred_lparams = param_objective(norm_unique_elevations, p3, p2, p1, p0)
        pred_lower_params.append([p3, p2, p1, p0])
        # loss = calculate_loss(pred_lower_params, lparams)

        norm_unique_elevations = uscaler.fit_transform(unique_elevations.reshape(-1, 1), uparams)
        popt, _ = curve_fit(param_objective, norm_unique_elevations.squeeze(), uparams)
        p3, p2, p1, p0 = popt
        pred_uparams = param_objective(norm_unique_elevations, p3, p2, p1, p0)
        pred_upper_params.append([p3, p2, p1, p0])
        # loss = calculate_loss(pred_upper_params, uparams)

        # data = [go.Scatter(x=unique_elevations, y=upper_params[:, i], name=f'Parameter {i}') for i in range(len(upper_params))]
        # data = [go.Scatter(x=unique_elevations, y=params, name=f'Parameter {j}'),
        #         go.Scatter(x=unique_elevations, y=pred_params.squeeze(), name=f'Pred Parameter {j}')]
        # fig2 = go.Figure(data=data)
        # fig2.update_layout(
        #     title=f'Param - Elev | Loss: {loss:.3f}',
        #     xaxis_title='Elevation (clicks)',
        #     yaxis_title=f'Parameter {j}',
        # )
        # fig2.show()

    elev = np.array([[9.7748]])
    lower_tilt_pan_params = []
    for i in range(len(pred_lower_params)):
        p3, p2, p1, p0 = pred_lower_params[i]
        norm_elevation = lscaler.transform(elev)
        param = param_objective(norm_elevation, p3, p2, p1, p0)
        lower_tilt_pan_params.append(param.item())
    a, b, c, d, e, f, g, h, k, l = lower_tilt_pan_params
    aa,bb,cc,dd,ee,ff,gg,hh,kk,ll =lower_params[0]
    pan_angles = np.arange(-60, 60, .5).reshape(-1, 1)
    norm_pan_angles = l_tiltscaler.transform(pan_angles)
    pred_tilt_angles = tilt_pan_objective(norm_pan_angles, a, b, c, d, e, f, g, h, k, l)
    original_pred_tilt_angles = tilt_pan_objective(norm_pan_angles, aa,bb,cc,dd,ee,ff,gg,hh,kk,ll)

    batch = groups[0]
    lx = batch['pan'].to_numpy().reshape(-1, 1)
    ly = batch['tilt'].to_numpy().reshape(-1, 1)

    fig = go.Figure(data=[
        go.Scatter(
            x=pan_angles.squeeze(), y=pred_tilt_angles.squeeze(),
            mode='markers',
            marker=dict(size=2, color='blue', opacity=1)
        ),
        go.Scatter(
                x=lx.squeeze(), y=ly.squeeze(),
                mode='markers',
                marker=dict(size=2, color='red', opacity=1)
            ),
        go.Scatter(
            x=pan_angles.squeeze(), y=original_pred_tilt_angles.squeeze(),
            mode='markers',
            marker=dict(size=2, color='green', opacity=1)
        ),

    ])
    fig.update_layout(
        title=f'Pan - Tilt - Elev:{elev} ',
        scene=dict(
            xaxis=dict(title='Pan (degrees)'),
            yaxis=dict(title='Tilt (degrees)'),
            zaxis=dict(title='Elevation (clicks)')
        ),
        xaxis_title='Pan (degrees)',
        yaxis_title='Tilt (degrees)',
    )
    fig.show()

    # print(f'Total Loss:{np.sum(losses)}')
    print('gr')
