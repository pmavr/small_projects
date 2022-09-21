import numpy as np
import pandas as pd
import plotly.graph_objs as go


def spheric2cartesian(r, theta, phi):
    theta = np.radians(theta)
    phi = np.radians(phi)
    x = r*np.cos(theta) * np.sin(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z=  r*np.sin(phi)
    return x, y, z


def cone(tip, radius, h, tilt_angle_range, pan_angle_range, resolution=100):
    left_pan_limit, right_pan_limit = pan_angle_range
    left_pan_limit = np.radians(left_pan_limit)
    right_pan_limit = np.radians(right_pan_limit)

    left_tilt_limit, right_tilt_limit = tilt_angle_range
    left_tilt_limit = np.radians(left_tilt_limit)
    right_tilt_limit = np.radians(right_tilt_limit)

    pan_angles, tilt_angles = np.mgrid[left_pan_limit:right_pan_limit:resolution * 1j, left_tilt_limit:right_tilt_limit:resolution * 1j]
    x = radius * np.cos(pan_angles) * np.sin(tilt_angles) + tip[0]
    y = radius * np.sin(pan_angles) * np.sin(tilt_angles) + tip[1]
    z = radius * np.sin(tilt_angles) + tip[2]
    return x, y, z


if __name__ == '__main__':
    # tilt_range = np.arange(-16.69, -5.35, .1)
    tilt_range = np.arange(-16, -5, .1)
    pan_range = np.arange(-65, 65, 1)
    min_focal_length = 1400
    max_focal_length = 10000
    focals = np.array([10660, 12340, 10660, 12340, 8440, 10660])
    pans = np.array([-30., -18.3, 30., 18.3, 0., 0.])
    tilts = np.array([-15., -9.5, -15, -9.5, -17.2, -10.])

    # xx,yy,zz = [],[],[]
    # for tilt in tilt_range:
    #     for pan in pan_range:
    #         # x, y, z = spheric2cartesian(min_focal_length, pan, tilt)
    #         xx.append(tilt)
    #         yy.append(pan)
    #         zz.append(min_focal_length)

    # x = np.array(xx)
    # y = np.array(yy)
    # z = np.array(zz)

    # h = max_focal_length - min_focal_length
    # tip = np.array([0, 0, min_focal_length])
    # radius = 1
    # x, y, z = cone(tip, radius, h, tilt_range, pan_range)

    fig = go.Figure(data=[
        # go.Surface(x=x, y=y, z=z),
        # go.Scatter3d(
        #     x=x, y=y, z=z,
        #     mode='markers',
        #     marker=dict(size=3, color='green', opacity=1)
        # ),
        go.Scatter3d(
            x=tilts, y=pans, z=focals,
            mode='markers',
            marker=dict(size=3, color='green', opacity=1)
        ),
    ])
    fig.update_layout(
        title=f'x - y - z',
        scene=dict(
            xaxis=dict(title='Tilt (degrees)'),
            yaxis=dict(title='Pan (degrees)'),
            zaxis=dict(title='Focal Length (clicks)'),
            # aspectmode='data'
        ),
    )
    fig.show()
