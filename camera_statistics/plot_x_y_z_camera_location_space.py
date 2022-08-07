import numpy as np
import pandas as pd
import plotly.graph_objs as go

import utils


def bleachers_line(y, slope):
    z = y * (-slope)
    return z


if __name__ == '__main__':
    data = utils.read_data(['forward_shift_elevation_tilt_samples_v2'])

    cc_mean = np.array([[52.36618474, -45.15650112, 16.82156705]])
    cc_std = np.array([[1.23192608, 9.3825635, 2.94875254]])
    cc_min = np.array([[50.05679141, -66.0702037, 10.13871263]])
    cc_max = np.array([[54.84563315, -16.74178234, 23.01126126]])

    camera_num = 10000
    camera_centers = np.random.normal(cc_mean, cc_std, (camera_num, 3))
    camera_centers = pd.DataFrame(camera_centers, columns=['x', 'y', 'z'])

    z = []
    y = np.arange(data['y'].min(), data['y'].max(), .005)
    inclinations = np.arange(.3, .45, .01)
    for incl in inclinations:
        tmp = bleachers_line(y, incl)
        z.append(np.stack([np.ones_like(tmp) * y, tmp], axis=1))
    lines = np.concatenate(z, axis=0)
    num_of_x_samples = lines.shape[0]
    x_samples = np.random.normal(cc_mean[:, 0], cc_std[:, 0], (num_of_x_samples, 1))
    samples = pd.DataFrame(np.concatenate([x_samples, lines], axis=1), columns=['x', 'y', 'z'])

    fig = go.Figure(data=[
        go.Scatter3d(
            x=camera_centers['x'], y=camera_centers['y'], z=camera_centers['z'],
            mode='markers',
            marker=dict(size=3, color='green', opacity=1)
        ),
        go.Scatter3d(
            x=samples['x'], y=samples['y'], z=samples['z'],
            mode='markers',
            marker=dict(size=3, color='red', opacity=1)
        ),
        # go.Scatter(
        #     x=lines['y'], y=lines['z'],
        #     mode='markers',
        #     marker=dict(size=3, color='blue', opacity=1)
        # ),

    ])
    fig.update_layout(
        title=f'X-Y-Z camera location',
        scene=dict(
            xaxis=dict(title='X (clicks)'),
            yaxis=dict(title='Y (clicks)'),
            zaxis=dict(title='Z (clicks)')
        ),
    )
    fig.show()

    # print(f'Total Loss:{np.sum(losses)}')
    print('gr')
