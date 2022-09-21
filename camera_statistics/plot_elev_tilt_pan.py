
import numpy as np
import pandas as pd
import plotly.graph_objs as go

import utils


if __name__ == '__main__':
    files = [
        'pan-tilt-z_samples_1',
        'pan-tilt-z_samples_2',
        'pan-tilt-z_samples_mirror'
    ]
    data = utils.read_data(files)

    unique_elevations = data['z'].unique()
    forward_shift = data['y'].unique()

    x = data['pan'].to_numpy()
    y = data['tilt'].to_numpy()
    z = data['z'].to_numpy()

    # tilt_at_max_pan = utils.get_tilt_angles_at_max_pan_angle(data)

    # curves = utils.predict_pan_tilt_curves(data)

    # uniform_samples = utils.uniform_samples_per_elevation(curves)
    print('gr')

    # curves = pd.concat([curves[e] for e in list(curves.keys())], axis=0)
    plots = [
        # go.Scatter3d(
        #     x=lower_data['pan'], y=lower_data['tilt'], z=lower_data['z'], name='lower camera_space',
        #     mode='markers',
        #     marker=dict(size=5, color='cyan', opacity=1)
        # ),
        go.Scatter3d(
            x=x, y=y, z=z, name='camera space',
            mode='markers',
            marker=dict(size=1, color='red', opacity=1)
        ),
        # go.Scatter3d(
        #     x=tilt_at_max_pan['pan'], y=tilt_at_max_pan['tilt'], z=tilt_at_max_pan.index, name='upper-lower bound',
        #     marker=dict(size=7, color='blue', opacity=1),
        #     line=dict(color='darkblue', width=5)
        # ),
        # go.Scatter3d(
        #     x=uniform_samples['pan'], y=uniform_samples['tilt'], z=uniform_samples['z'], name='uniform samples',
        #     mode='markers',
        #     marker=dict(size=2, color='green', opacity=1),
        # ),
        # go.Scatter3d(
        #     x=curves['pan'], y=curves['tilt'], z=curves['z'], name='interpolated curves',
        #     mode='markers',
        #     marker=dict(size=3, color='cyan', opacity=1),
        # ),
    ]


    fig = go.Figure(data=plots)
    fig.update_layout(
        title=f'Pan - Tilt - Elev ',
        scene=dict(
            xaxis=dict(title='Pan (degrees)'),
            yaxis=dict(title='Tilt (degrees)'),
            zaxis=dict(title='Elevation (clicks)')
        ),
    )
    fig.show()