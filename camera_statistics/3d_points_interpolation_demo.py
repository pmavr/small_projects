# Importing the necessary modules
import numpy as np
from scipy.interpolate import interp2d
import plotly.graph_objs as go


if __name__ == '__main__':

    n_points = 20
    pan = np.linspace(34.4, 36.2, n_points)
    tilt = np.linspace(-16.23482, -16.39415, n_points)
    z = np.linspace(26.42808, 24.96796, n_points)
    # pts = interpolate_between_2_points(p1, p2, n_points)
    print('gr')

    plots = [
        go.Scatter3d(
            x=pan, y=tilt, z=z, name='camera space',
            mode='markers',
            marker=dict(size=5, color='red', opacity=1)
        ),
    ]
    fig = go.Figure(data=plots)
    fig.update_layout(
        title=f'X - Y - Z ',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
    )
    fig.show()