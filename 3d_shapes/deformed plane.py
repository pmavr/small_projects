import numpy as np
import plotly.graph_objs as go


def interpolate_between_2_3d_points(p1, p2, n_points):
    x = np.linspace(p1[0], p2[0], n_points)
    y = np.linspace(p1[1], p2[1], n_points)
    z = np.linspace(p1[2], p2[2], n_points)
    interpolated_points = np.stack([x, y, z], axis=1)
    return interpolated_points


if __name__ == '__main__':
    focals = np.array([10660, 12340, 8440, 10660])
    pans = np.array([-30., -18.3, 0., 0.])
    tilts = np.array([-15., -9.5, -17.2, -10.])

    pts = np.stack([focals, pans, tilts], axis=1)
    # for range(pts.shape[0])
    inter_pts = interpolate_between_2_3d_points(pts[0,:], pts[1,:], 10)

    fig = go.Figure(data=[
        go.Scatter3d(
            x=t.flatten(), y=p.flatten(), z=f.flatten(),
            mode='markers',
            marker=dict(size=3, color='red', opacity=1)
        ),
    ])
    fig.update_layout(
        title=f'x - y - z',
        scene=dict(
            xaxis=dict(title='Tilt (degrees)'),
            yaxis=dict(title='Pan (degrees)'),
            zaxis=dict(title='Focal Length (clicks)'),
        ),
    )
    fig.show()
    print('gr')
