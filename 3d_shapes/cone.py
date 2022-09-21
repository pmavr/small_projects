import numpy as np
import pandas as pd
import plotly.graph_objs as go


def cone(tip, radius, h, angle_range, resolution=100):
    left_limit, right_limit = angle_range
    left_limit = np.radians(left_limit)
    right_limit = np.radians(right_limit)
    u, v = np.mgrid[left_limit:right_limit:resolution * 1j, 0:np.pi:resolution * 1j]
    x = radius * np.cos(u) * np.sin(v) + tip[0]
    y = radius * np.sin(u) * np.sin(v) + tip[1]
    z = h * np.sin(v) + tip[2]
    return x, y, z


if __name__ == '__main__':
    angle_range = (0, 360)
    h = 2
    tip = np.array([0, 0, 0])
    radius = 1
    x, y, z = cone(tip, radius, h, angle_range)

    fig = go.Figure(data=[
        go.Surface(x=x, y=y, z=z),
    ])
    fig.update_layout(
        title=f'x - y - z',
        scene=dict(
            xaxis=dict(title='X (clicks)'),
            yaxis=dict(title='Y (clicks)'),
            zaxis=dict(title='Z (clicks)'),
            aspectmode='data'
        ),
    )
    fig.show()
