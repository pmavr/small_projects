import numpy as np
import plotly.graph_objs as go

def plane(points):
    p1 = pts[:, 0]
    p2 = pts[:, 1]
    p3 = pts[:, 2]
    p4 = pts[:, 3]

    p1p2 = p2 - p1
    p3p4 = p4 - p3
    normal_vector = np.cross(p1p2, p3p4)
    return normal_vector

if __name__ == '__main__':

    focals = np.array([10660, 12340, 8440, 10660])
    pans = np.array([-30., -18.3, 0., 0.])
    tilts = np.array([-15., -9.5, -17.2, -10.])

    pts = np.stack([focals, pans, tilts])
    a, b, c = plane(pts)

    fl = []
    # pan = np.arange(0, 30, 1)
    # tilt = np.arange(-17, -10, 1)
    p, t = np.mgrid[-30:0:1, -17:-10:.1]
    f = (-b * p - c * t) / a
            # fl.append(f)
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
