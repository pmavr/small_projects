import numpy as np
import pandas as pd
import plotly.graph_objs as go

def ms(x, y, z, radius, resolution=180):
    """Return the coordinates for plotting a sphere centered at (x,y,z)"""
    u, v = np.mgrid[0:2*np.pi:resolution * 1j, 0:-np.pi:resolution * 1j]
    X = radius * np.cos(u) * np.sin(v) + x
    Y = radius * np.sin(u) * np.sin(v) + y
    Z = radius * np.cos(v) + z
    return (X, Y, Z)

if __name__=='__main__':
    r = 30000
    x, y, z = ms(0, 0, 0, r)

    fig = go.Figure(data=[
        go.Surface(
            x=x, y=y, z=z,
            # mode='markers',
            # marker=dict(size=3, color='red', opacity=1)
        ),
    ])
    fig.update_layout(
        title=f'x - y - z',
        scene=dict(
            zaxis=dict(title='Z (clicks)'),
            xaxis=dict(title='X (clicks)'),
            yaxis=dict(title='Y (clicks)'),
            aspectmode='data'
        ),
    )
    fig.show()