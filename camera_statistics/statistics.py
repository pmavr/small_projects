from scipy.stats import multivariate_normal
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def get_vector(v0, v1):
    return dict(
                x=v1[0].item(),
                y=v1[1].item(),
                xref="x", yref="y",
                text="",
                showarrow=True,
                axref="x", ayref='y',
                ax=v0[0].item(),
                ay=v0[1].item(),
                arrowhead=3,
                arrowwidth=1.5,
                arrowcolor='black'
            )


if __name__ == '__main__':
    files = [
        '2022_06_26_17_45_38-6055',
        '2022_06_26_18_10_11-4568',
        '2022_06_26_18_17_04-5651',
        '2022_06_26_18_25_43-5924'
    ]
    dataset = [np.load(f'{f}.npy') for f in files]
    dataset = np.concatenate(dataset, axis=0)
    data = pd.DataFrame(dataset, columns=['u', 'v', 'fl', 'tilt', 'pan', 'roll', 'x', 'y', 'z'])
    tilt_pan_z = np.transpose(data[['tilt', 'pan', 'z']].to_numpy())
    cov_mat = np.cov(tilt_pan_z)
    mean_mat = np.mean(tilt_pan_z, axis=1)
    # samples = multivariate_normal.rvs(cov=cov_mat, mean=mean_mat, size=10000)

    fig = go.Figure(data=[
        go.Scatter3d(
            x=data['tilt'].to_numpy(), y=data['pan'].to_numpy(), z=data['z'].to_numpy(),
            mode='markers',
            marker=dict(size=2, color='red', opacity=1)
        ),
        # go.Scatter(
        #     x=data['tilt'].to_numpy(), y=data['z'].to_numpy(), mode='markers',
        #     marker=dict(size=2, color='blue', opacity=1)
        # )
    ])
    fig.update_layout(
        title='Tilt - Pan - LocYZ.4',
        scene=dict(
            xaxis=dict(title='Tilt (degrees)'),
            yaxis=dict(title='Pan (degrees)'),
            zaxis=dict(title='Elevation (clicks)')
        ),
        xaxis_title='Tilt (degrees)',
        yaxis_title='Elevation (clicks)',
    )
    fig.show()

    print('gr')