import plotly.graph_objects as go
import numpy as np
from numpy import pi, sin, cos

def spheric2cartesian(r, theta, phi):
    x = r*cos(theta) *sin(phi)
    y = r*sin(theta)*sin(phi)
    z= r*cos(phi)
    return x, y, z
n=250 #number of points to be generated
R = 1.5 #sphere radius
theta_m = pi/12  #min theta  val
theta_M = theta_m + pi/6 #max theta val
phi_m= pi/5    #min phi
phi_M=  phi_m + pi/6  #max phi
#generate n points within a spherical sector
r = R*np.random.rand(n)
theta = theta_m + (theta_M-theta_m) *np.random.rand(n)
phi= phi_m +(phi_M-phi_m)* np.random.rand(n)
x, y, z = spheric2cartesian (r, theta, phi)
fig = go.Figure(go.Scatter3d(x=x, y=y, z=z, mode="markers", marker_size=3))
fig.update_layout(width=600, height=600, scene_camera_eye=dict(x=1.5, y=-1.5, z=1),          scene_aspectmode="data")
fig.show()