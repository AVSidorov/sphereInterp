import numpy as np

def carthesian2spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    phi[phi < 0] = phi[phi < 0] + 2*np.pi
    return r, theta, phi

def spherical2carthesian(theta, phi, r=None):
    if r is None:
        r = np.ones_like(theta)
    z = r * np.cos(theta)
    x = np.cos(phi) * r*np.sin(theta)
    y = np.sin(phi) * r*np.sin(theta)
    return x, y, z