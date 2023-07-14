from helpers import spherical2carthesian

import numpy as np
from scipy.interpolate import LSQSphereBivariateSpline, SmoothSphereBivariateSpline

from typing import Any, Callable

import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.graph_objects as go

thetaR, phiR = np.pi/180 * np.linspace(0, 90, 91), np.pi/180 * np.linspace(0, 360, 361)
thetaR[0] += 0.0001
thetaR[-1] -= 0.0001
phiR[0] += 0.0001
phiR[-1] -= 0.0001
PHI_regular, THETA_regular = np.meshgrid(phiR, thetaR)


def loadFile(filename: str) -> (np.ndarray, np.ndarray, np.ndarray):
    data = np.genfromtxt(filename, delimiter=";")
    phi = data[0, 1:]
    theta = data[1:, 0]
    data = data[1:, 1:]

    theta = theta/180 * np.pi
    phi = phi/180 * np.pi

    return data, theta, phi


def makeInterp(data: np.ndarray, theta: np.ndarray, phi: np.ndarray, s: float = 3.3e-8)\
        -> Callable[[np.ndarray, np.ndarray], None]:
    bool = ~np.isnan(data)
    PHI, THETA = np.meshgrid(phi, theta)
    lut = SmoothSphereBivariateSpline(THETA[bool].ravel(), PHI[bool].ravel(), data[bool].ravel(), s=s)

    return lut

def plot1D(data: np.ndarray, theta: np.ndarray, phi: np.ndarray, interpolator: Callable = None) -> None:
    plt.subplots(1, 2)
    plt.title("1D")

    plt.subplot(121)
    plt.plot(theta * 180/np.pi, data, '.-', linewidth=0.5, markersize=2)

    plt.subplot(122)
    plt.plot(phi * 180/np.pi, data.T, '.-', linewidth=0.5, markersize=2)

    if interpolator is not None:
        data_i = interpolator(theta, phi)

        plt.subplot(121)
        plt.gca().set_prop_cycle(None)
        plt.plot(theta * 180/np.pi, data_i)

        plt.subplot(122)
        plt.gca().set_prop_cycle(None)
        plt.plot(phi * 180/np.pi, data_i.T)

    plt.show()


def plot2D(data: np.ndarray, theta: np.ndarray, phi: np.ndarray, interpolator: Callable = None) -> None:
    PHI, THETA = np.meshgrid(phi, theta)
    plt.subplots(2, 2)

    plt.title("2D")

    plt.subplot(221)
    plt.pcolormesh(PHI * 180 / np.pi, THETA * 180 / np.pi, data, cmap=mpl.cm.jet, alpha=0.7)
    plt.gca().invert_yaxis()

    plt.subplot(222)
    plt.contourf(PHI*180/np.pi, THETA*180/np.pi, data, cmap=mpl.cm.jet, alpha=0.7, origin='upper', levels=25)
    plt.gca().invert_yaxis()

    if interpolator is not None:
        data_i = interpolator(thetaR, phiR)

        plt.subplot(223)
        plt.pcolormesh(PHI_regular * 180 / np.pi, THETA_regular * 180 / np.pi, data_i, cmap=mpl.cm.jet, alpha=0.7)
        plt.gca().invert_yaxis()

        plt.subplot(224)
        plt.contourf(PHI_regular*180/np.pi, THETA_regular*180/np.pi, data_i, cmap=mpl.cm.jet, alpha=0.7, origin='upper',
                     levels=25)
        plt.gca().invert_yaxis()

    plt.show()


def plot3D(data: np.ndarray, theta: np.ndarray, phi: np.ndarray, interpolator: Callable = None) -> None:
    PHI, THETA = np.meshgrid(phi, theta)

    X, Y, Z = spherical2carthesian(THETA, PHI, data)
    surf_orig = go.Surface(x=X, y=Y, z=Z, opacity=0.35)


    if interpolator is not None:
        data_i = interpolator(thetaR, phiR)
        X, Y, Z = spherical2carthesian(THETA_regular, PHI_regular, data_i)
        surf_intrp = go.Surface(x=X, y=Y, z=Z, opacity=1, surfacecolor=np.full_like(Z, 0))
        fig = go.Figure(data=[surf_intrp, surf_orig])
    else:
        fig = go.Figure(data=[surf_orig])

    fig.show()


def saveData(filename, data, theta, phi):
    thetaD = theta * 180/np.pi
    phiD = phi * 180/np.pi
    data_s = np.concatenate([np.atleast_2d(thetaD).T, data], axis=1)
    phi_s = np.append(np.nan, phiD)
    data_s = np.concatenate([np.atleast_2d(phi_s), data_s], axis=0)
    np.savetxt(filename, data_s, delimiter=";")


def processDef(filename, s=3.3e-8):
    data, theta, phi = loadFile(filename)
    lut = makeInterp(data, theta, phi, s=s)

    plot1D(data, theta, phi, lut)
    plot2D(data, theta, phi, lut)
    plot3D(data, theta, phi, lut)
