from helpers import spherical2carthesian

from typing import Tuple

import numpy as np
from scipy.interpolate import LSQSphereBivariateSpline, SmoothSphereBivariateSpline, interp1d, LinearNDInterpolator,\
    RegularGridInterpolator

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

    # remove nan rows
    bool = np.all(np.isnan(data), axis=1)
    data = data[~bool, :]
    theta = theta[~bool]

    # remove cycle val
    if phi[-1]-phi[0] == 2*np.pi:
        phi = phi[:-1]
        data = data[:, :-1]

    return data, theta, phi


def toPointList(data: np.ndarray,
              theta: np.ndarray,
              phi: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray]:

    northVal = None
    if theta[0] == 0:
        northVal = data[0, 0]
        theta = theta[1:]
        data = data[1:, :]

    southVal = None
    if theta[-1] == np.pi:
        southVal = data[-1, -1]
        theta = theta[:-1]
        data = data[:-1, :]

    PHI, THETA = np.meshgrid(phi, theta)
    X, Y, Z = spherical2carthesian(THETA, PHI)

    points = np.hstack((X.reshape((-1, 1)), Y.reshape((-1, 1)), Z.reshape((-1, 1))))
    data = data.ravel()

    if northVal is not None:
        points = np.insert(points, 0, (0., 0., 1.), axis=0)
        data = np.insert(data, 0, northVal)

    if southVal is not None:
        points = np.append(points, np.atleast_2d((0., 0., -1.)), axis=0)
        data = np.append(data, southVal)

    return points, data




def fillAround(data: np.ndarray,
              theta: np.ndarray,
              phi: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    pi_ngrid = np.argmin(np.abs(np.pi-phi))
    # it assumes here that grid along phi is regular and full num(phi) < pi == num(phi) > phi
    if np.abs(pi_ngrid - phi.size/2) > 1 or (np.max(np.diff(phi)) - np.min(np.diff(phi)))/np.mean(np.diff(phi)) > 0.001:
        print("\033[31mPhi Grid is not regular\033[0m")

    data_ud = np.roll(np.flipud(data), -pi_ngrid, axis=1)
    theta_h = -theta
    theta_l = 2 * np.pi - theta

    if theta[0] == 0:
        data_ud = data_ud[:-1, :]
        theta_h = theta_h[1:]
        theta_l = theta_l[1:]

    if theta[-1] == np.pi:
        data_ud = data_ud[1:, :]
        theta_h = theta_h[:-1]
        theta_l = theta_l[:-1]

    theta = np.concatenate((theta_h, theta, theta_l))
    theta.sort()

    phi = np.concatenate((phi - 2*np.pi, phi, phi + 2*np.pi))
    data = np.vstack((data_ud, data, data_ud))
    data = np.hstack((data, data, data))
    return data, theta, phi

def fillSphere(data: np.ndarray,
              theta: np.ndarray,
              phi: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_l = np.zeros_like(data)
    data_h = np.zeros_like(data)

    theta_n = np.pi-theta

    bool = np.logical_xor(theta_n > theta.max(), theta_n < theta.min())
    data_h = data_h[theta_n < theta.min(), :]
    data_l = data_l[theta_n > theta.max(), :]

    theta_n = theta_n[bool]

    data_h = np.flipud(data_h)
    data_l = np.flipud(data_l)
    theta = np.sort(np.concatenate((theta, theta_n)))

    data = np.concatenate((data_h, data, data_l))

    return data, theta, phi


def fillNaN(data: np.ndarray,
              theta: np.ndarray,
              phi: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bool = ~np.isnan(data)
    data_row = np.ndarray((0, data.shape[1]))
    for (row, br) in zip(data, bool):
        interp = interp1d(phi[br], row[br])
        brl = np.logical_and(phi >= phi[br].min(), phi <= phi[br].max()) # in case nan points on ends row
        row = np.full_like(phi, np.nan)
        row[brl] = interp(phi[brl])
        row = row.reshape((1, -1))
        data_row = np.append(data_row, row, axis=0)

    data_col = np.ndarray((data.shape[0], 0))
    for (col, br) in zip(data.T, bool.T):
        interp = interp1d(theta[br], col[br])
        brc = np.logical_and(theta >= theta[br].min(), theta <= theta[br].max()) # in case nan points on ends col
        col = np.full_like(theta, np.nan)
        col[brc] = interp(theta[brc])
        col = col.reshape((-1, 1))
        data_col = np.append(data_col, col, axis=1)

    data = (data_col + data_row) / 2

    return data, theta, phi


def cutNaN(data: np.ndarray,
              theta: np.ndarray,
              phi: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bool = np.isnan(data)
    while(bool[bool].size > 0):
        if np.any(bool[0, :]):
            data = data[1:, :]
            theta = theta[1:]
        bool = np.isnan(data)
        if np.any(bool[-1, :]):
            data = data[:-1, :]
            theta = theta[:-1]
        bool = np.isnan(data)
        if np.any(bool[:, 0]):
            data = data[:, 1:]
            phi = phi[1:]
        bool = np.isnan(data)
        if np.any(bool[:, -1]):
            data = data[:, :-1]
            phi = phi[:-1]
        bool = np.isnan(data)
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
    surf_orig = go.Surface(x=X, y=Y, z=Z, opacity=1)

    X, Y, Z = spherical2carthesian(THETA, PHI, np.full_like(THETA, data[~np.isnan(data)].max()))
    sphere = go.Surface(x=X, y=Y, z=Z, opacity=0.1, surfacecolor=np.full_like(X, 0))

    if interpolator is not None:
        data_i = interpolator(thetaR, phiR)
        X, Y, Z = spherical2carthesian(THETA_regular, PHI_regular, data_i)
        surf_intrp = go.Surface(x=X, y=Y, z=Z, opacity=1, surfacecolor=np.full_like(Z, 0))
        fig = go.Figure(data=[surf_intrp, surf_orig, sphere])
    else:
        fig = go.Figure(data=[surf_orig, sphere])

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
