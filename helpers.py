import numpy as np
from typing import Any, Callable, Optional, Tuple


def carthesian2spherical(x: [float, np.ndarray],
                         y: [float, np.ndarray],
                         z: [float, np.ndarray]
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)

    phi[phi < 0] = phi[phi < 0] + 2*np.pi
    return r, theta, phi


def spherical2carthesian(theta: [float, np.ndarray],
                         phi: [float, np.ndarray],
                         r: [float, np.ndarray] = None
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)
    r = np.atleast_1d(r)

    if r is None:
        r = np.ones_like(theta)
    z = r * np.cos(theta)
    x = np.cos(phi) * r*np.sin(theta)
    y = np.sin(phi) * r*np.sin(theta)
    return x, y, z


def xyz2sph(func: Callable[[[float, np.ndarray], [float, np.ndarray], [float, np.ndarray]],
                            Tuple[np.ndarray, np.ndarray, np.ndarray]]
            ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    def new(x: [float, np.ndarray], y: [float, np.ndarray], z: [float, np.ndarray]
            ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        r, theta, phi = carthesian2spherical(x, y, z)
        return func(theta, phi, r)
    return new


def sph2xyz(func: Callable[[[float, np.ndarray], [float, np.ndarray], [float, np.ndarray]],
                            Tuple[np.ndarray, np.ndarray, np.ndarray]]
            ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    def new(theta: np.ndarray, phi: np.ndarray, r: np.ndarray
            ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        x, y, z = spherical2carthesian(theta, phi, r)
        return func(x, y, z)
    return new


class F1:
    def __init__(self, base_point: np.ndarray, sigma: float = np.pi/4) -> None:
        self._base = base_point/np.linalg.norm(base_point)
        self.sigma = sigma

    def __call__(self, point: np.ndarray) -> Any:
        point = point / np.linalg.norm(point, axis=0)
        alpha = np.arccos(np.dot(self._base, point))
        value = (np.exp(-alpha**2/2/self.sigma**2) - np.exp(-np.pi**2/2/self.sigma**2)) /\
                (1 - np.exp(-np.pi**2/2/self.sigma**2))
        return value


class F2:
    def __init__(self, base_point: np.ndarray, sigma: float = np.pi/4) -> None:
        self._base = base_point/np.linalg.norm(base_point)
        self.sigma = sigma

    def __call__(self, point: np.ndarray) -> Any:
        point = point / np.linalg.norm(point, axis=0)
        alpha = np.arccos(np.dot(self._base, point))
        value = np.exp(-alpha**2/2/self.sigma**2)
        return value

# TODO
# point2xyz
# xyz3point

class F3:
    def __init__(self, base_point: np.ndarray, sigma: float = np.pi/4) -> None:
        self._base = base_point/np.linalg.norm(base_point)
        self.sigma = sigma

    def __call__(self, point: np.ndarray) -> Any:
        point = point / np.linalg.norm(point, axis=0)
        coss = np.dot(self._base, point)
        value = np.exp(-coss**2/2/self.sigma**2)
        return value
