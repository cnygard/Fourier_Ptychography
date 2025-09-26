import numpy as np
from zernfun import zernfun

def gzn(tpixel, NApixel, m, n):
    # generate Zernike mode of (n,m)
    # tpixel is the total num of the image;
    # NApixel is diameter of the NA.
    x = np.linspace(-tpixel / NApixel, tpixel / NApixel, num=tpixel)
    X, Y = np.meshgrid(x, x)
    print(f"X:{X.shape} Y:{Y.shape}")
    r, theta = cart2pol(X, Y)
    idx = r <= 1
    z = np.zeros_like(X)
    z[idx] = zernfun(n, m, np.array(r[idx])[:, np.newaxis], np.array(theta[idx])[:, np.newaxis]).squeeze()


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    print(f"rho:{rho.shape} phi:{phi.shape}")
    return(rho, phi)