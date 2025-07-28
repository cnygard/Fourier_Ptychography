import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from gzn import gzn


def himrecover(imseqlow, kx, ky, NA, wlength, spsize, psize, z, _opts):
    #     Input:
    #        imseqlow: low-res measurements, [m1 x n1 x numim] matrix
    #        kx,ky: normalized wavevector of each LED, which should times k0 later
    #        NA: objective NA
    #        wlength: central peak wavelength of LED, in m
    #        spsize: pixel size of low-res image on sample plane, in m
    #        psize: pixel size of high-res image on sample plane, in m
    #        z: known defocus distance, in m
    #        opts: other optimization parameters
    #     Output:
    #        him: recovered high-res image, [m x n] matrix
    #        tt: recorded intensity ratio between estimated and measured low-res amplitude, used for intensity correction
    #        fmaskpro: recovered pupil function
    #        imseqlow: low-res amplitudes after intensity correction

    # TODO: add in opts definitions / if statements
    aberration = [1, 2, 3] # replace with opts definition of aberration

    # k-spare parameterization
    m1, n1, numim = imseqlow.shape
    pratio = np.round(spsize / psize) # upsampling ratio
    m = pratio * m1
    n = pratio * n1
    k0 = (2 * np.pi) / wlength
    kx = k0 * kx
    ky = k0 * ky
    NAfilx = NA * (1 / wlength) * n * psize
    NAfily = NA * (1 / wlength) * m * psize # m1 * spize = m * psize
    kmax = np.pi / psize # max wave vector of the OTF
    dkx = (2 * np.pi) / (psize * n)
    dky = (2 * np.pi) / (psize * m)
    kx2 = np.arange(-kmax, kmax + 1, (n - 1) / 2) # odd N (number of points?)
    ky2 = np.arange(-kmax, kmax + 1, (m - 1) / 2)
    kxm, kym = np.meshgrid(kx2, ky2)
    kzm = np.sqrt(k0**2 - np.square(kxm) - np.square(kym))

    # prior knowledge of aberration
    H2 = np.exp(1j * z * np.real(kzm)) * np.exp((-1 * np.abs(z)) * np.abs(np.imag(kzm)))
    astigx = 0 # define the astigmatism aberration if it is known or you want to test it
    astigy = 0
    M1, N1 = np.meshgrid(np.arange(1, m1 + 1), np.arange(1, n1 + 1)) # add 1 because matlab's : range is inclusive on upper end
    zn = (astigx * gzn[np.max(m1, n1), 2 * np.maximum(np.round(NAfily), np.round(NAfilx)), 2, 2] +
     astigy * gzn[np.max(m1,n1), 2 * np.maximum(np.round(NAfily), np.round(NAfilx)), -2, 2])
    zn = cv.resize(zn, (n1, m1)) # switched m1 and n1 because OpenCV uses (width, height)
    if np.any(aberration != 0):
        fmaskpro = aberration # pre-calibrated aberration