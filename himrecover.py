import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
from gzn import gzn

# -1 = "-1 because python is 0-indexed"

def himrecover(imseqlow, kx, ky, NA, wlength, spsize, psize, z, opts):
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

    # Set default values for opts if not present (Matlab isfield equivalent)
    opts = opts or {}
    loopnum = opts.loopnum
    alpha = opts.alpha
    beta = opts.beta
    gamma_obj = opts.gamma_obj
    gamma_p = opts.gamma_p
    eta_obj = opts.eta_obj
    eta_p = opts.eta_p
    T = opts.T
    aberration = opts.aberration

    print("B1")
    # k-spare parameterization
    m1, n1, numim = imseqlow.shape
    pratio = np.round(spsize / psize) # upsampling ratio
    m = pratio * m1
    n = pratio * n1
    k0 = (2 * np.pi) / wlength
    kx = k0 * kx
    ky = k0 * ky
    print("B2")
    NAfilx = NA * (1 / wlength) * n * psize
    NAfily = NA * (1 / wlength) * m * psize # m1 * spize = m * psize
    kmax = np.pi / psize # max wave vector of the OTF
    dkx = (2 * np.pi) / (psize * n)
    dky = (2 * np.pi) / (psize * m)
    print("B3")
    kx2 = np.arange(-kmax, kmax + 1, kmax/((n-1)/2)) # odd N (number of points?)
    ky2 = np.arange(-kmax, kmax + 1, kmax/((m-1)/2))
    print("B4")
    print(f"kx2:{kx2.shape} ky2:{ky2.shape}")
    kxm, kym = np.meshgrid(kx2, ky2)
    print("B5")
    print(f"kxm:{kxm.shape} kym:{kym.shape}")
    kzm = np.sqrt(k0**2 - np.square(kxm) - np.square(kym))
    print("B6")

    print("C1")
    # prior knowledge of aberration
    print(f"z:{z.shape} kzm:{kzm.shape}")
    H2 = np.exp(1j * z * np.real(kzm)) * np.exp((-1 * np.abs(z)) * np.abs(np.imag(kzm)))
    print("C2")
    astigx = 0 # define the astigmatism aberration if it is known or you want to test it
    astigy = 0
    M1, N1 = np.meshgrid(np.arange(1, m1 + 1), np.arange(1, n1 + 1)) # add 1 because matlab's : range is inclusive on upper end
    print("C3")
    print(f"m1:{m1} type:{type(m1)}")
    zn = (astigx * gzn(max(m1, n1), 2 * np.maximum(np.round(NAfily), np.round(NAfilx)), np.array([2]), np.array([2])) +
     astigy * gzn(max(m1,n1), 2 * np.maximum(np.round(NAfily), np.round(NAfilx)), np.array([-2]), np.array([2])))
    print("C4")
    zn = cv.resize(zn, (n1, m1)) # switched m1 and n1 because OpenCV uses (width, height)
    print("C5")
    if np.any(aberration != 0):
        fmaskpro = aberration # pre-calibrated aberration
    else:
        fmaskpro = np.multiply(1, float(np.power(np.power(((N1 - ((m1 + 1) / 2)) / NAfily), 2 + ((M1 - ((n1 + 1) / 2)) / NAfilx)), 2) <= 1)) # low pass filter
        # defocus aberration, astigmatism aberration
        fmaskpro = np.multiply(H2[np.round(((m + 1) / 2) - ((m1 - 1) / 2)):np.round(((m + 1) / 2) + ((m1 - 1) / 2)),
                      np.round(((n + 1) / 2) - ((n1 - 1) / 2)):np.round(((n + 1) / 2) + ((n1 - 1) / 2))], np.exp(math.pi * np.multiply(1j, zn)))
        
    print("D")
    # initialization
    him = cv.resize(np.sum(imseqlow, axis=2), (n, m)) # 2 since matlab dimensions are 1-indexed (?), switched n and m since OpenCV is (w, h)
    himFT = np.fft.fftshift(np.fft.fft2(him))
    print("Pre loops")

    # main part to optimize estimate of high-res image
    for i in range(1, 3):
        print(i)
        for i3 in range(1, numim + 1):
            # when the image size is even, there will be a half pixel displacement for the center
            kxc = np.round((n + 1) / 2 - (kx[0, i3 - 1] / dkx)) # -1
            kyc = np.round((m + 1) / 2 - (ky[0, i3 - 1] / dky))
            kyl = np.round(kyc - (m1 - 1) / 2)
            kyh = np.round(kyc + (m1 - 1) / 2)
            kxl = np.round(kxc - (n1 - 1) / 2)
            kxh = np.round(kxc + (n1 - 1) / 2)
            O_j = himFT[kyl - 1:kyh, kxl - 1:kxh] # -1
            lowFT = O_j * fmaskpro
            im_lowFT = np.fft.ifft2(np.fft.ifftshift(lowFT))
            updatetemp = (pratio**2) * imseqlow[:, :, i3 - 1]
            im_lowFT = updatetemp * np.exp(1j * np.angle(im_lowFT))
            lowFT_p = np.fft.fftshift(np.fft.fft2(im_lowFT))
            himFT[kyl - 1:kyh, kxl - 1:kxh] = himFT[kyl - 1:kyh, kxl - 1:kxh] + (np.conj(fmaskpro) / (np.max(np.abs(fmaskpro)**2))) * (lowFT_p - lowFT)
    
    countimg = 0
    tt = np.ones((1, loopnum * numim))

    # for momentum method
    vobj0 = np.zeros((m, n))
    vp0 = np.zeros((m1, n1))
    ObjT = himFT
    PT = fmaskpro

    for i in range(1, loopnum + 1):
        print(i)
        for i3 in range(1, numim + 1):
            countimg = countimg + 1
            kxc = np.round((n + 1) / 2 - (kx[0, i3 - 1] / dkx)) # -1
            kyc = np.round((m + 1) / 2 - (ky[0, i3 - 1] / dky))
            kyl = np.round(kyc - (m1 - 1) / 2)
            kyh = np.round(kyc + (m1 - 1) / 2)
            kxl = np.round(kxc - (n1 - 1) / 2)
            kxh = np.round(kxc + (n1 - 1) / 2)
            O_j = himFT[kyl - 1:kyh, kxl - 1:kxh] # -1
            lowFT = O_j * fmaskpro
            im_lowFT = np.fft.ifft2(np.fft.ifftshift(lowFT))
            tt[0, i3 + (i - 2) * numim - 1] = np.mean(np.abs(im_lowFT)) / np.mean((pratio ** 2) * np.abs(imseqlow[:, :, i3 - 1]))
            if i >= 2:
                imseqlow[:, :, i3 - 1] = imseqlow[:, :, i3 - 1] * tt[0, i3 + (i - 2) * numim - 1]

            updatetemp = (pratio**2) * imseqlow[:, :, i3 - 1]
            im_lowFT = updatetemp * np.exp(1j * np.angle(im_lowFT))
            lowFT_p = np.fft.fftshift(np.fft.fft2(im_lowFT))

            himFT[kyl - 1:kyh, kxl - 1:kxh] = himFT[kyl - 1:kyh, kxl - 1:kxh] + (gamma_obj * np.conj(fmaskpro) * (lowFT_p - lowFT)) / ((1 - alpha) * np.abs(fmaskpro)**2 + alpha * np.max(np.abs(fmaskpro)**2))
            fmaskpro = fmaskpro + gamma_p * np.conj(O_j) * (lowFT_p - lowFT) / ((1 - beta) * np.abs(O_j)**2 + beta * np.max(np.abs(O_j)**2))

            if countimg == T: # momentum method
                vobj = eta_obj * vobj0 + (himFT - ObjT)
                himFT = ObjT + vobj
                vobj0 = vobj
                ObjT = himFT

                vp = eta_p * vp0 + (fmaskpro - PT)
                fmaskpro = PT + vp
                vp0 = vp
                PT = fmaskpro

                countimg = 0
    
    him = np.fft.ifft2(np.fft.ifftshift(himFT))

    return him, tt, fmaskpro, imseqlow
