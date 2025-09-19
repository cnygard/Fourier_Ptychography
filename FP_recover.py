import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os
from LED_location import LED_location
from k_vector import k_vector
from himrecover import himrecover

# Prepare the experimental data
# Load data file
data_name = 'MouseKidney_green'
data_dir = 'Data/'+data_name+'.mat'
data = scipy.io.loadmat(data_dir) # refer to 'data_description.txt' for more details
# Display raw images
imlow_HDR = data['imlow_HDR'] # 'center' shows the first low-res raw image; 'all' dynamically shows all low-res images

show_mode = 'first'
if show_mode == 'first':
  plt.title('raw image 1')
  plt.imshow(imlow_HDR[:,:,0],cmap='gray')
  plt.show()
elif show_mode == 'all':
  for slide in range(int(np.ceil(imlow_HDR.shape[2]/40))):
    for i in range(40):
      if i+slide*40 > imlow_HDR.shape[2]-1:
        break
      plt.subplot(5,8,i+1)
      plt.title('raw image '+str(i+slide*40+1))
      plt.imshow(imlow_HDR[:,:,i+slide*40],cmap='gray')
      plt.axis('off')
    plt.show()

# Set up the experiment parameters
xstart = 18; ystart = 20 # absolute coordinate of initial LED
arraysize = 15 # side length of lit LED array
xlocation, ylocation = LED_location(xstart, ystart, arraysize)
H = 90.88 # distance between LEDs and sample, in mm
LEDp = 4 # distance between adjacent LEDs, in mm
nglass = 1.52 # refraction index of glass substrate
t = 1 # glass thickness, in mm
kx, ky, NAt = k_vector(xlocation-xstart, ylocation-ystart, H, LEDp, nglass, t, data['theta'][0][0], data['xint'][0][0], data['yint'][0][0], arraysize^2)

# Reconstruct by FP algorithm
NA          = 0.1      # objective NA
spsize      = 1.845e-6 # pixel size of low-res image on sample plane, in m
upsmp_ratio = 4        # upsampling ratio
psize       = spsize/upsmp_ratio # pixel size of high-res image on sample plane, in m

class Opts:
    def __init__(self, loopnum=10, alpha=1, beta=1, gamma_obj=1, gamma_p=1, eta_obj=0.2, eta_p=0.2, T=1):
        self.loopnum   = loopnum   # iteration number
        self.alpha     = alpha     # '1' for ePIE, other value for rPIE
        self.beta      = beta      # '1' for ePIE, other value for rPIE
        self.gamma_obj = gamma_obj # the step size for object updating
        self.gamma_p   = gamma_p   # the step size for pupil updating
        self.eta_obj   = eta_obj   # the step size for adding momentum to object updating
        self.eta_p     = eta_p     # the step size for adding momentum to pupil updating
        self.T         = T         # do momentum every T images. '0' for no momentum during the recovery; integer, generally (0, arraysize^2].
        self.aberration = data['aberration'][0][0]

used_idx = np.arange(0, arraysize^2, 1) # choose which raw image is used, for example, 1:2:arraysize^2 means do FPM recovery with No1 image, No3 image, No5 image......
imlow_used = imlow_HDR[:,:,used_idx]
kx_used = kx[0,used_idx]
ky_used = ky[0,used_idx]
opts = Opts()
him, tt, fprobe, imlow_HDR1 = himrecover(imlow_used, kx_used, ky_used, NA, data['wlength'][0][0], spsize, psize, data['z'][0][0], opts)

# display
plt.subplot(121)
plt.title('Amplitude')
plt.imshow(np.abs(him[50:-50,50:-50]),cmap='gray')
plt.subplot(122)
plt.title('Phase')
plt.imshow(np.angle(him[50:-50,50:-50]),cmap='gray')
plt.show()
print(f'Wavelength: {wlength*1e+9} nm, Loop: {opts.loopnum}')
print(f'Maximum illumination NA = {np.max(NAt[used_ix])}')

# save results
out_dir = 'Results'
os.mkdir(out_dir)
out_name = f'{data_name}_result.mat'
out_data = {
    'him': him,
    'fprobe': fprobe,
    'tt': tt,
    'imlow_HDR1': imlow_HDR1
}
savemat(f'{out_dir}/{out_name}', out_data)
# CJCJCJ: Left off here. Try to test with output from real himrecover from matlab
