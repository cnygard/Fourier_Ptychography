import matplotlib.pyplot as plt
import numpy as np
import scipy.io

data_name = 'MouseKidney_green'
data_dir = 'Data/'+data_name+'.mat'
data = scipy.io.loadmat(data_dir)

imlow_HDR = data['imlow_HDR']

show_mode = 'all'
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
