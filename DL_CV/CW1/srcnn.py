import time
import os
import matplotlib.pyplot as plt
import warnings

import numpy as np
import tensorflow as tf
import scipy
import pdb
from skimage import measure
import skimage
warnings.filterwarnings('ignore')


def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def modcrop(image, scale=3):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def preprocess(path, scale=3):
  """
  Preprocess single image file 
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Apply image file with bicubic interpolation
  Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
  """
  image = imread(path, is_grayscale=True)
  label_ = modcrop(image, scale)

  # Must be normalized
  image = image / 255.
  label_ = label_ / 255.

  input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
  input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)

  return input_, label_

"""Set the image hyper parameters
"""
c_dim = 1
input_size = 255

"""Define the model weights and biases 
"""

# define the placeholders for inputs and outputs
inputs = tf.placeholder(tf.float32, [None, input_size, input_size, c_dim], name='inputs')

## ------ Add your code here: set the weight of three conv layers
# replace '0' with your hyper parameter numbers 
# conv1 layer with biases: 64 filters with size 9 x 9
# conv2 layer with biases and relu: 32 filters with size 1 x 1
# conv3 layer with biases and NO relu: 1 filter with size 5 x 5
weights = {
    'w1': tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1'),
    'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
    'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')
    }

biases = {
      'b1': tf.Variable(tf.zeros([64]), name='b1'),
      'b2': tf.Variable(tf.zeros([32]), name='b2'),
      'b3': tf.Variable(tf.zeros([1]), name='b3')
    }

"""Define the model layers with three convolutional layers
"""
## ------ Add your code here: to compute feature maps of input low-resolution images
# replace 'None' with your layers: use the tf.nn.conv2d() and tf.nn.relu()
# conv1 layer with biases and relu : 64 filters with size 9 x 9

conv1 = tf.nn.relu(tf.nn.conv2d(inputs, weights['w1'], strides=[1,1,1,1], padding='VALID') + biases['b1'])
##------ Add your code here: to compute non-linear mapping
# conv2 layer with biases and relu: 32 filters with size 1 x 1

conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['w2'], strides=[1,1,1,1], padding='VALID') + biases['b2'])
##------ Add your code here: compute the reconstruction of high-resolution image
# conv3 layer with biases and NO relu: 1 filter with size 5 x 5
conv3 = tf.nn.conv2d(conv2, weights['w3'], strides=[1,1,1,1], padding='VALID') + biases['b3']


"""Load the pre-trained model file
"""
model_path='./model/model.npy'
model = np.load(model_path, encoding='latin1', allow_pickle=True).item()

##------ Add your code here: show the weights of model and try to visualisa
# variabiles (w1, w2, w3)

weight_w1 = model['w1']
weight_w2 = model['w2']
weight_w3 = model['w3']
filter_w1 = model['w1'].shape[3]
filter_w2 = model['w2'].shape[3]
filter_w3 = model['w3'].shape[3]

print("*"*25)
print("filter_w1",filter_w1,"filter_w2",filter_w2,"filter_w3",filter_w3)

print("*"*25)
print("Plotting Weights")
fig = plt.figure(figsize = (8,8))
for i in range(filter_w1):
  sub_image = fig.add_subplot(8, 8, i+1)
  sub_image.imshow(weight_w1[:, :, 0, i] ,interpolation = None, cmap = 'gray')
plt.show()
fig.savefig('weight_w1_subplot.jpg')


fig = plt.figure(figsize = (8,8))
for i in range(filter_w2):
  sub_image = fig.add_subplot(8, 8, i+1)
  sub_image.imshow(weight_w2[:, :, 0, i] ,interpolation = None, cmap = 'gray')
plt.show()
fig.savefig('weight_w2_subplot.jpg')


plt.imshow(weight_w3[:, :, 0, 0] ,interpolation = None, cmap = 'gray')
plt.show()
#plt.savefig('weight_w3_subplot.jpg') # aprrantly this does not save the image
scipy.misc.imsave('weight_3.jpg', weight_w3[:,:,0,0]) # saving image

"""Initialize the model variabiles (w1, w2, w3, b1, b2, b3) with the pre-trained model file
"""
# launch a session
sess = tf.Session()

for key in weights.keys():
  sess.run(weights[key].assign(model[key]))

for key in biases.keys():
  sess.run(biases[key].assign(model[key]))

"""Read the test image
"""
blurred_image, groudtruth_image = preprocess('./image/butterfly_GT.bmp')

"""Run the model and get the SR image
"""
# transform the input to 4-D tensor
input_ = np.expand_dims(np.expand_dims(blurred_image, axis =0), axis=-1)

# run the session
# here you can also run to get feature map like 'conv1' and 'conv2'
output_ = sess.run(conv3, feed_dict={inputs: input_})

##------ Add your code here: save the blurred and SR images and compute the psnr
# hints: use the 'scipy.misc.imsave()'  and ' skimage.meause.compare_psnr()'
print("*"*25)
print("Saving Super resolution image, blurred_image and ground_truth image ")
scipy.misc.imsave('SR.jpg',output_[0,:,:,0])
scipy.misc.imsave('ground_truth.jpg',groudtruth_image)
scipy.misc.imsave('blurred.jpg',blurred_image)
print("*"*25)
print("groudtruth_image.shape",groudtruth_image.shape)
print("blurred_image.shape",blurred_image.shape)
print("SR_image.shape",output_.shape)

groudtruth_image_cropped = groudtruth_image[6:-6,6:-6] # cropping the image to size of SR image size
blurred_image_cropped = blurred_image[6:-6,6:-6] # croping the image to size of SR image
sr_float = output_[0,:,:,0].astype(np.float64)  # to numpy float 64 as ground truth and 
print("*"*25)
print("Calculating PSNR values on ground_truth vs blurred_image and ground_truth vs SR image")
psnr_sr_br = skimage.measure.compare_psnr(groudtruth_image_cropped,blurred_image_cropped)
psnr_sr_gt = skimage.measure.compare_psnr(groudtruth_image_cropped,sr_float)
print("*"*25)

print("blurred image PSNR", str(psnr_sr_br))
print("SR image PSNR", str(psnr_sr_gt))


"""
Results
*************************
filter_w1 64 filter_w2 32 filter_w3 1
*************************
Plotting Weights
2020-02-23 12:56:54.015994: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-02-23 12:56:54.062759: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:998] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-02-23 12:56:54.063220: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x141f4e0 executing computations on platform CUDA. Devices:
2020-02-23 12:56:54.063238: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): GeForce GT 740M, Compute Capability 3.5
2020-02-23 12:56:54.082053: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-02-23 12:56:54.082631: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x1499540 executing computations on platform Host. Devices:
2020-02-23 12:56:54.082662: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
2020-02-23 12:56:54.082823: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties: 
name: GeForce GT 740M major: 3 minor: 5 memoryClockRate(GHz): 1.0325
pciBusID: 0000:01:00.0
totalMemory: 1.96GiB freeMemory: 1.74GiB
2020-02-23 12:56:54.082847: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2020-02-23 12:56:54.083578: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-02-23 12:56:54.083597: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2020-02-23 12:56:54.083606: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2020-02-23 12:56:54.083699: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1553 MB memory) -> physical GPU (device: 0, name: GeForce GT 740M, pci bus id: 0000:01:00.0, compute capability: 3.5)
*************************
Saving Super resolution image, blurred_image and ground_truth image 
*************************
groudtruth_image.shape (255, 255)
blurred_image.shape (255, 255)
SR_image.shape (1, 243, 243, 1)
*************************
Calculating PSNR values on ground_truth vs blurred_image and ground_truth vs SR image
*************************
blurred image PSNR 20.453967418499577
SR image PSNR 21.77124773032378

"""