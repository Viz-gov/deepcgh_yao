#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:18:10 2020

@author: hoss
"""
import tensorflow as tf
from deepcgh import DeepCGH_Datasets, DeepDCGH
import numpy as np
from glob import glob
import scipy.io as scio
from utils import GS3D, display_results, get_propagate
import matplotlib.pyplot as plt
from datetime import datetime
import imageio

# Define params
retrain = True
frame_path = 'DeepCGH_Frames/*.mat'
coordinates = False

data = {
        'path' : 'DeepCGH_Datasets/Disks',
        'shape' : (1024, 1024, 3),
        'object_type' : 'Dot',
        'object_size' : 10,
        'object_count' : [27, 48],
        'intensity' : [0.8, 1],
        'normalize' : True,
        'centralized' : True,
        'N' : 2000,
        'train_ratio' : 1900/2000,
        'compression' : 'GZIP',
        'name' : 'target',
        }


model = {
        'path' : 'DeepCGH_Models/Disks',
        'num_frames':3,
        'quantization':8,
        'int_factor':16,
        'n_kernels':[64, 128, 256],
        'plane_distance':0.05,
        'focal_point':0.2,
        'wavelength':1.03e-6,
        'pixel_size': 8e-6,
        'input_name':'target',
        'output_name':'phi_slm',
        'lr' : 1e-4,
        'batch_size' : 16,
        'epochs' : 100,
        'token' : 'DCGH',
        'shuffle' : 16,
        'max_steps' :2000,
        # 'HMatrix' : hstack
        }

# Get data
dset = DeepCGH_Datasets(data)

dset.getDataset()

# Estimator
dcgh = DeepDCGH(data, model)

if retrain:
    dcgh.train(dset)

#%% This is a sample test. You can generate a random image and get the results
model['HMatrix'] = dcgh.Hs # For plotting we use the exact same H matrices that DeepCGH used

# Get a function that propagates SLM phase to different planes according to your setup's characteristics
propagate = get_propagate(data, model)

# Generate a random sample
image = dset.get_randSample()[np.newaxis,...]
# Get the phase for your target using a trained and loaded DeepCGH

# save tge target illumination image
image_squeezed = image.squeeze()

for i in range(image_squeezed.shape[-1]):
    plt.figure(figsize=(30, 20))
    plt.imshow(image_squeezed[:,:, i], cmap='gray')
    plt.axis('off')
plt.show()

image_array_reshaped = np.transpose(image_squeezed, (2, 0, 1))
if image_array_reshaped.dtype != np.uint8:
    image_array_uint8 = (image_array_reshaped * 255).astype(np.uint8)
else:
    image_array_uint8 = image_array_reshaped

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = r'C:\Users\myadmin\Vaziri Dropbox\Yao Wang\Project-SLM_Holography\DeepCGH\Experiment data'
output_path = f'{base_dir}/{timestamp}_DeepCGHTargetImage.tif'
imageio.mimwrite(output_path, image_array_uint8, format='TIFF')
print('target 3D image is saved')

phase = dcgh.get_hologram(image)

phaseMask = phase.squeeze()

phaseMask1 = phaseMask[0,:,:]
plt.figure(figsize=(30, 20))
plt.imshow(phaseMask1, cmap='gray')
plt.axis('off')

normalized_phase_mask = phaseMask1 % (2 * np.pi)
normalized_phase_mask = (normalized_phase_mask - np.min(normalized_phase_mask)) / (np.max(normalized_phase_mask) - np.min(normalized_phase_mask))
normalized_phase_mask = np.round(normalized_phase_mask * 255)
normalized_phase_mask = normalized_phase_mask.astype(np.uint8)

# Save the phase mask matrix as a bmp file

output_path = f'{base_dir}/{timestamp}_DeepCGHTrainedPhaseMask.bmp'
imageio.imwrite(output_path, normalized_phase_mask)
print('phase mask is saved')

# save log file for this simulation
# Specify the file to write to
log_file = f'{base_dir}/{timestamp}_DeepCGHTrainedPhaseMaskLog.txt'

# Open the file in write mode
with open(log_file, "w") as f:
    # Write the variable names and their values to the file
    f.write(f"field shape: {data['shape']}\n")
    f.write(f"object_type: {data['object_type']}\n")
    f.write(f"object_size: {data['object_size']}\n")
    f.write(f"object_count range: {data['object_count']}\n")
    f.write(f"intensity range: {data['intensity']}\n")
    f.write(f"normalize: {data['normalize']}\n")
    f.write(f"centralized: {data['centralized']}\n")
    f.write(f"num_frames: {model['num_frames']}\n")
    f.write(f"quantization: {model['quantization']}\n")
    f.write(f"plane_distance: {model['plane_distance']} m\n")
    f.write(f"focal_point EFL: {model['focal_point']} m\n")
    f.write(f"wavelength EFL: {model['wavelength']*1e6} um\n")
    f.write(f"SLM_pixel_size: {model['pixel_size']*1e6} um\n")

print(f"Variables have been written to {log_file}")



# Simulate what the solution would look like
reconstruction = propagate(phase).numpy()

#%% display simulation results
'''
plt.figure(figsize=(30, 20))
Z = [-50, 0, 50]
for i in range(reconstruction.shape[-1]):
    plt.subplot(231+i)
    plt.imshow(reconstruction[0, :,:, i], cmap='gray')
    plt.axis('off')
    plt.title('Simulation @ {}mm'.format(Z[i]))
    plt.subplot(234+i)
    plt.imshow(image[0, :,:, i], cmap='gray')
    plt.axis('off')
    plt.title('Target @ {}mm'.format(Z[i]))
plt.savefig('example.jpg')
plt.show()

#%%
'''