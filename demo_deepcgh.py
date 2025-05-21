#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:18:10 2020

@author: hoss
"""
import tensorflow as tf
from deepcgh import DeepCGH_Datasets, DeepCGH
import numpy as np
from glob import glob
import scipy.io as scio
from utils import GS3D, display_results, get_propagate
# Define params
retrain = True
frame_path = 'DeepCGH_Frames/*.mat'
coordinates = False

data = {
        'path' : 'DeepCGH_Datasets/Disks',
        'shape' : (512, 512, 3),
        'object_type' : 'Disk',
        'object_size' : 16,
        'object_count' : [3, 5],
        'intensity' : 1,
        'normalize' : True,
        'centralized' : False,
        'N' : 2500,
        'train_ratio' : 1900/2000,
        'compression' : 'GZIP',
        'name' : 'target',
        }


model = {
        'path' : 'DeepCGH_Models/Disks',
        'num_frames':5,
        'int_factor':16,
        'quantization':8,
        'n_kernels':[64, 128, 256],
        'plane_distance':0.05,
        'focal_point':0.2,
        'wavelength':1.04e-6,
        'pixel_size': 9.2e-6,
        'input_name':'target',
        'output_name':'phi_slm',
        'lr' : .0002,
        'batch_size' : 4,
        'epochs' : 100,
        'token' : 'DCGH',
        'shuffle' : 16,
        'max_steps' : 4000,
        # 'HMatrix' : hstack
        }


# Get data
dset = DeepCGH_Datasets(data)

dset.getDataset()

# Estimator
dcgh = DeepCGH(data, model)

if retrain:
    dcgh.train(dset)

#%% This is a sample test. You can generate a random image and get the results
model['HMatrix'] = dcgh.Hs # For plotting we use the exact same H matrices that DeepCGH used

# Get a function that propagates SLM phase to different planes according to your setup's characteristics
propagate = get_propagate(data, model)

# Generate a random sample
image = dset.get_randSample()[np.newaxis,...]
# Get the phase for your target using a trained and loaded DeepCGH
phase = dcgh.get_hologram(image)

# Simulate what the solution would look like
reconstruction = propagate(phase).numpy()
#GPT
#Debugging
# from utils import accuracy
# acc = accuracy(tf.constant(reconstruction[None,...]), tf.constant(sample[None,...]))
# print("Test accuracy:", acc.numpy())

print("Target image min/max:", image.min(), image.max())
print("Phase min/max:", phase.min(), phase.max())
print("Reconstruction min/max:", reconstruction.min(), reconstruction.max())
print("Reconstruction mean:", reconstruction.mean())
print("Reconstruction stddev:", reconstruction.std())


#%% Show the results
display_results(image, phase, reconstruction, 1)
