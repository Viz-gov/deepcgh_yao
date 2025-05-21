# sanity_check.py
import os
import numpy as np
import matplotlib.pyplot as plt
from deepcgh import DeepCGH_Datasets, DeepCGH
from utils   import get_propagate, GS3D

# 1) Prepare output folders
os.makedirs('sanity_results/targets',  exist_ok=True)
os.makedirs('sanity_results/recon',    exist_ok=True)
os.makedirs('sanity_results/phases',   exist_ok=True)

# 2) Load one sample from your existing dataset
from demo_deepcgh import data, model  # reuse your dicts
dset   = DeepCGH_Datasets(data)
dset.getDataset()
sample = dset.get_randSample()         # shape (H, W, planes)
planes = sample.shape[-1]

# Save each target plane
for p in range(planes):
    img = sample[:, :, p]
    plt.imsave(f'sanity_results/targets/plane_{p}.png',
               img, cmap='gray', vmin=0, vmax=1)

# 3) DeepCGH inference
dcgh  = DeepCGH(data, model)
dcgh.train(dset, max_steps=1)  

model['HMatrix'] = dcgh.Hs
prop  = get_propagate(data, model)

# wrap sample into batch of 1
batch_sample = sample[np.newaxis, ...]    # shape (1,H,W,planes)
# 3a) A simple TF input_fn for predict()
import tensorflow as tf
def simple_input_fn():
    return tf.data.Dataset.from_tensors(
        {'target': tf.constant(batch_sample, dtype=tf.float32)}
    )

# 3b) Run the Estimator predict
preds = dcgh.estimator.predict(input_fn=simple_input_fn,
                                yield_single_examples=False)
phase_dcgh = next(preds)

# 3c) Propagate and drop batch axis
recon_dcgh = prop(phase_dcgh).numpy()[0, ...]

# Save DeepCGH phase and recon
plt.imsave('sanity_results/phases/deepcgh_phase.png',
           np.squeeze(phase_dcgh[0]), cmap='twilight',
           vmin=-np.pi, vmax=np.pi)
plt.imsave('sanity_results/recon/recon_deepcgh.png',
           recon_dcgh[:, :, 0] / (recon_dcgh.max()+1e-8),
           cmap='gray', vmin=0, vmax=1)

# 4) Gerchberg–Saxton baseline
gs    = GS3D(data, model)
phase_gs = gs.get_phase(sample, K=10)[..., np.newaxis]  # (H,W,1)
recon_gs = get_propagate(data, {**model, 'HMatrix': gs.Hs})(phase_gs[np.newaxis,...]).numpy()[0,...]

# Save GS phase and recon
plt.imsave('sanity_results/phases/gs_phase.png',
           phase_gs[:,:,0], cmap='twilight',
           vmin=-np.pi, vmax=np.pi)
plt.imsave('sanity_results/recon/recon_gs.png',
           recon_gs[:, :, 0] / (recon_gs.max()+1e-8),
           cmap='gray', vmin=0, vmax=1)

print("Saved all sanity‐check images under sanity_results/")