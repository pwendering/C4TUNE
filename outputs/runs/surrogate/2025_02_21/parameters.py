# -*- coding: utf-8 -*-
"""
Continued training from 2025_02_20
"""

# =================== Network parameters ===================
num_params = 236
num_co2_steps = 11
num_light_steps = 6
param_encoding_dim = 128
env_encoding_dim = 64
n_hidden = 3
activation_fun = "ReLU"
weight_init = None

# =================== Training parameters ===================

# seed for random number generator (torch)
rng_seed = 93

# optimizer
learning_rate = 0.0273204
weight_decay = 0
momentum = 0.776007
nesterov = True

# learning rate scheduler
linearlr_scheduler_steps = 15
f_lr = 0.0100639

steplr_scheduler_step_size  = 15
steplr_gamma = 0.5

# training data
train_pct = 0.8
shuffle = True

# training loop
batch_size = 8
gradient_clipping = True
gradient_clipping_norm = 2.39349
gradient_clipping_max = 1
loss_fn = "combined_loss"

# training epochs
epochs = 60
start_epoch = 51

# progress display and checkpoint saving
display_step = 10000
checkpoint_every = 5
checkpoint_dir = "C:/Users/pw543/OneDrive - University of Cambridge/LearnPhotParams/parameter_prediction/runs_surrogate/"

# for enrichment with high MSE values
cp_for_enrichment = None  # checkpoint_dir + "surrogate-model-full-epoch-30.pth"


# =================== hyperparameter tuning ===================

# number of sampled models
n_samples_ray = 10

# =================== HPC settings ===================

# number of CPUs for pytorch (intraop)
n_cpu_torch = 4

# number of CPUs for ray tune
n_cpu_ray = 4