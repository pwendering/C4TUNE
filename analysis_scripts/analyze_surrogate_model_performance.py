"""

Assess the performance of the trained surrogate model.

"""

import torch
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay

sys.path.append(str(Path().resolve().parents[0]))

from src.models.model_surrogate import SurrogateModel
from src.prediction.surrogate_predictor import SurrogatePredictor
from src.utils.env_setup import set_training_environment, get_config
from src.utils.utils import load_param_names
from src.data.data import PhotResponseDataset


# =============================================================================
#%% Load model and data
# =============================================================================

# get surrogate model configuration
base_config_file = "../config/base.yaml"
model_config_file = "../config/surrogate.yaml"
config = get_config(base_config_file, model_config_file)

# numpy random seed 
np.random.seed(config.training.rng_seed)

# model weights after training
surrogate_checkpoint = os.path.join(config.paths.run_dir, "2025_02_21", "surrogate-epoch-60.pth")

# create surrogate model predictor
set_training_environment(config)
device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
model = SurrogateModel(config.model)
surrogate = SurrogatePredictor(model, surrogate_checkpoint, device, config)

# read the dataset and indices of the test set
dataset = PhotResponseDataset(config.paths.datasets)
idx_test = np.load(config.paths.datasets.test_idx)
co2_steps = dataset.a_co2.columns.to_numpy(dtype='int')
light_steps = dataset.a_light.columns.to_numpy(dtype='int')
n_co2 = len(co2_steps)
n_light = len(light_steps)
n_params = dataset.params.shape[1]

# indices of dataset subsample for testing
n_samples = 10000
idx_samples = np.random.choice(idx_test,
                               np.min((n_samples, idx_test.shape[0]-1)), 
                               replace=False)

# output directory
result_dir = os.path.join(config.paths.base_dir, "results", "surrogate_model")

# =============================================================================
#%% Predict curves for random subset of the test set 
# =============================================================================

env_input = {
    "co2_steps": dataset.co2_steps,
    "light_a_co2": dataset.light_a_co2,
    "light_steps": dataset.light_steps,
    "co2_a_light": dataset.co2_a_light,
    }

pred_curves_subset = surrogate.predict(
    dataset.params.iloc[idx_samples, :].to_numpy(),
    env_input
    )

# =============================================================================
#%% Calculate mean squared errors 
# =============================================================================

y_subset = np.concatenate((dataset.a_co2.iloc[idx_samples, :].to_numpy(),
                           dataset.a_light.iloc[idx_samples, :].to_numpy()),
                          axis=1)
pred_curves_subset_concat = np.concatenate((pred_curves_subset[0],
                                           pred_curves_subset[1]),  axis=1)
mse = ((pred_curves_subset_concat - y_subset)**2).mean(axis=1)
mse = np.expand_dims(mse, 1)


mse_a_co2 = ((pred_curves_subset[0] - y_subset[:, :n_co2])**2).mean(axis=1)
mse_a_light = ((pred_curves_subset[1] - y_subset[:, n_co2:])**2).mean(axis=1)

print("Median MSE +/- MAD surrogate model:")
print(f"Overall: {np.median(mse):.2f} +/- {np.median(np.abs(mse-np.median(mse))):.2f}")
print(f"A/CO2: {np.median(mse_a_co2):.2f} +/- {np.median(np.abs(mse_a_co2-np.median(mse_a_co2))):.2f}")
print(f"A/light: {np.median(mse_a_light):.2f} +/- {np.median(np.abs(mse_a_light-np.median(mse_a_light))):.2f}")

# =============================================================================
#%% Plot random example curve to compare surrogate model prediction with sampled curve
# =============================================================================

# idx_plot = np.argmax(mse)  # curve with maximum prediction error
idx_plot = np.random.choice(range(0, n_samples))  # random index in subset

fig_mse_subset, axs_mse_subset = plt.subplots(1, 2, figsize=(8,4), layout='tight')
axs_mse_subset[0].plot(co2_steps, y_subset[idx_plot, :n_co2])
axs_mse_subset[0].plot(co2_steps, pred_curves_subset[0][idx_plot])
axs_mse_subset[0].legend(["obs.",
                          f"pred. (MSE={mse[idx_plot]:.2f})"],
                         loc='best', frameon=False)
axs_mse_subset[0].set_xlabel(r"$p(CO_{2})\ (\mu bar)$")
axs_mse_subset[0].set_ylabel(r"$A_{net}\ (\mu mol\ m^{-2}\ s^{-1})$")
axs_mse_subset[0].set_ylim([-5, axs_mse_subset[0].get_ylim()[1]])

axs_mse_subset[1].plot(light_steps, y_subset[idx_plot, n_co2:])
axs_mse_subset[1].plot(light_steps, pred_curves_subset[1][idx_plot])
axs_mse_subset[1].set_xlabel(r"$light\ intensity\ (\mu mol\ m^{-2}\ s^{-1})$")
axs_mse_subset[1].set_ylim(axs_mse_subset[0].get_ylim())

fig_mse_subset.savefig(os.path.join(result_dir,
                                    "surrogate_model_example_curves_random.png"),
                       dpi=300)
# fig_mse_subset.savefig(os.path.join(result_dir,
#                                     "surrogate_model_example_curves_max_mse.png"),
#                        dpi=300)

# =============================================================================
#%% Generate scatter plots of surrogate model predictions
# =============================================================================

fig_surrogate_performance, axs_surrogate_performance = plt.subplots(1, 3, layout='constrained', figsize=(10, 4))

min_val_aco2 = np.min((pred_curves_subset[0], y_subset[:, :n_co2]))
max_val_aco2 = np.max((pred_curves_subset[0], y_subset[:, :n_co2]))
min_val_alight = np.min((pred_curves_subset[1], y_subset[:, n_co2:]))
max_val_alight = np.max((pred_curves_subset[1], y_subset[:, n_co2:]))

axes_limits = [np.min([min_val_aco2, min_val_alight]),
            np.max([max_val_aco2, max_val_alight])]
ticks = range(int(axes_limits[0].round(-1)+10), int(axes_limits[1].round(-1)+10), 10)

# A/CO2
axs_surrogate_performance[0].scatter(y_subset[:, :n_co2], pred_curves_subset[0],
    c=np.log10(mse.repeat(n_co2, axis=1)))
axs_surrogate_performance[0].set_xlim(axes_limits)
axs_surrogate_performance[0].set_ylim(axes_limits)
axs_surrogate_performance[0].set_xticks(ticks)
axs_surrogate_performance[0].set_yticks(ticks)

# A/light
sp2 = axs_surrogate_performance[1].scatter(y_subset[:, n_co2:],
    pred_curves_subset[1], c=np.log10(mse.repeat(n_light, axis=1)))
axs_surrogate_performance[1].set_xlim(axes_limits)
axs_surrogate_performance[1].set_ylim(axes_limits)
axs_surrogate_performance[1].set_xticks(ticks)
axs_surrogate_performance[1].set_yticks([])

nbins = 100

N, bins = np.histogram(np.log10(mse), bins=nbins, density=True)
xcenters = (bins[:-1] + bins[1:]) / 2
norm = colors.Normalize(np.log10(mse).min(), np.log10(mse).max())
axs_surrogate_performance[2].barh(xcenters, 1.5*np.ones(len(xcenters)),
        color=[plt.cm.viridis(norm(xc)) for xc in xcenters])

axs_surrogate_performance[2].set_box_aspect(3)
axs_surrogate_performance[2].hist(np.log10(mse),
         bins=nbins,
         density=True,
         histtype='step',
         orientation='horizontal',
         color=[0, 0, 0])
axs_surrogate_performance[2].yaxis.tick_right()
axs_surrogate_performance[2].yaxis.set_label_position('right')

axs_surrogate_performance[2].set_ylabel(r'$log_{10}\ MSE$')
axs_surrogate_performance[2].set_xlabel('density')
axs_surrogate_performance[2].set_ylim([xcenters[0], xcenters[-1]])
axs_surrogate_performance[2].set_xlim([N.min(), 1.5])
axs_surrogate_performance[2].set_box_aspect(7)

fig_surrogate_performance.supxlabel(r"$A_{net}\ (sampling)$"
                "\n"
                r"$(\mu mol\ m^{-2}\ s^{-1})$")
fig_surrogate_performance.supylabel(r"$A_{net}\ (prediction)$"
                "\n"
                r"$(\mu mol\ m^{-2}\ s^{-1})$")

fig_surrogate_performance.savefig(os.path.join(result_dir,
                                               "surrogate_model_mse_n1e4.png"),
                                  dpi=300)

# =============================================================================
#%% Repeat MSE estimation over entire test set
# =============================================================================

pred_curves_test_set = surrogate.predict(
    dataset.params.iloc[idx_test, :].to_numpy(),
    env_input
    )

y_test = np.concatenate((
    dataset.a_co2.iloc[idx_test, :], 
    dataset.a_light.iloc[idx_test, :]), axis=1)
pred_concat_test = np.concatenate((
    pred_curves_test_set[0],
    pred_curves_test_set[1]), axis=1)

# mean squared errors
mse_test_set = ((pred_concat_test - y_test)**2).mean(axis=1)
mse_test_set = np.expand_dims(mse_test_set, 1)

mse_test_set_a_co2 = ((pred_curves_test_set[0] - y_test[:, :n_co2])**2).mean(axis=1)
mse_test_set_a_light = ((pred_curves_test_set[1] - y_test[:, n_co2:])**2).mean(axis=1)

print("Median MSE +/- MAD surrogate model (test set):")
print(f"Overall: {np.median(mse_test_set):.2f} +/- {np.median(np.abs(mse_test_set-np.median(mse_test_set))):.2f}")
print(f"A/CO2: {np.median(mse_test_set_a_co2):.2f} +/- {np.median(np.abs(mse_test_set_a_co2-np.median(mse_test_set_a_co2))):.2f}")
print(f"A/light: {np.median(mse_test_set_a_light):.2f} +/- {np.median(np.abs(mse_test_set_a_light-np.median(mse_test_set_a_light))):.2f}")


# coefficients of determination
r2_test_set = 1 - np.sum(((pred_concat_test - y_test)**2), axis=1) \
    / np.sum((y_test - np.expand_dims(y_test.mean(axis=1), 1))**2, axis=1)

r2_test_set_a_co2 = 1 - np.sum(((pred_curves_test_set[0] - y_test[:, :n_co2])**2), axis=1) \
    / np.sum((y_test[:, :n_co2] - np.expand_dims(y_test[:, :n_co2].mean(axis=1), 1))**2, axis=1)

r2_test_set_a_light = 1 - np.sum(((pred_curves_test_set[1] - y_test[:, n_co2:])**2), axis=1) \
    / np.sum((y_test[:, n_co2:] - np.expand_dims(y_test[:, n_co2:].mean(axis=1), 1))**2, axis=1)

print("Median R2 +/- MAD surrogate model (test set):")
print(f"Overall: {np.median(r2_test_set):.2f} +/- {np.median(np.abs(r2_test_set-np.median(r2_test_set))):.2f}")
print(f"A/CO2: {np.median(r2_test_set_a_co2):.2f} +/- {np.median(np.abs(r2_test_set_a_co2-np.median(r2_test_set_a_co2))):.2f}")
print(f"A/light: {np.median(r2_test_set_a_light):.2f} +/- {np.median(np.abs(r2_test_set_a_light-np.median(r2_test_set_a_light))):.2f}")

# Pearson correlation between performances for A/CO2 and A/light curves
r_p_mse = np.corrcoef(mse_test_set_a_co2, mse_test_set_a_light)[0, 1]
r_p_r2 = np.corrcoef(r2_test_set_a_co2, r2_test_set_a_light)[0, 1]

print(f"Pearson correlation of MSE values between curve types: r={r_p_mse:.2f}")
print(f"Pearson correlation of R2 values between curve types: r={r_p_r2:.2f}")

# =============================================================================
#%% Train random forest model to find factors that impact surrogate model MSE
# =============================================================================

param_names = load_param_names(config)
param_ids = dataset.params.columns.values

np.random.seed(1)

# balance the high and low MSE indices
n = 90  # number of instances per MSE bin
h_mse = np.histogram(np.log(mse_test_set), bins=5)[1]  # histogram bins
data_idx = [np.random.choice(
    np.where(np.all((np.log(mse_test_set)>=h_mse[i], np.log(mse_test_set)<h_mse[i+1]), axis=0))[0],
    n, replace=False) for i in range(0, len(h_mse)-1)]
y = np.array([mse_test_set[idx] for idx in data_idx]).ravel()
X = np.reshape(dataset.params.iloc[idx_test, :].to_numpy()[data_idx],
               ((len(h_mse)-1)*n, n_params))

# train random forest regressor
rf = RandomForestRegressor()
rf_r2 = [0]*10
feature_importances_regression = np.zeros((10, dataset.params.shape[1]))
for i in range(0, 10):
    rf.fit(X, y)
    rf_r2[i] = rf.score(X, y)
    feature_importances_regression[i, :] = rf.feature_importances_

feat_order = np.argsort(-feature_importances_regression.mean(axis=0))

# plot feature importances
plt.rc('font', size=10)
fig_rf_features, axs_rf_features = plt.subplots(1, 2, layout='constrained', figsize=(5, 2))

# plot feature importances
axs_rf_features[0].barh(param_ids[feat_order][:10],
        feature_importances_regression.mean(axis=0)[feat_order][:10],
        xerr=feature_importances_regression.std(axis=0)[feat_order][:10])
axs_rf_features[0].set_xlabel("feature importance")
axs_rf_features[0].set_yticks(axs_rf_features[0].get_yticks(), [param_names[i] for i in feat_order[:10]])

# plot partial dependence of the two most important parameters
f_names = param_ids[feat_order][:2]
f = [np.where(param_ids==x)[0][0] for x in f_names]

features = [f[0]]
disp = PartialDependenceDisplay.from_estimator(
    rf, X, features, feature_names=dataset.params.columns.values,
    ax=axs_rf_features[1:])
axs_rf_features[1].set_xlabel(param_names[feat_order[0]])
    
fig_rf_features.savefig(
    os.path.join(result_dir, "features_partial_dependence_rf_regression_surrogate_mse.png"),
    dpi=300)
