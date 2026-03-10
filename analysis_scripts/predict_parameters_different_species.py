"""
Predict model parameters for different species and genotypes from the following
publications

* 
* 

"""

import os
import sys
import time
from copy import deepcopy
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch import FloatTensor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy.cluster import hierarchy
import fastcluster
from omegaconf import OmegaConf

sys.path.append(str(Path().resolve().parents[0]))

from src.models.model_c4tune import ParameterPredictionModel
from src.prediction.c4tune_predictor import C4tunePredictor
from src.models.model_surrogate import SurrogateModel
from src.prediction.surrogate_predictor import SurrogatePredictor
from src.utils.env_setup import set_training_environment, get_config
from src.utils.utils import load_param_names
from src.data.data import PhotResponseDataset
from src.c4_kinetic_model.c4model import C4DynamicModel
from src.utils.paths import PROJECT_ROOT, resolve_config_paths


np.random.seed(123)
torch.manual_seed(321)

plt.rc('font', size=14)
plt.rc('legend', fontsize=10)

def filter_response_curves(anet):
    a_max = 70
    a_min = -10
    
    remove_bool = np.any(anet>a_max, axis=1) | np.sum(anet==0, axis=1)>1 | \
        np.all(anet<0, axis=1) | np.any(anet<a_min, axis=1) | \
        (np.sum(np.diff(np.sign(np.diff(anet, n=1, axis=1)), n=1, axis=1), axis=1)>0)
        
    return remove_bool
    
def coeff_var(x):
    return x.std()/x.mean()

def rvcoeff(x, y):
    return np.trace(x@x.T@y@y.T)/np.sqrt(np.trace(x@x.T@x@x.T)*np.trace(y@y.T@y@y.T))
 


# https://gist.github.com/thriveth/8560036#file-cbcolors-py
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3',
          '#999999', '#e41a1c', '#dede00']

# load base configuration
base_config_file = os.path.join(PROJECT_ROOT, "config/base.yaml")
base_config = resolve_config_paths(OmegaConf.load(base_config_file))

result_dir = os.path.join(PROJECT_ROOT, "results",
                         "parameters_prediction_different_species")
data_dir = os.path.join(PROJECT_ROOT, "data",
                         "anet_measurements")
publication_dirs = ["Almeida2025"]

#%% Load synthetic dataset
dataset = PhotResponseDataset(base_config.paths.datasets)
co2_steps = dataset.a_co2.columns.to_numpy(dtype='int')
light_steps = dataset.a_light.columns.to_numpy(dtype='int')
n_co2 = len(co2_steps)
n_light = len(light_steps)
n_params = dataset.params.shape[1]

# parameter names
param_names = load_param_names()
n_params = len(param_names)

#%% Create C4TUNE predictor

c4tune_config_file = os.path.join(PROJECT_ROOT, "config/c4tune.yaml")
config_c4tune = get_config(base_config_file, c4tune_config_file)

c4tune_checkpoint = os.path.join(config_c4tune.paths.run_dir, "2025-03-21", "c4tune-epoch-60.pth")

# Load Cholesky decomposition matrix and change the model's property
L = np.loadtxt(config_c4tune.paths.cholesky_test, delimiter=',')

# create C4TUNE and surrogate model predictors
set_training_environment(config_c4tune)
device = torch.device(config_c4tune.training.device if torch.cuda.is_available() else "cpu")

c4tune_model = ParameterPredictionModel(config_c4tune.model, L=FloatTensor(L))

c4tune = C4tunePredictor(c4tune_model, c4tune_checkpoint, device, config_c4tune)

#%% Create Surrogate model predictor

surrogate_config_file = os.path.join(PROJECT_ROOT, "config/surrogate.yaml")
surrogate_config = get_config(base_config_file, surrogate_config_file)

# numpy random seed 
np.random.seed(surrogate_config.training.rng_seed)

# model weights after training
surrogate_checkpoint = os.path.join(surrogate_config.paths.run_dir, "2025_02_21", "surrogate-epoch-60.pth")

# create surrogate model predictor
set_training_environment(surrogate_config)
surrogate_model = SurrogateModel(surrogate_config.model)
surrogate = SurrogatePredictor(surrogate_model, surrogate_checkpoint, device, surrogate_config)

#%% C4 kinetic model

# create Wrapper for C4 kinetic model simulation written in Matlab
c4model = C4DynamicModel(base_config)
co2_order = c4model.order_co2_steps
light_order = c4model.order_light_steps

#%% Load experimental data

print("Parsing experimental data...")

geno_row_labels = []
species_row_labels = []
publication_row_labels = []

exp_temperatures_gas_exchange = []

a_co2 = []
light_a_co2 = []

a_light = []
co2_a_light = []

p_sheet_names = ["EnzymeActivities", "SLM", "Rd", "Phi_PSII", "FvFm",
                 "Leakiness"]
measured_params = {}

for i in range(0, len(publication_dirs)):
    
    tmp_data_file = os.path.join(data_dir, publication_dirs[i], "Data.xlsx")
    
    xl = pd.ExcelFile(tmp_data_file)
    
    # Read A/Ci and A/Q data
    aci_tmp = xl.parse(sheet_name="ACi", comment="#")
    geno_tmp = aci_tmp.loc[:, "Genotype"].unique()
    
    aq_tmp = xl.parse(sheet_name="AQ", comment="#")
    
    # Get irradiance (A/Ci)
    if aci_tmp["Irradiance (µmol m^-2 s^-1)"].unique().shape[0]==1:
        light_a_co2.append(aci_tmp["Irradiance (µmol m^-2 s^-1)"][0])
    else:
        print(f"{publication_dirs[i]}: Multiple irradiances in A/Ci data.")
                
    # Get Ca (A/Q)
    if aq_tmp["Ca (µbar)"].unique().shape[0]==1:
        co2_a_light.append(aq_tmp["Ca (µbar)"][0])
    else:
        print(f"{publication_dirs[i]}: Multiple Ca values in A/Q data.")
    
    if np.isnan(aci_tmp["Ca (µbar)"]).all():
        cica_flag = True
        # Ci need to be converted to Ca
        cica_tmp = xl.parse(sheet_name="CiCa", comment="#")
    else:
        cica_flag = False
            
    for j in range(0, len(geno_tmp)):
        
        # A/Q for current genotype
        idx_geno_aq = aq_tmp["Genotype"] == geno_tmp[j]
        aq_geno = aq_tmp["Anet (µmol m^-2 s^-1)"][idx_geno_aq].values
        
        # Light steps (A/Q) for current genotype
        light_steps_geno = aq_tmp["Irradiance (µmol m^-2 s^-1)"][idx_geno_aq].values
        light_order_geno = np.argsort(light_steps_geno)
        
        # A/Ci for current genotype
        idx_geno_aci = aci_tmp["Genotype"] == geno_tmp[j]
        aci_geno = aci_tmp["Anet (µmol m^-2 s^-1)"][idx_geno_aci].values
        
        
        
        # Ca steps (A/Ci) for current genotype
        if cica_flag:
            
            # If needed, transform Ci to Ca
            idx_geno_cica = cica_tmp["Genotype"] == geno_tmp[j]
            idx_light = cica_tmp["Irradiance (µmol m^-2 s^-1)"] == light_a_co2[-1]
            
            cica_geno = cica_tmp["CiCa"][idx_geno_cica & idx_light].values
            ci_cica_geno = cica_tmp["Ci (µbar)"][idx_geno_cica & idx_light].values
            ci_cica_geno_order = np.argsort(ci_cica_geno)
        
            # Interpolate Ci/Ca for Ci values in A/Ci curve to get conversion factors
            # For now, piecewice linear interpolation because I don't know what function 
            # the ratios follow.
            # Alternative: scipy.interpolate.CubicSpline
            ci_geno = aci_tmp.loc[idx_geno_aci, "Ci (µbar)"].values
            cica_geno = np.interp(ci_geno, ci_cica_geno[ci_cica_geno_order],
                                  cica_geno[ci_cica_geno_order])
            ca_geno = ci_geno / cica_geno
            
        else:
            ca_geno = aci_tmp.loc[idx_geno_aci, "Ca (µbar)"].values
        
        # Map A/Ci and A/Q curves to steps in the dataset
        ca_geno_order = np.argsort(ca_geno)
        
        # Interpolate Anet for the same CO2 and light steps as in the training
        # dataset
        a_co2.append(np.interp(co2_steps,
                               ca_geno[ca_geno_order],
                               aci_geno[ca_geno_order]).round(2))
        a_light.append(np.interp(light_steps,
                                 light_steps_geno[light_order_geno],
                                 aq_geno[light_order_geno]).round(2))
        
        # Temperature of gas exchange measurements
        exp_temperatures_gas_exchange.append(aci_tmp["Temperature (°C)"].unique()[0])
        
        # Append row labels
        geno_row_labels.append(geno_tmp[j])
        species_row_labels.append(aci_tmp.loc[idx_geno_aci, "Species"].values[0])
        publication_row_labels.append(publication_dirs[i])
        
        # Store measurements of enzyme kinetic and other parameters
    tmp_params = {k: {} for k in geno_tmp}
    for p in p_sheet_names:
        if p in xl.sheet_names:
            if p == "Leakiness":
                df = xl.parse(p, comment="#")
                df_new = pd.DataFrame(columns=df.columns)
                for g in geno_tmp:
                    
                    # Find Leakiness at ambient CO2 used for A/Q curves and maximum 
                    # light intensity
                    geno_ca_idx = np.where((df["Genotype"]==g) \
                                           & (df["Ca (µbar)"]==co2_a_light[i]))[0]
                    geno_ca_max_pfd_idx = np.argmax(
                        df.loc[geno_ca_idx, "Irradiance (µmol m^-2 s^-1)"])
                    # df.loc[geno_ca_idx(geno_ca_max_pfd_idx), "Leakiness (mol m^-2 s^-1)"]
                    # Remove all other rows for this genotype
                    df_new.loc[len(df_new)] = df.loc[geno_ca_idx[geno_ca_max_pfd_idx], :]
                df = df_new.T
            else:
                df = xl.parse(p, comment="#").T                
            df.columns = df.loc["Genotype", :].values
            d = df.to_dict()
            for g in df.columns.values:
                if g not in tmp_params.keys():
                    tmp_params[g] = {}
                tmp_params[g] |= d[g]
                    
        
        
    measured_params[publication_dirs[i]] = tmp_params
        
    xl.close()


#%% Predict parameters

print("Predicting parameters...")

predicted_parameters = {}

for i in range(0, len(publication_dirs)):
    
    key_pub = publication_dirs[i]
    
    env_input = {
        "co2_steps": dataset.co2_steps,
        "light_a_co2": (light_a_co2[i]-light_steps.mean())/light_steps.std(),
        "light_steps": dataset.light_steps,
        "co2_a_light": (co2_a_light[i]-co2_steps.mean())/co2_steps.std()
        }
       
    idx_pub = [j for j, x in enumerate(publication_row_labels) if x == key_pub]
    curve_input = {
        "a_co2": np.array([a_co2[i] for i in idx_pub]),
        "a_light": np.array([a_light[i] for i in idx_pub]),
    }
    
    predicted_parameters[key_pub] = c4tune.predict(curve_input, env_input)
    
    tmp_param_file = os.path.join(result_dir, publication_dirs[i]+"_params.txt")
    
    pd.DataFrame(predicted_parameters[key_pub], columns=param_names,
                 index=[geno_row_labels[i] \
                        for i, p in enumerate(publication_row_labels) if p==key_pub])
    np.savetxt(tmp_param_file, predicted_parameters[key_pub], delimiter=",")
    

#%% Curve simulations

print("Simulating A/Ci and A/Q curves...")

curves_surrogate = {}
curves_ode = {}

for i in range(0, len(publication_dirs)):
    
    print(f"> {publication_dirs[i]}")
    
    key_pub = publication_dirs[i]
    n_geno = sum([x in key_pub for x in publication_row_labels])
    
    env_input = {
        "co2_steps": dataset.co2_steps,
        "light_a_co2": (light_a_co2[i]-light_steps.mean())/light_steps.std(),
        "light_steps": dataset.light_steps,
        "co2_a_light": (co2_a_light[i]-co2_steps.mean())/co2_steps.std()
        }
    
    # Surrogate model
    curves_surrogate[key_pub] = \
        surrogate.predict(predicted_parameters[key_pub], env_input)
    
    # ODE model
    tmp_curves = [np.zeros((n_geno, n_co2)), np.zeros((n_geno, n_light))]

    for j in range(0, n_geno):
        
        if not np.isnan(predicted_parameters[key_pub][j]).all():
            # simulate curves with predicted parameters using ODE model
            try:
                aci_tmp, aq_tmp = c4model.simulate(predicted_parameters[key_pub][j].tolist())
                tmp_curves[0][j, :] = aci_tmp.T
                tmp_curves[1][j, :] = aq_tmp.T
            except:
                print(f"\tCurve {j}: Simulation failed.")

    curves_ode[key_pub] = tmp_curves

#%% Set failed curves to nan
for k in curves_ode.keys():
    for i in [0, 1]:
        curves_ode[k][i][np.any(curves_ode[k][i]>=1e5, axis=1), :] = np.nan

#%% Calculate MSE

mse = {}

for key_pub in publication_dirs:
    
    idx_pub = [j for j, x in enumerate(publication_row_labels) if x == key_pub]
    
    pred_ode = np.concatenate(curves_ode[key_pub], axis=1)
    meas = np.concatenate(
        ([a_co2[i] for i in idx_pub], [a_light[i] for i in idx_pub]), axis=1)
    meas[np.isnan(pred_ode).any(axis=1), :] = np.nan
    pred_ode[np.isnan(meas).any(axis=1), :] = np.nan
    mse[key_pub] = np.nanmean((pred_ode-meas)**2, axis=1)
    
#%% Plot example curves

fig_pdf = os.path.join(result_dir, 'curves_nadpme_species.pdf')

with PdfPages(fig_pdf) as pdf:
    
    for pub_idx in range(0, len(publication_dirs)):
        
        key_pub = publication_dirs[pub_idx]
        idx_pub = [j for j, x in enumerate(publication_row_labels) if x == key_pub]
        
        for curve_idx in range(0, len(predicted_parameters[key_pub])):
            
            genotype = [geno_row_labels[i] for i in idx_pub][curve_idx]
            species = "\ ".join([species_row_labels[i] for i in idx_pub][curve_idx].split(" "))
            
            fig_ex_crv, axs_ex_crv = plt.subplots(1, 2, layout='tight', figsize=(8, 4))
            
            fig_ex_crv.suptitle(", ".join([key_pub, f"$\it{{{species}}}$", genotype]))
            
            # A/Ci
            axs_ex_crv[0].plot(co2_steps, curves_surrogate[key_pub][0][curve_idx])
            axs_ex_crv[0].plot(co2_steps, curves_ode[key_pub][0][curve_idx])
            axs_ex_crv[0].scatter(co2_steps, [a_co2[i] for i in idx_pub][curve_idx],
                                  color='k')
            
            axs_ex_crv[0].legend(["Surrogate Model", "ODE Model", "Experimental"],
                                 ncol=1, frameon=False, loc="lower right")
            
            axs_ex_crv[0].set_xlabel(r"$p(CO_{2})\ (\mu bar)$")
            axs_ex_crv[0].set_ylabel(r"$A_{net}\ (\mu mol\ m^{-2}\ s^{-1})$")
            
            axs_ex_crv[0].text(0.3, 0.4,
                               f"$I = {light_a_co2[pub_idx]}\ \mu mol\ m^{{-2}}\ s^{{-1}}$",
                               fontsize=12,
                               transform=axs_ex_crv[0].transAxes)
            
            # A/Q
            axs_ex_crv[1].plot(light_steps, curves_surrogate[key_pub][1][curve_idx])
            axs_ex_crv[1].plot(light_steps, curves_ode[key_pub][1][curve_idx])
            axs_ex_crv[1].scatter(light_steps, [a_light[i] for i in idx_pub][curve_idx],
                                  color='k')
            axs_ex_crv[1].set_xlabel(r"$light\ intensity\ (\mu mol\ m^{-2}\ s^{-1})$")
            
            axs_ex_crv[1].text(0.3, 0.4,
                               f"$C_a = {co2_a_light[pub_idx]}\ \mu mol\ m^{{-2}}\ s^{{-1}}$",
                               fontsize=12,
                               transform=axs_ex_crv[1].transAxes)
            
            # unify y axes
            ylim_all = [ax.get_ylim() for ax in axs_ex_crv]
            ylim = [np.min(ylim_all), np.max(ylim_all)]
            [ax.set_ylim(ylim) for ax in axs_ex_crv]
            
            pdf.savefig()
            
            plt.close()

#%% Extract predicted model parameters that correspond to measurements

# Define which model parameters should be compared with which measurements

meas2model = {
    "VPMAX": "PEPC Vmax",
    "PEPC":  "PEPC Vmax",
    "NADPMDH": "NADP-MDH Vmax",
    "NADPME": "NADP-ME Vmax", 
    "PPDK": "PPDK Vmax",
    "VCMAX": "RuBisCO Vmax (CO2)",
    "Rubisco": "RuBisCO Vmax (CO2)",
    "PRK": "PRK Vmax",
    "GAPDH": "PGK;GAPDH Vmax",
    "PGK": "PGK;GAPDH Vmax",
    "JMAX": "Jmax",
    "Rd": "mit. respiration",
    "Leakiness": "plasmodesmata Perm (CO2)",
    "Phi_PSII": ["ATPS [MC] X32", "ATPS [MC] F32"]
    }

# Create dict for predictions
pred_params = deepcopy(measured_params)

for i in range(0, len(publication_dirs)):
    
    key_pub = publication_dirs[i]
    keys_geno = measured_params[key_pub].keys()

    for j, key_geno in enumerate(keys_geno):
        
        if key_pub == "Almeida2025":
            if any(x in key_geno for x in ["top", "middle", "bottom"]):
                geno_split = key_geno.split("_")
                key_geno_tmp = "_".join(geno_split[:-1])
                measured_params[key_pub][key_geno] |= measured_params[key_pub][key_geno_tmp]
        
        keys_param = measured_params[key_pub][key_geno].keys()
                
        for key_param in keys_param:
            
            key_param_split = key_param.split(" ")[0].split("_")
            if len(key_param_split) > 1:
                p_string = "_".join(key_param_split[:-1])
            else:
                p_string = key_param_split[0]
            
            # Genotypes in A/Ci curves
            geno_tmp = [geno for geno in keys_geno if geno in geno_row_labels]
            
            if p_string in meas2model.keys() and key_geno in geno_tmp:
                
                key_model_param = meas2model[p_string]
                if p_string == "Phi_PSII":
                    idx_model_param_1 = param_names.index(key_model_param[0])
                    idx_model_param_2 = param_names.index(key_model_param[1])
                    
                    p_pred = predicted_parameters[key_pub][j][idx_model_param_1] \
                        * predicted_parameters[key_pub][j][idx_model_param_2]
                else:
                    idx_model_param = param_names.index(key_model_param)
                    
                    p_pred = predicted_parameters[key_pub][j][idx_model_param]
                
                pred_params[key_pub][key_geno][key_param] = p_pred
            else:
                pred_params[key_pub][key_geno][key_param] = np.nan
            
#%% Plot agreement between measured and predicted parameters per study

cmap = plt.colormaps["tab20c"]

fig_pdf = os.path.join(result_dir, 'parameters_nadpme_species.pdf')

with PdfPages(fig_pdf) as pdf:

    for i in range(0, len(publication_dirs)):
        
        key_pub = publication_dirs[i]
        keys_geno = measured_params[key_pub].keys()
    
        plot_df = pd.DataFrame(columns=["Genotype", "Parameter", "Measured", "Predicted"])
        plot_df_uncertainty = pd.DataFrame(columns=["Genotype", "Parameter", "Min", "Max", "Error"])
        
        for j, key_geno in enumerate(keys_geno):
            
            keys_param = measured_params[key_pub][key_geno].keys()
            
            species = [species_row_labels[i] \
                       for i in range(0, len(publication_row_labels)) \
                           if geno_row_labels[i] == key_geno \
                               and publication_row_labels[i] == key_pub]
            
            for key_param in keys_param:
                
                key_param_split = key_param.split(" ")[0].split("_")
                
                if len(key_param_split) > 1:
                    param_str = "_".join(key_param_split[:-1])
                else:
                    param_str = key_param_split[0]
                
                p_min = np.nan
                p_max = np.nan
                p_err = np.nan
                
                if np.isnan(pred_params[key_pub][key_geno][key_param]):
                    pass
                
                elif any(x in key_param_split for x in ["Q25", "Q75", "SEM", "SE"]):
                    
                    if "Q25" in key_param_split:
                        p_min = measured_params[key_pub][key_geno][key_param]
                    elif "Q75" in key_param_split:
                        p_max = measured_params[key_pub][key_geno][key_param]
                    else:
                        p_err = measured_params[key_pub][key_geno][key_param]
                    
                    plot_df_uncertainty.loc[len(plot_df_uncertainty)] = \
                        [key_geno, param_str, p_min, p_max, p_err]
                else:
                    meas = measured_params[key_pub][key_geno][key_param]
                    pred = pred_params[key_pub][key_geno][key_param]
                    
                    # Convert prediction to FvCB model parameter
                    if any(x in param_str for x in ["VCMAX", "Rubisco"]):
                        if "Zea mays" in species:
                            f = 1000 * 0.67
                        elif "Sorghum bicolor" in species:
                            f = 1000 * 0.56
                        else:
                            f = 1000 * 0.67
                    elif any(x in param_str for x in ["VPMAX", "PEPC"]):
                        if "Zea mays" in species:
                            f = 1000 * 0.72
                        elif "Sorghum bicolor" in species:
                            f = 1000 * 0.68
                        else:
                            f = 1000 * 0.72
                    else:
                        f = 1                        
                    pred = pred * f
                    plot_df.loc[len(plot_df)] = [key_geno, param_str, meas, pred]
                
        for p in plot_df["Parameter"].unique():
            
            p_idx = np.where(plot_df["Parameter"]==p)[0]
            
            tmp_geno = plot_df.loc[p_idx, "Genotype"]
            
            colors = cmap(np.linspace(0, 1, len(tmp_geno)))
            
            if (len(plot_df_uncertainty) > 0) & any(p == x for x in plot_df_uncertainty["Parameter"]):
                unc_geno = plot_df_uncertainty.loc[plot_df_uncertainty["Parameter"]==p]
                
                if not np.isnan(unc_geno["Error"]).any():
                    p_min = plot_df.loc[p_idx, "Measured"].to_numpy() - unc_geno["Error"].to_numpy()
                    p_max = plot_df.loc[p_idx, "Measured"].to_numpy() + unc_geno["Error"].to_numpy()
                elif not np.isnan(unc_geno["Min"]).all():
                    p_min = unc_geno.loc[~np.isnan(unc_geno["Min"]), "Min"].to_numpy()
                    p_max = unc_geno.loc[~np.isnan(unc_geno["Max"]), "Max"].to_numpy()
            else:
                p_min = plot_df.loc[p_idx, "Measured"].to_numpy()
                p_max = plot_df.loc[p_idx, "Measured"].to_numpy()
                
            
            fig = plt.figure(figsize=(4, 4), layout='constrained')
            ax = fig.add_subplot()
            fig.suptitle(", ".join([key_pub]))
            for j in range(0, len(tmp_geno)):
                
                xerr = np.zeros((2, 1))
                yerr = np.zeros((2, 1))
                xerr[:, 0] = np.abs(plot_df.loc[p_idx[j], "Measured"] \
                                    - [p_min[j], p_max[j]])
                ax.errorbar(plot_df.loc[p_idx[j], "Measured"],
                           plot_df.loc[p_idx[j], "Predicted"],
                           xerr=xerr, yerr=yerr,
                           color=colors[j],
                           marker="o")
            
            if any(x == p for x in ["VCMAX", "Rubisco", "VPMAX", "PEPC"]):
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                ax.set_xlim([np.min((xlim, ylim)), np.max((xlim, ylim))])
                ax.set_ylim(ax.get_xlim())
                ax.set_yticks(ax.get_yticks())
                ax.set_yticklabels(ax.get_yticklabels())
                ax.set_xticks(ax.get_yticks())
                ax.set_xticklabels(ax.get_yticklabels())
                
            ax.set_xlabel("Measured")
            ax.set_ylabel("Predicted")
            
            ax.text(0.1, 0.8, p, transform=ax.transAxes)
            if len(tmp_geno) < 10:
                fig.legend(tmp_geno.to_list(),
                          bbox_to_anchor=[0.5, 0],
                          ncol=3,
                          loc='upper center',
                          frameon=False)
            if len(tmp_geno) >= 10:
                fig.legend(tmp_geno.to_list(),
                          bbox_to_anchor=[0.5, 0],
                          frameon=False, ncol=3,
                          loc='upper center')
            
            fig.savefig(os.path.join(result_dir, "param_figs", "_".join([key_pub, p])),
                        bbox_inches='tight', dpi=300)
            pdf.savefig(bbox_inches='tight')
            
            plt.close()
                
                
#%% Identify limiting factors

def find_relevant_correlations(corr_mat, t):
    is_high_corr = np.all(np.abs(corr_mat)>t, axis=1)
    is_consistent = np.abs(np.sum(np.sign(corr_mat), axis=1))==corr_mat.shape[1]
    idx_relevant = np.where(is_high_corr&is_consistent)[0]
    corr_mat_relevant = corr_mat[idx_relevant, :]
    corr_order = np.argsort(np.median(np.abs(corr_mat_relevant), axis=1))[::-1]
    return idx_relevant[corr_order]

def get_corr_threshold(pairwise_corr):
    return np.min(np.abs(pairwise_corr), axis=1)

def add_indicator_high_corr(ax, corr, p_names, corr_threshold):
    yticklabels = [x.get_text() for x in ax.get_yticklabels()]
    param_plot_idx = [p_names.index(x) for x in yticklabels]
    x, y = np.where(abs(corr[param_plot_idx, :])>=corr_threshold)
    ax.scatter(y+0.5, x+0.5, c="gray", s=3)
    return ax

# Correlation between steps of A/CO2 and A/light curves

for p in publication_dirs:
    
    p_idx_all = [i for i in range(0, len(publication_row_labels)) if publication_row_labels[i] == p]
    
    species_pub = [species_row_labels[i] for i in p_idx_all]
    species_uniq = np.unique(species_pub)
    
    for s in species_uniq:
        
        s_idx = [i for i in range(0, len(species_pub)) if species_pub[i]==s]
        p_idx = [p_idx_all[i] for i in s_idx]
        
    
        # pairwise Pearson correlation between steps of A/CO2 and A/light curves
        tmp_a_co2 = [a_co2[i] for i in p_idx]
        tmp_a_light = [a_light[i] for i in p_idx]
        r_p_a_co2 = pd.DataFrame(tmp_a_co2).corr().to_numpy()
        r_p_a_light = pd.DataFrame(tmp_a_light).corr().to_numpy()
        
        # thresholds for relevant correlations
        t_a_co2 = get_corr_threshold(r_p_a_co2)
        t_a_light = get_corr_threshold(r_p_a_light)
        
        # Calculate correlations
        corr_a_co2 = pd.DataFrame(
            np.concatenate((np.log(predicted_parameters[p][s_idx, :]), tmp_a_co2),
                           axis=1)).corr().to_numpy()[:n_params, -n_co2:]
        corr_a_light = pd.DataFrame(
            np.concatenate((np.log(predicted_parameters[p][s_idx, :]), tmp_a_light),
                           axis=1)).corr().to_numpy()[:n_params, -n_light:]
        
        # joint plot with Anet pairwise correlations
        n_plot = 15
    
        fig_corr, axs_corr = plt.subplots(2, 2, figsize=(8, 5), layout='constrained',
                                gridspec_kw={'height_ratios': [1.3, 2]})
        
        fig_corr.suptitle(p+"_"+"_".join(s.split(" ")))
        
        sns.set(rc={'font.size': 10, 'legend.fontsize': 10})
        sns.heatmap(r_p_a_co2, ax=axs_corr[0, 0],
                    xticklabels=[],
                    yticklabels=co2_steps,
                    cbar_kws={"label": "Pearson r", "aspect": 5, "location":"left"},
                    cmap=sns.color_palette("mako", as_cmap=True),
                    vmin=-1.0, vmax=1.0,
                    cbar=True)
        axs_corr[0, 0].set_ylabel(r"$p(CO_{2})\ (\mu bar)$")
    
        axs_corr[0, 1] = sns.heatmap(r_p_a_light, ax=axs_corr[0, 1],
                    xticklabels=[],
                    yticklabels=light_steps,
                    cmap=sns.color_palette("mako", as_cmap=True),
                    vmin=-1.0, vmax=1.0,
                    cbar=False)
        axs_corr[0, 1].set_ylabel("light intensity"
                     "\n"
                     r"$(\mu mol\ m^{-2}\ s^{-1})$")
    
        plot_idx = np.argsort(np.max(np.abs(corr_a_co2), axis=1))[-n_plot:]
        plot_data = corr_a_co2[plot_idx, :]
        linkage = fastcluster.linkage(
            plot_data, method="average", metric="cosine")
        leaf_order = hierarchy.dendrogram(linkage, no_plot=True)['leaves']
        yticklabels = [param_names[i] for i in plot_idx]
        yticklabels = [yticklabels[i] for i in leaf_order]
        axs_corr[1, 0] = sns.heatmap(plot_data[leaf_order, :],
                        ax=axs_corr[1, 0],
                        vmin=-1, vmax=1,
                        cmap=sns.color_palette("mako", as_cmap=True),
                        xticklabels=co2_steps,
                        yticklabels=yticklabels,
                        cbar=False)
        axs_corr[1, 0] = add_indicator_high_corr(axs_corr[1, 0], 
                                                      corr_a_co2, 
                                                      param_names, t_a_co2)
        axs_corr[1, 0].set_xticklabels(axs_corr[1, 0].get_xticklabels(),
                                            rotation=90, ha="center")
        axs_corr[1, 0].set_xlabel(r"$p(CO_{2})\ (\mu bar)$")
    
        plot_idx = np.argsort(np.max(np.abs(corr_a_light), axis=1))[-n_plot:]
        plot_data = corr_a_light[plot_idx, :]
        linkage = fastcluster.linkage(
            plot_data, method="average", metric="cosine")
        leaf_order = hierarchy.dendrogram(linkage, no_plot=True)['leaves'][::-1]
        yticklabels = [param_names[i] for i in plot_idx]
        yticklabels = [yticklabels[i] for i in leaf_order]
        axs_corr[1, 1] = sns.heatmap(plot_data[leaf_order, :],
                        ax=axs_corr[1, 1],
                        vmin=-1, vmax=1,
                        cmap=sns.color_palette("mako", as_cmap=True),
                        xticklabels=light_steps,
                        yticklabels=yticklabels,
                        cbar=False)
        axs_corr[1, 1] = add_indicator_high_corr(axs_corr[1, 1],
                                                      corr_a_light, 
                                                      param_names, t_a_light)
        axs_corr[1, 1].set_xticklabels(axs_corr[1, 1].get_xticklabels(),
                                            rotation=90, ha="center")
        axs_corr[1, 1].set_xlabel("light intensity"
                     "\n"
                     r"$(\mu mol\ m^{-2}\ s^{-1})$")
    
        fig_corr.savefig(os.path.join(result_dir, "corr_param_anet_relevant_"+p+"_"+"_".join(s.split(" "))+".png"), dpi=300)
