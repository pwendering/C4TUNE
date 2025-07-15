"""

Determine partial correlations between Anet measurements and parameter 
predictions for the population of maize genotypes.

"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin
from multiprocessing import Pool
from functools import partial
from omegaconf import OmegaConf

sys.path.append(str(Path().resolve().parents[0]))

from src.utils.utils import load_param_names

def empirical_pvalue(corr_ref, p_idx, params, anet, n_samples_pval=1000):
    
    n_acc = params.shape[0]
    
    # compute empirical p-values
    corr_counter = 0
    params_permuted = params.copy()
    corr_data_permuted = pd.concat([
        pd.DataFrame(np.log(params_permuted), index=anet.index),
        anet], axis=1)
    
    for i in range(0, n_samples_pval):
        
        # permute parameter values across accessions
        param_rand_idx = np.random.choice(np.arange(0, n_acc), n_acc, replace=False)

        corr_data_permuted.iloc[:, p_idx] = params_permuted[param_rand_idx, p_idx]
        
        # calculate partial correlation
        tmp_corr = corr_data_permuted.pcorr().iloc[p_idx, -1]
        corr_counter += np.int8(tmp_corr>=corr_ref)

    # empirical p-value: probability that correlation with permuted parameter
    # values exceeds correlation with true parameters
    return (corr_counter+1)/(n_samples_pval+1)

def pcorr_params_anet(params, anet, ncpu=6):
    
    n_params = params.shape[1]
    n_co2 = anet.shape[1]
    
    pcorr = np.zeros((n_co2, n_params))
    pval = np.zeros((n_co2, n_params))
    
    for i in range(0, n_co2):
        
        print(f"Step {i+1}")
        
        corr_data = pd.concat([
            pd.DataFrame(np.log(params), index=anet.index), anet.iloc[:, i]],
            axis=1)
        
        # calculate partial correlation (Pearson)
        pcorr[i, :] = corr_data.pcorr().iloc[0:-1, -1]
        
        if ncpu > 1:
            f = partial(empirical_pvalue, params=params, anet=anet.iloc[:, i])
            with Pool(ncpu) as p:
                tmp_pval = p.starmap(f, zip(pcorr[i, :], np.arange(0, n_params)))
            pval[i, :] = tmp_pval
        else:
            for j in range(0, n_params):
                pval[i, j] = empirical_pvalue(pcorr[i, j], j, params, anet.iloc[:, i])
    
    return pcorr, pval

def plot_clustermap(corr_data, pval, xticklabels, param_names, cb=False,
                    row_cluster=True):
    
    # significance level
    alpha = 0.05
    
    # font size
    sns.set(font_scale=1.5)
    
    # color bar labels
    cbar_kws = {"label": "Pearson r"}
    
    # identify parameters with the most significant p-values
    plot_idx = np.argsort(np.sum(pval<alpha, axis=0))[-20:]
    
    # remove non-significant correlations
    plot_data = corr_data[:, plot_idx].T
    plot_data[pval[:, plot_idx].T>=alpha] = np.nan
    corr_data[pval>=alpha] = np.nan
    
    if not row_cluster or np.any(np.isnan(corr_data)):
        row_cluster = False
    else:
        row_cluster = True
    
    # plot clustermap
    g = sns.clustermap(corr_data[:, plot_idx].T,
                       col_cluster=False,
                       row_cluster=row_cluster,
                       cbar_kws=cbar_kws,
                       method='average',
                       metric='cosine',
                       xticklabels=xticklabels,
                       yticklabels=[param_names[i] for i in plot_idx],
                       cmap=sns.color_palette("mako", as_cmap=True),
                       figsize=(8, 8),
                       vmin=-1.0, vmax=1.0)
    
    # axes labels
    g.ax_heatmap.set_ylabel(r"$parameters$")
    
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(),
                                 rotation=45,
                                 ha="right")
    
    # remove dendrogram
    g.ax_row_dendrogram.set_visible(False)
    
    # remove colorbar
    if not cb:
        g.cax.set_visible(False)
    
    return g

def main(config, n_samples_pval):
    
    result_dir = os.path.join(config.paths.base_dir, "results",
                             "analysis_limiting_factors", "partial_correlation")
    param_dir = os.path.join(config.paths.base_dir, "results",
                             "parameter_prediction_maize_genotypes")
    data_dir = os.path.join(config.paths.base_dir, "data",
                             "anet_measurements")
    
    param_names = load_param_names(config)
    
    #%% Load predicted parameters
    params_2022 = pd.read_csv(os.path.join(param_dir, "params_2022.csv"), index_col=0).to_numpy()
    params_2023 = pd.read_csv(os.path.join(param_dir, "params_2023.csv"), index_col=0).to_numpy()

    #%% Load average Anet measurements
    a_co2_2022 = pd.read_csv(os.path.join(data_dir, "a_co2_maize_2022.csv"), index_col=0)
    a_co2_2023 = pd.read_csv(os.path.join(data_dir, "a_co2_maize_2023.csv"), index_col=0)
    a_light_2022 = pd.read_csv(os.path.join(data_dir, "a_light_maize_2022.csv"), index_col=0)
    a_light_2023 = pd.read_csv(os.path.join(data_dir, "a_light_maize_2023.csv"), index_col=0)
    
    co2_steps = a_co2_2022.columns.values
    light_steps = a_light_2022.columns.values
    
    #%% Partial (Pearson) correlations between log-transformed parameters and Anet
    
    print("A/CO2 2022")
    pcorr_a_co2_2022, pval_a_co2_2022 = pcorr_params_anet(params_2022, a_co2_2022)
    np.savetxt(os.path.join(result_dir, "pcorr_a_co2_2022_n" + str(n_samples_pval) + ".csv"),
               pcorr_a_co2_2022, delimiter=",")
    np.savetxt(os.path.join(result_dir, "pval_a_co2_2022_n" + str(n_samples_pval) + ".csv"),
               pval_a_co2_2022, delimiter=",")
    
    print("A/light 2022")
    pcorr_a_light_2022, pval_a_light_2022 = pcorr_params_anet(params_2022, a_light_2022)
    np.savetxt(os.path.join(result_dir, "pcorr_a_light_2022_n" + str(n_samples_pval) + ".csv"),
               pcorr_a_light_2022, delimiter=",")
    np.savetxt(os.path.join(result_dir, "pval_a_light_2022_n" +str(n_samples_pval) + ".csv"),
               pval_a_light_2022, delimiter=",")

    print("A/CO2 2023")
    pcorr_a_co2_2023, pval_a_co2_2023 = pcorr_params_anet(params_2023, a_co2_2023)
    np.savetxt(os.path.join(result_dir, "pcorr_a_co2_2023_n" + str(n_samples_pval) + ".csv"),
               pcorr_a_co2_2023, delimiter=",")
    np.savetxt(os.path.join(result_dir, "pval_a_co2_2023_n" +str(n_samples_pval) + ".csv"),
               pval_a_co2_2023, delimiter=",")

    print("A/light 2023")
    pcorr_a_light_2023, pval_a_light_2023 = pcorr_params_anet(params_2023, a_light_2023) 
    np.savetxt(os.path.join(result_dir, "pcorr_a_light_2023_n" + str(n_samples_pval) + ".csv"),
               pcorr_a_light_2023, delimiter=",")
    np.savetxt(os.path.join(result_dir, "pval_a_light_2023_n" + str(n_samples_pval) + ".csv"),
               pval_a_light_2023, delimiter=",")

    # Create figures
    g_a_co2_param_pcorr_2022 = plot_clustermap(pcorr_a_co2_2022, pval_a_co2_2022,
                                    co2_steps, param_names, cb=True)
    g_a_co2_param_pcorr_2022.ax_heatmap.set_xlabel(r"$p(CO_{2})\ (\mu bar)$")

    g_a_light_param_pcorr_2022 = plot_clustermap(pcorr_a_light_2022, pval_a_light_2022,
                                      light_steps, param_names)
    g_a_light_param_pcorr_2022.ax_heatmap.set_xlabel(r"$light\ intensity\ (\mu mol\ m^{-2}\ s^{-1})$")

    g_a_co2_param_pcorr_2023 = plot_clustermap(pcorr_a_co2_2023, pval_a_co2_2023,
                                    co2_steps, param_names)
    g_a_co2_param_pcorr_2023.ax_heatmap.set_xlabel(r"$p(CO_{2})\ (\mu bar)$")

    g_a_light_param_pcorr_2023 = plot_clustermap(pcorr_a_light_2023, pval_a_light_2023,
                                    light_steps, param_names)
    g_a_light_param_pcorr_2023.ax_heatmap.set_xlabel(r"$light\ intensity\ (\mu mol\ m^{-2}\ s^{-1})$")


    # Save figures
    g_a_co2_param_pcorr_2022.savefig(
        os.path.join(result_dir,
                     "partial_correlation_parameters_a_co2_2022_n" + str(n_samples_pval) + ".png"),
        dpi=300)
    g_a_light_param_pcorr_2022.savefig(
        os.path.join(result_dir,
                     "partial_correlation_parameters_a_light_2022_n" + str(n_samples_pval) + ".png"),
        dpi=300)
    g_a_co2_param_pcorr_2023.savefig(
        os.path.join(result_dir,           
                     "partial_correlation_parameters_a_co2_2023_n" + str(n_samples_pval) + ".png"),
        dpi=300)
    g_a_light_param_pcorr_2023.savefig(
        os.path.join(result_dir,
                     "partial_correlation_parameters_a_light_2023_n" + str(n_samples_pval) + ".png"),
        dpi=300)
    

if __name__ == '__main__':
    
    np.random.seed(123)
    torch.manual_seed(321)

    plt.rc('font', size=14)
    plt.rc('legend', fontsize=10)
    
    # load base configuration
    base_config_file = "../config/base.yaml"
    base_config = OmegaConf.load(base_config_file)
    
    main(base_config, n_samples_pval=10000)

    
    
    