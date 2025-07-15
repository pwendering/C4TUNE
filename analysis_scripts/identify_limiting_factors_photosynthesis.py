# -*- coding: utf-8 -*-
"""

Identify parameters that are potentially limiting photosynthesis in a population 
of maize genotypes. The relevant parameters are determined by their correlation
to Anet measurements and whether it is sign consistent and above the minimum
correlation threshold at each CO2 or light level.

"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.cluster import hierarchy
import torch
from torch import FloatTensor
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator, FormatStrFormatter
import seaborn as sns
import fastcluster
from omegaconf import OmegaConf

sys.path.append(str(Path().resolve().parents[0]))

from src.models.model_c4tune import ParameterPredictionModel
from src.prediction.c4tune_predictor import C4tunePredictor
from src.utils.env_setup import set_training_environment, get_config
from src.utils.utils import load_param_names
from src.data.data import PhotResponseDataset
from src.c4_kinetic_model.c4model import C4DynamicModel


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

def simulate_parameter_updates(P, corr_mat, idx_relevant, idx_acc, c4model, p_all=None):
    
    p_all_flag = p_all is not None
        
    n_targets = len(idx_relevant)
    a_co2 = np.zeros((n_targets, len(c4model.co2_steps_simulation)))
    a_light = np.zeros((n_targets, len(c4model.light_steps_simulation)))
    
    for i in range(n_targets):
        
        # generate updated parameter set
        params_inc = P[idx_acc, :].copy()
        if corr_mat[idx_relevant[i], :].min()<0:
            if p_all_flag:
                params_inc[idx_relevant[i]] = p_all[:, idx_relevant[i]].min()
            else:
                params_inc[idx_relevant[i]] = P[:, idx_relevant[i]].min()
        else:
            if p_all_flag:
                params_inc[idx_relevant[i]] = p_all[:, idx_relevant[i]].max()
            else:
                params_inc[idx_relevant[i]] = P[:, idx_relevant[i]].max()

        # Simulate A/CO2 and A/light curves
        aci_tmp, aq_tmp = c4model.simulate(params_inc.tolist())
        a_co2[i, :] = aci_tmp.copy()
        a_light[i, :] = aq_tmp.copy()
    
    return a_co2, a_light

def explore_parameter_range(p_acc, p_all, corr_mat, idx_relevant):
    '''
    Test improvement of changing a parameter in the direction of the correlation 
    over a range of values.
    '''
    
    n_steps = 10
    
    if corr_mat[idx_relevant, :].min()<0:
        p_end = p_all[:, idx_relevant].min()
    else:
        p_end = p_all[:, idx_relevant].max()
    steps = np.linspace(p_acc[idx_relevant], p_end, n_steps)
    a_co2 = np.zeros((n_steps, len(c4model.light_steps_simulation)))
    a_light = np.zeros((n_steps, len(c4model.co2_steps_simulation)))
    
    for i in range(n_steps):
        
        # generate updated parameter set
        p_acc[idx_relevant] = steps[i]

        # Simulate A/CO2 and A/light curves
        aci_tmp, aq_tmp = c4model.simulate(p_acc.tolist())
        a_co2[i, :] = aci_tmp[0, :].copy()
        a_light[i, :] = aq_tmp[0, :].copy()
        
    return a_co2, a_light, steps   

def get_plot_idx(a_co2_ref, a_light_ref, a_co2_updated, a_light_updated):

    plot_idx = np.argmax(np.max(
        np.concatenate((a_co2_updated, a_light_updated), axis=1)
               - np.concatenate((a_co2_ref, a_light_ref), axis=1), axis=1))
    return plot_idx

def plot_updated_curves(ax, a_net_ref, a_net_updated, x_levels,
                        colors=['#377eb8', '#ff7f00', 'gray'], line_styles=['-', '--', '-']):
    
    if len(a_net_ref.shape)==1:
        a_net_ref = np.expand_dims(a_net_ref, 1)
    
    if len(a_net_updated.shape)==1:
        a_net_updated = np.expand_dims(a_net_updated, 1)
        
    if len(x_levels.shape)==1:
        x_levels = np.expand_dims(x_levels, 1)
    
    # a_net_change = 100*(a_net_updated/a_net_ref.T-1)
    a_net_change = a_net_updated-a_net_ref.T
    
    ax.plot(x_levels.T, a_net_ref, colors[0], linestyle=line_styles[0], linewidth=2)
    ax.plot(x_levels.T, a_net_updated.T, colors[1], linestyle=line_styles[1], linewidth=2)  
    
    ax_right = ax.twinx()
    ax_right.plot(x_levels.T, a_net_change.T, colors[2], linestyle=line_styles[2],
                  linewidth=2, alpha=0.5)
    
    ax.set_xlim(np.min(x_levels), np.max(x_levels))
    
    return ax, ax_right

def add_letter(ax, letter):
    ax.text(0.1, 0.87, letter, fontsize=10, ha='left', transform=ax.transAxes,
            weight="bold")
    return ax

def create_result_df(P, param_names, corr_mat, idx_relevant, idx_acc, a_co2_change,
                     a_light_change, a_co2_diff, a_light_diff, all_params):
    
    p_names = [param_names[i] for i in idx_relevant]
    
    updated_values = np.zeros((len(idx_relevant)))
    potential_values = np.zeros((len(idx_relevant)))
    for i in range(len(idx_relevant)):
        if corr_mat[idx_relevant[i], :].min()<0:
            updated_values[i] = P[:, idx_relevant[i]].min()
            potential_values[i] = all_params[:, idx_relevant[i]].min()
        else:
            updated_values[i] = P[:, idx_relevant[i]].max()
            potential_values[i] = all_params[:, idx_relevant[i]].max()

    df = pd.DataFrame({"Parameter": p_names,
                  "Predicted value": P[idx_acc, idx_relevant].T,
                  "Updated value":  updated_values,
                  "Potential value": potential_values,
                  "Pearson r": corr_mat[
                      idx_relevant,
                      np.argmax(np.abs(corr_mat[idx_relevant, :]), axis=1)],
                  "A/CO2 change (%)": 
                      [f"{x:.2f}-{y:.2f}"
                       for x, y in zip(a_co2_change.min(axis=1), a_co2_change.max(axis=1))], 
                  "A/light change (%)": 
                      [f"{x:.2f}-{y:.2f}"
                       for x, y in zip(a_light_change.min(axis=1), a_light_change.max(axis=1))],
                  "A/CO2 change (umol/m2/s)":
                      [f"{x:.2f}-{y:.2f}"
                       for x, y in zip(a_co2_diff.min(axis=1), a_co2_diff.max(axis=1))],
                  "A/light change (umol/m2/s)":
                      [f"{x:.2f}-{y:.2f}"
                       for x, y in zip(a_light_diff.min(axis=1), a_light_diff.max(axis=1))]
                      })
    return df

def get_plot_settings():
    sns.set(rc={'font.size': 14, 'legend.fontsize': 12, 'axes.labelsize': 14})
    sns.set_style("white")
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True
    text_fsz = 12
    return text_fsz

def get_percent_increase(a, b):
    return 100*(a/b-1)

def plot_anet_improvements_targets_individual(
        a_co2, a_light, pred_params, corr_a_co2, corr_a_light, 
        idx_relevant_a_co2, idx_relevant_a_light, p_names,
        co2_steps, light_steps, c4model, p_minmax=None, acc_performance='min'):
    
    # convert steps into numeric arrays
    co2_steps = np.array(co2_steps, dtype='int')
    # co2_steps = np.expand_dims(np.array(co2_steps, dtype='int'), 1).T
    light_steps = np.array(light_steps, dtype='int')
    # light_steps = np.expand_dims(np.array(light_steps, dtype='int'), 1).T

    # find accession with minimum overall Anet values across both curves
    if acc_performance == 'min':
        acc_order_idx = 0  # accession with overall minimum Anet
    elif acc_performance == 'average':
        acc_order_idx = int(np.floor(a_co2_2022.shape[0]/2))  # medium performing accession
    elif acc_performance == 'max':
        acc_order_idx = a_co2_2022.shape[0]-1  # accession with highest overall Anet
    
    median_anet = np.median(np.concatenate((a_co2, a_light), axis=1), axis=1)
    acc_idx = np.argsort(median_anet)[acc_order_idx]
    
    # ODE simulation with predicted parameter for the experimental measurement
    a_co2_ode_ref, a_light_ode_ref = c4model.simulate(pred_params[acc_idx, :].tolist())
    
    # replace each parameter identified in correlation analysis with the maximum or 
    # minimum across all accessions in the population and year
    if acc_performance != 'max':
        p_minmax = None
        print('Replacing parameter with min/max value of the predictions.')
        
    else:
        print('Replacing parameter with min/max value of the dataset.')
        
    a_co2_ode_targets_aco2, a_light_ode_targets_aco2 = simulate_parameter_updates(
        pred_params.copy(), corr_a_co2, idx_relevant_a_co2, acc_idx, c4model, p_minmax)
    
    a_co2_ode_targets_alight, a_light_ode_targets_alight = simulate_parameter_updates(
        pred_params.copy(), corr_a_light, idx_relevant_a_light, acc_idx, c4model,
        p_minmax)
    
    # get/set plotting parameters
    text_fsz = get_plot_settings()
    line_styles = ['-', '--', '-']
    colors = ['black', 'lightcoral', 'gray']
    
    # initialize figure
    fig, axs = plt.subplots(2, 2, layout='constrained', figsize=(6, 6))
    
    # define paramter that should be plotted (maximum change across all steps)
    idx_target_plot_a_co2 = get_plot_idx(
        a_co2_ode_ref, a_light_ode_ref,
        a_co2_ode_targets_aco2, a_light_ode_targets_aco2)

    # A/CO2 curve with updated targets from A/CO2 curve correlation
    axs[0, 0], ax_right_11 = plot_updated_curves(
        axs[0, 0], a_co2_ode_ref, 
        a_co2_ode_targets_aco2[idx_target_plot_a_co2, :].T,
        co2_steps, colors=colors)
    axs[0, 0].set_xticks([])
    
    axs[0, 0].text(0.95, 0.1,
                   p_names[idx_relevant_a_co2[idx_target_plot_a_co2]],
                   fontsize=text_fsz, ha='right', 
                   transform=axs[0, 0].transAxes)
    
    # A/light curve with updated targets from A/CO2 curve correlation
    axs[0, 1], ax_right_12 = plot_updated_curves(
        axs[0, 1], a_light_ode_ref,
        a_light_ode_targets_aco2[idx_target_plot_a_co2, :], 
        light_steps, colors=colors)
    axs[0, 1].set_xticks([])
    
    # A/CO2 curve with updated targets from A/light curve correlation
    idx_target_plot_a_light = get_plot_idx(
        a_co2_ode_ref, a_light_ode_ref, 
        a_co2_ode_targets_alight, a_light_ode_targets_alight)
    
    axs[1, 0], ax_right_21 = plot_updated_curves(
        axs[1, 0], a_co2_ode_ref, 
        a_co2_ode_targets_alight[idx_target_plot_a_light, :], 
        co2_steps, colors=colors)
    axs[1, 0].text(0.95, 0.1, p_names[idx_relevant_a_light[idx_target_plot_a_light]],
                               fontsize=text_fsz, ha='right', 
                               transform=axs[1, 0].transAxes)
    
    tmp_xlim = axs[1, 0].get_xlim()
    axs[1, 0].set_xticks(axs[1, 0].get_xticks()[1:],
                         axs[1, 0].get_xticklabels()[1:],
                         rotation=45, ha='center')
    axs[1, 0].set_xlim(tmp_xlim)
    axs[1, 0].set_xlabel(r"$p(CO_{2})\ (\mu bar)$")
    
    # A/light curve with updated targets from A/light curve correlation
    axs[1, 1], ax_right_22 = plot_updated_curves(
        axs[1, 1], a_light_ode_ref, 
        a_light_ode_targets_alight[idx_target_plot_a_light, :],
        light_steps,
        colors=colors)
    tmp_xlim = axs[1, 1].get_xlim()
    axs[1, 1].set_xticks(axs[1, 1].get_xticks()[1:],
                         axs[1, 1].get_xticklabels()[1:],
                         rotation=45, ha='center')
    axs[1, 1].set_xlim(tmp_xlim)
    axs[1, 1].set_xlabel("light intensity"
                         "\n"
                         r"$(\mu mol\ m^{-2}\ s^{-1})$")
    
    fig.supylabel(r"$A_{net}\ (\mu mol\ m^{-2}\ s^{-1})$", y=0.6)
    ax_right_22.yaxis.set_label_position('right')
    ax_right_22.set_ylabel(r"$Difference\ in\ A_{net}\ (\mu mol\ m^{-2}\ s^{-1})$", y=1)
    
    labels = ["predicted", "updated", "difference"]
    handles = [plt.Line2D((0, 0), (0, 0), color=colors[i], linestyle=line_styles[i])
                for i in range(len(colors))]
    axs[1, 1].legend(handles, labels, ncol=1, frameon=False, loc='lower right')
    
    # left y-axes
    ylim_left = (
        np.min([x[0].get_ylim() for x in axs]),
        np.max([x[0].get_ylim() for x in axs])
    )
    
    yticks_left = AutoLocator().tick_values(0, ylim_left[1])
    
    axs[0, 0].set_yticks(yticks_left)
    axs[0, 1].set_yticks(yticks_left, [])
    axs[1, 0].set_yticks(yticks_left)
    axs[1, 1].set_yticks(yticks_left, [])
    
    # right y-axes
    ylim_aco2_targets_right = [
        np.max((np.min((ax_right_11.get_ylim(), ax_right_12.get_ylim(), (0,0))), -100)),
        np.min((np.max((ax_right_11.get_ylim(), ax_right_12.get_ylim())), 100))
        ]
    
    ylim_alight_targets_right = [
        np.max((np.min((ax_right_21.get_ylim(), ax_right_22.get_ylim(), (0,0))), -100)),
        np.min((np.max((ax_right_21.get_ylim(), ax_right_22.get_ylim())), 100))
        ]
    
    yticks_aco2_targets_right = AutoLocator().tick_values(
        np.min((ylim_aco2_targets_right[0], 0.0)), ylim_aco2_targets_right[1])
    ax_right_11.set_yticks(yticks_aco2_targets_right, [])
    ax_right_12.set_yticks(yticks_aco2_targets_right)
    yticks_alight_targets_right = AutoLocator().tick_values(
        np.min((ylim_alight_targets_right[0], 0.0)), ylim_alight_targets_right[1])
    ax_right_21.set_yticks(yticks_alight_targets_right, [])
    ax_right_22.set_yticks(yticks_alight_targets_right)
    
    ax_right_11.set_ylim(ylim_aco2_targets_right)
    ax_right_12.set_ylim(ylim_aco2_targets_right)
    ax_right_21.set_ylim(ylim_alight_targets_right)
    ax_right_22.set_ylim(ylim_alight_targets_right)
    
    sim_results = {
        "a_co2_ode_ref": a_co2_ode_ref,
        "a_light_ode_ref": a_light_ode_ref,
        "a_co2_ode_targets_aco2": a_co2_ode_targets_aco2,
        "a_light_ode_targets_aco2": a_light_ode_targets_aco2,
        "a_co2_ode_targets_alight": a_co2_ode_targets_alight,
        "a_light_ode_targets_alight": a_light_ode_targets_alight
        }
    
    return fig, axs, sim_results, acc_idx

def get_parameter_optimization_summary(sim_results, corr_a_co2, corr_a_light,
                                       idx_relevant_a_co2, idx_relevant_a_light,
                                       acc_idx, pred_params, param_names, p_minmax=None):
    
    
    idx_target_plot_a_co2 = get_plot_idx(
        sim_results['a_co2_ode_ref'], sim_results['a_light_ode_ref'],
        sim_results['a_co2_ode_targets_aco2'],
        sim_results['a_light_ode_targets_aco2'])
    
    idx_target_plot_a_light = get_plot_idx(
        sim_results['a_co2_ode_ref'], sim_results['a_light_ode_ref'], 
        sim_results['a_co2_ode_targets_alight'],
        sim_results['a_light_ode_targets_alight'])
    
    # increases in Anet
    a_co2_increase_aco2 = get_percent_increase(
        sim_results['a_co2_ode_targets_aco2'][idx_target_plot_a_co2],
        sim_results['a_co2_ode_ref'].T)
    
    a_light_increase_aco2 = get_percent_increase(
        sim_results['a_light_ode_targets_aco2'][idx_target_plot_a_co2],
        sim_results['a_light_ode_ref'].T)
    
    a_co2_increase_alight = get_percent_increase(
        sim_results['a_co2_ode_targets_alight'][idx_target_plot_a_light],
        sim_results['a_co2_ode_ref'].T)
    
    a_light_increase_alight = get_percent_increase(
        sim_results['a_light_ode_targets_alight'][idx_target_plot_a_light],
        sim_results['a_light_ode_ref'].T)
    
    if corr_a_co2[idx_relevant_a_co2[idx_target_plot_a_co2], :].min()>0:
        if p_minmax is not None:
            new_value = p_minmax[:, idx_relevant_a_co2[idx_target_plot_a_co2]].max()
        else:
            new_value = pred_params[:, idx_relevant_a_co2[idx_target_plot_a_co2]].max()
    else:
        if p_minmax is not None:
            new_value = p_minmax[:, idx_relevant_a_co2[idx_target_plot_a_co2]].min()
        else:
            new_value = pred_params[:, idx_relevant_a_co2[idx_target_plot_a_co2]].min()
    
    print(f"Updating {param_names[idx_relevant_a_co2[idx_target_plot_a_co2]]} from "
          f"{pred_params[acc_idx, idx_relevant_a_co2[idx_target_plot_a_co2]]:.2f} to {new_value:.2f} "
          f"resulted in a change in Anet\nbetween {a_co2_increase_aco2.min():.2f} "
          f"and {a_co2_increase_aco2.max():.2f}% for A/CO2 curves and\nbetween "
          f"{a_light_increase_aco2.min():.2f} and {a_light_increase_aco2.max():.2f}% "
          "for A/light curves")
    
    if corr_a_light[idx_relevant_a_light[idx_target_plot_a_light], :].min()>0:
        if p_minmax is not None:
            new_value = p_minmax[:, idx_relevant_a_light[idx_target_plot_a_light]].max()
        else:
            new_value = pred_params[:, idx_relevant_a_light[idx_target_plot_a_light]].max()
    else:
        if p_minmax is not None:
            new_value = p_minmax[:, idx_relevant_a_light[idx_target_plot_a_light]].min()
        else:
            new_value = pred_params[:, idx_relevant_a_light[idx_target_plot_a_light]].min()
            
    print(f"Updating {param_names[idx_relevant_a_light[idx_target_plot_a_light]]} from "
          f"{pred_params[acc_idx, idx_relevant_a_light[idx_target_plot_a_light]]:.2f} to {new_value:.2f} "
          f"resulted in a change in Anet\nbetween {a_co2_increase_alight.min():.2f} "
          f"and {a_co2_increase_alight.max():.2f}% for A/CO2 curves and\nbetween "
          f"{a_light_increase_alight.min():.2f} and {a_light_increase_alight.max():.2f}% "
          "for A/light curves")
    
    df_targets_aco2 = create_result_df(pred_params.copy(), param_names, corr_a_co2,
                     idx_relevant_a_co2, acc_idx,
                     get_percent_increase(sim_results['a_co2_ode_targets_aco2'],
                                          sim_results['a_co2_ode_ref']),
                     get_percent_increase(sim_results['a_light_ode_targets_aco2'],
                                          sim_results['a_light_ode_ref']),
                     sim_results['a_co2_ode_targets_aco2'] - sim_results['a_co2_ode_ref'],
                     sim_results['a_light_ode_targets_aco2'] - sim_results['a_light_ode_ref'],
                     p_minmax)
    df_targets_alight = create_result_df(pred_params.copy(), param_names, corr_a_light,
                     idx_relevant_a_light, acc_idx,
                     get_percent_increase(sim_results['a_co2_ode_targets_alight'],
                                          sim_results['a_co2_ode_ref']),
                     get_percent_increase(sim_results['a_light_ode_targets_alight'],
                                         sim_results['a_light_ode_ref']),
                     sim_results['a_co2_ode_targets_alight'] - sim_results['a_co2_ode_ref'],
                     sim_results['a_light_ode_targets_alight'] - sim_results['a_light_ode_ref'],
                     p_minmax)
    
    df_combined = pd.concat((df_targets_aco2, df_targets_alight))
    
    return df_combined
    

def simulate_improvements_targets_all_accessions(
        a_co2, a_light, pred_params, corr_a_co2, corr_a_light, 
        idx_relevant_a_co2, idx_relevant_a_light, co2_steps, light_steps, 
        c4model, p_minmax=None):
    
    n_accessions = a_co2.shape[0]
    n_targets_aco2 = len(idx_relevant_a_co2)
    n_targets_alight = len(idx_relevant_a_light)
    
    n_co2 = len(co2_steps)
    n_light = len(light_steps)
    
    a_co2_ode_ref = np.zeros((n_accessions, n_co2))
    a_light_ode_ref = np.zeros((n_accessions, n_light))
    
    a_co2_ode_targets_aco2 = np.zeros((n_accessions, n_targets_aco2, n_co2))
    a_light_ode_targets_aco2 = np.zeros((n_accessions, n_targets_aco2, n_light))
    a_co2_ode_targets_alight = np.zeros((n_accessions, n_targets_alight, n_co2))
    a_light_ode_targets_alight = np.zeros((n_accessions, n_targets_alight, n_light))
    
    if p_minmax is not None:
        print('Replacing parameter with min/max value of the dataset.')
        
    for i in range(n_accessions):
        
        # ODE simulation with predicted parameter for the experimental measurement
        tmp_a_co2, tmp_a_light = c4model.simulate(pred_params[i, :].tolist())
        a_co2_ode_ref[i, :] = tmp_a_co2[0, :].copy()
        a_light_ode_ref[i, :] = tmp_a_light[0, :].copy()
        
        # replace each parameter identified in correlation analysis with the maximum or 
        # minimum across all accessions in the population and year            
        a_co2_ode_targets_aco2[i, :, :], a_light_ode_targets_aco2[i, :, :] = \
            simulate_parameter_updates(
            pred_params.copy(), corr_a_co2, idx_relevant_a_co2, i,
            c4model, p_minmax)
        
        a_co2_ode_targets_alight[i, :, :], a_light_ode_targets_alight[i, :, :] = \
            simulate_parameter_updates(
            pred_params.copy(), corr_a_light, idx_relevant_a_light, i,
            c4model, p_minmax)

        if (i+1)%10 == 0:
            print(f"Done with {i+1} accessions.")
    
    sim_results = {
        "a_co2_ode_ref": a_co2_ode_ref,
        "a_light_ode_ref": a_light_ode_ref,
        "a_co2_ode_targets_aco2": a_co2_ode_targets_aco2,
        "a_light_ode_targets_aco2": a_light_ode_targets_aco2,
        "a_co2_ode_targets_alight": a_co2_ode_targets_alight,
        "a_light_ode_targets_alight": a_light_ode_targets_alight
        }
    
    return sim_results
    

def plot_sim_results_all_accessions(sim_results, param_names, a_co2, a_light,
                                    idx_relevant_a_co2, idx_relevant_a_light,
                                    co2_steps, light_steps):
    
    n_co2 = a_co2.shape[1]
    n_light = a_light.shape[1]
    n_accessions = a_co2.shape[0]

    # select best target
    anet_diff_a_co2_targets = np.concatenate((
        sim_results['a_co2_ode_targets_aco2']-np.expand_dims(sim_results['a_co2_ode_ref'], 1),
        sim_results['a_light_ode_targets_aco2']-np.expand_dims(sim_results['a_light_ode_ref'], 1)),
        axis=2)
    median_anet_diff_a_co2_targets = np.nanmedian(anet_diff_a_co2_targets, axis=(0, 2))
    idx_best_target_a_co2 = np.argmax(median_anet_diff_a_co2_targets)
    
    anet_diff_a_light_targets = np.concatenate((
        sim_results['a_co2_ode_targets_alight']-np.expand_dims(sim_results['a_co2_ode_ref'], 1),
        sim_results['a_light_ode_targets_alight']-np.expand_dims(sim_results['a_light_ode_ref'], 1)),
        axis=2)
    median_anet_diff_a_light_targets = np.nanmedian(anet_diff_a_light_targets, axis=(0,2))
    idx_best_target_a_light = np.argmax(median_anet_diff_a_light_targets)
    # idx_best_target_a_light = np.argsort(median_anet_diff_a_light_targets)[-2]  # to select the second best target
    
    anet_data = np.concatenate((a_co2, a_light), axis=1)
    _, a_net_bins = np.histogram(np.median(anet_data, axis=1), bins=5)
    cdata = np.digitize(np.median(anet_data, axis=1), a_net_bins)
    
    # get/set plotting parameters
    get_plot_settings()
    text_fsz = 10
    
    def plot_anet_changes(x, y, c, ax):
        ax.scatter(np.tile(x, (n_accessions, 1)), y, c=c,
                    cmap=sns.color_palette("mako", as_cmap=True),
                    alpha=0.5, marker="s")
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        return ax
    
    fig, axs = plt.subplots(2, 3, figsize=(5, 7),
                            layout='constrained',
                            gridspec_kw={'width_ratios': [6, 6, 1]})
    
    axs[0, 0] = plot_anet_changes(co2_steps,
                                  anet_diff_a_co2_targets[:, idx_best_target_a_co2, :n_co2],
                                  np.repeat(cdata, n_co2), axs[0, 0])
    
    axs[0, 1] = plot_anet_changes(light_steps,
                                  anet_diff_a_co2_targets[:, idx_best_target_a_co2, n_co2:],
                                  np.repeat(cdata, n_light), axs[0, 1])
    
    axs[1, 0] = plot_anet_changes(co2_steps,
                                  anet_diff_a_light_targets[:, idx_best_target_a_light, :n_co2],
                                  np.repeat(cdata, n_co2), axs[1, 0])
    
    axs[1, 1] = plot_anet_changes(light_steps,
                                  anet_diff_a_light_targets[:, idx_best_target_a_light, n_co2:],
                                  np.repeat(cdata, n_light), axs[1, 1])
    
    axs[0, 1].text(0.95, 0.9, param_names[idx_relevant_a_co2[idx_best_target_a_co2]],
                               fontsize=text_fsz, ha='right', 
                               transform=axs[0, 1].transAxes)
    if idx_best_target_a_co2 != idx_best_target_a_light:
        axs[1, 1].text(0.95, 0.9, param_names[idx_relevant_a_light[idx_best_target_a_light]],
                                   fontsize=text_fsz, ha='right', 
                                   transform=axs[1, 1].transAxes)
    
    axs[1, 0].set_xlabel(r"$p(CO_{2})\ (\mu bar)$")
    axs[1, 1].set_xlabel("light intensity"
                         "\n"
                         r"$(\mu mol\ m^{-2}\ s^{-1})$")
    
    # add colorbar
    axs[0, 2].axis('off')
    axs[1, 2].axis('off')
    cbar_kwargs = {"label": r"$median\ measured\ A_{net}\ (\mu mol\ m^{-2}\ s^{-1})$",
                   "aspect": 40,
                   "pad":-0.05
                   }
    cbar = fig.colorbar(axs[1, 1].collections[0], ax=axs, **cbar_kwargs)
    cbar.ax.set_yticks(range(1, len(a_net_bins)+1))
    cbar.ax.set_yticklabels([f"{x:.1f}" for x in a_net_bins.round(2)])
    
    tmp_ylim = [
        np.min((axs[0, 0].get_ylim(), axs[0, 1].get_ylim())),
        np.max((axs[0, 0].get_ylim(), axs[0, 1].get_ylim()))
        ]
    axs[0, 0].set_ylim(tmp_ylim)
    axs[0, 1].set_ylim(tmp_ylim)
    
    tmp_ylim = [
        np.min((axs[1, 0].get_ylim(), axs[1, 1].get_ylim())),
        np.max((axs[1, 0].get_ylim(), axs[1, 1].get_ylim()))
        ]
    axs[1, 0].set_ylim(tmp_ylim)
    axs[1, 1].set_ylim(tmp_ylim)
    
    axs[0, 0].set_xticklabels([])
    axs[0, 1].set_xticklabels([])
    axs[0, 1].set_yticklabels([])
    axs[1, 1].set_yticklabels([])
    
    fig.supylabel(r"$Difference\ in\ A_{net}\ (\mu mol\ m^{-2}\ s^{-1})$",
                  y=0.5, ha='center', x=-0.05)
    
    return fig, axs
    
def filter_sim_results(sim_results):
    
    for k in sim_results.keys():
        if len(sim_results[k].shape)==2:
            more_then_one_zero_idx = np.sum(sim_results[k]==0, axis=1)>1
            infeasible_idx = np.any(sim_results[k]>70, axis=1)
            remove_idx = more_then_one_zero_idx | infeasible_idx
            sim_results[k][remove_idx, :] = np.nan
        if len(sim_results[k].shape)>2:
            for i in range(sim_results[k].shape[1]):
                more_then_one_zero_idx = np.sum(sim_results[k][:, i, :]==0, axis=1)>1
                infeasible_idx = np.any(sim_results[k][:, i, :]>70, axis=1)
                remove_idx = more_then_one_zero_idx | infeasible_idx
                sim_results[k][remove_idx, i, :] = np.nan
    
    return sim_results
    
def write_optimization_summary_all_acc_excel(
        file_name, sim_results, idx_relevant_a_co2, idx_relevant_a_light,
        param_names, co2_steps, light_steps):
    
    with pd.ExcelWriter(file_name) as writer:
        
        # differences
        anet_diff_a_co2_targets = np.concatenate((
            sim_results['a_co2_ode_targets_aco2']-np.expand_dims(sim_results['a_co2_ode_ref'], 1),
            sim_results['a_light_ode_targets_aco2']-np.expand_dims(sim_results['a_light_ode_ref'], 1)),
            axis=2)
        
        anet_diff_a_light_targets = np.concatenate((
            sim_results['a_co2_ode_targets_alight']-np.expand_dims(sim_results['a_co2_ode_ref'], 1),
            sim_results['a_light_ode_targets_alight']-np.expand_dims(sim_results['a_light_ode_ref'], 1)),
            axis=2)
        
        a_net_diff = np.concatenate((anet_diff_a_co2_targets, anet_diff_a_light_targets), axis=1)
        
        # percent changes
        anet_ratios_a_co2_targets = np.concatenate((
            get_percent_increase(sim_results['a_co2_ode_targets_aco2'],
                                 np.expand_dims(sim_results['a_co2_ode_ref'], 1)),
            get_percent_increase(sim_results['a_light_ode_targets_aco2'],
                                 np.expand_dims(sim_results['a_light_ode_ref'], 1))),
            axis=2)
        
        anet_ratios_a_light_targets = np.concatenate((
            get_percent_increase(sim_results['a_co2_ode_targets_alight'],
                                 np.expand_dims(sim_results['a_co2_ode_ref'], 1)),
            get_percent_increase(sim_results['a_light_ode_targets_alight'],
                                 np.expand_dims(sim_results['a_light_ode_ref'], 1))),
            axis=2)
        
        a_net_ratios = np.concatenate((anet_ratios_a_co2_targets, anet_ratios_a_light_targets), axis=1)
        
        comb_relevant_idx = np.concatenate((idx_relevant_a_co2, idx_relevant_a_light))
        tmp_df = pd.concat(
            {
                "A/CO2 (umol/m2/s)": pd.DataFrame(
                    {"min": np.nanmin(a_net_diff[:, :, :n_co2], axis=(0,2)),
                     "max": np.nanmax(a_net_diff[:, :, :n_co2], axis=(0,2)),
                     "median": np.nanmedian(a_net_diff[:, :, :n_co2], axis=(0,2))
                    }, index=[param_names[i] for i in comb_relevant_idx]),
                "A/light (umol/m2/s)": pd.DataFrame(
                    {"min": np.nanmin(a_net_diff[:, :, n_co2:], axis=(0,2)),
                     "max": np.nanmax(a_net_diff[:, :, n_co2:], axis=(0,2)),
                     "median": np.nanmedian(a_net_diff[:, :, n_co2:], axis=(0,2))
                    }, index=[param_names[i] for i in comb_relevant_idx]),
                "both curves": pd.DataFrame(
                    {"min": np.nanmin(a_net_diff, axis=(0,2)),
                     "max": np.nanmax(a_net_diff, axis=(0,2)),
                     "median": np.nanmedian(a_net_diff, axis=(0,2))
                    }, index=[param_names[i] for i in comb_relevant_idx])
                }, axis=1)
        
        tmp_df.to_excel(writer, 
                        sheet_name="Differences",
                        float_format='%.2f', startrow=0, index_label="Parameter")
        
        
        tmp_df = pd.concat(
            {
                "A/CO2 (%)": pd.DataFrame(
                    {"min": np.nanmin(a_net_ratios[:, :, :n_co2], axis=(0,2)),
                     "max": np.nanmax(a_net_ratios[:, :, :n_co2], axis=(0,2)),
                     "median": np.nanmedian(a_net_ratios[:, :, :n_co2], axis=(0,2))
                    }, index=[param_names[i] for i in comb_relevant_idx]),
                "A/light (%)": pd.DataFrame(
                    {"min": np.nanmin(a_net_ratios[:, :, n_co2:], axis=(0,2)),
                     "max": np.nanmax(a_net_ratios[:, :, n_co2:], axis=(0,2)),
                     "median": np.nanmedian(a_net_ratios[:, :, n_co2:], axis=(0,2))
                    }, index=[param_names[i] for i in comb_relevant_idx]),
                "both curves": pd.DataFrame(
                    {"min": np.nanmin(a_net_ratios, axis=(0,2)),
                     "max": np.nanmax(a_net_ratios, axis=(0,2)),
                     "median": np.nanmedian(a_net_ratios, axis=(0,2))
                    }, index=[param_names[i] for i in comb_relevant_idx])
                }, axis=1)
        
        tmp_df.to_excel(writer, 
                        sheet_name="Percent changes",
                        float_format='%2.2f', startrow=0, index_label="Parameter")
        
        
        tmp_df = pd.concat(
            {
                "median difference A/CO2 (umol/m2/s)": pd.DataFrame(
                     np.nanmedian(a_net_diff[:, :, :n_co2], axis=0),
                     columns=co2_steps,
                     index=[param_names[i] for i in comb_relevant_idx]),
                "median difference A/light (umol/m2/s)": pd.DataFrame(
                    np.nanmedian(a_net_diff[:, :, n_co2:], axis=0),
                    columns=light_steps,
                    index=[param_names[i] for i in comb_relevant_idx])
            }, axis=1)
        
        tmp_df.to_excel(writer, 
                        sheet_name="Differences stepwise",
                        float_format='%.2f', startrow=0, index_label="Parameter")
    
    
        tmp_df = pd.concat(
            {
                "median change A/CO2 (%)": pd.DataFrame(
                     np.nanmedian(a_net_ratios[:, :, :n_co2], axis=0),
                     columns=co2_steps,
                     index=[param_names[i] for i in comb_relevant_idx]),
                "median change A/light (%)": pd.DataFrame(
                    np.nanmedian(a_net_ratios[:, :, n_co2:], axis=0),
                    columns=light_steps,
                    index=[param_names[i] for i in comb_relevant_idx])
            }, axis=1)
        
        tmp_df.to_excel(writer, 
                        sheet_name="Percent changes stepwise",
                        float_format='%.2f', startrow=0, index_label="Parameter")

np.random.seed(123)
torch.manual_seed(321)

plt.rc('font', size=14)
plt.rc('legend', fontsize=10)

# load base configuration
base_config_file = "../config/base.yaml"
base_config = OmegaConf.load(base_config_file)

result_dir = os.path.join(base_config.paths.base_dir, "results",
                         "analysis_limiting_factors", "test")
param_dir = os.path.join(base_config.paths.base_dir, "results",
                         "parameter_prediction_maize_genotypes")
data_dir = os.path.join(base_config.paths.base_dir, "data",
                         "anet_measurements")

#%% Load experimental data
a_co2_2022 = pd.read_csv(os.path.join(data_dir, "a_co2_maize_2022.csv"), index_col=0)
a_co2_2023 = pd.read_csv(os.path.join(data_dir, "a_co2_maize_2023.csv"), index_col=0)
a_light_2022 = pd.read_csv(os.path.join(data_dir, "a_light_maize_2022.csv"), index_col=0)
a_light_2023 = pd.read_csv(os.path.join(data_dir, "a_light_maize_2023.csv"), index_col=0)

#%% Load predicted parameters
params_2022 = pd.read_csv(os.path.join(param_dir, "params_2022.csv"), index_col=0).to_numpy()
params_2023 = pd.read_csv(os.path.join(param_dir, "params_2023.csv"), index_col=0).to_numpy()

#%% Create C4TUNE predictor

base_config_file = "../config/base.yaml"
c4tune_config_file = "../config/c4tune.yaml"
config_c4tune = get_config(base_config_file, c4tune_config_file)

c4tune_checkpoint = os.path.join(config_c4tune.paths.run_dir, "2025-03-21", "c4tune-epoch-60.pth")

# Load Cholesky decomposition matrix and change the model's property
L = np.loadtxt(config_c4tune.paths.cholesky_test, delimiter=',')

# create C4TUNE and surrogate model predictors
set_training_environment(config_c4tune)
device = torch.device(config_c4tune.training.device if torch.cuda.is_available() else "cpu")

c4tune_model = ParameterPredictionModel(config_c4tune.model, L=FloatTensor(L))

c4tune = C4tunePredictor(c4tune_model, c4tune_checkpoint, device, config_c4tune)

#%% Load sampling results

# read the dataset and indices of the test set
dataset = PhotResponseDataset(config_c4tune.paths.datasets)
idx_test = np.load(config_c4tune.paths.datasets.test_idx)
co2_steps = np.array(dataset.a_co2.columns)
light_steps = np.array(dataset.a_light.columns)
n_co2 = len(co2_steps)
n_light = len(light_steps)

# parameter names
param_names = load_param_names(base_config)
param_ids = dataset.params.columns.values
n_params = len(param_names)

#%% Correlation between steps of A/CO2 and A/light curves

# pairwise Pearson correlation between steps of A/CO2 and A/light curves
r_p_a_co2_2022 = np.corrcoef(a_co2_2022.T)
r_p_a_light_2022 = np.corrcoef(a_light_2022.T)
r_p_a_co2_2023 = np.corrcoef(a_co2_2023.T)
r_p_a_light_2023 = np.corrcoef(a_light_2023.T)

#%%  Pearson correlation between log-transformed parameters values and Anet 

# thresholds for relevant correlations
t_a_co2_2022 = get_corr_threshold(r_p_a_co2_2022)
t_a_light_2022 = get_corr_threshold(r_p_a_light_2022)
t_a_co2_2023 = get_corr_threshold(r_p_a_co2_2023)
t_a_light_2023 = get_corr_threshold(r_p_a_light_2023)

# Calculate correlations
corr_a_co2_2022 = np.corrcoef(
    np.concatenate((np.log(params_2022), a_co2_2022), axis=1).T)[:n_params, -n_co2:]
corr_a_light_2022 = np.corrcoef(
    np.concatenate((np.log(params_2022), a_light_2022), axis=1).T)[:n_params, -n_light:]
corr_a_co2_2023 = np.corrcoef(
    np.concatenate((np.log(params_2023), a_co2_2023), axis=1).T)[:n_params, -n_co2:]
corr_a_light_2023 = np.corrcoef(
    np.concatenate((np.log(params_2023), a_light_2023), axis=1).T)[:n_params, -n_light:]

# joint plot with Anet pairwise correlations

# number of parameters to plot
n_plot = 15

# 2022
fig_corr_2022, axs_corr_2022 = plt.subplots(2, 2, figsize=(8, 5), layout='constrained',
                        gridspec_kw={'height_ratios': [1.3, 2]})
sns.set(rc={'font.size': 10, 'legend.fontsize': 10})
sns.heatmap(r_p_a_co2_2022, ax=axs_corr_2022[0, 0],
            xticklabels=[],
            yticklabels=co2_steps,
            cbar_kws={"label": "Pearson r", "aspect": 5, "location":"left"},
            cmap=sns.color_palette("mako", as_cmap=True),
            vmin=-1.0, vmax=1.0,
            cbar=True)
axs_corr_2022[0, 0].set_ylabel(r"$p(CO_{2})\ (\mu bar)$")

axs_corr_2022[0, 1] = sns.heatmap(r_p_a_light_2022, ax=axs_corr_2022[0, 1],
            xticklabels=[],
            yticklabels=light_steps,
            cmap=sns.color_palette("mako", as_cmap=True),
            vmin=-1.0, vmax=1.0,
            cbar=False)
axs_corr_2022[0, 1].set_ylabel("light intensity"
             "\n"
             r"$(\mu mol\ m^{-2}\ s^{-1})$")

plot_idx = np.argsort(np.max(np.abs(corr_a_co2_2022), axis=1))[-n_plot:]
plot_data = corr_a_co2_2022[plot_idx, :]
linkage = fastcluster.linkage(
    plot_data, method="average", metric="cosine")
leaf_order = hierarchy.dendrogram(linkage, no_plot=True)['leaves']
yticklabels = [param_names[i] for i in plot_idx]
yticklabels = [yticklabels[i] for i in leaf_order]
axs_corr_2022[1, 0] = sns.heatmap(plot_data[leaf_order, :],
                ax=axs_corr_2022[1, 0],
                vmin=-1, vmax=1,
                cmap=sns.color_palette("mako", as_cmap=True),
                xticklabels=co2_steps,
                yticklabels=yticklabels,
                cbar=False)
axs_corr_2022[1, 0] = add_indicator_high_corr(axs_corr_2022[1, 0], 
                                              corr_a_co2_2022, 
                                              param_names, t_a_co2_2022)
axs_corr_2022[1, 0].set_xticklabels(axs_corr_2022[1, 0].get_xticklabels(),
                                    rotation=90, ha="center")
axs_corr_2022[1, 0].set_xlabel(r"$p(CO_{2})\ (\mu bar)$")

plot_idx = np.argsort(np.max(np.abs(corr_a_light_2022), axis=1))[-n_plot:]
plot_data = corr_a_light_2022[plot_idx, :]
linkage = fastcluster.linkage(
    plot_data, method="average", metric="cosine")
leaf_order = hierarchy.dendrogram(linkage, no_plot=True)['leaves'][::-1]
yticklabels = [param_names[i] for i in plot_idx]
yticklabels = [yticklabels[i] for i in leaf_order]
axs_corr_2022[1, 1] = sns.heatmap(plot_data[leaf_order, :],
                ax=axs_corr_2022[1, 1],
                vmin=-1, vmax=1,
                cmap=sns.color_palette("mako", as_cmap=True),
                xticklabels=light_steps,
                yticklabels=yticklabels,
                cbar=False)
axs_corr_2022[1, 1] = add_indicator_high_corr(axs_corr_2022[1, 1],
                                              corr_a_light_2022, 
                                              param_names, t_a_light_2022)
axs_corr_2022[1, 1].set_xticklabels(axs_corr_2022[1, 1].get_xticklabels(),
                                    rotation=90, ha="center")
axs_corr_2022[1, 1].set_xlabel("light intensity"
             "\n"
             r"$(\mu mol\ m^{-2}\ s^{-1})$")

fig_corr_2022.savefig(os.path.join(result_dir, "corr_param_anet_2022.png"), dpi=300)

# 2023
fig_corr_2023, axs_corr_2023 = plt.subplots(2, 2, figsize=(8, 5), layout='constrained',
                        gridspec_kw={'height_ratios': [1.3, 2]})
sns.set(rc={'font.size': 10, 'legend.fontsize': 10})
axs_corr_2023[0, 0] = sns.heatmap(r_p_a_co2_2023, ax=axs_corr_2023[0, 0],
            xticklabels=[],
            yticklabels=co2_steps,
            cbar_kws={"label": "Pearson r", "aspect": 5, "location":"left"},
            cmap=sns.color_palette("mako", as_cmap=True),
            vmin=-1.0, vmax=1.0,
            cbar=True)
axs_corr_2023[0, 0].set_ylabel(r"$p(CO_{2})\ (\mu bar)$")

axs_corr_2023[0, 1] = sns.heatmap(r_p_a_light_2023, ax=axs_corr_2023[0, 1],
            xticklabels=[],
            yticklabels=light_steps,
            cmap=sns.color_palette("mako", as_cmap=True),
            vmin=-1.0, vmax=1.0,
            cbar=False)
axs_corr_2023[0, 1].set_ylabel("light intensity"
             "\n"
             r"$(\mu mol\ m^{-2}\ s^{-1})$")

plot_idx = np.argsort(np.max(np.abs(corr_a_co2_2023), axis=1))[-n_plot:]
plot_data = corr_a_co2_2023[plot_idx, :]
linkage = fastcluster.linkage(
    plot_data, method="average", metric="cosine")
leaf_order = hierarchy.dendrogram(linkage, no_plot=True)['leaves']
yticklabels = [param_names[i] for i in plot_idx]
yticklabels = [yticklabels[i] for i in leaf_order]
axs_corr_2023[1, 0] = sns.heatmap(plot_data[leaf_order, :],
                ax=axs_corr_2023[1, 0],
                vmin=-1, vmax=1,
                cmap=sns.color_palette("mako", as_cmap=True),
                xticklabels=co2_steps,
                yticklabels=yticklabels,
                cbar=False)
axs_corr_2023[1, 0] = add_indicator_high_corr(axs_corr_2023[1, 0],
                                              corr_a_co2_2023,
                                              param_names, t_a_co2_2023)
axs_corr_2023[1, 0].set_xticklabels(axs_corr_2023[1, 0].get_xticklabels(), 
                                    rotation=45, ha="right")
axs_corr_2023[1, 0].set_xlabel(r"$p(CO_{2})\ (\mu bar)$")


plot_idx = np.argsort(np.max(np.abs(corr_a_light_2023), axis=1))[-n_plot:]
plot_data = corr_a_light_2023[plot_idx, :]
linkage = fastcluster.linkage(
    plot_data, method="average", metric="cosine")
leaf_order = hierarchy.dendrogram(linkage, no_plot=True)['leaves'][::-1]
yticklabels = [param_names[i] for i in plot_idx]
yticklabels = [yticklabels[i] for i in leaf_order]
axs_corr_2023[1, 1] = sns.heatmap(plot_data[leaf_order, :],
                ax=axs_corr_2023[1, 1],
                vmin=-1, vmax=1,
                cmap=sns.color_palette("mako", as_cmap=True),
                xticklabels=light_steps,
                yticklabels=yticklabels,
                cbar=False)
axs_corr_2023[1, 1] = add_indicator_high_corr(axs_corr_2023[1, 1],
                                              corr_a_light_2023,
                                              param_names, t_a_light_2023)
axs_corr_2023[1, 1].set_xticklabels(axs_corr_2023[1, 1].get_xticklabels(), 
                                    rotation=45, ha="right")
axs_corr_2023[1, 1].set_xlabel(r"$light\ intensity\ (\mu mol\ m^{-2}\ s^{-1})$")

fig_corr_2023.savefig(os.path.join(result_dir, "corr_param_anet_2023.png"), dpi=300)

#%% Sign consistent correlations above threshold

# indices of CO2 steps relevant for A/CO2 responses
co2_idx = range(0, n_co2)  # there is no difference beween step 6:end or considering all steps

idx_relevant_a_co2_2022 = find_relevant_correlations(corr_a_co2_2022[:, co2_idx], t_a_co2_2022[co2_idx])
idx_relevant_a_light_2022 = find_relevant_correlations(corr_a_light_2022, t_a_light_2022)
idx_relevant_a_co2_2023 = find_relevant_correlations(corr_a_co2_2023[:, co2_idx], t_a_co2_2023[co2_idx])
idx_relevant_a_light_2023 = find_relevant_correlations(corr_a_light_2023, t_a_light_2023)

p_names_relevant_a_co2_2022 = [param_names[i] for i in idx_relevant_a_co2_2022]
p_names_relevant_a_light_2022 = [param_names[i] for i in idx_relevant_a_light_2022]
p_names_relevant_a_co2_2023 = [param_names[i] for i in idx_relevant_a_co2_2023]
p_names_relevant_a_light_2023 = [param_names[i] for i in idx_relevant_a_light_2023]

# joint plot with Anet pairwise correlations
n_plot = 15

# 2022
fig_corr_2022_relevant, axs_corr_2022_relevant = plt.subplots(2, 2, figsize=(15, 10), layout='constrained',
                        gridspec_kw={'height_ratios': [1.5, 2]})
sns.set(font_scale=1.5)
g = sns.heatmap(r_p_a_co2_2022, ax=axs_corr_2022_relevant[0, 0],
            xticklabels=[],
            yticklabels=co2_steps,
            cbar_kws={"label": "Pearson r"},
            cmap=sns.color_palette("mako", as_cmap=True),
            vmin=-1.0, vmax=1.0,
            cbar=False)
g.set_ylabel(r"$p(CO_{2})\ (\mu bar)$")

g = sns.heatmap(r_p_a_light_2022, ax=axs_corr_2022_relevant[0, 1],
            xticklabels=[],
            yticklabels=light_steps,
            cbar_kws={"label": "Pearson r", "location": "right"},
            cmap=sns.color_palette("mako", as_cmap=True),
            vmin=-1.0, vmax=1.0,
            cbar=True)
g.set_ylabel("light intensity"
             "\n"
             r"$(\mu mol\ m^{-2}\ s^{-1})$")

plot_data = corr_a_co2_2022[idx_relevant_a_co2_2022[:n_plot], :]
linkage = fastcluster.linkage(
    plot_data, method="average", metric="cosine")
leaf_order = hierarchy.dendrogram(linkage, no_plot=True)['leaves'][::-1]
yticklabels = p_names_relevant_a_co2_2022[:n_plot]
yticklabels = [yticklabels[i] for i in leaf_order]
g = sns.heatmap(plot_data[leaf_order, :],
                ax=axs_corr_2022_relevant[1, 0],
                vmin=-1, vmax=1, cbar_kws={"label": "Pearson r"},
                cmap=sns.color_palette("mako", as_cmap=True),
                xticklabels=co2_steps,
                yticklabels=yticklabels,
                cbar=False)
g.set_xticklabels(g.get_xticklabels(), rotation=45, ha="right")
g.set_xlabel(r"$p(CO_{2})\ (\mu bar)$")

plot_data = corr_a_light_2022[idx_relevant_a_light_2022[:n_plot], :]
linkage = fastcluster.linkage(
    plot_data, method="average", metric="cosine")
leaf_order = hierarchy.dendrogram(linkage, no_plot=True)['leaves'][::-1]
yticklabels = p_names_relevant_a_light_2022[:n_plot]
yticklabels = [yticklabels[i] for i in leaf_order]
g = sns.heatmap(plot_data[leaf_order, :],
                ax=axs_corr_2022_relevant[1, 1],
                vmin=-1, vmax=1, cbar_kws={"label": "Pearson r"},
                cmap=sns.color_palette("mako", as_cmap=True),
                xticklabels=light_steps,
                yticklabels=yticklabels,
                cbar=False)
g.set_xticklabels(g.get_xticklabels(), rotation=45, ha="right")
g.set_xlabel(r"$light\ intensity\ (\mu mol\ m^{-2}\ s^{-1})$")

fig_corr_2022_relevant.savefig(os.path.join(result_dir, "corr_param_anet_2022_relevant.png"), dpi=300)

# most limiting (relevant) parameters across both curve types
idx_intersect_2022 = np.intersect1d(idx_relevant_a_co2_2022, idx_relevant_a_light_2022)
median_intersect_a_light_2022 = np.median(corr_a_light_2022[idx_intersect_2022, :], axis=1)
median_intersect_a_co2_2022 = np.median(corr_a_co2_2022[idx_intersect_2022, :], axis=1)
av_corr_intersect_2022 = np.mean([median_intersect_a_light_2022, median_intersect_a_co2_2022], axis=0)
order_corr_intersect_2022 = np.argsort(np.abs(av_corr_intersect_2022))[::-1]
param_names_intersect_2022 = [param_names[i] for i in idx_intersect_2022[order_corr_intersect_2022]]
param_ids_intersect_2022 = [param_ids[i] for i in idx_intersect_2022[order_corr_intersect_2022]]
av_corr_intersect_2022_sort = av_corr_intersect_2022[order_corr_intersect_2022]

# 2023
fig_corr_2023_relevant, axs_corr_2023_relevant = plt.subplots(2, 2, figsize=(15, 10), layout='constrained',
                        gridspec_kw={'height_ratios': [1.5, 2]})
sns.set(font_scale=1.5)
g = sns.heatmap(r_p_a_co2_2023, ax=axs_corr_2023_relevant[0, 0],
            xticklabels=[],
            yticklabels=co2_steps,
            cbar_kws={"label": "Pearson r"},
            cmap=sns.color_palette("mako", as_cmap=True),
            vmin=-1.0, vmax=1.0,
            cbar=False)
g.set_ylabel(r"$p(CO_{2})\ (\mu bar)$")

g = sns.heatmap(r_p_a_light_2023, ax=axs_corr_2023_relevant[0, 1],
            xticklabels=[],
            yticklabels=light_steps,
            cbar_kws={"label": "Pearson r", "location": "right"},
            cmap=sns.color_palette("mako", as_cmap=True),
            vmin=-1.0, vmax=1.0,
            cbar=True)
g.set_ylabel("light intensity"
             "\n"
             r"$(\mu mol\ m^{-2}\ s^{-1})$")

plot_data = corr_a_co2_2023[idx_relevant_a_co2_2023[:n_plot], :]
linkage = fastcluster.linkage(
    plot_data, method="average", metric="cosine")
leaf_order = hierarchy.dendrogram(linkage, no_plot=True)['leaves'][::-1]
yticklabels = p_names_relevant_a_co2_2023[:n_plot]
yticklabels = [yticklabels[i] for i in leaf_order]
g = sns.heatmap(plot_data[leaf_order, :],
                ax=axs_corr_2023_relevant[1, 0],
                vmin=-1, vmax=1, cbar_kws={"label": "Pearson r"},
                cmap=sns.color_palette("mako", as_cmap=True),
                xticklabels=co2_steps,
                yticklabels=yticklabels,
                cbar=False)
g.set_xticklabels(g.get_xticklabels(), rotation=45, ha="right")
g.set_xlabel(r"$p(CO_{2})\ (\mu bar)$")

plot_data = corr_a_light_2023[idx_relevant_a_light_2023[:n_plot], :]
linkage = fastcluster.linkage(
    plot_data, method="average", metric="cosine")
leaf_order = hierarchy.dendrogram(linkage, no_plot=True)['leaves'][::-1]
yticklabels = p_names_relevant_a_light_2023[:n_plot]
yticklabels = [yticklabels[i] for i in leaf_order]
g = sns.heatmap(plot_data[leaf_order, :],
                ax=axs_corr_2023_relevant[1, 1],
                vmin=-1, vmax=1, cbar_kws={"label": "Pearson r"},
                cmap=sns.color_palette("mako", as_cmap=True),
                xticklabels=light_steps,
                yticklabels=yticklabels,
                cbar=False)
g.set_xticklabels(g.get_xticklabels(), rotation=45, ha="right")
g.set_xlabel(r"$light\ intensity\ (\mu mol\ m^{-2}\ s^{-1})$")

fig_corr_2023_relevant.savefig(os.path.join(result_dir, "corr_param_anet_2023_relevant.png"), dpi=300)


# most limiting (relevant) parameters across both curve types
idx_intersect_2023 = np.intersect1d(idx_relevant_a_co2_2023, idx_relevant_a_light_2023)
median_intersect_a_light_2023 = np.median(corr_a_light_2023[idx_intersect_2023, :], axis=1)
median_intersect_a_co2_2023 = np.median(corr_a_co2_2023[idx_intersect_2023, :], axis=1)
av_corr_intersect_2023 = np.mean([median_intersect_a_light_2023, median_intersect_a_co2_2023], axis=0)
order_corr_intersect_2023 = np.argsort(np.abs(av_corr_intersect_2023))[::-1]
param_names_intersect_2023 = [param_names[i] for i in idx_intersect_2023[order_corr_intersect_2023]]
param_ids_intersect_2023 = [param_ids[i] for i in idx_intersect_2023[order_corr_intersect_2023]]
av_corr_intersect_2023_sort = av_corr_intersect_2023[order_corr_intersect_2023]

#%% Update parameters identified by correlation analysis and re-run ODE model simulations

# create Wrapper for C4 kinetic model simulation written in Matlab
c4model = C4DynamicModel(base_config)

# =============================================================================
# Test targets
# =============================================================================

p_minmax = np.array([dataset.params.min().to_numpy(),
                      dataset.params.max().to_numpy()])

vcmax_idx = 169  # Vm6
for year in ["2022", "2023"]:
    
        print(year)
        
        for acc_performance in ["min", "average", "max"]:
        
            print(acc_performance + " accession")
            
            # selected targets
            if year == "2022":
                tmp_fig, _, sim_results, acc_idx = plot_anet_improvements_targets_individual(
                    a_co2_2022, a_light_2022, params_2022, corr_a_co2_2022,
                    corr_a_light_2022, idx_relevant_a_co2_2022, idx_relevant_a_light_2022,
                    param_names, co2_steps, light_steps, c4model,
                    p_minmax=p_minmax, acc_performance=acc_performance)
            else:
                tmp_fig, _, sim_results, acc_idx = plot_anet_improvements_targets_individual(
                    a_co2_2023, a_light_2023, params_2023, corr_a_co2_2023,
                    corr_a_light_2023, idx_relevant_a_co2_2023, idx_relevant_a_light_2023,
                    param_names, co2_steps, light_steps, c4model, 
                    p_minmax=p_minmax, acc_performance=acc_performance)
                    
            
            tmp_fig.savefig(
                os.path.join(result_dir, "anet_updated_params_" + year + "_" + acc_performance + ".png"),
                dpi=300)
            
            if year == "2022":
                tmp_df_combined = get_parameter_optimization_summary(
                        sim_results, corr_a_co2_2022, corr_a_light_2022,
                        idx_relevant_a_co2_2022, idx_relevant_a_light_2022,
                        acc_idx, params_2022, param_names, p_minmax=p_minmax)
            else:
                tmp_df_combined = get_parameter_optimization_summary(
                        sim_results, corr_a_co2_2023, corr_a_light_2023,
                        idx_relevant_a_co2_2023, idx_relevant_a_light_2023,
                        acc_idx, params_2023, param_names, p_minmax=p_minmax)
            
            tmp_df_combined.to_csv(
                os.path.join(result_dir, "param_update_simulation_results_" + 
                             year + "_" + acc_performance + ".csv"))
            print("")
            
            # Rubicso Vcmax
            if year == "2022":
                tmp_fig, _, sim_results, acc_idx = plot_anet_improvements_targets_individual(
                    a_co2_2022, a_light_2022, params_2022, corr_a_co2_2022,
                    corr_a_light_2022, [vcmax_idx], [vcmax_idx],
                    param_names, co2_steps, light_steps, c4model,
                    p_minmax=p_minmax, acc_performance=acc_performance)
            else:
                tmp_fig, _, sim_results, acc_idx = plot_anet_improvements_targets_individual(
                    a_co2_2023, a_light_2023, params_2023, corr_a_co2_2023,
                    corr_a_light_2023, [vcmax_idx], [vcmax_idx],
                    param_names, co2_steps, light_steps, c4model,
                    p_minmax=p_minmax, acc_performance=acc_performance)
            
            tmp_fig.savefig(
                os.path.join(result_dir, "anet_updated_vcmax_" + year + "_" + acc_performance + ".png"),
                dpi=300)
            
            if year == "2022":
                tmp_df_combined = get_parameter_optimization_summary(
                        sim_results, corr_a_co2_2022, corr_a_light_2022,
                        [vcmax_idx], [vcmax_idx], acc_idx, params_2022.numpy(), 
                        param_names, p_minmax=p_minmax)
            else:
                tmp_df_combined = get_parameter_optimization_summary(
                    sim_results, corr_a_co2_2023, corr_a_light_2023,
                    [vcmax_idx], [vcmax_idx], acc_idx, params_2023.numpy(), 
                    param_names, p_minmax=p_minmax)
            
            tmp_df_combined.to_csv(
                os.path.join(result_dir, "vcmax_update_simulation_results_" + 
                             year + "_" + acc_performance + ".csv"))
            print("")

# =============================================================================
# Simulate parameter updates across all genotypes
# =============================================================================
# across all accessions

# selected targets, 2022
sim_results_2022 = simulate_improvements_targets_all_accessions(
    a_co2_2022, a_light_2022, params_2022.numpy(), corr_a_co2_2022,
    corr_a_light_2022, idx_relevant_a_co2_2022, idx_relevant_a_light_2022,
    co2_steps, light_steps, c4model, None)
np.save(os.path.join(result_dir, "param_opt_all_acc_2022"), sim_results_2022)

sim_results_2022 = np.load(os.path.join(result_dir, "param_opt_all_acc_2022.npy"), 
                           allow_pickle=True).item()
sim_results_2022 = filter_sim_results(sim_results_2022)

fig_opt_all_acc_2022, _ = plot_sim_results_all_accessions(
    sim_results_2022, param_names, a_co2_2022, a_light_2022,
    idx_relevant_a_co2_2022, idx_relevant_a_light_2022, co2_steps, light_steps)
fig_opt_all_acc_2022.savefig(os.path.join(result_dir, "anet_change_selected_targets_all_acc_2022.png"),
                              bbox_inches='tight',
                              dpi=300)

file_name = os.path.join(result_dir, "parmeter_optimization_summary_2022.xlsx")
write_optimization_summary_all_acc_excel(
    file_name, sim_results_2022, idx_relevant_a_co2_2022, 
    idx_relevant_a_light_2022, param_names, co2_steps, light_steps)

# Vcmax, 2022
sim_results_vcmax_2022 = simulate_improvements_targets_all_accessions(
    a_co2_2022, a_light_2022, params_2022.numpy(), corr_a_co2_2022,
    corr_a_light_2022, [vcmax_idx], [vcmax_idx], co2_steps, light_steps, c4model, None)
np.save(os.path.join(result_dir, "param_opt_all_acc_vcmax_2022"), sim_results_vcmax_2022)

sim_results_vcmax_2022 = np.load(os.path.join(result_dir, "param_opt_all_acc_vcmax_2022.npy"),
                                 allow_pickle=True).item()
sim_results_vcmax_2022 = filter_sim_results(sim_results_vcmax_2022)

fig_opt_all_acc_vcmax_2022, _ = plot_sim_results_all_accessions(
    sim_results_vcmax_2022, param_names, a_co2_2022, a_light_2022,
    [vcmax_idx], [vcmax_idx])
fig_opt_all_acc_vcmax_2022.savefig(os.path.join(result_dir, "anet_change_vcmax_all_acc_2022.png"),
                                    bbox_inches='tight',
                                    dpi=300)

file_name = os.path.join(result_dir, "parmeter_optimization_vcmax_summary_2022.xlsx")
write_optimization_summary_all_acc_excel(
    file_name, sim_results_vcmax_2022, [vcmax_idx], [vcmax_idx], param_names,
    co2_steps, light_steps)

# selected targets, 2022, dataset values
sim_results_dataset_2022 = simulate_improvements_targets_all_accessions(
    a_co2_2022, a_light_2022, params_2022, corr_a_co2_2022,
    corr_a_light_2022, idx_relevant_a_co2_2022, idx_relevant_a_light_2022,
    co2_steps, light_steps, c4model, p_minmax)
np.save(os.path.join(result_dir, "param_opt_all_acc_dataset_2022"), sim_results_dataset_2022)

sim_results_dataset_2022 = np.load(os.path.join(result_dir, "param_opt_all_acc_dataset_2022.npy"), 
                                          allow_pickle=True).item()
sim_results_dataset_2022 = filter_sim_results(sim_results_dataset_2022)

fig_opt_all_acc_dataset_2022, _ = plot_sim_results_all_accessions(
    sim_results_dataset_2022, param_names, a_co2_2022, a_light_2022,
    idx_relevant_a_co2_2022, idx_relevant_a_light_2022, co2_steps, light_steps)
fig_opt_all_acc_dataset_2022.savefig(
    os.path.join(result_dir, "anet_change_selected_targets_all_acc_dataset_2022.png"),
    dpi=300, bbox_inches='tight')

# Vcmax, 2022, dataset values
sim_results_vcmax_dataset_2022 = simulate_improvements_targets_all_accessions(
    a_co2_2022, a_light_2022, params_2022, corr_a_co2_2022,
    corr_a_light_2022, [vcmax_idx], [vcmax_idx], co2_steps, light_steps, c4model, 
    p_minmax)
np.save(os.path.join(result_dir, "param_opt_all_acc_vcmax_dataset_2022"),
        sim_results_vcmax_dataset_2022)

sim_results_vcmax_dataset_2022 = np.load(
    os.path.join(result_dir, "param_opt_all_acc_vcmax_dataset_2022.npy"), 
    allow_pickle=True).item()
sim_results_vcmax_dataset_2022 = filter_sim_results(sim_results_vcmax_dataset_2022)

fig_opt_all_acc_vcmax_dataset_2022, _ = plot_sim_results_all_accessions(
    sim_results_vcmax_dataset_2022, param_names, a_co2_2022, a_light_2022,
    [vcmax_idx], [vcmax_idx], co2_steps, light_steps)
fig_opt_all_acc_vcmax_dataset_2022.savefig(
    os.path.join(result_dir, "anet_change_vcmax_dataset_all_acc_2022.png"),
    dpi=300, bbox_inches='tight')



# def save_complete_optimization_results_excel(
#         filename, sim_results, idx_relevant_a_co2, idx_relevant_a_light):
    
#     anet_diff_a_co2_targets = np.concatenate((
#         sim_results['a_co2_ode_targets_aco2']-np.expand_dims(sim_results['a_co2_ode_ref'], 1),
#         sim_results['a_light_ode_targets_aco2']-np.expand_dims(sim_results['a_light_ode_ref'], 1)),
#         axis=2)
    
#     anet_diff_a_light_targets = np.concatenate((
#         sim_results['a_co2_ode_targets_alight']-np.expand_dims(sim_results['a_co2_ode_ref'], 1),
#         sim_results['a_light_ode_targets_alight']-np.expand_dims(sim_results['a_light_ode_ref'], 1)),
#         axis=2)
    
#     anet_ratios_a_co2_targets = np.concatenate((
#         sim_results['a_co2_ode_targets_aco2']-np.expand_dims(sim_results['a_co2_ode_ref'], 1),
#         sim_results['a_light_ode_targets_aco2']-np.expand_dims(sim_results['a_light_ode_ref'], 1)),
#         axis=2)
    
#     anet_ratios_a_light_targets = np.concatenate((
#         sim_results['a_co2_ode_targets_alight']-np.expand_dims(sim_results['a_co2_ode_ref'], 1),
#         sim_results['a_light_ode_targets_alight']-np.expand_dims(sim_results['a_light_ode_ref'], 1)),
#         axis=2)
    
    
#     sim_q, q_order, sim_co2, co2_order = get_env_inputs()
#     light_steps = [sim_q[i] for i in q_order]
#     co2_steps = [sim_co2[i] for i in co2_order]
    
    
#     with pd.ExcelWriter(file_name) as writer:
#         for i in range(len(idx_relevant_a_co2_2022)):
#             tmp_df = pd.concat(
#                 {
#                     "A/CO2 (umol/m2/s)": pd.DataFrame(anet_diff_a_light_targets[:, i, :n_co2],
#                                                       columns=co2_steps,
#                                                       index=a_co2_2022.index),
#                     "A/light (umol/m2/s)": pd.DataFrame(anet_diff_a_light_targets[:, i, n_co2:],
#                                                         columns=light_steps,
#                                                         index=a_light_2022.index)
#                 }, axis=1)
#             tmp_df.to_excel(writer, 
#                             sheet_name=param_names[idx_relevant_a_co2_2022[i]].replace("[", "(").replace("]", ")"),
#                             float_format='%.2f', startrow=0, index_label="Accession")


# file_name = os.path.join(result_dir, "anet_diff_targets_results_2022.xlsx")
# save_complete_optimization_results_excel(file_name)