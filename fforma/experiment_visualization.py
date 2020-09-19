#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import ast

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

#############################################################################
# PARSE EXPERIMENT
#############################################################################

def load_experiment_df():
    pass

#############################################################################
# DISTRIBUTION PLOTS
#############################################################################

def plot_single_cat_distributions(distributions_dict, ax,
                                  txt_dict=None, fig_title=None, xlabel=None):
    n_distributions = len(distributions_dict.keys())

    n_colors = len(distributions_dict.keys())
    colors = sns.color_palette("hls", n_colors)
    #colors = sns.color_palette("plasma", n_colors)

    for idx, dist_name in enumerate(distributions_dict.keys()):
        if np.var(distributions_dict[dist_name])>0.000001:
            train_dist_plot = sns.distplot(distributions_dict[dist_name],
                                          #bw='silverman',
                                          #kde=False,
                                          #rug=True,
                                          label=dist_name,
                                          color=colors[idx],
                                          ax=ax)
        if txt_dict is not None:
            ax.text(0.5, 0.9, txt_dict[dist_name],
                    horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title(fig_title, fontsize=15.5)
    ax.grid(True)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

def plot_single_distribution(distribution, ax, txt=None,
                             color=None, fig_title=None, xlabel=None):

    dist_plot = sns.distplot(distribution,
                              #bw='silverman',
                              #kde=False,
                              #rug=True,
                              color=color,
                              ax=ax)

    if txt is not None:
      ax.text(0.5, 0.9, txt,
              horizontalalignment='center',
              verticalalignment='center', transform=ax.transAxes)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=14)

    ax.set_ylabel('Density', fontsize=14)
    ax.set_title(fig_title, fontsize=15.5)
    ax.grid(True)
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

def plot_grid_cat_distributions(df, cats, var): #, experiment_name=None
    cols = int(np.ceil(len(cats)/2))
    fig, axs = plt.subplots(2, cols, figsize=(4*cols, 5.5), sharex=True)
    plt.subplots_adjust(wspace=0.95)
    plt.subplots_adjust(hspace=0.5)

    for idx, cat in enumerate(cats):
        unique_cats = df[cat].unique()
        cat_dict = {}
        for c in unique_cats:
            values = df[df[cat]==c][var].values
            values = values[~np.isnan(values)]
            if len(values)>0:
                plot_noise = -np.abs(np.random.normal(loc=0.0,
                                                      scale=0.0001, size=values.shape))
                cat_dict[c] = values + plot_noise

        row = int(np.round((idx/len(cats))+0.001, 0))
        col = idx % cols
        plot_single_cat_distributions(cat_dict, axs[row, col],
                                      fig_title=cat, xlabel=var)

    best_score = math.floor(df[var].max() * 1000) / 1000
    suptitle = 'best ' + var + ': ' + str(best_score)
    fig.suptitle(suptitle, fontsize=18)
    if experiment_name is not None:
        plot_file = './results/plots/{}_{}_exploration.png'.format(experiment_name, var)
        plt.savefig(plot_file, bbox_inches = "tight", dpi=100)
    plt.show()
