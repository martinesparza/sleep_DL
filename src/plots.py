"""
Plotting functions
author: @mesparza
created on: 03-10-2022
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sns.set(font='Arial')
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def confusion_mat(confusion_data, clasification_data, title='',
                  save_path=None, file_name='confusion.eps'):

    for i in range(5):
        confusion_data[i] = np.round(confusion_data[i] / np.sum(
            confusion_data[i]), 3)

    acc = []
    for i in range(len(clasification_data)):
        acc.append(clasification_data[i]['accuracy'])
    acc = np.mean(acc)

    fig = plt.figure(constrained_layout=True, figsize=(5, 5))
    ax = sns.heatmap(confusion_data, annot=True, cmap='binary', fmt='.2f',
                     vmin=0,
                     vmax=1, xticklabels=['W', 'N1', 'N2', 'N3', 'R'],
                     yticklabels=['W', 'N1', 'N2', 'N3', 'R'],
                     cbar=True);
    ax.set_ylabel('True Label', labelpad=10)
    ax.set_xlabel('Predicted Label', labelpad=10)
    fig.suptitle(title+" {:.3f}".format(acc))

    if save_path != None:
        plt.savefig(save_path+file_name, dpi=300)

    return None

def bar_plot(classification, title='', save_path=None,
             file_name='barplot.eps'):
    fig = plt.figure(constrained_layout=False, figsize=(6, 3))
    df = pd.DataFrame({'Stage': [], 'Value': [], 'Metric': []})
    ignore_keys = ['accuracy', 'macro avg', 'weighted avg']
    for i in range(len(classification)):
        for key in classification[i].keys() - ignore_keys:
            for key2 in classification[i][key].keys() - ['support']:
                df.loc[len(df.index)] = [key, classification[i][key][key2],
                                         key2]
    with sns.axes_style("whitegrid"):
        ax = sns.barplot(data=df, x="Stage", y='Value', hue="Metric",
                    order=["Wake", "N1", "N2", "N3", "REM"], palette='binary',
                    capsize=0.1, errwidth=2)
        plt.legend([], [], frameon=False)
        ax.set_ylabel('')
        ax.set_ylim((0, 1))
        for bar in ax.patches:
            bar.set_zorder(3)
    fig.suptitle(title)
    #sns.move_legend(ax, "upper right")

    if save_path != None:
        plt.savefig(save_path + file_name, dpi=300)


    return None

def plot_epochs_metrics(csvs, save_path=None, file_name='lineplot.eps'):
    csvs_mean = np.mean(np.array(csvs), axis=0)
    csvs_se = stats.tstd(csvs, axis=0)

    with sns.axes_style("whitegrid"):
        fig = plt.figure(constrained_layout=False, figsize=(6, 3))
        ax = fig.add_subplot(111)
        plt.plot(csvs_mean[:, 0], csvs_mean[:, 1], color='black', linewidth=2)
        plt.plot(csvs_mean[:, 0], csvs_mean[:, 3], color='#1B73EB', linewidth=2)
        plt.fill_between(csvs_mean[:, 0], csvs_mean[:, 1] - csvs_se[:, 1],
                         csvs_mean[:, 1] + csvs_se[:, 1], alpha=0.2,
                         edgecolor=None, facecolor='black')
        plt.fill_between(csvs_mean[:, 0], csvs_mean[:, 3] - csvs_se[:, 3],
                         csvs_mean[:, 3] + csvs_se[:, 3], alpha=0.2,
                         edgecolor=None, facecolor='#1B73EB')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
    if save_path != None:
        plt.savefig(save_path + file_name, dpi=300)

    return None