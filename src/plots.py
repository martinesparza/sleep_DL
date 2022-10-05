"""
Plotting functions
author: @mesparza
created on: 03-10-2022
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set(font='Arial')
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def confusion_mat(confusion_data, clasification_data, title='', save=False,
                  viz=True):

    for i in range(5):
        confusion_data[i] = np.round(confusion_data[i] / np.sum(
            confusion_data[i]), 3)

    acc = []
    for i in range(len(clasification_data)):
        acc.append(clasification_data[i]['accuracy'])
    acc = np.mean(acc)

    fig = plt.figure(constrained_layout=False, figsize=(5, 5))
    ax = sns.heatmap(confusion_data, annot=True, cmap='binary', fmt='.2f',
                     vmin=0,
                     vmax=1, xticklabels=['W', 'N1', 'N2', 'N3', 'R'],
                     yticklabels=['W', 'N1', 'N2', 'N3', 'R'])
    ax.set_ylabel('True Label', labelpad=10)
    ax.set_xlabel('Predicted Label', labelpad=10)
    fig.suptitle(title+" {:.2f}".format(acc))
    if viz:
        plt.show()

    plt.savefig(save+'confusion_control2.eps', dpi=300)

    return None

