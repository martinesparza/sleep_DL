"""
Plotting functions
author: @mesparza
created on: 03-10-2022
"""

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(font='Arial')

def confusion_mat(data):

    fig = plt.figure(constrained_layout=False, figsize=(5, 5))
    ax = sns.heatmap(data, annot=True, cmap='binary', fmt='.2f', vmin=0,
                     vmax=1)
    plt.show()

    return None

