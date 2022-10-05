"""
Visualization related testing
author: @mesparza
created on: 03-10-2022
"""
import numpy as np
import pickle
from src.plots import confusion_mat
from src.settings import SleepEDF_data_path, DODH_data_path, \
    ISRUC_data_path, DODO_data_path, Temp_figures_path

# Load data
datasets = [SleepEDF_data_path, DODH_data_path, ISRUC_data_path,
            DODO_data_path]

# Initialize confusion variable
confusion = np.zeros((5, 5))
classification = []
folds = 10

for n in range(len(datasets)):
    for i in range(1, folds+1):
        with open(datasets[n] + '/confusion_fold_'+str(i), 'rb') as f:
            confusion += pickle.load(f)
        with open(datasets[n] + '/classification_fold_'+str(i), 'rb') as f:
            classification.append(pickle.load(f))


confusion_mat(confusion_data=confusion, clasification_data=classification,
              title='Control test accuracy:', save=Temp_figures_path)



