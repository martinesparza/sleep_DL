"""
Visualization related testing
author: @mesparza
created on: 03-10-2022
"""
import numpy as np
import pickle
from src.plots import confusion_mat

# Load data
data_path = '/Volumes/GoogleDrive-101271366273470520077/My ' \
            'Drive/PaperEdu/Datos/SleepEDF/Results/Control'

# Initialize confusion variable
confusion = np.zeros((5, 5))
classification = np.zeros((5, 5))


folds = 10
for i in range(1, 2):
    with open(data_path + '/confusion_fold_'+str(i), 'rb') as f:
        confusion += pickle.load(f)

for i in range(5):
    confusion[i] = np.round (confusion[i]/np.sum(confusion[i]),3)

confusion_mat(confusion)

