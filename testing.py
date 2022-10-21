"""
Visualization related testing
author: @mesparza
created on: 03-10-2022
"""

from src.plots import confusion_mat, bar_plot, plot_epochs_metrics
from src.settings import SleepEDF_data_path, DODH_data_path, \
    ISRUC_data_path, DODO_data_path, SleepEDF_raw_path, Temp_figures_path
import numpy as np
import pickle


if __name__ == "__main__":
    # Load paths
    datasets = [SleepEDF_data_path, DODH_data_path, ISRUC_data_path,
                DODO_data_path]

    # Initialize variables
    confusion = np.zeros((5, 5))
    classification = []
    csvs = []
    folds = 10

    # Load Data
    for n in range(len(datasets)):
        for i in range(1, folds+1):
            with open(datasets[n] + '/confusion_fold_'+str(i), 'rb') as f:
                confusion += pickle.load(f)
            with open(datasets[n] + '/classification_fold_'+str(i), 'rb') as f:
                classification.append(pickle.load(f))
            '''if n == 0:
                csvs.append(pd.read_csv(datasets[n] + '/SleepEDF_CV_'+str(
                    i)+'.csv').to_numpy())
            elif n == 1:
                csvs.append(pd.read_csv(datasets[n] + '/DREEM_H_CV_' + str(
                    i) + '.csv').to_numpy())
            elif n == 2:
                csvs.append(pd.read_csv(datasets[n] + '/DREEM_H_CV_' + str(
                    i) + '.csv').to_numpy())
            elif n == 3:
                csvs.append(pd.read_csv(datasets[n] + '/DREEM_H_CV_' + str(
                    i) + '.csv').to_numpy())
            else:
                pass'''



    '''plot_epochs_metrics(csvs, save_path=None,
                        file_name='control_lineplot.pdf')'''

    # Confusion matrix
    confusion_mat(confusion_data=confusion,
    clasification_data=classification,
                  title='Control test acc:',
                  save_path=None,
                  file_name='control_confusion.eps')

    # Bar plots
    bar_plot(classification, save_path=Temp_figures_path,
             file_name='Level1_2_barplot.eps', title='Occipital channel')



    print("Done")