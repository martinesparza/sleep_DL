# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 18:42:02 2022

@author: mesparza

DODH all tests
"""

# %% Import libraries

from tD_Global import Global
from models import get_model_cnn
import numpy as np
from supple import gen, chunker, rescale_array, gen_fit
from keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from glob import glob
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
from scipy.io import loadmat


# %% Pick test
# --------------------------------------------------------------------------
# Pick test. 
#   0 = Fpz-Cz (Control)
#   2 = Ppz-Oz
#   3 = Fpz-Cz + Pz-Oz
#   4 = Fpz-Cz + Pz-Oz + EMG + EOG
#   5 = Fpz-Cz (offline)
# --------------------------------------------------------------------------

for test in range(4,5):
    
    print("\n-----------------------------\n Begin test: %s "% test)
    

    # Addpath of preprocessed data
    if test == 0:
        base_path = '.\SleepEDF\Data\Control'
        output_folder = '.\SleepEDF\Results\Control'
        channels = 1 # F3-C1
        online = 1 # online
        
    elif test == 1:
        base_path = '.\SleepEDF\Data\Level 1_1'
        output_folder = '.\SleepEDF\Results\Level 1_1'
        channels = 1 # 
        online = 1 
        
    elif test == 2:
        base_path = '.\SleepEDF\Data\Level 1_2'
        output_folder = '.\SleepEDF\Results\Level 1_2'   
        channels = 1 # O1-M2
        online = 1
        
    elif test == 3:
        base_path = '.\SleepEDF\Data\Level 2'
        output_folder = '.\SleepEDF\Results\Level 2'
        channels = 2 # Fpz-Cz + Pz-Oz
        online = 1
        
    elif test == 4:
        base_path = '.\SleepEDF\Data\Level 3'
        output_folder = '.\SleepEDF\Results\Level 3'
        channels = 4 # EEG (2) + EMG + EOG 
        online = 1
    
    elif test == 5:
        base_path = '.\SleepEDF\Data\Level 4'
        output_folder = '.\SleepEDF\Results\Level 4'
        channels = 1 # F3-M2
        online = 0 # offline
        
    # %% Establish folds
    
    # Number of folds
    folds = 10
    
    # Number of nights
    #n = 25
    
    # Load files and select handles
    files = sorted(glob(os.path.join(base_path, "*.npz")))
    ids = sorted(list(set([x.split("\\")[-1][:5] for x in files])))
    
    # Initialize accuracy list for each fold
    accuracy_list = []
    
    
    # Load cross-validation indices
    cross_val_indx = loadmat('cross_val_indx.mat')
    cross_val_indx = cross_val_indx['cross_val_indx']
    cross_val_indx = cross_val_indx.tolist()
    cross_val_indx = cross_val_indx[0]

    
    # %% Main loop
    
    for i in range(1,folds+1):
        
        print("------ Begin fold: %s "% i)
        
        test_values = []
        train_values = []
        
        # Assign participants to train or test
        for j in range(77):
            if cross_val_indx[j] == i:
                test_values.append(j)
            else:
                train_values.append(j)
                
        
        train_ids = [ids[n] for n in train_values]
        test_ids = [ids[n] for n in test_values]
        
        train_val, test = [x for x in files if x.split("\\")[-1][:5] in train_ids],\
                      [x for x in files if x.split("\\")[-1][:5] in test_ids]
    
        train, val = train_test_split(train_val, test_size=0.20, random_state=1000)
    
        train_dict = {k: np.load(k) for k in train}
        test_dict = {k: np.load(k) for k in test}
        val_dict = {k: np.load(k) for k in val}
        
        # Calculate avg and std to normalize data
        x_cumul = [] 
        for record in tqdm(train_dict):
            x_cumul =+ train_dict[record]['x'].ravel()
            
        avg = np.mean(x_cumul)
        std = np.std(x_cumul)
        
        # Define model and callbacks
        model = get_model_cnn(channels)
        file_path = output_folder + '\SleepEDF_CV_'+ str(i) + '.h5'
        
        checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        csv_logger = CSVLogger(output_folder + '\SleepEDF_CV_'+ str(i) + '.csv')
        callbacks_list = [checkpoint, csv_logger]
        
        # Generate data
        #x_train, y_train = gen_fit(train_dict, avg, std)
    
    
        # Begin training
        model.fit_generator(gen(train_dict, avg, std, aug=False), validation_data=gen(val_dict,avg, std), epochs=100, verbose=1,
                        steps_per_epoch=1000, validation_steps=200, callbacks=callbacks_list)
        #model.fit(x=x_train[:,0:2000,:,:], y=y_train[:,0:2000,:], epochs = 100, verbose = 1, batch_size = 1000)

        # Load best weights
        model.load_weights(file_path)
    
        # Initialize result variables
        preds = []
        gt = []
        individual_preds = []
        individual_gt = []
    
        
        # Iterate over test dictionary
        for record in tqdm(test_dict):
            all_rows = test_dict[record]['x']
    
            j = 0
            individual_preds_temp = [] # Initialize variable to save a single night. Not used in this script. 
            individual_gt_temp = []
            
            # Feed epochs in real-time or in non-overlapping windows
            for batch_hyp in chunker(range(all_rows.shape[0]), online):
        
                X = all_rows[min(batch_hyp):max(batch_hyp)+1, ...]
                Y = test_dict[record]['y'][min(batch_hyp):max(batch_hyp)+1]
        
                X = np.expand_dims(X, 0)
        
                X = rescale_array(X,avg,std)
        
                Y_pred = model.predict(X)
                Y_pred = Y_pred.argmax(axis=-1).ravel().tolist()
                
                if online == 1:
                    # If this is an online test we use all 5 epochs in the first 
                    # iteration and then we only append the last epoch (sliding)
        
                    if j == 0:
                        gt += Y.ravel().tolist()
                        individual_gt_temp += Y.ravel().tolist()
                        
                        preds += Y_pred
                        individual_preds_temp += Y_pred
                    else:
                        individual_preds_temp.append(Y_pred[-1])
                        preds.append(Y_pred[-1])
                        
                        gt.append(Y.ravel().tolist()[-1])
                        individual_gt_temp.append(Y.ravel().tolist()[-1])
                    j += 1
                    
                elif online == 0:
                    # If this is offline we append all new epochs. 
                    gt += Y.ravel().tolist()
                    preds += Y_pred
            
            # for each subject save labels as a separate list. 
            individual_preds.append(individual_preds_temp)
            individual_gt.append(individual_gt_temp)
            
        
        acc = accuracy_score(gt, preds)
        accuracy_list.append(acc)
        
        print("\nSeq Test accuracy score : %s "% acc)
        
        target_names = ['Wake','N1','N2','N3','REM']
        confusion = confusion_matrix(gt,preds)
        
        classification = classification_report(gt, preds,target_names = target_names, output_dict=(True))
        
        with open(output_folder + '\confusion_fold_'+str(i), 'wb') as f:
            pickle.dump(confusion, f)
        with open(output_folder + '\classification_fold_'+str(i), 'wb') as f:
            pickle.dump(classification, f)
            
    print('\n10-fold Cross-Validation average accuracy: %s'% np.mean(accuracy_list))       
            

# %% Load acc. 

accs = []
for i in range(1,11):
     with open('classification_fold_'+str(i), 'rb') as f:
         x = pickle.load(f)
         accs.append(x['accuracy'])
         
print("Acc: "+str(np.mean(accs)))
    
        
        
        
        
    
  