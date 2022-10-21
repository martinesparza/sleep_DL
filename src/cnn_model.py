from models import get_model_cnn
import numpy as np
#from utils import gen, chunker, WINDOW_SIZE, rescale_array
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from glob import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle


#base_path = "/media/ml/data_ml/EEG/deepsleepnet/data_npy"

base_path = "D:\Martin\TFG\ISRUCsleep\out_ISRUC"

files = sorted(glob(os.path.join(base_path, "*.npz")))

ids = sorted(list(set([x.split("\\")[-1][:3] for x in files])))
#split by test subject


train_ids, test_ids = train_test_split(ids, test_size=0.15,random_state = 100)

#Clinic_01, random_state = 1000

train_val, test = [x for x in files if x.split("\\")[-1][:3] in ids],\
                  [x for x in files if x.split("\\")[-1][:3] in test_ids]

train, val = train_test_split(train_val, test_size=0.15,random_state = 1000)


train_dict = {k: np.load(k) for k in train}
test_dict = {k: np.load(k) for k in test}
val_dict = {k: np.load(k) for k in val}

model = get_model_cnn()

file_path = "test_ISRUC_full5.h5"
#model.load_weights(file_path)

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=75, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=10, verbose=2)
csv_logger = CSVLogger('test_ISRUC_full5.csv')
callbacks_list = [checkpoint, early, csv_logger]  # early
#callbacks_list = [checkpoint,csv_logger]


#model.fit_generator(gen(train_dict, aug=False), epochs=100, verbose=1,
                    #steps_per_epoch=2500, validation_steps=200, callbacks=callbacks_list)

model.fit_generator(gen(train_dict, aug=False), validation_data=gen(val_dict), epochs=100, verbose=1,
                    steps_per_epoch=2500, validation_steps=200, callbacks=callbacks_list)
#model.fit_generator(gen(train_dict, aug=False), epochs=15, verbose=1,
 #                   steps_per_epoch=250, callbacks=callbacks_list)


model.load_weights(file_path)


preds = []
gt = []
individual_preds = []
individual_gt = []
total_X = []
total = np.array(total_X)



for record in tqdm(test_dict):
    all_rows = test_dict[record]['x']
    total_X.append(all_rows)
    j = 0
    individual_preds_temp = []
    individual_gt_temp = []
    for batch_hyp in chunker(range(all_rows.shape[0])):


        X = all_rows[min(batch_hyp):max(batch_hyp)+1, ...]
        Y = test_dict[record]['y'][min(batch_hyp):max(batch_hyp)+1]

        X = np.expand_dims(X, 0)

        X = rescale_array(X)

        Y_pred = model.predict(X)
        Y_pred = Y_pred.argmax(axis=-1).ravel().tolist()
        
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
    
    individual_preds.append(individual_preds_temp)
    individual_gt.append(individual_gt_temp)
    



f1 = f1_score(gt, preds, average="macro")

print("Seq Test f1 score : %s "% f1)

acc = accuracy_score(gt, preds)

print("Seq Test accuracy score : %s "% acc)

print(classification_report(gt, preds))

confusion = (confusion_matrix(gt,preds))

for i in range (5):
    print(np.round(confusion[i]/np.sum(confusion[i]),2))
          
'''    
confusion = (confusion_matrix(gt,preds))
classification = classification_report(gt, preds)
   
with open('confusion', 'wb') as f:
    pickle.dump(confusion, f)
        
with open('classification', 'wb') as f:
    pickle.dump(classification, f)
          
'''

# SINGLE CHUNKER
'''
preds = []
gt = []

for record in tqdm(test_dict):
     all_rows = test_dict[record]['x']
     Y = test_dict[record]['y']
     
     gt += Y.ravel().tolist()
     
     
     for i in range(all_rows.shape[0]):
         X = test_dict[record]['x'][i, ...]
         X = np.expand_dims(X, 0)
         X = np.expand_dims(X, 0)
         
         X = rescale_array(X)

         Y_pred = model.predict(X)
         Y_pred = Y_pred.argmax(axis=-1).ravel().tolist()
         
         preds += Y_pred
         
         
         

'''
'''
classification_SEDF07 = classification_report(gt, preds)
confusion_SEDF07 = confusion_matrix(gt,preds)
    
with open('confusion_SEDF07', 'wb') as f:
    pickle.dump(confusion_SEDF07, f)
        
with open('classification_SEDF07', 'wb') as f:
    pickle.dump(classification_SEDF07, f)


test_night = 8

idx_3 = []
idx_1 = []
for i in range(len(individual_gt[test_night])):
    if individual_gt[test_night][i] == 0:
        individual_gt[test_night][i] = 5
    if individual_gt[test_night][i] == 3:
        idx_3.append(i)
    if individual_gt[test_night][i] == 1:
        idx_1.append(i)
        
for i in idx_1:
    individual_gt[test_night][i] = 3
 
for i in idx_3:
    individual_gt[test_night][i] = 1
    
        
idx_3 = []
idx_1 = []
for i in range(len(individual_preds[test_night])):
    if individual_preds[test_night][i] == 0:
        individual_preds[test_night][i] = 5
    if individual_preds[test_night][i] == 3:
        idx_3.append(i)
    if individual_preds[test_night][i] == 1:
        idx_1.append(i)
        
for i in idx_1:
    individual_preds[test_night][i] = 3
 
for i in idx_3:
    individual_preds[test_night][i] = 1
    
        
    
acc = accuracy_score(individual_gt[test_night], individual_preds[test_night])
print("Seq Test accuracy score : %s "% acc)        

fig = plt.figure(figsize=(45,8))
ax = fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(4)
plt.rc('font', size=30)
plt.plot(np.linspace(0,len(individual_gt[test_night])/2,len(individual_gt[test_night])),individual_preds[test_night],color = 'red',label = 'Predicted Hypnogram',linewidth=2.25)
plt.plot(np.linspace(0,len(individual_gt[test_night])/2,len(individual_gt[test_night])),individual_gt[test_night],color = 'black',label = 'Ground Truth',linewidth=4.0)
#plt.xlabel('Min')
#plt.legend(loc="upper right")
#plt.title('Test night: 0'+str(test_night)+', Accuracy = '+str(round(acc,4)))
plt.yticks((1,2,3,4,5), ('','','','',''))
plt.xticks((0,100,200,300,400,500), ('','','','','',''))
plt.xlim(0,500)
plt.show()


acc_list = []
for i in range(0,32):
    temp = accuracy_score(individual_gt[i], individual_preds[i])
    acc_list.append(temp)
    
'''    
    




