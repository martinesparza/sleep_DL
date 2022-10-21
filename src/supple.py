import numpy as np
import random
from tqdm import tqdm

sliding_WINDOW_SIZE = 5
WINDOW_SIZE = 100
slide_velocity = 1
iterations = 100


def rescale_array(X,avg=20,std=3):
    X = X - avg
    X = X / std 
    return X


def aug_X(X):
    scale = 1 + np.random.uniform(-0.1, 0.1)
    offset = np.random.uniform(-0.1, 0.1)
    noise = np.random.normal(scale=0.05, size=X.shape)
    X = scale * X + offset + noise
    return X

def gen(dict_files, avg, std, aug=False):
    while True:
        record_name = random.choice(list(dict_files.keys()))
        batch_data = dict_files[record_name]
        all_rows = batch_data['x']

        for i in range(iterations):
            start_index = random.choice(range(all_rows.shape[0]-WINDOW_SIZE))

            X = all_rows[start_index:start_index+WINDOW_SIZE, ...]
            Y = batch_data['y'][start_index:start_index+WINDOW_SIZE]

            X = np.expand_dims(X, 0)
            Y = np.expand_dims(Y, -1)
            Y = np.expand_dims(Y, 0)

            if aug:
                X = aug_X(X)
            X = rescale_array(X,avg,std)

            yield X, Y
            
def gen_fit(dict_files, avg=20, std=3):
    
    x_train = []
    for record_name in tqdm(dict_files):  
        if record_name == list(dict_files.keys())[0]:
            batch_data = dict_files[record_name]
            x_train = batch_data['x']
            y_train = batch_data['y']
        else:
            x_train = np.vstack([x_train, batch_data['x']])
            y_train = np.append(y_train, batch_data['y'])
            
    x_train = np.expand_dims(x_train, 0)
    y_train = np.expand_dims(y_train, -1)
    y_train = np.expand_dims(y_train, 0)
    
    x_train = rescale_array(x_train,avg,std)
    
    return x_train, y_train
            
            
def chunker(seq, online, size=sliding_WINDOW_SIZE, vel = slide_velocity):
    if online == 1:
        return (seq[pos-size:pos] for pos in range(size, len(seq), vel))
    elif online == 0:
        return (seq[pos:pos + WINDOW_SIZE] for pos in range(0, len(seq), WINDOW_SIZE))


