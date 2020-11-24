import os, os.path
import sys
import pickle
import numpy as np
from sklearn.preprocessing import normalize
from scipy import stats

# -----------------------------------------------------------------------------
# write_to_file() - Small auxiliary function to write data to a file
# -----------------------------------------------------------------------------
def write_to_file(dic, filename):
    with open('material/saved_data/{}.txt'.format(filename), 'wb') as write_f:
        pickle.dump(dic, write_f)
    return

# -----------------------------------------------------------------------------
# read_from_file() - Small auxiliary function to read data from a file
# -----------------------------------------------------------------------------
def read_from_file(filename):
    with open('material/saved_data/{}.txt'.format(filename), 'rb') as read_f:
        return pickle.load(read_f) 

# -----------------------------------------------------------------------------
# normalize_dic() - Normalizes a dic 
# -----------------------------------------------------------------------------
def normalize_dic(dic, **kwargs):
    result_dic = {}

    values = dic.values()
    value_it = iter(values)

    if type(next(value_it)) == str:
        values_list = np.array([len(doc) for doc in values])
    else:
        values_list = np.array([score for score in values])
    
    if 'norm_method' not in kwargs or kwargs['norm_method'] == '1': 
        values_list = values_list / np.linalg.norm(values_list)

    elif kwargs['norm_method'] == '2':
        values_list = normalize(values_list[:,np.newaxis], axis=0).ravel()

    elif kwargs['norm_method'] == 'zscore':
        values_list = stats.zscore(values_list)

    keys = list(dic.keys())
    for i in range(len(keys)):
        result_dic[keys[i]] = values_list[i]

    return result_dic
