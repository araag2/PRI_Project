import os, os.path
import sys
import pickle

# --------------------------------------------------------------------------
# write_to_file() - Small auxiliary function to write a dictionary to
# a file
# -----------------------------------------------------------------------------
def write_to_file(dic, filename):
    with open('material/saved_data/{}.txt'.format(filename), 'wb') as write_f:
        pickle.dump(dic, write_f)
    return

# --------------------------------------------------------------------------
# read_from_file() - Small auxiliary function to read a dictionary from
# a file
# -----------------------------------------------------------------------------
def read_from_file(filename):
    with open('material/saved_data/{}.txt'.format(filename), 'rb') as read_f:
        return pickle.load(read_f) 