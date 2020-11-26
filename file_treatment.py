# --------------------------------
# 2nd Delivery of the PRI project
# 86379 - Ana Evans
# 86389 - Artur Guimarães
# 86417 - Francisco Rosa
# --------------------------------
import os, os.path
import sys
import pickle
import re
from bs4 import BeautifulSoup

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

#--------------------------------------------------
# get_xml_files_recursively - Auxiliary function to get_files_from_directory
#
# Input: path - The path to the parent directory or file from which to start our recursive function
#               
# Behaviour: Creates a list with the path to every file that's an hierarquical child of parent directory path,
# recursively going through each child in Post-Order traversing
#
# Output: A List with the paths to each file child
#--------------------------------------------------
def get_xml_files_recursively(path, judged_documents, **kwargs):
    files_list = []
    directory_list = os.listdir(path)
    for f in directory_list:
        n_path = '{}{}/'.format(path,f)
        if os.path.isdir(n_path):
            files_list.extend(get_xml_files_recursively(n_path, **kwargs))

        elif 'judged' in kwargs and kwargs['judged']:
                if int(f.split('news')[0]) in judged_documents:
                    files_list.append(re.sub('//','/','{}/{}'.format(path,f)))
        else:
            files_list.append(re.sub('//','/','{}/{}'.format(path,f)))
    return files_list

# -------------------------------------------------
# get_files_from_directory - Recursively gets all files from directory or file path, parsing the files from xml to objects
# and spliting them in D_Test and D_Train in the conditions specified by our project
#
# Input: path - The path to the parent directory or file from which to start our search
#               
# Behaviour: It starts by creating a list with the path to every file that's an hierarquical child of parent directory path,
# recursively going through each child in Post-Order traversing. Afterwards it parses each and every file from xml to a runtime
# object using the BeautifulSoup library. At last after having all files in object form it splits the dataset in D_Test and D_Train
# sets, according to their identifier (D_Test -> identifier > 1996-09-30   D_Train -> identifier <= 1996-09-30)
#
# Output: A List with the Lists of file objects present in D_Test and D_Train
# -------------------------------------------------
def get_files_from_directory(path, judged_documents, **kwargs):
    file_list = get_xml_files_recursively(path, judged_documents, **kwargs)

    parsed_files_test = []
    parsed_files_train = []

    #TODO: You can remove this afterwards, just makes things faster
    if 'set' in kwargs and kwargs['set'] == 'test':
        for f in file_list:
            date_identifier = int(f.split('/')[2])

            if date_identifier <= 19960930:
                continue

            open_file = open(f, 'r')
            parsed_file = BeautifulSoup(open_file.read(), 'lxml')
            
            if parsed_file.copyright != None:
                parsed_file.copyright.decompose()

            if parsed_file.codes != None:
                parsed_file.codes.decompose()
                
            parsed_files_test += [parsed_file,]

    elif 'set' in kwargs and kwargs['set'] == 'train':
        for f in file_list:
            date_identifier = int(f.split('/')[2])

            if date_identifier > 19960930:
                break

            open_file = open(f, 'r')
            parsed_file = BeautifulSoup(open_file.read(), 'lxml')
            
            if parsed_file.copyright != None:
                parsed_file.copyright.decompose()

            if parsed_file.codes != None:
                parsed_file.codes.decompose()
                
            parsed_files_train += [parsed_file,]

    else:
        for f in file_list:
            date_identifier = int(f.split('/')[2])

            open_file = open(f, 'r')
            parsed_file = BeautifulSoup(open_file.read(), 'lxml')
            
            if parsed_file.copyright != None:
                parsed_file.copyright.decompose()

            if parsed_file.codes != None:
                parsed_file.codes.decompose()
                
            if date_identifier <= 19960930:
                parsed_files_train += [parsed_file,]
            else:
                parsed_files_test += [parsed_file,]

    return (parsed_files_test, parsed_files_train)