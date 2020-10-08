import sklearn
import spacy
import nltk
import whoosh
import os
import glob

def inverted_index(directory):
    os.chdir(directory)
    files = glob.glob('*.txt')
    print(files)

inverted_index('lab2_material')