# --------------------------------
# 2nd Delivery of the PRI project
# 86379 - Ana Evans
# 86389 - Artur Guimarães
# 86417 - Francisco Rosa
# --------------------------------
import os, os.path
import re
import sys
import time
import nltk
import spacy
import whoosh
import shutil
import sklearn
import math
import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
from heapq import nlargest 
from bs4 import BeautifulSoup
from lxml import etree
from whoosh import index
from whoosh import scoring
from whoosh.qparser import *
from whoosh.fields import *
from sklearn.metrics import *
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

# File imports
from _main_ import get_files_from_directory
from _main_ import process_collection
from proj_utilities import *

# --------------------------------------------------------------------------
# cosine_similarity_list - Computes the cosine similarity between a document
# and a given list of documents
#
# Input: 
#
# Behaviour: 
#
# Output: 
# -----------------------------------------------------------------------------
def cosine_similarity_list(doc, doc_dic, theta, **kwargs):
    result = []
    doc_keys = doc_dic.keys()
    doc_list = []

    for doc in doc_keys:
        doc_list += [doc_dic[doc], ]

    # TODO: **kwargs for norm, min_df and max_df
    vec = TfidfVectorizer()
    doc_list_vectors = vec.fit_transform(doc_list)
    print(doc_list_vectors)
    doc_vector = vec.transform([doc])

    similarity_list = cosine_similarity(doc_vector, doc_list_vectors)
    print(similarity_list)

    return result

# --------------------------------------------------------------------------
# build_graph - Builds a document graph from document collection D using
# the similarity measure in sim agains theta threshold
#
# Input: D - The document collection to build our graph with
#        sim - The similarity measure used
#        theta - The similarity threshold 
#        kwargs -
#
# Behaviour: 
#
# Output: A dictionary representing the weighted undirect graph that
# connect all documents on the basis of the given similarity measure
# -----------------------------------------------------------------------------
def build_graph(D, sim, theta, **kwargs):
    #doc_dic = process_collection(D, False, **kwargs)
    doc_dic = read_dic_from_file('test_dic')
    print(doc_dic.keys())

    '''
    graph = {}
    for doc in doc_list:
        graph[int(doc)] = {}

    if sim == None or sim == 'cosine':
        #for doc in doc_list:
        similarity_list = cosine_similarity_list(doc_list['4929'], doc_list, theta, **kwargs)
    '''

    return

# -----------------------------------------------------------------------
# undirected_page_rank - 
#
# Input: 
# 
# Behaviour: 
#
# Output:
# -----------------------------------------------------------------------
def undirected_page_rank(q, D, p, sim, theta, **kwargs):
    return

# --------------------------------------------------------------------------------
# ~ Just the Main Function ~
# --------------------------------------------------------------------------------
def main():
    build_graph(None, 'cosine', 0.3)
    return


main()