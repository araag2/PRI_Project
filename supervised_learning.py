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
from copy import deepcopy
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
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from textblob import TextBlob

# File imports
from _main_ import get_files_from_directory
from _main_ import process_collection
from _main_ import get_judged_docs
from _main_ import get_topics
from _main_ import ranking_page_rank
from proj_utilities import *

#Global variables
topics = {}


# training 
#
# Input: q - topic document
#        d_train - training collection
#        r_train - judgements
#    
# Behaviour: Learns a classification model to predict the relevance of documents on the
# topic q using Dtrain and Rtrain, where the training process is subjected to
# proper preprocessing, classifier’s selection and hyperparameterization
#    
# Output: q-conditional classification model

def training(q, d_train, r_train, **kwargs):
    
    r_labels = {}
    for doc in r_train:
        r_labels[doc] = r_train[doc]
    
    vectorizer = TfidfVectorizer()

    return

# classify
# 
# Input: d - document
#        q - topic
#        M - classification model
#
# Behaviour: classifies the probability of document d to be relevant for topic q given M
#
# Output: probabilistic classification output on the relevance of document d to the
# topic t

def classify(d, q, M, **kwargs):

    return

# evaluate 
# 
# Input: q_test - subset of topics
#        d_test - testing document collection
#        r_test - judgements
# 
# Behaviour: evaluates the behavior of the IR system in the presence and absence of
# relevance feedback. In the presence of relevance feedback, training and
# testing functions are called for each topic in Qtest for a more comprehensive assessment
#
# Output: performance statistics regarding the underlying classification system and
# the behavior of the aided IR system

def evaluate(q_test, d_test, r_test, **kwargs):

    return

# --------------------------------------------------------------------------------
# ~ Just the Main Function ~
# --------------------------------------------------------------------------------
def main():
    return


main()