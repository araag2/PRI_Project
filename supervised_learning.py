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
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC

# File imports
from _main_ import get_files_from_directory
from _main_ import process_collection
from _main_ import get_judged_docs
from _main_ import get_topics
from _main_ import ranking_page_rank
from _main_ import find_R_test_labels
from _main_ import tfidf_process
from proj_utilities import *

#Global variables
topics = {}
d_train = {}
d_test = {}
r_train = {}
r_test = {}

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
    classifiers = {'multinomialnb': MultinomialNB, 'kneighbors': KNeighborsClassifier, 'perceptron': Perceptron, 'linearsvc' :LinearSVC}
    r_labels = find_R_test_labels(r_train[q])
    
    #TODO: replace with tfidf_process

    norm = 'l2' if 'norm' not in kwargs else kwargs['norm']
    min_df = 2 if 'min_df' not in kwargs else kwargs['min_df']
    max_df = 0.8 if 'max_df' not in kwargs else kwargs['max_df']
    max_features = None if 'max_features' not in kwargs else kwargs['max_features']
    classifier = MultinomialNB() if 'classifier' not in kwargs else kwargs['classifier']

    vec = TfidfVectorizer(norm=norm, min_df=min_df, max_df=max_df, max_features=max_features)

    d_train_vec = vec.fit(d_train)
    #r_labels = vec.fit(r_labels)

    classifier.fit(X=d_train_vec, y=r_labels)

    return classifier

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

    vec = tfidf_process({'doc':d})[0]

    return M.predict_proba(vec)

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

    for q in q_test:
        classifier = training(q, d_train, r_train)
        for d in d_test:
            prob = classify(d, q, classifier)

    return

# --------------------------------------------------------------------------------
# ~ Just the Main Function ~
# --------------------------------------------------------------------------------
def main():
    global topics
    global d_train
    global d_test
    global r_train
    global r_test

    d_set = [None, None]
    d_test = d_set[0]
    d_train = d_set[1]

    r_set = [None, None]
    r_test = r_set[0]
    r_train = r_set[1]

    q_test = []

    evaluate(q_test, d_test, r_test)

    r_train = find_R_test_labels('material/')[1]
    topics = get_topics('material/')

    return


main()