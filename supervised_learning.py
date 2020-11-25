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

from file_treatment import read_from_file
from data_set_treatment import tfidf_process
from data_set_treatment import get_R_set
from data_set_treatment import find_R_test_labels
from data_set_treatment import get_topics

#Global variables
topics = {}
d_train = {}
d_test = {}
r_train = {}
r_test = {}
topic_vectorizers = {}

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
    global topic_vectorizers

    classifiers = {'multinomialnb': MultinomialNB, 'kneighbors': KNeighborsClassifier}
    classifier = classifiers['multinomialnb']() if 'classifier' not in kwargs else classifiers[kwargs['classifier']]()

    r_labels = find_R_test_labels(r_train[q])

    subset_dtrain = {}
    for doc in r_labels:
        subset_dtrain[doc] = d_train[doc]

    vec_results = tfidf_process(subset_dtrain, **kwargs)
    topic_vectorizers[q] = vec_results[0]
    d_train_vec = vec_results[2]
    
    r_labels = list(r_labels.values())
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
    vec = topic_vectorizers[q].transform(d)
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
            prob = classify([d_test[d]], q, classifier)
            print(prob)
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

    '''
    'collections_processed/Dtest_judged_collection_processed' -> Dtest judged docs
    'collections_processed/Dtrain_judged_collection_processed' -> Dtrain judged docs
    'collections_processed/Dtrain_collection_processed' -> Dtrain completo
    '''

    d_test = read_from_file('collections_processed/Dtest_judged_collection_processed')
    d_train = read_from_file('collections_processed/Dtrain_judged_collection_processed')

    r_set = get_R_set('material/')[0]

    r_test = r_set[0]
    r_train = r_set[1]

    q_test = [120,123]

    evaluate(q_test, d_test, r_test)

    topics = get_topics('material/')

    return


main()