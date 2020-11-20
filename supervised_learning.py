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
from proj_utilities import *

#Global variables
topics = {}
trained_classifiers ={}

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
    
    r_labels = find_R_test_labels(r_train)
    
    norm = 'l2' if 'norm' not in kwargs else kwargs['norm']
    min_df = 1 if 'min_df' not in kwargs else kwargs['min_df']
    max_df = 1.0 if 'max_df' not in kwargs else kwargs['max_df']
    max_features = None if 'max_features' not in kwargs else kwargs['max_features']

    classifiers = [MultinomialNB()]
    #classifiers = [MultinomialNB(), KNeighborsClassifier(), Perceptron(), LinearSVC()]

    vec = TfidfVectorizer(norm=norm, min_df=min_df, max_df=max_df, max_features=max_features)

    #select q relevant documents in r_train
    #select same documents in d_train (?)

    d_train_vec = vec.fit(d_train)
    r_train_vec = vec.fit(r_train)

    trained = []
    
    for classifier in classifiers:
        classifier.fit(X=d_train_vec, y=r_train_vec)
        trained.append(classifier)

    trained_classifiers[q] = trained

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
    global topics

    d_set = [None, None]
    d_test = d_set[0]
    d_train = d_set[1]

    r_set = [None, None]
    r_test = r_set[0]
    r_train = r_set[1]

    q_test = []

    for q in q_test:
        training(q, d_train, r_train)

    evaluate(q_test, d_test, r_test)

    r_train = find_R_test_labels('material/')[1]
    topics = get_topics('material/')
    

    return


main()