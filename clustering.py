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
#import pandas as pd
#import matplotlib as mpl 
#import matplotlib.pyplot as plt
from copy import deepcopy
from heapq import nlargest 
from bs4 import BeautifulSoup
#from lxml import etree
from whoosh import index
from whoosh import scoring
from whoosh.qparser import *
from whoosh.fields import *
from sklearn.metrics import *
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.cluster import *
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
#from textblob import TextBlob

# File imports
from _main_ import get_files_from_directory
from _main_ import process_collection
from _main_ import get_judged_docs
from _main_ import get_topics
from _main_ import ranking_page_rank
from _main_ import get_R_set
from _main_ import tfidf_process
from proj_utilities import *

topics = {}
index_id = 1

# --------------------------------------------------------------------------------
# trainsKmeans - trains the KMeans algorithm
#
# Input: vec_D - the set of document or topics to be clustered
#       y - the set of ids and relevance info
#       clusters - the list with the number of clusters to be attempted
#       distance - parameter for the evaluation measures
# 
# Behaviour: creates the KMeans clusters for the number of clusters previously
# defined, and then applies a set of evaluation measures to select the best one
#
# Output: A list containing te best number of clusters to use, the list of labels
# and the rands score
# --------------------------------------------------------------------------------
def trainKmeans(vec_D, y, clusters, distance):

    best_result = [None, None, 0]
    for i in clusters:
        clustering_kmeans = KMeans(n_clusters=i).fit(vec_D)
        labels_kmeans = clustering_kmeans.labels_

        sil_score = silhouette_score(vec_D, labels_kmeans, metric=distance)
        rands_score = adjusted_rand_score(y, labels_kmeans)
        vs_score = v_measure_score(y, labels_kmeans)

        score_mean = (sil_score + rands_score + vs_score) / 3

        if score_mean > best_result[2]:
            best_result = [clustering_kmeans, labels_kmeans, score_mean]

    return best_result

# --------------------------------------------------------------------------------
# trainsAgglomerative - trains the Agglomerative Clustering algorithm
#
# Input: vec_D - the set of document or topics to be clustered
#       y - the set of ids and relevance info
#       clusters - the list with the number of clusters to be attempted
#       distance - parameter for the evaluation measures
# 
# Behaviour: creates the Agglomerative clusters for the number of clusters previously
# defined, and then applies a set of evaluation measures to select the best one
#
# Output: A list containing te best number of clusters to use, the list of labels
# and the rands score
# --------------------------------------------------------------------------------
def trainAgglomerative(vec_D, y, clusters, distance):

    vec_D = vec_D.toarray()
    best_result = [None, None, 0]
    for i in clusters:
        clustering_agg = AgglomerativeClustering(n_clusters=i).fit(vec_D)
        labels_agg = clustering_agg.labels_

        sil_score = silhouette_score(vec_D, labels_agg, metric=distance)
        rands_score = adjusted_rand_score(y, labels_agg)
        vs_score = v_measure_score(y, labels_agg)

        score_mean = (sil_score + rands_score + vs_score) / 3

        if score_mean > best_result[2]:
            best_result = [clustering_kmeans, labels_agg, score_mean]

    return best_result

# ----------------------------------------------------------------------------------------------------
# Clustering: Applies Clustering, a unsupervised learning technique to classify 
# sets of documents instead of individual documents
#
# Input: D - set of documents or topics to be clustered
#       **kwargs - Optional parameters with the following functionality (default 
#       values prefixed by *)
#           clusters []: List with the number of clusters to attempt in the clustering algorithms
#           distance [TODO]: 
#
# Behaviour: Starts by obtaining the processed collection of documents. Then vectorizes
# them throught the function tfidf_process and obtains the vectorizer itself, the document
# keys and the document vector. Afterwards, we obtain the r_set entries relevant to the doc keys,
# and deal with the kwargs information. These arguments are sent to the clustering training functions
# which return the information regarding the best clustering they found. We compare KM and AC through
# their best rand score, and then process the information to output.
#
# Output: A list of cluster results. These results consist in a pair per cluster, with the cluster 
# centroid and the set of document/topic ids which comprise it
# ----------------------------------------------------------------------------------------------------
def clustering(D, **kwargs):
    #doc_dic = process_collection(D, False, **kwargs)
    doc_dic = read_from_file('collections_processed/Dtrain_collection_processed')

    tfidf_vec_info = tfidf_process(doc_dic, **kwargs)
    vectorizer = tfidf_vec_info[0]
    doc_keys = tfidf_vec_info[1]
    doc_vectors = tfidf_vec_info[2]

    r_set = get_R_set('material/',index='doc_id')[1]
    y = []

    #TODO: if documents not topics and topics case
    for i in range(len(doc_keys)):
        y.append([])
        for r in r_set[doc_keys[i]]:
            y[i].append('{}_{}'.format(r, r_set[doc_keys[i]][r]))
    y = np.array(y, dtype=object)

    #print('Doc dic has len:{}'.format(len(doc_dic)))
    #print('r_set has len:{}'.format(len(r_set)))
    #print('doc_keys has len:{}'.format(len(doc_keys)))
    #print('doc_vectors has len:{}'.format(doc_vectors.getnnz()))
    #print('y has len:{}'.format(len(y)))

    clusters = [2] if 'clusters' not in kwargs else kwargs['clusters']
    distance = 'euclidean' if 'distance' not in kwargs else kwargs['distance']
    best_KM = trainKmeans(doc_vectors, y, clusters, distance)
    best_AC = trainAgglomerative(doc_vectors, y, clusters, distance)

    print(best_KM)
    print(best_AC)

    if best_KM[2] < best_AC[2]:
        best = best_AC
    else:
        best = best_KM
        centroids = best.cluster_centers

    result = []
    i = 0
    while i < best[0]:
        id_set = []
        labels = best[1]
        j = 0
        while labels[j] == labels[j+1]:
            id_set += str(doc_keys[j])
            j+=1
        id_set += str(doc_keys[j])
        result.append((centroids[i],id_set))
        i += 1

    return result

# --------------------------------------------------------------------------------
# ~ Just the Main Function ~
# --------------------------------------------------------------------------------
def main():
    global topics
    topics = get_topics('material/')
    material_dic = 'material/'

    #D_set = get_files_from_directory('../rcv1_test/19961001')[1]
    #D = read_from_file('Dtrain_collection')
    
    result = clustering(None)
    print(result)

    #print("Hello world uwu")

main()