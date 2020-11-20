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
import pandas as pd
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
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.cluster import *
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from textblob import TextBlob

# File imports
from _main_ import get_files_from_directory
from _main_ import process_collection
from _main_ import get_judged_docs
from _main_ import get_topics
from _main_ import ranking_page_rank
from _main_ import get_R_set
from proj_utilities import *

topics = {}
index_id = 1

# -----------------------------------------------------------------------------------------------------
# tfidf_process - Processes our entire document collection with a tf-idf vectorizer 
# and transforms the entire collection into tf-idf spaced vectors 
#
# Input: doc_dic - The entire document collection in dictionary form
#        **kwargs - Optional parameters with the following functionality (default values prefixed by *)
#               norm [*l2 | l1]: Method to calculate the norm of each output row
#               min_df [*1 | float | int]: Ignore the terms which have a freq lower than min_df
#               max_df [*1.0 | float | int]: Ignore the terms which have a freq higher than man_df
#               max_features [*None | int]: 
#
# Behaviour: Creates a tf-idf vectorizer and fits the entire document collection into it. 
# Afterwards, transforms the entire document collection into vector form, allowing it to be 
# directly used to calculate similarities. It also converts structures into to an easy form to manipulate 
# at the previous higher level.
#
# Output: The tf-idf vectorizer created, a list of document keys (ids) and the entire doc
# collection in vector form.
# -----------------------------------------------------------------------------------------------------
def tfidf_process(doc_dic, **kwargs):
    doc_keys = list(doc_dic.keys())
    doc_list = []

    for doc in doc_keys:
        doc_list += [doc_dic[doc], ]

    norm = 'l2' if 'norm' not in kwargs else kwargs['norm']
    min_df = 1 if 'min_df' not in kwargs else kwargs['min_df']
    max_df = 1.0 if 'max_df' not in kwargs else kwargs['max_df']
    max_features = None if 'max_features' not in kwargs else kwargs['max_features']

    vec = TfidfVectorizer(norm=norm, min_df=min_df, max_df=max_df, max_features=max_features)
    vec.fit(doc_list)

    doc_list_vectors = vec.transform(doc_list)

    return [vec, doc_keys, doc_list_vectors]

# --------------------------------------------------------------------------------
# Kmeans Clustering
# supposedly rand score will get /something/ to compare to the obtained clustering.
# what????????? should it be?????? im ?????????
# topics could go with relevance but how is that supposed to help in this case?????
# --------------------------------------------------------------------------------
def trainKmeans(vec_D, y, clusters, distance):

    silhouettes = []
    rands = []
    vs = [] 

    for i in clusters:
        clustering_kmeans = KMeans(n_clusters=i).fit(vec_D)
        labels_kmeans = clustering_kmeans.labels_

        silhouettes.append(silhouette_score(vec_D, labels_kmeans, metric=distance))
        rands.append(adjusted_rand_score(y, labels_kmeans))
        vs.append(v_measure_score(y, labels_kmeans))

    best_cluster = clusters[np.argmax(rands)]
    clustering_kmeans = KMeans(n_clusters=best_cluster).fit(vec_D)

    return [best_cluster, clustering_kmeans, np.max(rands)]

# --------------------------------------------------------------------------------
# Agglomerative Clustering
# --------------------------------------------------------------------------------
def trainAgglomerative(vec_D, array_D, clusters, distance):

    silhouettes = []
    rands = []
    vs = [] 

    for i in clusters:
        clustering_agg = AgglomerativeClustering(n_clusters=i).fit(vec_D)
        labels_agg = clustering_agg.labels_

        silhouettes.append(silhouette_score(vec_D, labels_agg, metric=distance))
        rands.append(adjusted_rand_score(y, labels_agg))
        vs.append(v_measure_score(y, labels_agg))

    best_cluster = clusters[np.argmax(rands)]
    clustering_agg = AgglomerativeClustering(n_clusters=best_cluster).fit(vec_D)

    return [best_cluster, clustering_agg, np.max(rands)]

# --------------------------------------------------------------------------------
# clustering i guess shrugs
# --------------------------------------------------------------------------------
def clustering(D, **kwargs):
    doc_dic = process_collection(D, False, **kwargs)

    tfidf_vec_info = tfidf_process(doc_dic, **kwargs)

    vectorizer = tfidf_vec_info[0]
    doc_keys = tfidf_vec_info[1]
    doc_vectors = tfidf_vec_info[2]

    r_set = get_R_set('material/',index='doc_id')[1]
    y = []

    for key in doc_keys:
        for r in r_set[key]:
            y.append('{}_{}'.format(r, r_set[key][r]))

    clusters = [2, 5, 10, 25, 50] if 'clusters' not in kwargs else kwargs['clusters']
    distance = 'euclidean' if 'distance' not in kwargs else kwargs['distance']
    best_KM = trainKmeans(doc_vectors, y, clusters, distance)
    best_AC = trainAgglomerative(doc_vectors, y, clusters, distance)

    if best_KM[2] < best_AC[2]:
        best = best_AC
    else:
        best = best_KM
    
    

# --------------------------------------------------------------------------------
# ~ Just the Main Function ~
# --------------------------------------------------------------------------------
def main():
    global topics
    topics = get_topics('material/')
    material_dic = 'material/'

    D_set = get_files_from_directory('../rcv1_test/19961001')[1]

    print("Hello world uwu")

main()