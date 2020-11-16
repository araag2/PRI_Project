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
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from textblob import TextBlob

# File imports
from _main_ import get_files_from_directory
from _main_ import process_collection
from proj_utilities import *

# -----------------------------------------------------------------------------------------
# tfidf_process - Processes our entire document collection with a tf-idf vectorizer 
# and transforms our query doc and the entire collection into tf-idf spaced vectors 
#
# Input: doc_query - The query document that will be compared to the rest of the collection
#        doc_dic - The entire document collection in dictionary form
#        kwargs - Parameters for the tf-idf vectorizer
#
# Behaviour: Creates a tf-idf vectorizer and fits the entire document collection into it. 
# Afterwards, transforms both the query document and the entire collection into vector form,
# allowing them to be directly used to calculate similarities. It also converts structures
# into to an easy form to manipulate at the previous higher level.
#
# Output: A list of document keys (ids), the query doc in vector form and the entire doc
# collection in vector form.
# -----------------------------------------------------------------------------------------
def tfidf_process(doc_query, doc_dic, **kwargs):
    doc_keys = list(doc_dic.keys())
    doc_list = []

    for doc in doc_keys:
        doc_list += [doc_dic[doc], ]

    # TODO: **kwargs for norm, min_df and max_df
    vec = TfidfVectorizer()
    vec.fit(doc_list)

    doc_list_vectors = vec.transform(doc_list)
    doc_vector = vec.transform([doc_query])

    return [doc_keys, doc_vector, doc_list_vectors]


# -----------------------------------------------------------------------------
# manhattan_distance_dic - Computes the cosine similarity between a document
# and a given list of documents
#
# Input: 
#
# Behaviour: 
#
# Output: 
# -----------------------------------------------------------------------------
def manhattan_distance_dic(doc_query, doc_dic, theta, **kwargs):
    result = {}
    tfidf_processed_list = tfidf_process(doc_query, doc_dic, **kwargs)

    doc_keys = tfidf_processed_list[0]
    doc_vector = tfidf_processed_list[1]
    doc_list_vectors = tfidf_processed_list[2]

    distance_list = manhattan_distances(doc_vector, doc_list_vectors)[0]

    for i in range(len(distance_list)):
        if distance_list[i] != 0 and 1/distance_list[i] >= theta:
            result[int(doc_keys[i])] = 1/distance_list[i]

    return result

# -----------------------------------------------------------------------------
# eucledian_distance_dic - Computes the cosine similarity between a document
# and a given list of documents
#
# Input:  - The document collection to build our graph with
#        sim - The similarity measure used
#        theta - The similarity threshold 
#        kwargs -
#
# Behaviour: 
#
# Output: 
# -----------------------------------------------------------------------------
def eucledian_distance_dic(doc_query, doc_dic, theta, **kwargs):
    result = {}
    tfidf_processed_list = tfidf_process(doc_query, doc_dic, **kwargs)

    doc_keys = tfidf_processed_list[0]
    doc_vector = tfidf_processed_list[1]
    doc_list_vectors = tfidf_processed_list[2]

    distance_list = euclidean_distances(doc_vector, doc_list_vectors)[0]

    for i in range(len(distance_list)):
        if distance_list[i] != 0 and 1/distance_list[i] >= theta:
            result[int(doc_keys[i])] = 1/distance_list[i]

    return result

# -----------------------------------------------------------------------------
# cosine_similarity_dic - Computes the cosine similarity between a document
# and a given list of documents
#
# Input: 
#
# Behaviour: 
#
# Output: 
# -----------------------------------------------------------------------------
def cosine_similarity_dic(doc_query, doc_dic, theta, **kwargs):
    result = {}
    tfidf_processed_list = tfidf_process(doc_query, doc_dic, **kwargs)

    doc_keys = tfidf_processed_list[0]
    doc_vector = tfidf_processed_list[1]
    doc_list_vectors = tfidf_processed_list[2]

    distance_list = cosine_similarity(doc_vector, doc_list_vectors)[0]

    for i in range(len(distance_list)):
        if distance_list[i] >= theta:
            result[int(doc_keys[i])] = distance_list[i]

    return result

# ------------------------------------------------------------------------------
# build_graph - Builds a document graph from document collection D using
# the similarity measure in sim agains theta threshold
#
# Input: D - The document collection to build our graph with
#        sim - [*cosine | eucledian | manhattan] : The similarity measure used
#        theta - The similarity threshold 
#        kwargs -
#
# Behaviour: 
#
# Output: A dictionary representing the weighted undirect graph that
# connect all documents on the basis of the given similarity measure
# ------------------------------------------------------------------------------
def build_graph(D, sim, theta, **kwargs):
    sim_methods = {'cosine': cosine_similarity_dic, 'euclidean': eucledian_distance_dic, 'manhattan': manhattan_distance_dic}

    #doc_dic = process_collection(D, False, **kwargs)
    doc_dic = read_from_file('test_dic')

    graph = {}
    for doc in doc_dic:
        graph[int(doc)] = {}

    sim_method = None

    if sim == None:
        sim_method = sim_methods['cosine']

    elif sim == 'cosine' or sim == 'euclidean' or sim == 'manhattan':
        sim_method = sim_methods[sim]
    
    else:
        print("Error: similarity measure not recognized.")
        return

    for doc in doc_dic:
        doc_id = int(doc)
        similarity_dic = sim_method(doc_dic[doc], doc_dic, theta, **kwargs)

        for simil_doc in similarity_dic:
            if doc_id != simil_doc:
                graph[doc_id][simil_doc] = similarity_dic[simil_doc]
                graph[simil_doc][doc_id] = similarity_dic[simil_doc]

    return graph

# -----------------------------------------------------------------------
# page_rank - 
#
# Input: 
# 
# Behaviour: 
#
# Output:
# -----------------------------------------------------------------------
def page_rank(result_graph, link_graph, damping, max_iters, **kwargs):
    for _ in range(max_iters):
        iter_graph = {}

        for doc in result_graph:
            cumulative_prob = 0
            link_n = len(result_graph[doc])

            for link in result_graph[doc]:
                cumulative_prob += result_graph[link] / link_n 
            iter_graph[doc] = cumulative_prob

        result_graph = iter_graph

    return result_graph

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
    result_graph = {}
    link_graph = build_graph(D, sim, theta, **kwargs) 

    uniform_residue = 0.15
    damping = 1 - uniform_residue
    for doc in link_graph:
        result_graph[doc] = uniform_residue

    max_iters = 50
    if 'iter' in kwargs:
        max_iters = kwargs['iter']

    result_graph = page_rank(result_graph, link_graph, damping, max_iters, **kwargs)
    print(result_graph)

    return

# --------------------------------------------------------------------------------
# ~ Just the Main Function ~
# --------------------------------------------------------------------------------
def main():
    print(build_graph(None, 'cosine', 0.3))
    return


main()