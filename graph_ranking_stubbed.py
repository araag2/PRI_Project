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
from _main_ import ranking_page_rank
from proj_utilities import *

#Global variables
topics = {}

# -----------------------------------------------------------------------------------------------------
# tfidf_process - Processes our entire document collection with a tf-idf vectorizer 
# and transforms our query doc and the entire collection into tf-idf spaced vectors 
#
# Input: doc_query - The query document that will be compared to the rest of the collection
#        doc_dic - The entire document collection in dictionary form
#        **kwargs - Optional parameters with the following functionality (default values prefixed by *)
#               norm [*l2 | l1]: Method to calculate the norm of each output row
#               min_df [*1 | float | int]: Ignore the terms which have a freq lower than min_df
#               max_df [*1.0 | float | int]: Ignore the terms which have a freq higher than man_df
#               max_features [*None | int]: 
#
# Behaviour: Creates a tf-idf vectorizer and fits the entire document collection into it. 
# Afterwards, transforms both the query document and the entire collection into vector form,
# allowing them to be directly used to calculate similarities. It also converts structures
# into to an easy form to manipulate at the previous higher level.
#
# Output: A list of document keys (ids), the query doc in vector form and the entire doc
# collection in vector form.
# -----------------------------------------------------------------------------------------------------
def tfidf_process(doc_query, doc_dic, **kwargs):
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
# sim_method_helper - Short helper method to encapsulate choosing the correct
# similarity function
#
# Input: sim - The string which represents which similarity function to use
#
# Behaviour: Matches the string with a dictionary of functions we have available
#
# Output: The function to use
# ------------------------------------------------------------------------------
def sim_method_helper(sim):
    sim_methods = {'cosine': cosine_similarity_dic, 'euclidean': eucledian_distance_dic, 'manhattan': manhattan_distance_dic}
    sim_method = None
    
    if sim == None:
        sim_method = sim_methods['cosine']

    elif sim == 'cosine' or sim == 'euclidean' or sim == 'manhattan':
        sim_method = sim_methods[sim]
    
    else:
        print("Error: similarity measure not recognized.")

    return sim_method

# ------------------------------------------------------------------------------
# build_graph - Builds a document graph from document collection D using
# the similarity measure in sim agains theta threshold
#
# Input: D - The document collection to build our graph with
#        sim - [*cosine | eucledian | manhattan] : The similarity measure used
#        theta - The similarity threshold 
#        kwargs -
#
# Behaviour: This function starts by creating the necessary structures for each
# of the give graph entries, and then proceeds to calculate the necessary
# pairwise similarity measures. It does so by treating each individual document
# as the query document and comparing it to all the rest.
#
# Output: A dictionary representing the weighted undirect graph that
# connect all documents on the basis of the given similarity measure
# ------------------------------------------------------------------------------
def build_graph(D, sim, theta, **kwargs):
    doc_dic = process_collection(D, False, **kwargs)
    #doc_dic = read_from_file('judged_docs_processed')

    graph = {}
    for doc in doc_dic:
        graph[doc] = {}

    sim_method = sim_method_helper(sim)

    #print(len(doc_dic))
    for doc in doc_dic:
        start_time = time.time()
        doc_id = doc
        similarity_dic = sim_method(doc_dic[doc], doc_dic, theta, **kwargs)

        for simil_doc in similarity_dic:
            if doc_id != simil_doc:
                graph[doc_id][simil_doc] = similarity_dic[simil_doc]
                graph[simil_doc][doc_id] = similarity_dic[simil_doc]

        #print("One iteration takes:{}".format(time.time()-start_time))

    #write_to_file(graph, 'judged_docs_linked_graph')
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
def page_rank(link_graph, q, D, **kwargs):
    max_iters = 50 if 'iter' not in kwargs else kwargs['iter']
    p = 0.15 if 'p' not in kwargs else kwargs['p']
    N = len(link_graph)
    damping = 1 - p
    prior = None

    if 'prior' not in kwargs:
        prior = p / N

        result_graph = {}
        # Setting uniform priors
        for doc in link_graph:
            result_graph[doc] = prior

        # Dictionary to save max_iters * (N-1) len() operations
        link_count = {}
        for doc in link_graph:
            link_count[doc] = len(link_graph[doc])

        for _ in range(max_iters):
            iter_graph = {}

            for doc in result_graph:
                cumulative_post = 0

                for link in link_graph[doc]:
                    cumulative_post += (result_graph[link] / link_count[doc]) 
                iter_graph[doc] = prior + damping * cumulative_post 

            result_graph = iter_graph

    elif 'prior' in kwargs and kwargs['prior'] == 'non-uniform':
        ranked_dic = ranking_page_rank(query, len(link_graph), D, **kwargs)
        prior_dic = {}

        # Initialize prior using original IR system
        for doc in link_graph:
            if doc in ranked_dic:
                prior_dic[doc] = ranked_dic[doc]
            else:
                prior_dic[doc] = 0

        result_graph = deepcopy(prior_dic)

        # Dictionary to save max_iters * (N-1) cum_sum operations
        link_weighted_count = {}
        for doc in link_graph:
            link_weighted_count[doc] = 0
            for link in link_graph[doc]:
                link_weighted_count[doc] += link_graph[doc][link]

        for _ in range(max_iters):
            iter_graph = {}

            #TODO: CHECK MATH HERE 
            for doc in result_graph:
                cumulative_prior = 0
                cumulative_post = 0

                for link in link_graph[doc]:
                    cumulative_prior += prior_dic[link]
                    cumulative_post += ((result_graph[link] * link_graph[link][doc]) / link_weighted_count[doc]) 

                iter_graph[doc] = p*cumulative_prior + damping * cumulative_post 

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
    init_graph = {}
    link_graph = build_graph(D, sim, theta, **kwargs)
    D_processed = process_collection(D, False)

    #TODO: Remove this
    #link_graph = read_from_file('link_graph')
    ranked_graph = page_rank(link_graph, q, D, **kwargs)

    query = topics[q]
    sim_method = sim_method_helper(sim)
    sim_dic = sim_method(query, D_processed, theta, **kwargs)

    sim_weight = 0.15 if 'sim_weight' not in kwargs else kwargs['sim_weight']
    pr_weight = 1 - sim_weight

    # Rebalances similarity based on page rank
    for doc in sim_dic:
        sim_dic[doc] = sim_weight * sim_dic[doc] + pr_weight * ranked_graph[doc]

    # Retrieve top p documents
    sorted_dic = sorted(sim_dic, key = sim_dic.get, reverse=True)
    result = []

    for i in range(p):
        doc = sorted_dic[i]
        result += [(doc, sim_dic[doc]),]

    print(result)
    return result

# --------------------------------------------------------------------------------
# ~ Just the Main Function ~
# --------------------------------------------------------------------------------
def main():
    global topics 
    topics = read_from_file('topics_processed')

    #D_judged = read_from_file('judged_docs_processed')
    #build_graph(None, 'cosine', 0.3)
    D = get_files_from_directory('../rcv1_test/19960820/')

    undirected_page_rank(140, D, 5, 'cosine', 0.3, prior='non-uniform')

    

    
    return


main()