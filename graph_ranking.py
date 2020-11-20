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
from _main_ import tfidf_process
from _main_ import process_collection
from _main_ import get_judged_docs
from _main_ import get_topics
from _main_ import ranking_page_rank
from proj_utilities import *

#Global variables
topics = {}

# -------------------------------------------------------------------------------------
# manhattan_distance_dic - Computes the manhattan distance between a document
# and a given list of documents
#
# Input: doc_query - Document that serves as basis to compare all other documents to
#        vectorizer - The structure that contains our tf-idf vectorizer
#        doc_keys - A list of all document keys
#        doc_vectors - All documents in vector form contained in the vectorizer space
#        theta - The similarity threshold 
#
# Behaviour: It starts by transforming the query document into its vector notion in the
# vectorizer space. Afterwards, it calculates pairwise similarity based on the inverse 
# manhattan distance between all document vectors. In the end returns a dictionary with 
# all documents that have their similarity values (1/distance) greater than or equal to theta.
#
# Output: Dictionary with all documents that pass the similarity treshold
# -------------------------------------------------------------------------------------
def manhattan_distance_dic(doc_query, vectorizer, doc_keys, doc_vectors, theta, **kwargs):
    result = {}

    doc_vector = vectorizer.transform(doc_query)
    distance_list = manhattan_distances(doc_vector, doc_vectors)[0]

    for i in range(len(distance_list)):
        if distance_list[i] != 0 and 1/distance_list[i] >= theta:
            result[int(doc_keys[i])] = 1/distance_list[i]

    return result

# -----------------------------------------------------------------------------
# eucledian_distance_dic -  Computes the eucledian distance between a document
# and a given list of documents
#
# Input: doc_query - Document that serves as basis to compare all other documents to
#        vectorizer - The structure that contains our tf-idf vectorizer
#        doc_keys - A list of all document keys
#        doc_vectors - All documents in vector form contained in the vectorizer space
#        theta - The similarity threshold 
#
# Behaviour: It starts by transforming the query document into its vector notion in the
# vectorizer space. Afterwards, it calculates pairwise similarity based on the inverse 
# eucledian distance between all document vectors.  In the end returns a dictionary with 
# all documents that have their similarity values (1/distance) greater than or equal to theta.
#
# Output: Dictionary with all documents that pass the similarity treshold
# -----------------------------------------------------------------------------
def eucledian_distance_dic(doc_query, vectorizer, doc_keys, doc_vectors, theta, **kwargs):
    result = {}

    doc_vector = vectorizer.transform(doc_query)
    distance_list = euclidean_distances(doc_vector, doc_vectors)[0]

    for i in range(len(distance_list)):
        if distance_list[i] != 0 and 1/distance_list[i] >= theta:
            result[int(doc_keys[i])] = 1/distance_list[i]

    return result

# -----------------------------------------------------------------------------
# cosine_similarity_dic - Computes the cosine similarity between a document
# and a given list of documents
#
# Input: doc_query - Document that serves as basis to compare all other documents to
#        vectorizer - The structure that contains our tf-idf vectorizer
#        doc_keys - A list of all document keys
#        doc_vectors - All documents in vector form contained in the vectorizer space
#        theta - The similarity threshold 
#
# Behaviour: It starts by transforming the query document into its vector notion in the
# vectorizer space. Afterwards, it calculates pairwise similarity based on the cosine 
# similarity measure between all document vectors. In the end returns a dictionary with 
# all documents that have their similarity values greater than or equal to theta.
#
# Output: Dictionary with all documents that pass the similarity treshold
# -----------------------------------------------------------------------------
def cosine_similarity_dic(doc_query, vectorizer, doc_keys, doc_vectors, theta, **kwargs):
    result = {}

    doc_vector = vectorizer.transform(doc_query)
    distance_list = cosine_similarity(doc_vector, doc_vectors)[0]

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

    tfidf_vectorizer_info = tfidf_process(doc_dic, **kwargs)

    vectorizer = tfidf_vectorizer_info[0]
    doc_keys = tfidf_vectorizer_info[1]
    doc_vectors = tfidf_vectorizer_info[2]

    graph = {}
    for doc in doc_dic:
        graph[doc] = {}

    sim_method = sim_method_helper(sim)

    for doc in doc_dic:
        similarity_dic = sim_method([doc_dic[doc]], vectorizer, doc_keys, doc_vectors, theta, **kwargs)

        for simil_doc in similarity_dic:
            if doc != simil_doc:
                graph[doc][simil_doc] = similarity_dic[simil_doc]
                graph[simil_doc][doc] = similarity_dic[simil_doc]

    return graph

# ---------------------------------------------------------------------------------------------------------
# page_rank - Function that directly uses the Page Rank algorithm with a 
# variation for undirected graphs and uses it to calculate a score for each
# candidate based on the provided link_graph. 
#
# Input: link_graph - The undirected graph that contains all document links
#        and their correspondent weight.
#        q - A topic query in the form of topic identifier (int)
#        D - The document collection we built our graph with
#        **kwargs - Optional parameters with the following functionality (default values prefixed by *)
#               iter [*50 | int]: Number of iterations to run the algorithm in 
#               p [*0.15 | float]: Starting p value which represents the residual probability for each node
#               prior [*uniform | non-uniform]: Method to calculate priors in our algorithm 
# 
# Behaviour: This function starts by setting the default values for the Page Rank algorithm, and after 
# selecting which prior to use, it applies the algorithm max_iters number of times. It also builds some
# auxiliary structures like link_count to ensure we don't repeatly calculate const values. 
#
# Output: The resulting PageRank graph in dictionary form. 
# ---------------------------------------------------------------------------------------------------------
def page_rank(link_graph, q, D, **kwargs):
    result_graph = {}

    max_iters = 50 if 'iter' not in kwargs else kwargs['iter']
    p = 0.15 if 'p' not in kwargs else kwargs['p']
    N = len(link_graph)
    follow_p = 1 - p
    prior = None

    if 'prior' not in kwargs or kwargs['prior'] == 'uniform':
        prior = p / N

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
                iter_graph[doc] = prior + follow_p * cumulative_post 

            result_graph = iter_graph

    elif 'prior' in kwargs and kwargs['prior'] == 'non-uniform':

        # TODO: Check different priors
        ranked_dic = ranking_page_rank(topics[q], len(link_graph), D, **kwargs)
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

            for doc in result_graph:
                cumulative_prior = 0
                cumulative_post = 0

                for link in link_graph[doc]:
                    cumulative_prior += prior_dic[link]
                    cumulative_post += ((result_graph[link] * link_graph[link][doc]) / link_weighted_count[doc]) 

                iter_graph[doc] = p * cumulative_prior + follow_p * cumulative_post 

            result_graph = iter_graph

    return result_graph

# -------------------------------------------------------------------------------------------------------
# undirected_page_rank - This function applies a modified version of the PageRank algorithm for undirected
# graphs to the provided document collection, retriving the top p documents for topic q in regars to 
# similarity measure sim and treshold theta. 
#
# Input: q - A topic query in the form of topic identifier (int)
#        D - The document collection we built our graph with
#        p - The number of top documents to return
#        sim - [*cosine | eucledian | manhattan] : The similarity measure used
#        theta - The similarity threshold
#        **kwargs - Optional parameters with the following functionality (default values prefixed by *)
#               sim_weight [*0.5 | float in [0.0, 1.0] ]: The weight given to the base similarity measure
#               over the PageRank results 
# 
# Behaviour: This function serves primarily as an encapsulation for the PageRank algorithm, and as such
# it starts by creating the necessary structures for it to run, namely the link_graph. Afterwards, it 
# takes the PageRank results present in PageRank and weights the final results in accordance to the 
# results from the similarity measure sim given similiraty weight sim_weight. In the end it selects the
# top p perfoming documents for query q and returns them in list form.
#
# Output: A list of ordered top-documents with their corresponding score in the form (d, score), ordered
# in descending order of score.
# -------------------------------------------------------------------------------------------------------
def undirected_page_rank(q, D, p, sim, theta, **kwargs):
    link_graph = build_graph(D, sim, theta, **kwargs)
    ranked_graph = page_rank(link_graph, q, D, **kwargs)

    query = topics[q]
    sim_method = sim_method_helper(sim)

    tdidf_info = tfidf_process(process_collection(D, False), **kwargs)
    vectorizer = tdidf_info[0]
    doc_keys = tdidf_info[1]
    doc_vectors = tdidf_info[2]

    sim_dic = sim_method([query], vectorizer, doc_keys, doc_vectors, 0, **kwargs)

    sim_weight = 0.5 if 'sim_weight' not in kwargs else kwargs['sim_weight']
    pr_weight = 1 - sim_weight

    #TODO: Normalize weights (Zscore)
    # Rebalances similarity based on page rank
    for doc in sim_dic:
        sim_dic[doc] = sim_weight * sim_dic[doc] + pr_weight * ranked_graph[doc]

    # Retrieve top p documents
    sorted_dic = sorted(sim_dic, key = sim_dic.get, reverse=True)

    result = []
    result_range = range(p) if p <= len(sorted_dic) else range(len(sorted_dic))
    for i in result_range:
        doc = sorted_dic[i]
        result += [(doc, sim_dic[doc]),]

    return result

# --------------------------------------------------------------------------------
# ~ Just the Main Function ~
# --------------------------------------------------------------------------------
def main():
    global topics 
    topics = get_topics('material/')
    D = get_files_from_directory('../rcv1_test/19960820/')[1]

    print(build_graph(D, 'cosine', 0.3))
    #print(undirected_page_rank(101, D, 5, 'cosine', 0.3, prior='uniform'))

    return


main()