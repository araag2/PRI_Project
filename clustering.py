# --------------------------------
# 2nd Delivery of the PRI project
# 86379 - Ana Evans
# 86389 - Artur Guimarães
# 86417 - Francisco Rosa
# --------------------------------

import nltk
import sklearn
import math
import numpy as np
from sklearn.metrics import *
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import *

# File imports

from file_treatment import read_from_file
from data_set_treatment import tfidf_process
from data_set_treatment import get_topics
from data_set_treatment import get_R_set

topics = {}
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
            best_result = [clustering_agg, labels_agg, score_mean]

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
    doc_keys = tfidf_vec_info[1]
    doc_vectors = tfidf_vec_info[2]

    r_set = get_R_set('material/',index='doc_id')[0][1]
    y = []

    #TODO: if documents not topics and topics case
    for i in range(len(doc_keys)):
        y.append([])
        for r in r_set[doc_keys[i]]:
            y[i].append('{}_{}'.format(r, r_set[doc_keys[i]][r]))
    y = np.array(y, dtype=object)

    # trainAgglomerative
    clustering_methods = [trainKmeans] if 'methods' not in kwargs else kwargs['methods']
    # TODO: add more 
    clusters = [2] if 'clusters' not in kwargs else kwargs['clusters']
    distances = ['euclidean'] if 'distance' not in kwargs else kwargs['distance']
    
    best_clusters = [None, None, 0]
    for method in clustering_methods:
        for dist in distances:
            clustering = method(doc_vectors, y, clusters, dist)

            if clustering[2] > best_clusters[2]:
                best_clusters = clustering

    centroids = best_clusters[0].cluster_centers_
    doc_labels = best_clusters[1]

    result = []
    for i in range(len(centroids)):
        entry = (centroids[i], [])
        result.append(entry)

    for i in range(len(doc_keys)):
        result[doc_labels[i]][1].append(doc_keys[i]) 

    return result

# ----------------------------------------------------------------------------------------------------
# interpret(): Evaluates clusters in terms of median (centroid) and medoid criteria
#
# Input: cluster - A document/topic cluster
#        D - Set of documents or topics in cluster
#       **kwargs - Optional parameters with the following functionality (default 
#       values prefixed by *)
#
# Behaviour: it's a surprise :)
#
# Output: ;)
# ----------------------------------------------------------------------------------------------------
def interpret(cluster, D, **kwargs):

    documents = {}
    for doc_id in cluster[1]:
        documents[doc_id] = D[doc_id]

    doc_vectors = tfidf_process(documents, **kwargs)[2]
    distance_matrix = pairwise_distances(doc_vectors)

    centroid = cluster[0]
    medoid_index = np.argmin(distance_matrix.sum(axis=0))

    return [centroid, cluster[1][medoid_index]]

# ----------------------------------------------------------------------------------------------------
# evaluate: 
#
# Input: 
#
# Behaviour: 
#
# Output: 
# ----------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
# ~ Just the Main Function ~
# --------------------------------------------------------------------------------
def main():
    global topics
    material_dic = 'material/'
    topics = get_topics(material_dic)
    
    #D_set = get_files_from_directory('../rcv1_test/19961001')[1]
    D = read_from_file('collections_processed/Dtrain_judged_collection_processed')
    #print(read_from_file('topics_processed'))

    result = clustering(D)
    
    for cluster in result:
        print(interpret(cluster, D))

main()