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
from file_treatment import get_files_from_directory
from data_set_treatment import tfidf_process
from data_set_treatment import get_topics
from data_set_treatment import get_R_set
from data_set_treatment import process_collection

topics = {}
tfidf_vec_info = [None, None, None]
Labels_pred = []

# --------------------------------------------------------------------------------
# get_clustering_score() - Gets a clustering methods score based on supervised, unsupervised
# or mixed metrics. Uses a linear combination of all results.
# --------------------------------------------------------------------------------
def get_clustering_score(x, labels_true, labels_pred, target):
    unsupervised_methods = { 'sil_score' : silhouette_score, 'calin_score' : calinski_harabasz_score, 'davies_score' : davies_bouldin_score}

    supervised_methods = {'rands_score' : adjusted_rand_score, 'v_score' : v_measure_score, 'complete_score' : completeness_score, 
                          'fowlkes_score' : fowlkes_mallows_score, 'homogenity_score': homogeneity_score, 'mutual_score' : mutual_info_score}

    if target == 'supervised':
        unsupervised_methods = {}
    elif target == 'unsupervised':
        supervised_methods = {}

    result = 0
    for method in unsupervised_methods:
        result += unsupervised_methods[method](x, labels_pred)

    for method in supervised_methods:
        score = supervised_methods[method](labels_true, labels_pred)
        result += score
        print('{} = {}'.format(method, score))

    return result / (len(unsupervised_methods) + len(supervised_methods))

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

    array_D = vec_D.toarray()
    best_result = [None, None, 0]
    for i in clusters:
        clustering_kmeans = KMeans(n_clusters=i).fit(vec_D)
        labels_pred = clustering_kmeans.labels_

        score_mean = get_clustering_score(array_D, y, labels_pred, 'unsupervised')

        if score_mean > best_result[2]:
            best_result = [clustering_kmeans, labels_pred, score_mean]  

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
        labels_pred = clustering_agg.labels_

        score_mean = get_clustering_score(vec_D, y, labels_pred, 'unsupervised')

        if score_mean > best_result[2]:
            best_result = [clustering_agg, labels_pred, score_mean]

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
    global tfidf_vec_info
    global labels_pred

    mode = 'docs' if 'mode' not in kwargs else kwargs['mode']

    doc_keys = None
    doc_vectors = None
    y = []

    if mode == 'docs':
        tfidf_vec_info = tfidf_process(D, remove_stopwords='english', **kwargs)
        doc_keys = tfidf_vec_info[1]
        doc_vectors = tfidf_vec_info[2]

        r_set = get_R_set('material/',index='doc_id')[0][1]
    
        for i in range(len(doc_keys)):
            y.append([])
            if doc_keys[i] in r_set:
                for r in r_set[doc_keys[i]]:
                    y[i].append('{}_{}'.format(r, r_set[doc_keys[i]][r]))
        y = np.array(y, dtype=object)

    elif mode == 'topics':
        tfidf_vec_info = tfidf_process(D, min_df = 1, max_df=1.0, **kwargs)
        doc_keys = tfidf_vec_info[1]
        doc_vectors = tfidf_vec_info[2]

        r_set = get_R_set('material/')[0][1]

        for i in range(len(doc_keys)):
            y.append([])
            if doc_keys[i] in r_set:
                for r in r_set[doc_keys[i]]:
                    y[i].append('{}_{}'.format(r, r_set[doc_keys[i]][r]))
        y = np.array(y, dtype=object)


    # trainAgglomerative
    clustering_methods = [trainKmeans] if 'methods' not in kwargs else kwargs['methods']

    # TODO: add more 
    clusters = list(range(2,10)) if 'clusters' not in kwargs else kwargs['clusters']
    distances = ['euclidean'] if 'distance' not in kwargs else kwargs['distance']
    top_cluster_words = 5 if 'top_cluster_words' not in kwargs else kwargs['top_cluster_words']
    
    best_clusters = [None, None, 0]
    for method in clustering_methods:
        for dist in distances:
            clustering = method(doc_vectors, y, clusters, dist)

            if clustering[2] > best_clusters[2]:
                best_clusters = clustering

    #TODO: fixme
    centroids = best_clusters[0].cluster_centers_

    doc_labels = best_clusters[1]
    labels_pred = best_clusters[1]

    result = []
    for i in range(len(centroids)):
        aux_centroid = centroids[i][centroids[i] != 0.]
        sorted_args = np.argsort(aux_centroid)

        inverse_transformed = tfidf_vec_info[0].inverse_transform(centroids[i])[0]
        n_args = len(inverse_transformed)
        entry_range = top_cluster_words if n_args > top_cluster_words else n_args

        centroid_result = []
        for i in range(entry_range):
            centroid_result.append(inverse_transformed[sorted_args[i]])

        entry = (centroid_result, [])
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

    doc_vectors = tfidf_process(documents, min_df = 1, max_df=1.0, **kwargs)[2]
    distance_matrix = pairwise_distances(doc_vectors)

    centroid = cluster[0]
    medoid_index = np.argmin(distance_matrix.sum(axis=0))

    return [centroid, cluster[1][medoid_index]]

# ----------------------------------------------------------------------------------------------------
# get_topic_subset() - Retrieves topics from the global variable topics that are contained within
# Q 
# ----------------------------------------------------------------------------------------------------
def get_topic_subset(q_test):
    result = {}
    for topic in topics:
        if topic in q_test:
            result[topic] = topics[topic]
    return result

# ----------------------------------------------------------------------------------------------------
# get_category_ids() - Helper function to get category id's from collection D to serve as ground
# truth
# ----------------------------------------------------------------------------------------------------
def get_category_ids(doc_keys):
    clustered_docs = {}
    for doc in doc_keys:
        clustered_docs[doc] = True

    D = get_files_from_directory('../rcv1/', clustered_docs)
    D[0].extend(D[1])
    D = D[0]

    codes_y = []
    for doc in D:
        codes_y.append([])
        if len(doc.find_all('codes')) > 1:
            topic_section = doc.find_all('codes')[1]
            topic_ids = topic_section.find_all('code')

            for iden in topic_ids:
                codes_y[-1].append(iden['code'])
    codes_y = np.array(codes_y, dtype=object)

    return codes_y

# ----------------------------------------------------------------------------------------------------
# evaluate: 
#
# Input: 
#
# Behaviour: 
#
# Output: 
# ----------------------------------------------------------------------------------------------------
def evaluate(D, **kwargs):
    global tfidf_vec_info
    global labels_pred

    mode = 'docs' if 'mode' not in kwargs else kwargs['mode']
    doc_dic = D

    if mode == 'docs':
        doc_dic = process_collection(D, False, **kwargs)
    elif mode == 'topics':
        doc_dic = get_topic_subset(D)

    clusters = clustering(doc_dic, **kwargs)
    cluster_info = []

    for cluster in clusters:
        cluster_info.append(interpret(cluster, doc_dic, **kwargs))

    n_clusters = len(clusters)
    name = 'document' if mode == 'docs' else 'topic'

    print("The clustering solution has k = {} clusters".format(n_clusters))
    for i in range(n_clusters):
        print("\nCluster {}:".format(i+1))
        print("Centroid has top words {}".format(cluster_info[i][0]))
        print("Medoid is {} with id {}".format(name,cluster_info[i][1]))
        print("Cluster is composed by {} {}s".format(len(clusters[i][1]), name))

    if mode == 'docs' and 'external' in kwargs and kwargs['external']:
        print('\nExternal evaluation for clustering solution:')
        doc_keys = tfidf_vec_info[1]
        category_ids = get_category_ids(doc_keys)
        final_score = get_clustering_score(None, category_ids, labels_pred, 'supervised')
        print('\nFinal Mean supervised score = {}'.format(final_score))
    
    return


# --------------------------------------------------------------------------------
# ~ Just the Main Function ~
# --------------------------------------------------------------------------------
def main():
    global topics
    material_dic = 'material/'
    topics = get_topics(material_dic)
    
    #D_set = get_files_from_directory('../rcv1_test/19961001')[1]
    D = get_files_from_directory('../rcv1_test/19960821/', None)[1]
    #D = list(range(101, 201, 1))
    #print(read_from_file('topics_processed'))

    evaluate(D, mode='docs', external = True)


main()