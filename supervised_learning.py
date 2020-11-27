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
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from rank_bm25 import BM25Okapi

# File imports

from file_treatment import read_from_file
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

# -----------------------------------------------------------------------------------------------------
# create_vectorizer - Processes our entire document collection with a tf-idf vectorizer 
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
# Output: The created Vectorizer and the entire doc collection in vector form.
# -----------------------------------------------------------------------------------------------------
def create_vectorizer(doc_dic, feature_space, **kwargs):
    doc_keys = list(doc_dic.keys())
    doc_list = []

    for doc in doc_keys:
        doc_list.append(doc_dic[doc])

    norm = 'l2' if 'norm' not in kwargs else kwargs['norm']
    min_df = 2 if 'min_df' not in kwargs else kwargs['min_df']
    max_df = 0.8 if 'max_df' not in kwargs else kwargs['max_df']
    max_features = None if 'max_features' not in kwargs else kwargs['max_features']
    stop_words = None if 'remove_stopwords' not in kwargs else kwargs['remove_stopwords']
    vec = None
    doc_list_vectors = None

    if feature_space == 'tf':
        vec = CountVectorizer(min_df=min_df, max_df=max_df, max_features=max_features, stop_words= stop_words)
        vec.fit(doc_list)
        vec.transform(doc_list)
        
    elif feature_space == 'idf':
        vec = []
        vec.append(CountVectorizer(min_df=min_df, max_df=max_df, max_features=max_features, stop_words= stop_words))
        aux_vectors = vec[0].fit_transform(doc_list)

        vec.append(TfidfTransformer(smooth_idf=True, use_idf=True)) 
        doc_list_vectors = vec[1].fit_transform(aux_vectors)

    elif feature_space == 'tf-idf':
        vec = TfidfVectorizer(norm=norm, min_df=min_df, max_df=max_df, max_features=max_features, stop_words= stop_words)

    #TODO
    elif feature_space == 'bm25':
        return

    if type(vec) != list:
        vec.fit(doc_list)
        doc_list_vectors = vec.transform(doc_list)

    return [vec, doc_list_vectors]
    
# --------------------------------------------------------------------------------------------------------------------------------------
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
# --------------------------------------------------------------------------------------------------------------------------------------

def training(q, d_train, r_train, **kwargs):
    global topic_vectorizers

    classifiers = {'multinomialnb': MultinomialNB, 'kneighbors': KNeighborsClassifier, 'randomforest': RandomForestClassifier, 'mlp': MLPClassifier}
    classifier = classifiers['multinomialnb']() if 'classifier' not in kwargs else classifiers[kwargs['classifier']]()

    r_labels = find_R_test_labels(r_train[q])

    subset_dtrain = {}
    for doc in r_labels:
        subset_dtrain[doc] = d_train[doc]

    vec_results = create_vectorizer(subset_dtrain, 'idf', **kwargs)
    topic_vectorizers[q] = vec_results[0]
    d_train_vec = vec_results[1]
    
    r_labels = list(r_labels.values())
    
    classifier.fit(X=d_train_vec, y=r_labels)

    return classifier

# --------------------------------------------------------------------------------------------------------------------------------------
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
# --------------------------------------------------------------------------------------------------------------------------------------

def classify(d, q, M, **kwargs):
    vec = None
    vectorizers = topic_vectorizers[q]

    if type(vectorizers) != list:
        vec = vectorizers.transform(d)
    else:
        vec = vectorizers[1].transform(vectorizers[0].transform(d))

    return M.predict_proba(vec)[0][1]

# -------------------------------------------------------------------------------------------------
# display_results - Auxiliary function to display calculated statistical data
# -------------------------------------------------------------------------------------------------
def display_results(q, results):
    print("Result for search on Topic {}".format(q))
    result_str= ''
    for m in results:
        if m != 'ranking':
            result_str += '{} = {}, '.format(m, round(results[m],4)) 
    print("{}\n".format(result_str[:-2]))
  
    if 'ranking' in results:
        print("Top ranked documents on Topic {}".format(q))
        print("{}".format(results['ranking']))

    return

# --------------------------------------------------------------------------------------------------------------------------------------
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
# --------------------------------------------------------------------------------------------------------------------------------------
def evaluate(q_test, d_test, r_test, **kwargs):

    #classifiers = {'multinomialnb': MultinomialNB, 'kneighbors': KNeighborsClassifier, 'randomforest': RandomForestClassifier, 'mlp': MLPClassifier}
    #classifier = 'multinomialnb' if 'classifier' not in kwargs else kwargs['classifier']
    ranking = False if 'ranking' not in kwargs else kwargs['ranking']
    p = 5 if 'top_p' not in kwargs else kwargs['top_p']

    #for parametrization
    total_accuracy = 0

    for q in q_test:

        sol_labels = []
        o_labels_training = []
        trained_probs = {}

        trained_classifier = training(q, d_train, r_train, **kwargs)

        judged_docs = []
        for doc_id in r_test[q]:
            judged_docs.append(doc_id)
        
        for doc_id in judged_docs:

            trained_prob = classify([d_test[doc_id]], q, trained_classifier)
            trained_probs[doc_id] = trained_prob
            sol_labels.append(r_test[q][doc_id])
            o_labels_training.append(1 if trained_prob > 0.5 else 0)

        results = {}
        
        results['accuracy'] = accuracy_score(sol_labels, o_labels_training)
        results['precision-micro'] = precision_score(sol_labels, o_labels_training, average='micro', zero_division=1)
        results['precision-macro'] = precision_score(sol_labels, o_labels_training, average='macro', zero_division=1)
        results['recall-micro'] =  recall_score(sol_labels, o_labels_training, average='micro')
        results['recall-macro'] =  recall_score(sol_labels, o_labels_training, average='macro')
        results['f-beta-micro'] = fbeta_score(sol_labels, o_labels_training, average='micro', beta=0.5)
        results['f-beta-macro'] = fbeta_score(sol_labels, o_labels_training, average='macro', beta=0.5)

        if ranking:
            sort_probs = sorted(trained_probs, key = trained_probs.get, reverse=True)

            ranked_result = []
            result_range = range(p) if p <= len(sort_probs) else range(len(sort_probs))
            for i in result_range:
                doc_id = sort_probs[i]
                ranked_result.append((doc_id, round(trained_probs[doc_id], 4)))

            results['ranking'] = ranked_result

        #display_results(q, results)

        #for parametrization
        total_accuracy += results['accuracy']

    return total_accuracy / len(q_test)

# --------------------------------------------------------------------------------
# ~ Just the Main Function ~
# --------------------------------------------------------------------------------
def main():
    global topics
    global d_train
    global d_test
    global r_train
    global r_test

    d_test = read_from_file('collections_processed/Dtest_judged_collection_processed')
    d_train = read_from_file('collections_processed/Dtrain_judged_collection_processed')

    r_set = get_R_set('material/')[0]

    r_test = r_set[0]
    r_train = r_set[1]

    q_test = list(range(120,140))

    #evaluate(q_test, d_test, r_test, ranking=True, classifier='multinomialnb')

    #for parametrization
    for classifier in ['multinomialnb','kneighbors','randomforest','mlp']:
        
        best = (0,0)
        best_acc = 0

        min_df = 0
        while min_df <= 8:
            max_df = 0.6
            while max_df < 1:
                acc = evaluate(q_test, d_test, r_test, ranking=True, classifier=classifier, min_df=min_df, max_df=max_df)
                if acc > best_acc:
                    best = (min_df, max_df)
                    best_acc = acc
                max_df += 0.05
            min_df += 1

        topics = get_topics('material/')

        print(classifier)
        print(best)
        print(best_acc)

    return


main()