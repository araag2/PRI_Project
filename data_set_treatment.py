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
from scipy import stats
from bs4 import BeautifulSoup
from whoosh import index
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from textblob import TextBlob

# File imports
from file_treatment import get_files_from_directory

# -----------------------------------------------------------------------------------------------------------
# processing - Processes text in String form
#
# Input: text - The text in String form to be processed
#        **kwargs - Optional named arguments, with the following functionality (default values prefixed by *)
#               lowercasing [*True | False]: Flag to perform Lowercasing 
#               punctuation [*True | False]: Flag to remove punction
#               spellcheck [True | *False]: Flag to perform spell check using TextBlob
#               stopwords [*True | False]: Flag to remove Stop Words 
#               simplication [*lemmatization | stemming | None]: Flag to perform Lemmatization or Stemming
#               
# Behaviour: Procceses the text in the input argument text as refered to by the arguments in **kwargs,
# behaviour being completely dependent on them except for Tokenization which is always performed
#
# Output: A String with the processed text 
# ----------------------------------------------------------------------------------------------------------
def processing(text, **kwargs):

    p_text = text
    # Lowercasing the entire string
    if 'lowercasing' not in kwargs or kwargs['lowercasing']:
        p_text = p_text.lower()

    #Remove punctuation
    if 'punctuation' not in kwargs or kwargs['punctuation']:
        p_text = re.sub("[/-]"," ",p_text)
        p_text = re.sub("[.,;:\"\'!?`´()$£€]","",p_text)

    # Spell Check
    if "spellcheck" in kwargs and kwargs['spellcheck']:          
        p_text = str(TextBlob(p_text).correct())

    # Tokenization
    tokens = nltk.word_tokenize(p_text)
    string_tokens = ''

    # Spell Check correction
    if "spellcheck" in kwargs and kwargs['spellcheck']:
        n_tokens = []
        for word in tokens:           
            n_tokens += ' {}'.format(TextBlob(word).correct)

    # Lemmatization
    if 'simplification' not in kwargs or kwargs['simplification'] == 'lemmatization':
        lemma = WordNetLemmatizer()

        #Remove stopwords
        if 'stopwords' not in kwargs or kwargs['stopwords']:
            for word in tokens:
                if word not in stopwords.words('English'):   
                    string_tokens += ' {}'.format(lemma.lemmatize(word))
        else: 
            for word in tokens: 
                string_tokens += ' {}'.format(lemma.lemmatize(word))

    # Stemming
    elif kwargs['simplification'] == 'stemming':
        stemer = nltk.stem.snowball.EnglishStemmer()

        #Remove stopwords
        if 'stopwords' not in kwargs or kwargs['stopwords']:
            for word in tokens:
                if word not in stopwords.words('English'):   
                    string_tokens += ' {}'.format(stemer.stem(word))
        else: 
            for word in tokens: 
                string_tokens += ' {}'.format(stemer.stem(word))

    # Case for no simplification
    else:
        for word in tokens: 
            string_tokens += ' {}'.format(word)   

    # Removing the first whitespace in the output 
    return string_tokens[1:]

# -----------------------------------------------------------------------------
# process_collection - Small auxiliary function to externaly process a text 
# collection independently of program function
# -----------------------------------------------------------------------------

def process_collection(collection, tokenize, **kwargs):
    result = {}
    for doc in collection:
        item_id = int(doc.newsitem.get('itemid'))
        title = processing(re.sub('<[^<]+>', "", str(doc.title)), **kwargs)
        dateline = processing(re.sub('<[^<]+>|\w[0-9]+-[0-9]+-[0-9]+\w', "", str(doc.dateline)), **kwargs)
        text = processing(re.sub('<[^<]+>', "", str(doc.find_all('text')))[1:-1], **kwargs)
        
        if tokenize:
            result[item_id] = nltk.word_tokenize('{} {} {}'.format(title, dateline, text))
        else:
            result[item_id] = '{}\n{}\n{}'.format(title, dateline, text)

    return result

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
        doc_list.append(doc_dic[doc])

    norm = 'l2' if 'norm' not in kwargs else kwargs['norm']
    min_df = 2 if 'min_df' not in kwargs else kwargs['min_df']
    max_df = 0.8 if 'max_df' not in kwargs else kwargs['max_df']
    max_features = None if 'max_features' not in kwargs else kwargs['max_features']
    stop_words = None if 'remove_stopwords' not in kwargs else kwargs['remove_stopwords']

    vec = TfidfVectorizer(norm=norm, min_df=min_df, max_df=max_df, max_features=max_features, stop_words= stop_words)
    vec.fit(doc_list)

    doc_list_vectors = vec.transform(doc_list)

    return [vec, doc_keys, doc_list_vectors]

# -----------------------------------------------------------------------
# get_topics - Auxiliary function that gathers info on all topics
#
# Input: directory - Directory path for project materials
# 
# Behaviour: Extracts topic info from '{directory}topics.txt' and updates
# the global dictionary which stores topic info
#
# Output: None
# -----------------------------------------------------------------------
def get_topics(directory):
    topics = {}
    
    topic_f = open('{}topics.txt'.format(directory), 'r')
    parsed_file = BeautifulSoup(topic_f.read(), 'lxml')

    topic_list = parsed_file.find_all('top')

    for topic in topic_list:
        split_topic = topic.getText().split('\n')
        split_topic = list(filter(lambda x: x!='', split_topic))

        number = split_topic[0].split(' ')[2][1:]
        title = processing(split_topic[1])
        topics[int(number)] = re.sub(' +',' ',title)
    return topics

# -------------------------------------------------------------------------------------------------
# get_R_set - Auxiliary function that extracts the R set
#
# Input: directory - Directory path for project materials
# 
# Behaviour: Extracts the triplet (Topic id, Document id, Feedback) for each entry in the 
# R set, present in '{directory}qrels_test.txt' (R-test) and '{directory}qrels_test.txt' (R-train)
#
# Output: [R-Test, R-Train], each being a list of triplet entries
# -------------------------------------------------------------------------------------------------
def get_R_set(directory, **kwargs):
    judged_documents = {}

    r_test_f = open('{}qrels_test.txt'.format(directory), 'r')
    r_train_f = open('{}qrels_train.txt'.format(directory), 'r')

    r_test_lines = r_test_f.readlines()
    r_train_lines = r_train_f.readlines()

    r_test_lines = [r_test_lines, r_train_lines]
    r_set = [{},{}]
    
    if 'index' in kwargs and kwargs['index'] == 'doc_id':
        for i in range(2):
            for line in r_test_lines[i]:
                split_entry = line.split(' ')
                topic_id = int(split_entry[0][1:])
                doc_id = int(split_entry[1])

                if doc_id not in judged_documents: 
                    judged_documents[doc_id] = True

                feedback = int(split_entry[2])

                if doc_id not in r_set[i]:
                    r_set[i][doc_id] = {}
                r_set[i][doc_id][topic_id] = feedback

    else:
        for i in range(2):
            for line in r_test_lines[i]:
                split_entry = line.split(' ')
                topic_id = int(split_entry[0][1:])
                doc_id = int(split_entry[1])

                if doc_id not in judged_documents: 
                    judged_documents[doc_id] = True

                feedback = int(split_entry[2])

                if topic_id not in r_set[i]:
                    r_set[i][topic_id] = {}
                r_set[i][topic_id][doc_id] = feedback

    return [r_set, judged_documents]

# -------------------------------------------------------------------------------------------------
# find_R_test_labels - Function that finds the test labels for a given R_Set
#
# Input: R_test - The R_Test set 
#
# Behaviour: Extrapolates the feedback from the R_Test set to a dic or array
#
# Output: The R_Test set labels in dic or np array form
# -------------------------------------------------------------------------------------------------
def find_R_test_labels(R_test, **kwargs):
    r_labels = None

    if 'list' not in kwargs:
        r_labels = {}
        for doc in R_test:
            r_labels[doc] = R_test[doc]

    elif 'list' in kwargs and kwargs['list']:
        r_labels = []
        for doc in R_test:
            r_labels.append(R_test[doc])

    return r_labels

# -----------------------------------------------------------------------------
# get_judged_docs - Small auxiliary function that returns the judged documents
# in the given rcv1 directory
# -----------------------------------------------------------------------------
def get_judged_docs(material_dir, rcv1_dir):
    judged_documents = get_R_set(material_dir)[1]
    return get_files_from_directory(rcv1_dir, judged_documents, judged=True)

# -------------------------------------------------------------------------------------------------
# find_ranked_query_labels - Function that finds the test labels for given query_docs and r_labels
#
# Input: query_docs - The ranked query docs
#        r_labels - the labels R_Test set produced 
#
# Behaviour: Compares de R_Test set feedback with the ranked docs
#
# Output: The labels for the ranked query docs in np array form
# -------------------------------------------------------------------------------------------------
def find_ranked_query_labels(query_docs, r_labels):
    q_docs = np.array(query_docs)
    q_docs = q_docs[:,0]

    query_labels = []
    result_labels = []

    for doc in query_docs:
        if doc[0] in r_labels:
            query_labels += [[doc[0], 1], ]
            result_labels += [[doc[0], r_labels[doc[0]]], ]
    
    for doc in r_labels:
        if doc not in q_docs:
            query_labels += [[doc, 0], ]
            result_labels += [[doc, r_labels[doc]], ]

    return [np.array(query_labels), np.array(result_labels)]  

# -------------------------------------------------------------------------------------------------
# find_boolean_query_labels - Function that finds the test labels for given query_docs and r_labels
#
# Input: query_docs - The query docs
#        r_labels - the labels R_Test set produced 
#
# Behaviour: Compares the R_Test set feedback with the ranked docs
#
# Output: The labels for the query docs in np array form
# -------------------------------------------------------------------------------------------------
def find_boolean_query_labels(query_docs, r_labels):
    query_labels = []
    result_labels = []

    for doc in r_labels:
        if doc in query_docs:
            query_labels += [[doc, 1], ]
            result_labels += [[doc, r_labels[doc]]]
        else:
            query_labels += [[doc, 0], ]
            result_labels += [[doc, r_labels[doc]]]

    return [np.array(query_labels), np.array(result_labels)]  

# -----------------------------------------------------------------------------
# normalize_dic() - Normalizes a dic 
# -----------------------------------------------------------------------------
def normalize_dic(dic, **kwargs):
    result_dic = {}

    values = dic.values()
    value_it = iter(values)

    if type(next(value_it)) == str:
        values_list = np.array([len(doc) for doc in values])
    else:
        values_list = np.array([score for score in values])
    
    if 'norm_method' not in kwargs or kwargs['norm_method'] == '1': 
        values_list = values_list / np.linalg.norm(values_list)

    elif kwargs['norm_method'] == '2':
        values_list = normalize(values_list[:,np.newaxis], axis=0).ravel()

    elif kwargs['norm_method'] == 'zscore':
        values_list = stats.zscore(values_list)

    keys = list(dic.keys())
    for i in range(len(keys)):
        result_dic[keys[i]] = values_list[i]

    return result_dic