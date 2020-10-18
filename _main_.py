# --------------------------------
# 1st Delivery of the PRI project
# 86379 - Ana Evans
# 86389 - Artur Guimarães
# 86417 - Francisco Rosa
# --------------------------------
import sys
import os, os.path
import shutil
import re
import time
import sklearn
import spacy
import whoosh
from whoosh import index
from whoosh.fields import *
import nltk
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from bs4 import BeautifulSoup

#--------------------------------------------------
# Get files recursively
#--------------------------------------------------
def get_xml_files_recursively(path):
    files_list = []
    directory_list = os.listdir(path)
    for f in directory_list:
        n_path = '{}{}/'.format(path,f)
        if os.path.isdir(n_path):
            files_list.extend(get_xml_files_recursively(n_path))
        else:
            files_list.append('{}{}'.format(path,f))
    return files_list

# -------------------------------------------------
# Auxiliary function to get all files from a folder
# -------------------------------------------------
def get_files_from_directory(path):
    file_list = get_xml_files_recursively(path)
    parsed_files = []

    for f in file_list:
        open_file = open(f, 'r')
        parsed_file = BeautifulSoup(open_file.read(), 'lxml')
        
        if parsed_file.copyright != None:
            parsed_file.copyright.decompose()

        if parsed_file.codes != None:
            parsed_file.codes.decompose()
              
        parsed_files += [parsed_file,]

    return parsed_files

# -------------------------------------------------
# Preprocessing function
# -------------------------------------------------
def processing(text):
    
    # Lowercasing the entire string
    text = text.lower()

    # Tokenization
    tokens = nltk.word_tokenize(text)

    p_tokens = []

    # TODO: Lem and Stem
    #nltk.stem.snowball.EnglishStemmer()

    lemma = WordNetLemmatizer()

    #Remove stopwords and punctuation
    for word in tokens:
        if word.isalpha() and word not in stopwords.words('English'):
            word = lemma.lemmatize(word)
            p_tokens += [word, ]
    return p_tokens

    
# TODO

# --------------------------------------------------------------------------------
# @input D and optional set of arguments on text preprocessing

# @behavior preprocesses each document in D and builds an efficient inverted index
# (with the necessary statistics for the subsequent functions)

# @output tuple with the inverted index I, indexing time and space required
# --------------------------------------------------------------------------------
def indexing(D, **kwargs):
    ind_id = '1'

    start_time = time.time()
    ind_name = 'index{}'.format(ind_id)
    ind_dir = '{}_dir'.format(ind_name)

    if os.path.exists(ind_dir):
        shutil.rmtree(ind_dir)
        os.mkdir(ind_dir)
    else:
        os.mkdir(ind_dir)

    schema = Schema(id= NUMERIC(stored=True), context =TEXT)
    ind = index.create_in(ind_dir, schema=schema, indexname=ind_name)
    ind_writer = ind.writer()

    if not index.exists_in(ind_dir, indexname=ind_name):
        print("Error creating index")
        return

    #for doc in D:
        #p_headline = 
        #print(doc.title)
        #print(doc.headline)
        #print(doc.byline)
        #print(doc.dateline)
        #print(doc.find_all('text'))
        #print(doc.get_text())

    time_required = 'Time Elapsed {:4f} seconds'.format(time.time() - start_time)
    size_required = 'Size required {} bytes'.format(os.path.getsize(ind_dir))
    print(time_required)
    print(size_required)

    return None

# ------------------------------------------------------------------------------------------
# @input topic q ∈ Q (identifier), inverted index I, number of top terms for the
# topic (k), and optional arguments on scoring

# @behavior selects the top-k informative terms in q against I using parameterizable scoring

# @output list of k terms (a term can be either a word or phrase)
# ------------------------------------------------------------------------------------------
def extract_topic_query(q, I, k, **kwargs):
    return

# ------------------------------------------------------------------------------------------
# @input topic q (identifer), number of top terms k, and index I

# @behavior maps the inputed topic into a simplified Boolean query using 
# extract topic query and then search for matching* documents using the Boolean IR model

# @output the altered collection, specifically an ordered list of document identifiers
# ------------------------------------------------------------------------------------------
def boolean_query(q, k, I, **kwargs):
    return

# ------------------------------------------------------------------------------------------------
# @input topic q ∈ Q (identifier), number of top documents to return (p), index I,
# optional arguments on IR models

# @behavior uses directly the topic text (without extracting simpler queries) to rank
# documents in the indexed collection using the vector space model or probabilistic retrieval model

# @output ordered set of top-p documents, specifically a list of pairs – (document
# identifier, scoring) – ordered in descending order of score
# -------------------------------------------------------------------------------------------------
def ranking(q, p, I, **kwargs):
    return

# -------------------------------------------------------------------------------------------------
# @input set of topics Qtest ⊆ Q, document collection D_test, relevance feedback
# R_test, arguments on text processing and retrieval models

# @behavior uses the aforementioned functions of the target IR system to test simple
# retrieval (Boolean querying) tasks or ranking tasks for each topic q ∈
# Q_test, and comprehensively evaluates the IR system against the available
# relevance feedback

# @output extensive evaluation statistics for the inpufied queries, including recalland-precision 
# curves at difierent output sizes, MAP, BPREF analysis, cumulative gains and eficiency
# -------------------------------------------------------------------------------------------------
def evaluation(Q_test, R_test, D_test, **kwargs):
    return

# --------------------------------------------------------------------------------
# ~ Just the Main Function ~
# --------------------------------------------------------------------------------
def main():

    D_set = get_files_from_directory('../rcv1_test/test/')    #test
    indexing(D_set)
    #print(processing("Mice and cheeses"))

main()