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
from whoosh.qparser import *
from whoosh.fields import *
import nltk
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from bs4 import BeautifulSoup
from lxml import etree

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

    string_tokens = ''

    # TODO: Lem and Stem
    #nltk.stem.snowball.EnglishStemmer()
    lemma = WordNetLemmatizer()

    #Remove stopwords and punctuation
    for word in tokens:
        if word.isalpha() and word not in stopwords.words('English'):
            word = lemma.lemmatize(word)
            string_tokens += ' {}'.format(word)
    return string_tokens[1:]

    
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

    schema = Schema(id= NUMERIC(stored=True), content= TEXT)
    ind = index.create_in(ind_dir, schema=schema, indexname=ind_name)
    ind_writer = ind.writer()

    if not index.exists_in(ind_dir, indexname=ind_name):
        print("Error creating index")
        return

    for doc in D:
        item_id = doc.newsitem.get('itemid')
        title = processing(re.sub('<[^<]+>', "", str(doc.title)))
        dateline = processing(re.sub('<[^<]+>|\w[0-9]+-[0-9]+-[0-9]+\w', "", str(doc.dateline)))
        text = processing(re.sub('<[^<]+>', "", str(doc.find_all('text')))[1:-1])

        ind_writer.add_document(id=item_id, content=u'{} {} {}'.format(title, dateline, text))
        print(item_id)

    ind_writer.commit()

    for i in ind.searcher().documents():
        print(i)
    
    time_required = round(time.time() - start_time, 6)
    
    #TODO: Fix me the size is wrong and wonky
    space_required = os.path.getsize(ind_dir)

    return (ind, time_required, space_required)

# ------------------------------------------------------------------------------------------
# @input topic q ∈ Q (identifier), inverted index I, number of top terms for the
# topic (k), and optional arguments on scoring

# @behavior selects the top-k informative terms in q against I using parameterizable scoring

# @output list of k terms (a term can be either a word or phrase)
# ------------------------------------------------------------------------------------------
def extract_topic_query(q, I, k, **kwargs):

    with I.searcher() as searcher:
        parser = QueryParser("content", I.schema, group=OrGroup).parse(q)
        results = searcher.search(parser, limit=None)
        print(results[0])
        res_list = [r.values()[0] for r in results]
        print(res_list)
        
    return res_list

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

    D_set = get_files_from_directory('../rcv1_test/19960820/')    #test
    index = indexing(D_set)
    extract_topic_query("USA is amaziiing", index[0], 5)

main()