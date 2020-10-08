# --------------------------------
# 1st Delivery of the PRI project
# 86379 - Ana Evans
# 86389 - Artur Guimarães
# 86417 - Francisco Rosa
# --------------------------------

import sklearn
import spacy
import nltk
import whoosh

# @input D and optional set of arguments on text preprocessing

# @behavior preprocesses each document in D and builds an efficient inverted index
# (with the necessary statistics for the subsequent functions)

# @output tuple with the inverted index I, indexing time and space required
def indexing(D, **kwargs):
    return








# TODO

# @input topic q ∈ Q (identifier), inverted index I, number of top terms for the
# topic (k), and optional arguments on scoring

# @behavior selects the top-k informative terms in q against I using parameterizable scoring

# @output list of k terms (a term can be either a word or phrase)
def extract_topic_query(q, I, k, **kwargs):
    return

# @input topic q (identifer), number of top terms k, and index I

# @behavior maps the inputed topic into a simplified Boolean query using 
# extract topic query and then search for matching* documents using the Boolean IR model

# @output the altered collection, specifically an ordered list of document identifiers
def boolean_query(q, k, I, **kwargs):
    return

# @input topic q ∈ Q (identifier), number of top documents to return (p), index I,
# optional arguments on IR models

# @behavior uses directly the topic text (without extracting simpler queries) to rank
# documents in the indexed collection using the vector space model or probabilistic retrieval model

# @output ordered set of top-p documents, specifically a list of pairs – (document
# identifier, scoring) – ordered in descending order of score
def ranking(q, p, I, **kwargs):
    return

# @input set of topics Qtest ⊆ Q, document collection D_test, relevance feedback
# R_test, arguments on text processing and retrieval models

# @behavior uses the aforementioned functions of the target IR system to test simple
# retrieval (Boolean querying) tasks or ranking tasks for each topic q ∈
# Q_test, and comprehensively evaluates the IR system against the available
# relevance feedback

# @output extensive evaluation statistics for the inpufied queries, including recalland-precision 
# curves at difierent output sizes, MAP, BPREF analysis, cumulative gains and eficiency
def evaluation(Q_test, R_test, D_test, **kwargs):
    return