# --------------------------------
# 1st Delivery of the PRI project
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
from bs4 import BeautifulSoup
from lxml import etree
from whoosh import index
from whoosh import scoring
from whoosh.qparser import *
from whoosh.fields import *
from sklearn.metrics import *
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer

topics = {}
index_id = 1
# -----------------------------------------------------------------------
# getTopics - Auxiliary function that gathers info on all topics
#
# Input: directory - Directory path for project materials
# 
# Behaviour: Extracts topic info from '{directory}topics.txt' and updates
# the global dictionary which stores topic info
#
# Output: None
# -----------------------------------------------------------------------
def getTopics(directory):
    global topics
    
    topic_f = open('{}topics.txt'.format(directory), 'r')
    parsed_file = BeautifulSoup(topic_f.read(), 'lxml')

    topic_list = parsed_file.find_all('top')

    for topic in topic_list:
        split_topic = topic.getText().split('\n')
        split_topic = list(filter(lambda x: x!='', split_topic))

        number = split_topic[0].split(' ')[2][1:]
        title = processing(split_topic[1])
        topics[int(number)] = re.sub(' +',' ',title)  
    return

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
def get_R_set(directory):

    r_test_f = open('{}qrels_test.txt'.format(directory), 'r')
    r_train_f = open('{}qrels_train.txt'.format(directory), 'r')

    r_test_lines = r_test_f.readlines()
    r_train_lines = r_train_f.readlines()

    r_test_lines = [r_test_lines, r_train_lines]
    r_set = [{},{}]
    
    for i in range(2):
        for line in r_test_lines[i]:
            split_entry = line.split(' ')
            topic_id = int(split_entry[0][1:])
            doc_id = int(split_entry[1])
            feedback = int(split_entry[2])
            
            if topic_id not in r_set[i]:
                r_set[i][topic_id] = {}
            r_set[i][topic_id][doc_id] = feedback

    return r_set

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
            files_list.append(re.sub('//','/','{}/{}'.format(path,f)))
    return files_list

# -------------------------------------------------
# Auxiliary function to get all files from a folder
# -------------------------------------------------
def get_files_from_directory(path):
    file_list = get_xml_files_recursively(path)

    parsed_files_test = []
    parsed_files_train = []

    for f in file_list:
        print(f)
        date_identifier = int(f.split('/')[2])

        open_file = open(f, 'r')
        parsed_file = BeautifulSoup(open_file.read(), 'lxml')
        
        if parsed_file.copyright != None:
            parsed_file.copyright.decompose()

        if parsed_file.codes != None:
            parsed_file.codes.decompose()
              
        if date_identifier <= 19960930:
            parsed_files_train += [parsed_file,]
        else:
            parsed_files_test += [parsed_file,]

    return (parsed_files_test, parsed_files_train)

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

    #Remove stopwords
    for word in tokens:
        if word not in stopwords.words('English'):
            word = lemma.lemmatize(word)
            string_tokens += ' {}'.format(word)

    #Remove punctuation
    string_tokens = re.sub("[/-]"," ",string_tokens)
    string_tokens = re.sub("[.,;:\"\'!?`´()$£€]","",string_tokens)
    return string_tokens[1:]

# --------------------------------------------------------------------------------
# @input D and optional set of arguments on text preprocessing

# @behavior preprocesses each document in D and builds an efficient inverted index
# (with the necessary statistics for the subsequent functions)

# @output tuple with the inverted index I, indexing time and space required
# --------------------------------------------------------------------------------
def indexing(D, **kwargs):
    global index_id

    start_time = time.time()
    ind_name = 'index{}'.format(str(index_id))
    ind_dir = '{}_dir'.format(ind_name)

    if os.path.exists(ind_dir):
        shutil.rmtree(ind_dir)
        os.mkdir(ind_dir)
    else:
        os.mkdir(ind_dir)

    schema = Schema(id= NUMERIC(stored=True), content= TEXT(stored=True))
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
        
        result = nltk.word_tokenize('{} {} {}'.format(title, dateline, text))
        ind_writer.add_document(id=item_id, content=result)

    ind_writer.commit()
    
    time_required = round(time.time() - start_time, 6)
    
    index_id += 1

    #TODO: Fixme the size is wrong and wonky
    space_required = os.path.getsize(ind_dir)
    print(ind_dir)
    print(space_required)

    return (ind, time_required, space_required)

# ------------------------------------------------------------------------------------------
# @input topic q ∈ Q (identifier), inverted index I, number of top terms for the
# topic (k), and optional arguments on scoring

# @behavior selects the top-k informative terms in q against I using parameterizable scoring

# @output list of k terms (a term can be either a word or phrase)
# ------------------------------------------------------------------------------------------
def extract_topic_query(q, I, k, **kwargs):
    global topics 
    topic = topics[q]

    topic_terms = [] 
    with I.searcher() as searcher:
        parser = QueryParser("content", I.schema, group=OrGroup).parse(topic)
        results = searcher.search(parser, limit=None)
        res_list = [int(r.values()[1]) for r in results]

        numbers_list = []
        for i in res_list:
            numbers_list += [searcher.document_number(id=i),]
        #TODO: Implement **kwargs for the model='' attribute
        topic_terms = searcher.key_terms(numbers_list, "content", numterms=k)
      
    result = []
    for term in topic_terms:
        result += [term[0], ]

    return result

# ----------------------------------------------------------------
# Auxiliary function that gathers all documents with the top terms
# ----------------------------------------------------------------
def boolean_query_aux(document_lists, k):
    miss_m = round(0.2*k)
    seen = []
    result_docs = []

    for term_docs in document_lists:
        for doc in term_docs:
            if doc not in seen:
                chances = miss_m
                flag = True
                for doc_list in document_lists:
                    if doc not in doc_list:
                        if chances == 0:
                            flag = False
                            break
                        chances -= 1
                if flag:
                    result_docs += [doc,]
                seen += [doc, ]

    result_docs.sort()
    return result_docs

# ------------------------------------------------------------------------------------------
# @input topic q (identifer), number of top terms k, and index I

# @behavior maps the inputed topic into a simplified Boolean query using 
# extract topic query and then search for matching* documents using the Boolean IR model

# @output the altered collection, specifically an ordered list of document identifiers
# ------------------------------------------------------------------------------------------
def boolean_query(q, k, I, **kwargs):
    terms = extract_topic_query(q, I, k)

    document_lists = []
    with I.searcher() as searcher:
        for term in terms:
            parser = QueryParser("content", I.schema, group=OrGroup).parse(term)
            results = searcher.search(parser, limit=None)
            term_list = [int(r.values()[1]) for r in results]
            document_lists += [term_list,]
            
    return boolean_query_aux(document_lists, k)

# ------------------------------------------------------------------------------------------------
# @input topic q ∈ Q (identifier), number of top documents to return (p), index I,
# optional arguments on IR models

# @behavior uses directly the topic text (without extracting simpler queries) to rank
# documents in the indexed collection using the vector space model or probabilistic retrieval model

# @output ordered set of top-p documents, specifically a list of pairs – (document
# identifier, scoring) – ordered in descending order of score
# -------------------------------------------------------------------------------------------------
def ranking(q, p, I, **kwargs):
    global topics
    topic = topics[q]

    weight_vector = scoring.BM25F(B=0.75, content_B=1.0, K1=1.5)
    with I.searcher(weighting=weight_vector) as searcher:
        parser = QueryParser("content", I.schema, group=OrGroup).parse(topic)
        results = searcher.search(parser, limit=p)
        
        term_list = []
        for i in range(p):
            term_list += [(results[i].values()[1], results.score(i)), ]

    return term_list

# -------------------------------------------------------------------------------------------------
# Auxiliry function
# -------------------------------------------------------------------------------------------------
def find_relevant_R_test(R_test):
    relevant_docs = []
    for doc in R_test:
        if R_test[doc] == 1:
            relevant_docs += [doc, ]

    return relevant_docs

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
    I = indexing(D_test)[0]

    results = {}

    for q in Q_test:
        boolean_docs = boolean_query(q, 2, I)
        ranked_docs = ranking(q, 10, I)
        relevant_docs = find_relevant_R_test(R_test[q])

        print(boolean_docs)
        print(ranked_docs)
        print(relevant_docs)

    return

# --------------------------------------------------------------------------------
# ~ Just the Main Function ~
# --------------------------------------------------------------------------------
def main():
    material_dic = 'material/'

    D_set = get_files_from_directory('../rcv1/19961001')    #test
    R_set = get_R_set(material_dic)
    getTopics(material_dic)

    Q_test = [101]
    evaluation(Q_test, R_set[0], D_set[0])    

main()