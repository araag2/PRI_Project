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
import math
import numpy as np
from bs4 import BeautifulSoup
from lxml import etree
from whoosh import index
from whoosh import scoring
from whoosh.qparser import *
from whoosh.fields import *
from sklearn.metrics import *
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from textblob import TextBlob

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
# get_xml_files_recursively - Auxiliary function to get_files_from_directory
#
# Input: path - The path to the parent directory or file from which to start our recursive function
#               
# Behaviour: Creates a list with the path to every file that's an hierarquical child of parent directory path,
# recursively going through each child in Post-Order traversing
#
# Output: A List with the paths to each file child
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
# get_files_from_directory - Recursively gets all files from directory or file path, parsing the files from xml to objects
# and spliting them in D_Test and D_Train in the conditions specified by our project
#
# Input: path - The path to the parent directory or file from which to start our search
#               
# Behaviour: It starts by creating a list with the path to every file that's an hierarquical child of parent directory path,
# recursively going through each child in Post-Order traversing. Afterwards it parses each and every file from xml to a runtime
# object using the BeautifulSoup library. At last after having all files in object form it splits the dataset in D_Test and D_Train
# sets, according to their identifier (D_Test -> identifier > 1996-09-30   D_Train -> identifier <= 1996-09-30)
#
# Output: A List with the Lists of file objects present in D_Test and D_Train
# -------------------------------------------------
def get_files_from_directory(path):
    file_list = get_xml_files_recursively(path)

    parsed_files_test = []
    parsed_files_train = []

    for f in file_list:
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

# --------------------------------------------------------------------------------
# indexing - Creates an index after processing all text on data set D
#
# Input: D - The data set we will be building the index with
#        **kwargs - Optional named arguments for text preprocessing, with the following functionality (default values prefixed by *)
#               lowercasing [*True | False]: Flag to perform Lowercasing 
#               punctuation [*True | False]: Flag to remove punction
#               spellcheck [True | *False]: Flag to perform spell check using TextBlob
#               stopwords [*True | False]: Flag to remove Stop Words 
#               simplication [*lemmatization | stemming | None]: Flag to perform Lemmatization or Stemming
#               
# Behaviour: This function starts by creating the directory for our Index, after initializing our Schema fields. It then
# processes all documents on data set D and stores valuable information from them on the index (identifier, title, dateline and text).
# At last it commits the resulting processed documents to our index and calculates the total computational time the function used and the
# Disk space required to store the index.
#
# Output: A triplet tuple with the Inverted Index in object structure, the computational time for the function and 
# the disk space required to store the Inverted Index 
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
        title = processing(re.sub('<[^<]+>', "", str(doc.title)), **kwargs)
        dateline = processing(re.sub('<[^<]+>|\w[0-9]+-[0-9]+-[0-9]+\w', "", str(doc.dateline)), **kwargs)
        text = processing(re.sub('<[^<]+>', "", str(doc.find_all('text')))[1:-1], **kwargs)
        
        result = nltk.word_tokenize('{} {} {}'.format(title, dateline, text))
        ind_writer.add_document(id=item_id, content=result)

    ind_writer.commit()
    
    time_required = round(time.time() - start_time, 6)
    
    index_id += 1

    #TODO: Fixme the size is wrong and wonky
    space_required = os.path.getsize(ind_dir)

    return (ind, time_required, space_required)

# --------------------------------------------------------------------------------------------------------------------------------------
# extract_topic_query - Return the top-k informative terms from the topic q agains I using parameterizable scoring
#
# Input: q - The identifier number of the topic we want to search about
#        I - The Index object in which we will perform our search
#        k - The number of top-k terms to return 
#        **kwargs - Optional named arguments to parameterize scoring, with the following functionality (default values prefixed by *)
#               scoring [freq | tf-idf | dfree | pl2 |*bm25] - Chooses the scoring model we will use to score our terms
#               C [float | *1.0] - Free parameter for the pl2 model
#               B [float | *0.75] - Free parameter for the BM25 model
#               content_B [float | *1.0] - Free parameter specific to the content field for the BM25 model
#               k1 [float | *1.5] - Free parameter for the BM25 model
#
# Behaviour: Extracting the relevant model information from **kwargs, this function uses the index I present in its arguments 
# to perform a scored search on the top-k informative terms for topic q. It does so by creating a QueryParser object to parse
# the entire lenght of terms from q we've stored in our global topics structure and by using searcher.key_terms() to return
# the top terms according to our scoring weight vector. 
#
# Output: A List that contains the top k terms 
# -----------------------------------------------------------------------------------------------------------------------------------------
def extract_topic_query(q, I, k, **kwargs):
    global topics 
    topic = topics[q]

    topic_terms = []
    weight_vector = None

    # Chooses which score model to use from kwargs
    if 'scoring' not in kwargs:
        weight_vector = scoring.BM25F(B=0.75, content_B=1.0, K1=1.5)

    elif kwargs['scoring'] == 'freq':
        weight_vector = scoring.Frequency()

    elif kwargs['scoring'] == 'tf-idf':
        weight_vector = scoring.TF_IDF()

    elif kwargs['scoring'] == 'dfree':
        weight_vector = scoring.DFree()

    elif kwargs['scoring'] == 'pl2':
        C = 1.0 if 'C' not in kwargs else kwargs['C']

        weight_vector = scoring.PL2(c=C)

    elif kwargs['scoring'] == 'bm25':
        b = 0.75 if 'B' not in kwargs else kwargs['B']
        content_b = 1.0 if 'content_B' not in kwargs else kwargs['content_B']
        k1 = 1.5 if 'K1' not in kwargs else kwargs['K1']

        weight_vector = scoring.BM25F(B=b, content_B=content_b, K1=k1)    

    with I.searcher(weighting=weight_vector) as searcher:
        parser = QueryParser("content", I.schema, group=OrGroup).parse(topic)
        results = searcher.search(parser, limit=None)
        res_list = [int(r.values()[1]) for r in results]

        numbers_list = []
        for i in res_list:
            numbers_list += [searcher.document_number(id=i),]

        topic_terms = searcher.key_terms(numbers_list, "content", numterms=k, normalize=True)
      
    result = []
    for term in topic_terms:
        result += [term[0], ]

    return result

# -------------------------------------------------------------------------------------------------------------------
# boolean_query_aux - Auxiliary function to boolean_query that will check repeated ocurrences of documents
#
# Input: document_lists - A List of Lists in which each inner List has all documents in which the n-th term appeared
#        k - The number of terms we are using
#
# Behaviour: The function starts by calculating our error margin, in other words the number of missmatches a document
# can have before we stop considering it as relevant. This function composes a very simple algorithmn, where for each
# document we find in a sublist (non repeated, we use the list 'seen' to check that) we check if it's contained within 
# all other sublists, until it's not contained in miss_m + 1 lists. When that's the case, the document is no longer 
# relevant and we move on to the next one, iterating upon all elements of all sublists. The Time Complexity of this 
# function is O(N^2) while the Space Complexity is O(N)
#
# Output: A List of all relevants docs that don't exceed miss_m missmatches
# -------------------------------------------------------------------------------------------------------------------
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
# boolean_query - Function that will query all documents in index I and find those who contain
# all top k-terms relevant to topic q allowing up to round(0.2*k) missmatches 
#
# Input: q - The identifier number of the topic we want to search about 
#        k - The number of top k-terms to check documents for
#        I - The Index object in which we will perform our search
#
# Behaviour: The function starts by running extract_topic_query to return top k-terms with which
# we will search for the relevant docs for topic q. Then we use the index I to perform a simple
# search on, parsing the result of our search per term to our auxiliary function. 
#
# Output: A List of all relevants docs that don't exceed miss_m missmatches
# ------------------------------------------------------------------------------------------
def boolean_query(q, k, I, **kwargs):
    terms = extract_topic_query(q, I, k, **kwargs)

    document_lists = []
    with I.searcher() as searcher:
        for term in terms:
            parser = QueryParser("content", I.schema, group=OrGroup).parse(term)
            results = searcher.search(parser, limit=None)
            term_list = [int(r.values()[1]) for r in results]
            document_lists += [term_list,]
            
    return boolean_query_aux(document_lists, k)


# ------------------------------------------------------------------------------------------
# cosine_scoring - Function that scores a document based on cosine similarity 
#
# Input: searcher - The searcher associated with the index I
#        all the other arguments are built-ins from FunctionWeighting() and old whoosh.scoring
#        documentation
#
# Behaviour: Uses the tf-idf result from searcher.idf() and applies cosine similarity formula
# to it
#
# Output: cosine similarity weight vector formula 
# ------------------------------------------------------------------------------------------
def cosine_scoring(searcher, fieldnum, text, docnum, weight, QTF=1):
    idf = searcher.idf(fieldnum, text)

    DTW = (1.0 + math.log(weight)) * idf
    QMF = 1.0
    QTW = ((0.5 + (0.5 * QTF/ QMF))) * idf
    return DTW * QTW

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

    weight_vector = None
    if 'ranking' not in kwargs:
        weight_vector = scoring.BM25F(B=0.75, content_B=1.0, K1=1.5)

    elif kwargs['ranking'] == 'cosine':
        weight_vector = scoring.FunctionWeighting(cosine_scoring)

    elif kwargs['ranking'] == 'bm25':
        b = 0.75 if 'B' not in kwargs else kwargs['B']
        content_b = 1.0 if 'content_B' not in kwargs else kwargs['content_B']
        k1 = 1.5 if 'K1' not in kwargs else kwargs['K1']

        weight_vector = scoring.BM25F(B=b, content_B=content_b, K1=k1)  


    with I.searcher(weighting=weight_vector) as searcher:
        parser = QueryParser("content", I.schema, group=OrGroup).parse(topic)
        results = searcher.search(parser, limit=p)
        
        term_list = []
        for i in range(p):
            term_list += [(results[i].values()[1], results.score(i)), ]

    return term_list

# -------------------------------------------------------------------------------------------------
# Auxiliary function
# -------------------------------------------------------------------------------------------------
def find_R_test_labels(R_test):
    r_labels = []
    for doc in R_test:
        r_labels += [[doc, R_test[doc]], ]

    return np.array(r_labels)

# -------------------------------------------------------------------------------------------------
# Auxiliary function
# -------------------------------------------------------------------------------------------------
def find_query_labels(query_docs, r_labels):
    query_labels = []
    for doc in r_labels:
        if doc[0] in query_docs:
            query_labels += [[doc[0], 1], ]
        else:
            query_labels += [[doc[0], 0], ]

    return np.array(query_labels)    

# -------------------------------------------------------------------------------------------------
# Evaluate
# -------------------------------------------------------------------------------------------------
def evaluate(topic, o_labels, sol_labels):

    print("Accuracy of {}: {}".format(topic, accuracy_score(sol_labels, o_labels)))
    print("Micro Precision of {}: {}".format(topic, precision_score(sol_labels, o_labels, average='micro')))
    print("Micro F1-score of  {}: {}".format(topic, recall_score(sol_labels, o_labels, average='micro')))

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
    I = indexing(D_test)[0]

    results = {}
    k_range = [1,2,4,6,8,10]

    for q in Q_test:
        r_labels = find_R_test_labels(R_test[q])

        for k in k_range:
            boolean_docs = boolean_query(q, k, I)
            query_labels = find_query_labels(boolean_docs, r_labels)

            evaluate(q, query_labels[:, 1], r_labels[:, 1])
    return

# --------------------------------------------------------------------------------
# ~ Just the Main Function ~
# --------------------------------------------------------------------------------
def main():
    material_dic = 'material/'

    R_set = get_R_set(material_dic)
    getTopics(material_dic)

    D_set = get_files_from_directory('../rcv1_test/19961001')    #test
    I = indexing(D_set[0])
    #extract_topic_query(101, I[0], 5, scoring='bm25')
    #print(ranking(101, 5, I[0], scoring='cosine'))

    Q_test = [101]
    evaluation(Q_test, R_set[0], D_set[0])    

main()