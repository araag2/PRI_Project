# --------------------------------
# 2nd Delivery of the PRI project
# 86379 - Ana Evans
# 86389 - Artur Guimarães
# 86417 - Francisco Rosa
# --------------------------------
import re
import nltk
import spacy
import whoosh
import shutil
import sklearn
import math
import matplotlib as mpl 
import matplotlib.pyplot as plt
from heapq import nlargest 
from whoosh import index
from whoosh import scoring
from whoosh.qparser import *
from whoosh.fields import *
from sklearn.metrics import *

# File imports
from data_set_treatment import *
from file_treatment import *

topics = {}
judged_documents = {}
index_id = 1

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

    processed_docs = process_collection(D, True, **kwargs)

    for doc in processed_docs:
        ind_writer.add_document(id=int(doc), content=processed_docs[doc])

    ind_writer.commit()
    
    time_required = round(time.time() - start_time, 6)
    
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

# -------------------------------------------------------------------------------------------------
# bpref - Function that runs the bpref evaluation metric
# -------------------------------------------------------------------------------------------------
def bpref(sol_labels):
    R = 0
    N = 0
    bpref = 0
    n_count = 0
    for label in sol_labels:
        if label == 0:
            N += 1
        else:
            R += 1

    for label in sol_labels:
        if label == 0:
            n_count += 1
        else:
            bpref += (1 - n_count/(min(R,N)))

    return (1 / R) * bpref

# -------------------------------------------------------------------------------------------------
# reciprocal_rank_fusion - Auxiliary function to calculate the RRF for the top-p documents
# Uses the formula RBF_score(f) = sum (1 / (50 + rank_f))
# -------------------------------------------------------------------------------------------------
def reciprocal_rank_fusion(p, ranking_lists):
    document_ranks = {}

    for rank_l in ranking_lists:
        for i in range(len(rank_l )):
            if rank_l[i][0] not in document_ranks:
                document_ranks[rank_l[i][0]] = 0
            document_ranks[rank_l[i][0]] += 1 / (50 + i+1)

    p_highest = None

    if p != None:
        p_highest = nlargest(p, document_ranks, key=document_ranks.get)
    else:
        p_highest = nlargest(len(document_ranks), document_ranks, key=document_ranks.get)
    
    results = []

    for p in p_highest:
        results += [[p, document_ranks[p]]]  

    return results

# ------------------------------------------------------------------------------------------------
# ranking - Function that will query all documents in index I and rank the top p ones
#
# Input: q - The identifier number of the topic we want to search about 
#        p - The number of top ranked documents we will return
#        I - The Index object in which we will perform our search
#        **kwargs - Optional named arguments to parameterize scoring, with the following functionality (default values prefixed by *)
#               ranking [cosine | RRF | tf-idf | *bm25] - Chooses the scoring model we will use to score our terms
#               B [float | *0.75] - Free parameter for the BM25 model
#               content_B [float | *1.0] - Free parameter specific to the content field for the BM25 model
#               k1 [float | *1.5] - Free parameter for the BM25 model
#
# Behaviour: The function uses the weight vector generated by its given scoring system to search and rank  
# the top-p documents in the index according to the full topic text.
#
# Output: A List of lists that contains pairs [document_id, score] in descending score ordering
# -------------------------------------------------------------------------------------------------
def ranking(q, p, I, **kwargs):
    global topics
    topic = topics[q]

    weight_vector = None
    if 'ranking' not in kwargs:
        weight_vector = scoring.BM25F(B=0.75, content_B=1.0, K1=1.5)

    elif kwargs['ranking'] == 'cosine':
        weight_vector = scoring.FunctionWeighting(cosine_scoring)

    elif kwargs['ranking'] == 'tf-idf':
        weight_vector = scoring.TF_IDF()

    elif kwargs['ranking'] == 'bm25':
        b = 0.75 if 'B' not in kwargs else kwargs['B']
        content_b = 1.0 if 'content_B' not in kwargs else kwargs['content_B']
        k1 = 1.5 if 'K1' not in kwargs else kwargs['K1']

        weight_vector = scoring.BM25F(B=b, content_B=content_b, K1=k1)  

    elif kwargs['ranking'] == 'RRF':
        bm25_ranking_1 = ranking(q, p, I, ranking="bm25")
        bm25_ranking_2 = ranking(q, p, I, ranking="bm25", b=0.5, content_b=1.25, k1=1.25)
        bm25_ranking_3 = ranking(q, p, I, ranking="bm25", b=0.5, content_b=1.5, k1=1.00)

        return reciprocal_rank_fusion(p, [bm25_ranking_1, bm25_ranking_2, bm25_ranking_3])

    with I.searcher(weighting=weight_vector) as searcher:
        parser = QueryParser("content", I.schema, group=OrGroup).parse(topic)
        results = searcher.search(parser, limit=p)
        
        term_list = []

        if p != None:
            for i in range(p):
                if i < len(results):
                    term_list += [(int(results[i].values()[1]), results.score(i)), ]
        else:
            for i in range(len(results)):
                term_list += [(int(results[i].values()[1]), results.score(i)), ]

    return term_list

# -------------------------------------------------------------------------------------------------
# evaluate_ranked_query - Auxiliary function to calculate statistical data
# -------------------------------------------------------------------------------------------------
def evaluate_ranked_query(topic, o_labels, sol_labels, **kwargs):
    results = {}

    results['accuracy'] = accuracy_score(sol_labels, o_labels)
    results['precision-micro'] = precision_score(sol_labels, o_labels, average='micro', zero_division=1)
    results['precision-macro'] = precision_score(sol_labels, o_labels, average='macro', zero_division=1)
    results['recall-micro'] =  recall_score(sol_labels, o_labels, average='micro')
    results['recall-macro'] =  recall_score(sol_labels, o_labels, average='macro')
    results['f-beta-micro'] = fbeta_score(sol_labels, o_labels, average='micro', beta=0.5)
    results['f-beta-macro'] = fbeta_score(sol_labels, o_labels, average='macro', beta=0.5)
    results['MAP'] = average_precision_score(sol_labels, o_labels)
    results['BPREF'] = bpref(sol_labels)

    if 'curves' in kwargs and kwargs['curves']:
        precision, recall, _ = precision_recall_curve(sol_labels, o_labels)
        PrecisionRecallDisplay(precision=precision, recall=recall).plot()
        plt.title('Precision Recall curve for Ranked topic {}'.format(topic))
        plt.show()

    return results

# -------------------------------------------------------------------------------------------------
# evaluate_boolean_query - Auxiliary function to calculate statistical data
# -------------------------------------------------------------------------------------------------
def evaluate_boolean_query(topic, o_labels, sol_labels, **kwargs):
    results = {}

    results['accuracy'] = accuracy_score(sol_labels, o_labels)
    results['precision-micro'] = precision_score(sol_labels, o_labels, average='micro', zero_division=1)
    results['precision-macro'] = precision_score(sol_labels, o_labels, average='macro', zero_division=1)
    results['recall-micro'] =  recall_score(sol_labels, o_labels, average='micro')
    results['recall-macro'] =  recall_score(sol_labels, o_labels, average='macro')
    results['f-beta-micro'] = fbeta_score(sol_labels, o_labels, average='micro', beta=0.5)
    results['f-beta-macro'] = fbeta_score(sol_labels, o_labels, average='macro', beta=0.5)
    results['MAP'] = average_precision_score(sol_labels, o_labels)

    
    if 'curves' in kwargs and kwargs['curves']:
        precision, recall, _ = precision_recall_curve(sol_labels, o_labels)
        PrecisionRecallDisplay(precision=precision, recall=recall).plot()
        plt.title('Precision Recall curve for Boolean topic {}'.format(topic))
        plt.show()


    return results

# -------------------------------------------------------------------------------------------------
# display_results_per_q - Auxiliary function to display calculated statistical data
# -------------------------------------------------------------------------------------------------
def display_results_per_q(q, results_ranked, results_boolean):
    print("Result for search on Topic {}".format(q))
    print("\nRanked Search:")
    for p in results_ranked:
        result_str= ''
        for m in results_ranked[p]:
            result_str += '{} = {}, '.format(m, round(results_ranked[p][m],4)) 
        print("For p={}: {}".format(p, result_str[:-2]))

    print("\nBoolean Search:")
    for k in results_boolean:
        result_str= ''
        for m in results_boolean[k]:
            result_str += '{} = {}, '.format(m, round(results_boolean[k][m],4)) 
        print("For k={}: {}".format(k, result_str[:-2]))

        print("Result for search on Topic {}".format(q))

    return
# -------------------------------------------------------------------------------------------------------
# evaluation - Function that fully evaluates our IR model, providing full statiscal analysis for several
# p and k values across multiple ranges and topics
#
# Input: Q_test - The set of topics we will evaluate the perform of our IR model on
#        R_test - The topic labels we are looking for
#        D_test - Our test set in collection form
#        **kwargs - The additional args in this function also refer to the additional args in indexing(),
#        ranking() and boolean_query(), for which documentation is provided above. Other than that, we have:
#               k_range [list of ints | *[1,2,4,6,8,10]] - List of k values our model will test
#               p_range [list of ints or None | *[100,200,300,400,500]] - List of p values our model will test
#               curves [True | *False] - Display the precision/recall curves
#
# Behaviour: The function provides full statistics for every topic in Q_test, using R_test and D_test
# to build an index. Then, for each p in p_range it will use ranking() to rank the top p documents
# and for each k in k_range it will use k to evaluate the relevant docs using boolean_query(). In the end,
# it uses retrival results to provide full statiscal analysis.
#
# Output: Full statistical analysis for the provided input args
# -----------------------------------------------------------------------------------------------------
def evaluation(Q_test, R_test, D_test, **kwargs):
    # Standard index execution
    I = index.open_dir("index_judged_docs_dir", indexname='index_judged_docs')

    #I = indexing(D_test, **kwargs)[0]

    results_ranked = {}
    results_boolean = {}
    k_range = [1,2,4,6,8,10] if 'k_range' not in kwargs else kwargs['k_range']
    p_range = [100,200,300,400,500, None] if 'p_range' not in kwargs else kwargs['p_range']


    for q in Q_test:
        r_labels = find_R_test_labels(R_test[q])

        for p in p_range:
            score_docs = ranking(q, p, I, **kwargs)
            ranked_labels = find_ranked_query_labels(score_docs, r_labels)

            results_ranked[p] = evaluate_ranked_query(q, ranked_labels[0][:, 1],ranked_labels[1][:, 1], **kwargs)

        for k in k_range:
            boolean_docs = boolean_query(q, k, I, **kwargs)
            query_labels = find_boolean_query_labels(boolean_docs, r_labels)

            results_boolean[k] = evaluate_boolean_query(q, query_labels[0][:, 1], query_labels[1][:, 1], **kwargs)

    display_results_per_q(q, results_ranked, results_boolean)
    return

# --------------------------------------------------------------------------------------------
# overlapping_terms() - Function that finds the overlapping terms for a given k range
#
# Input: 
#
# Behaviour: Queries the top terms for all k's in a given k range and checks them for overlap
#
# Output: Data about the overlaping terms
# --------------------------------------------------------------------------------------------
def overlapping_terms():
    # Standard index execution
    I = index.open_dir("index_judged_docs_dir", indexname='index_judged_docs')

    k_range = [2,3,5,7,10,15]

    for k in k_range:
        top_terms = {}
        for q in range(101,201,1):
            results = extract_topic_query(q, I, k)
            for r in results:
                if r not in top_terms:
                    top_terms[r] = 0
                top_terms[r] += 1

        r_terms = 0
        for term in top_terms:
            if top_terms[term] > 1:
                r_terms += 1
        print("\nNumber of overlapping terms: {}".format(r_terms))
        print("Percent of overlapping terms: {}%".format(round(r_terms/len(top_terms)*100,3)))
        print(top_terms)
    return

# --------------------------------------------------------------------------------
# ~ Just the Main Function ~
# --------------------------------------------------------------------------------
def main():
    global topics
    material_dic = 'material/'

    R_set = get_R_set(material_dic)
    topics = get_topics(material_dic)

    #evaluation([120], R_set[0], [None], ranking='RRF')
    print(topics)

    return

#main()