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
import matplotlib as mpl 
import matplotlib.pyplot as plt
from heapq import nlargest 
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

# File imports
from _main_ import get_files_from_directory
from _main_ import process_collection

# --------------------------------------------------------------------------
# build_graph - Builds a document graph from document collection D using
# the similarity measure in sim agains theta threshold
#
# Input: D - The document collection to build our graph with
#        sim - The similarity measure used
#        theta - The similarity threshold 
#        kwargs -
#
# Behaviour: 
#
# Output: A dictionary representing the weighted undirect graph that
# connect all documents on the basis of the given similarity measure
# -----------------------------------------------------------------------------
def build_graph(D, sim, theta, **kwargs):
    doc_list = process_collection(D, **kwargs)

    graph = {}
    #for doc in doc_list:

    return

# -----------------------------------------------------------------------
# undirected_page_rank - 
#
# Input: 
# 
# Behaviour: 
#
# Output:
# -----------------------------------------------------------------------
def undirected_page_rank(q, D, p, sim, theta, **kwargs):
    return

# --------------------------------------------------------------------------------
# ~ Just the Main Function ~
# --------------------------------------------------------------------------------
def main():
    D_set = get_files_from_directory('../rcv1/19960820')
    build_graph(D_set[1])
    return


main()