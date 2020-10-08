import sklearn
import spacy
import nltk
import whoosh
import os
import glob

## Exercise 4.1 from Lab1
def inverted_index(dic):
    os.chdir(dic)
    files = glob.glob('*.txt')
    
    index = {}
    doc_index = 0
        
    for _ in files:
        doc_index += 1
        with open(_, 'r') as f:
            words = nltk.word_tokenize(f.read())
            word_vector = nltk.FreqDist(words)
            for item in word_vector.items():
                if item[0] in index:
                    index[item[0]][0] += 1
                    index[item[0]][1] += [(doc_index, item[1]), ]

                if item[0] not in index:
                    index[item[0]] = [0, []]
                    index[item[0]][0] = 1
                    index[item[0]][1] = [(doc_index, item[1])]
    print(index)
    return index

## Exercise 4.2 from Lab1
def statistics(index):
    n_seen_id = []
    n_documents = 0
    n_terms = len(index.keys())
    n_occurences = 0

    for term_i in index:
        for term in index[term_i][1]:
            if term[0] not in n_seen_id:
                n_seen_id += [term[0], ]
                n_documents += 1
            n_occurences += term[1]
                    
    print("Number of Documents is",n_documents)
    print("Number of Terms is",n_terms)
    print("Number of Term Total Occurences is",n_occurences)

index = inverted_index('lab2_material')
statistics(index)