import sklearn
import spacy
import nltk
import whoosh
import os
import glob
import matplotlib.pyplot as plt

## Exercise 4.1 from Lab1
def inverted_index(dic):
    os.chdir(dic)
    files = glob.glob('*.txt')
    
    index = {}
    auxiliary_map = {}
    index['document_count'] = 0

    for _ in files:
        index['document_count'] += 1
        with open(_, 'r') as f:
            auxiliary_map[index['document_count']] = []

            words = nltk.word_tokenize(f.read())
            word_vector = nltk.FreqDist(words)
            token_count = 0
            term_count = 0
            for item in word_vector.items():
                if item[0] in index:
                    index[item[0]][0] += 1
                    index[item[0]] += [(index['document_count'], item[1]), ]
                else:
                    index[item[0]] = [1,]
                    index[item[0]] += [(index['document_count'], item[1]), ]
                token_count += item[1]
                term_count += 1
            auxiliary_map[index['document_count']] = (token_count,term_count)
    print(index)
    print(auxiliary_map)
    return index, auxiliary_map

## Exercise 4.2 from Lab1
def statistics(index):
    n_terms = len(index.keys())
    n_occurences = 0

    for term_i in index:
        if term_i != 'document_count':
            for term in index[term_i][1:]:
                n_occurences += term[1]
                    
    print("Number of Documents indexed is", index['document_count'])
    print("Number of Terms indexed is",n_terms)
    print("Number of Term Individual Occurences is",n_occurences)

index,auxiliary_map = inverted_index('lab2_material')
statistics(index)

print(auxiliary_map)
plt.hist(auxiliary_map)
plt.show()