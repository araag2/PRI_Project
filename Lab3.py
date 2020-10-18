import os, os.path
from whoosh import index
from whoosh.fields import *
from whoosh.qparser import *

global directory

def create_index():
    document_list = open('pri_cfc.txt', 'r')
    documents = document_list.readlines()

    if not os.path.exists("indexdir"):
        os.mkdir("indexdir")

    global directory
    directory = index.open_dir('indexdir')

    schema = Schema(id = NUMERIC(stored=True), content=TEXT)
    ix = index.create_in("indexdir", schema)
    writer = ix.writer()

    for doc in documents:
        id_text = doc.split(' ', 1)
        writer.add_document(id = id_text[0], content=u"{}".format(id_text[1]))
    writer.commit() 
    return

def search_index(search):
    global directory
    q = QueryParser("content", directory.schema, group=OrGroup).parse(u'{}'.format(search))
    results = directory.searcher().search(q, limit=100)
    for r in results:
        print(r)
    return

def main():
    create_index()

    while(True):
        i = input()
        if input == "exit":
            break
        search_index(i)

    return

main()