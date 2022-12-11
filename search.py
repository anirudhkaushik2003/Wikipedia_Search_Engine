from collections import defaultdict
import pdb
import sys
import threading
import math
import re
import os
import numpy as np
import sys
import timeit
from nltk.corpus import stopwords
from nltk.stem.porter import *
import re
import Stemmer
import threading
import numpy as np


stemmer = Stemmer.Stemmer("english")
# stemmer = PorterStemmer()


stopWords = set(stopwords.words("english"))
stop_map = defaultdict(int)

for word in stopWords:
    stop_map[word] = 1  # is bool faster than int?



num_files = 0

class TextProcesser:
    def Stem(words):  # vectorize with np arrays
        return stemmer.stemWords(
            words
        )  # apparently stemword-> (gives unhashable type list error) and stemwords are different

    def tokenize(text):
        text = text.encode("ascii", "ignore").decode()
        text = re.sub(r"http[^\ ]*\ ", r" ", text)  # removing urls
        text = re.sub(
            r"&nbsp;|&lt;|&gt;|&amp;|&quot;|&apos;", r" ", text
        )  # removing html entities
        text = re.sub(r'[^a-zA-Z0-9]',r' ',text)  # removing special characters
        # Add support for foreign language words (don't remove accents (aigu, grave, etc.))
        return text.split()  # rematch returns a list not a string

    def StopWordRemover(words):
        return [w for w in words if stop_map[w] != 1]


def getTotalFiles():
    return len(list(os.listdir("words/")))
def getTotalDocs():
    return len(list(os.listdir("docs/")))

def hash(word):
    global num_files
    for i in range(num_files):
        f = open("words/"+str(i)+".txt", 'r')
        content = f.readlines()
        low = content[0].split()[0]
        high = content[-1].split()[0]
        if word <= high and word >= low:
            return (len(content), i)

def binarySearchFile(arr,low, high, x):
    # Check base case

    if high >= low:
        mid = (high + low) // 2
        m = arr[mid].rstrip("\n").rstrip().split()[0]
        # If element is present at the middle itself
        if m == x:
            return mid

        # If element is smaller than mid, then it can only
        # be present in left subarray
        elif m > x:
            return binarySearchFile(arr, low, mid - 1, x)

        # Else the element can only be present in right subarray
        else:
            return binarySearchFile(arr, mid + 1, high, x)

    else:
        # Element is not present in the array
        return -1


def fieldQuery(words, fields):
    words_dict = defaultdict(lambda: defaultdict(list))
    for i in range(len(fields)):
        if fields[i] == "t":
            fields[i] = "title"
        if fields[i] == "b":
            fields[i] = "body"
        if fields[i] == "i":
            fields[i] = "infobox"
        if fields[i] == "c":
            fields[i] = "category"
        if fields[i] == "r":
            fields[i] = "reference"
        if fields[i] == "l":
            fields[i] = "link"
    l1 = len(words)
    words2 = []
    for i in range(1,l1+1):
        x = ""
        for j in range(0,i):
            x += words[j]
        words2.append(x)
    for i in range(len(words)):
        num_lines, fileNo = hash(words[i])
        fname = "words/"+str(fileNo)+".txt"
        arr = []
        f = open(fname, 'r') 
        lineNo = binarySearchFile(f.readlines(), 0, num_lines,words[i] )
        f.close()
        if lineNo != -1:
            with open(fields[i]+"/"+str(fileNo)+".txt", 'r') as f:
                data = f.readlines()[lineNo]
                data = data.split()
                for j in range(len(data)):
                    data[j] = data[j].split("-") #docID-tf for each field
                    words_dict[data[j][0]][fields[i]].append((words[i],str(float(data[j][1])*200)))
            f.close()
            fields2 = ['title', 'body', 'infobox', 'category', 'reference', 'link']
            for field in fields2:
                if field != fields[i]:
                    with open(field+"/"+str(fileNo)+".txt", 'r') as f:
                        data = f.readlines()[lineNo]
                        data = data.split()
                        for j in range(len(data)):
                            data[j] = data[j].split("-") #docID-tf for each field
                            words_dict[data[j][0]][field].append((words[i],str(float(data[j][1])*0.1)))
                    f.close()
    fields = ['title', 'body', 'infobox', 'category', 'reference', 'link']
    for word in words2:
        num_lines, fileNo = hash(word)
        fname = "words/"+str(fileNo)+".txt"
        f = open(fname, 'r') 
        lineNo = binarySearchFile(f.readlines(), 0, num_lines,word)
        f.close()
        if lineNo != -1:
            for field in fields:
                with open(field+"/"+str(fileNo)+".txt", 'r') as f:
                    data = f.readlines()[lineNo]
                    data = data.split()
                    for i in range(len(data)):
                        data[i] = data[i].split("-") #docID-tf for each field
                        words_dict[data[i][0]][field].append((word,str(float(data[i][1])*0.5)))
                f.close()
    rank(words_dict)
    # getTitles(words_dict)
    return

def simpleQuery(words):
    l1 = len(words)
    words2 = []
    for i in range(1,l1+1):
        x = ""
        for j in range(0,i):
            x += words[j]
        words2.append(x)
    global num_files
    global num_docs
    words_dict = defaultdict(lambda: defaultdict(list))
    fields = ['title', 'body', 'infobox', 'category', 'reference', 'link']
    for word in words:
        num_lines, fileNo = hash(word)
        fname = "words/"+str(fileNo)+".txt"
        f = open(fname, 'r') 
        lineNo = binarySearchFile(f.readlines(), 0, num_lines,word)
        f.close()
        if lineNo != -1:
            for field in fields:
                with open(field+"/"+str(fileNo)+".txt", 'r') as f:
                    data = f.readlines()[lineNo]
                    data = data.split()
                    for i in range(len(data)):
                        data[i] = data[i].split("-") #docID-tf for each field
                        words_dict[data[i][0]][field].append((word,data[i][1]))
                f.close()
    for word in words2:
        num_lines, fileNo = hash(word)
        fname = "words/"+str(fileNo)+".txt"
        f = open(fname, 'r') 
        lineNo = binarySearchFile(f.readlines(), 0, num_lines,word)
        f.close()
        if lineNo != -1:
            for field in fields:
                with open(field+"/"+str(fileNo)+".txt", 'r') as f:
                    data = f.readlines()[lineNo]
                    data = data.split()
                    for i in range(len(data)):
                        data[i] = data[i].split("-") #docID-tf for each field
                        words_dict[data[i][0]][field].append((word,str(float(data[i][1])*0.01)))
                f.close()            
    # getTitles(words_dict)
    rank(words_dict)
    return

def getTitles(topTen):
    fp = open("queries_op.txt", "a")
    for docID in topTen:
        fileNo = int(docID) // 16000
        if int(docID) >= (fileNo*16000) and int(docID) <= ((fileNo+1)*16000):
            with open("docs/"+str(fileNo)+".txt", 'r') as f:
                docs = f.readlines()[(int(docID) % 16000) - 1 ]
                fp.write(docs.split(" ",1)[0] +", " + docs.split(" ",1)[1])
    fp.close()

def rank(words_dict):
    nfiles = len(words_dict.keys())
    queryIdf = {}
    field_scores = defaultdict(float)
    fields = ['title', 'body', 'infobox', 'category', 'reference', 'link']
    for field in fields:
        if field == "title": 
            field_scores[field]  = 0.35
        elif field == "body":
            field_scores[field] = 0.25
        elif field == "infobox":
            field_scores[field] = 0.20
        elif field == "category":
            field_scores[field] = 0.1
        elif field == "reference":
            field_scores[field] = 0.05
        elif field == "link":
            field_scores[field] = 0.05

    docFreq = defaultdict(float)
    weighted_scores = defaultdict(float)
    for docs in words_dict.keys(): # for each document
        df = 0.0
        wf = 0.0
        for field in words_dict[docs].keys(): # for each field for that document
            for word in words_dict[docs][field]: # for each word in that field
                df += float(word[1])
                wf += float(word[1]) * field_scores[field]
        docFreq[docs] = df
        weighted_scores[docs] = wf
            
            
    for key in docFreq:
        queryIdf[key] = math.log(abs(float(nfiles) - float(docFreq[key]) + 0.5) / ( float(docFreq[key]) + 0.5))
        docFreq[key] = queryIdf[key]*weighted_scores[key]
    
    sorted_docFreq = dict(sorted(docFreq.items(), key=lambda item: item[1], reverse=True))
    ind = 0
    topTen = []
    for key in sorted_docFreq:
        ind += 1
        topTen.append(key)
        if ind == 10:
            break
    
    #for key in docs:
    #    docs[key] /= float(math.sqrt(s1[key])) * float(math.sqrt(s2[key]))

    getTitles(topTen)

def search():

    words = "words/"
    titleMap = "docs/"


    while True:
        query = input('\nEnter Query:\n')
        # fp = open(sys.argv[1],'r')
        # query = fp.readline()
        # if not query:
        #     break

        start = timeit.default_timer()
        query = query.lower()
        d = TextProcesser

        if re.match(r'[t|b|i|c|r|l]:', query):
            words = re.findall(r'[t|b|c|i|l|r]:([^:]*)(?!\S)', query)
            tempFields = re.findall(r'([t|b|c|i|l|r]):', query)
            tokens = []
            fields = []
            for i in range(len(words)):
                for word in words[i].split():
                    fields.append(tempFields[i])
                    tokens.append(word)
            tokens = d.StopWordRemover(tokens)
            tokens = d.Stem(tokens)
            fieldQuery(tokens, fields)
            # results = rank(results, docFreq, nfiles, 'f')
        else:
            tokens = d.tokenize(query)
            tokens = d.StopWordRemover(tokens)
            tokens = d.Stem(tokens)
            simpleQuery(tokens)
            # results = rank(results, docFreq, nfiles, 's')



        end = timeit.default_timer()
        print('Time taken =', end-start)


if __name__ == '__main__':
    num_files = getTotalFiles()
    num_docs = getTotalDocs()
    search()
