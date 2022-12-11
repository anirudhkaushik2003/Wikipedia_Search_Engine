from distutils.log import info
from operator import index
from os import getloadavg, link
import os
import sys
import xml.sax
from nltk.corpus import stopwords
from nltk.stem.porter import *
import re
from collections import defaultdict
import Stemmer
import threading
import numpy as np
import heapq
import threading
from tqdm import tqdm


stemmer = Stemmer.Stemmer("english")
# stemmer = PorterStemmer()


stopWords = set(stopwords.words("english"))
stop_map = defaultdict(int)
super_words = defaultdict(list)
docTitle = defaultdict(str)

for word in stopWords:
    stop_map[word] = 1  # is bool faster than int?

threads = []
pages = 0
total = 0
total_bef_stem = 0

fileNo = 0
indexNo = 0
docNo = 0


class Indexer:
    def __init__(self, filename):
        self.filename = filename
        self.super_words = np.array([])

    def index_creation(self, title, infobox, body, category, references, links, id):
        global pages
        global total
        global super_words
        print(pages)
        # self.f_id.write(str(id)+":"+str(pages)+"-")
        # title = np.array(title)
        # infobox = np.array(infobox)
        # body = np.array(body)
        # category = np.array(category)
        # references = np.array(references)
        # links = np.array(links)

        # unique, counts = np.unique(title, return_counts=True)
        # title_index = dict(zip(unique, counts))

        # unique, counts = np.unique(infobox, return_counts=True)
        # infobox_index = dict(zip(unique, counts))

        # unique, counts = np.unique(category, return_counts=True)
        # category_index = dict(zip(unique, counts))

        # unique, counts = np.unique(references, return_counts=True)
        # references_index = dict(zip(unique, counts))

        # unique, counts = np.unique(body, return_counts=True)
        # body_index = dict(zip(unique, counts))

        # unique, counts = np.unique(links, return_counts=True)
        # links_index = dict(zip(unique, counts))

        # global_merged = np.concatenate((title, infobox, body, category, references, links))
        # global_merged,counts = np.unique(global_merged,return_counts=True)
        # total += np.sum(counts)
        # for x in global_merged:
        #     temp = []
        #     temp.append("-d"+str(pages))
        #     if(x in title_index):
        #         temp.append("t"+str(title_index[x]))
        #     if(x in body_index):
        #         temp.append("b"+str(body_index[x]))
        #     if(x in infobox_index):
        #         temp.append("i"+str(infobox_index[x]))
        #     if(x in category_index):
        #         temp.append("c"+str(category_index[x]))
        #     if(x in references_index):
        #         temp.append("r"+str(references_index[x]))
        #     if(x in links_index):
        #         temp.append("l"+str(links_index[x]))

        #     super_words[x] += temp ### This method creates a memory issue and the process is killed

        words = defaultdict(int)
        mapper = defaultdict(int)

        for x in title:
            mapper[x] += 1
            words[x] += 1
            total += 1

        title = mapper
        mapper = defaultdict(int)

        for x in body:
            mapper[x] += 1
            words[x] += 1
            total += 1

        body = mapper
        mapper = defaultdict(int)

        for x in category:
            mapper[x] += 1
            words[x] += 1
            total += 1

        category = mapper
        mapper = defaultdict(int)

        for x in infobox:
            mapper[x] += 1
            words[x] += 1
            total += 1

        infobox = mapper
        mapper = defaultdict(int)

        for x in references:
            mapper[x] += 1
            words[x] += 1
            total += 1

        references = mapper
        mapper = defaultdict(int)

        for x in links:
            mapper[x] += 1
            words[x] += 1
            total += 1

        links = mapper
        mapper = defaultdict(int)

        for x in words.keys():
            t = title[x]
            b = body[x]
            c = category[x]
            i = infobox[x]
            r = references[x]
            l = links[x]

            temp = ["-d" + str(pages)]
            if t:
                temp.append("t" + str(t))
            if b:
                temp.append("b" + str(b))
            if c:
                temp.append("c" + str(c))
            if i:
                temp.append("i" + str(i))
            if r:
                temp.append("r" + str(r))
            if l:
                temp.append("l" + str(l))
            super_words[x] += temp

            temp = []

        global threads
        if pages % 16000 == 0:
            # w = Thread_write(super_words)
            # threads.append(Thread_write(super_words))
            # threads[-1].start()

            writer = NonThreadedWrite(super_words)
            writer.run()
            super_words = defaultdict(list)

    def long_mem_store(self, fileNo):
        files = {}
        global_merged = defaultdict(list)
        file_top = {}
        min_heap = []
        file_open = []
        words = 0
        prev_words = words
        first_word = {}
        global indexNo
        for i in range(fileNo):
            files[i] = open("index"+"/" + str(i) + ".txt", "r")
            file_top[i] = files[i].readline().strip()
            first_word[i] = file_top[i].split("-")
            file_open.append(1)
            if first_word[i][0] not in min_heap:
                heapq.heappush(min_heap, first_word[i][0])

        while any(file_open) == 1:
            min_word = heapq.heappop(min_heap) # get min word
            words += 1
            if words%160000 == 0: # every 160 thousand words
                # if avg posting list = 100 bytes to 1000 bytes, words = 10 bytes
                # writeWords(global_merged.keys(),indexNo)
                s = SecondaryIndex(global_merged)
                s.createIndex(words-prev_words)
                indexNo += 1
                global_merged = defaultdict(list) # write 160 * 10^7 or 8 bytes = 160 MB to 1.6 GB per file
                prev_words = words
            for i in range(fileNo):
                if file_open[i] == 1:
                    if first_word[i][0] == min_word: # each file is sorted so first entry is the min entry so word can only be present here
                        global_merged[min_word].extend(first_word[i][1:])
                        
                        file_top[i] = files[i].readline().strip() # read next line
                        if file_top[i] == "":
                            files[i].close()
                            file_open[i] = 0
                        else:
                            first_word[i] = file_top[i].split("-") # store the new first word of the file
                            if first_word[i][0] not in min_heap:
                                heapq.heappush(min_heap, first_word[i][0]) # push the word if it isn't already there
                                # won't add a word which has already been processed and removed
        s = SecondaryIndex(global_merged)
        s.createIndex(words-prev_words)
        print(f"words: {words}")

class NonThreadedWrite():
    def __init__(self, info):
        self.info = info

    def run(self):
        sorted_keys = sorted(self.info.keys())
        global fileNo
        with open("index"+"/"+str(fileNo) + ".txt", "w") as f:
            for key in sorted_keys:
                f.write(key + "".join(self.info[key])+"\n")

            f.close()
        fileNo += 1

def writeWords(words,indexNo):
    sorted_words = sorted(words)
    with open("words/"+str(indexNo) + ".txt", "w") as f:
        i = 0
        for key in sorted_words:
            f.write(key + " " + str(i)+"\n")
            i += 1

        f.close()

class Thread_write():
    def __init__(self, info):
        
        self.info = info

    def run(self):
        sorted_keys = sorted(self.info.keys())
        with open("words"+str(indexNo) + ".txt", "w") as f:
            for key in sorted_keys:
                f.write(key + "-" + "-".join(self.info[key])+"\n")

            f.close()

class SecondaryIndex():
    def __init__(self, postingList):
        self.postingList = postingList


    def createIndex(self,words):
        global indexNo
        title = defaultdict(dict)
        body = defaultdict(dict)
        category = defaultdict(dict)
        infobox = defaultdict(dict)
        reference = defaultdict(dict)
        link = defaultdict(dict)
        tf_list = defaultdict(int)
        for key in tqdm(sorted(self.postingList.keys())): # for each word
            tf_list[key] = 0 # get the tf of the word
            docs = self.postingList[key]
            for i in range(len(docs)):
                str_len = len(docs[i])
                d = ""
                t, b, c, info, r, l = 0, 0, 0, 0, 0, 0
                for j in range(str_len):
                    temp = ""
                    if j < str_len and docs[i][j] == "d":
                        j += 1
                        while j < str_len and (docs[i][j] != "t" and docs[i][j] != "b" and docs[i][j] != "c" and docs[i][j] != "i" and docs[i][j] != "r" and docs[i][j] != "l"):
                            temp += docs[i][j]
                            j += 1
                        d = temp
                    temp = ""
                    if j < str_len and docs[i][j] == "t":
                        j += 1
                        while j < str_len and (docs[i][j] != "b" and docs[i][j] != "c" and docs[i][j] != "i" and docs[i][j] != "r" and docs[i][j] != "l"):
                            temp += docs[i][j]
                            j += 1
                        if temp == "":
                            t = float(0)
                        else:
                            t = float(temp)
                    temp = ""
                    if j < str_len and docs[i][j] == "b":
                        j += 1
                        while j < str_len and (docs[i][j] != "c" and docs[i][j] != "i" and docs[i][j] != "r" and docs[i][j] != "l"):
                            temp += docs[i][j]
                            j += 1
                        if temp == "":
                            b = float(0)
                        else:
                            b = float(temp)
                    temp = ""
                    if j < str_len and docs[i][j] == "c":
                        j += 1
                        while j < str_len and (docs[i][j] != "i" and docs[i][j] != "r" and docs[i][j] != "l"):
                            temp += docs[i][j]
                            j += 1
                        if temp == "":
                            c = float(0)
                        else:
                            c = float(temp)
                    temp = ""
                    if j < str_len and docs[i][j] == "i":
                        j += 1
                        while j < str_len and (docs[i][j] != "r" and docs[i][j] != "l"):
                            temp += docs[i][j]
                            j += 1
                        if temp == "":
                            info = float(0)
                        else:
                            info = float(temp)
                    temp = ""
                    if j < str_len and docs[i][j] == "r":
                        j += 1
                        while j < str_len and (docs[i][j] != "l"):
                            temp += docs[i][j]
                            j += 1
                        if temp == "":
                            r = float(0)
                        else:
                            r = float(temp)
                    if j < str_len and docs[i][j] == "l":
                        j += 1
                        while j < str_len:
                            temp += docs[i][j]
                            j += 1
                        if temp == "":
                            l = float(0)
                        else:
                            l = float(temp)

                tf = 0
                if t != 0:
                    title[key][d] = t
                    tf += t
                if b != 0:
                    body[key][d] = b
                    tf += b
                if c != 0:
                    category[key][d] = c
                if info != 0:
                    infobox[key][d] = i
                if r != 0:
                    reference[key][d] = r
                    tf += r
                if l != 0:
                    link[key][d] = l
                    tf += l
                tf_list[key] += tf

        try:
            os.mkdir("./words")  
        except:
            pass
        try:
            os.mkdir("./title")
        except:
            pass
        try:
            os.mkdir("./body")
        except:
            pass
        try:
            os.mkdir("./infobox")
        except:
            pass
        try:
            os.mkdir("./category")
        except:
            pass
        try:
            os.mkdir("./reference")
        except:
            pass
        try:
            os.mkdir("./link")
        except:
            pass

        sorted_keys = sorted(self.postingList.keys())
        
        # dictionary
        tw = threading.Thread(target=self.distinctWords, args=(indexNo,sorted_keys,tf_list))
        tw.start()

        # indices (field queries)
        # docIndex(d)
        tt = threading.Thread(target=self.titleIndex, args=(title,indexNo,sorted_keys))
        tt.start()
        tb = threading.Thread(target=self.bodyIndex, args=(body,indexNo,sorted_keys))
        tb.start()
        tc = threading.Thread(target=self.categoryIndex, args=(category,indexNo,sorted_keys))
        tc.start()
        ti = threading.Thread(target=self.infoboxIndex, args=(infobox,indexNo,sorted_keys))
        ti.start()
        tr = threading.Thread(target=self.referenceIndex, args=(reference,indexNo,sorted_keys))
        tr.start()
        tl = threading.Thread(target=self.linkIndex, args=(link,indexNo,sorted_keys))
        tl.start()


        tw.join()
        tt.join()
        tb.join()
        tc.join()
        ti.join()
        tr.join()
        tl.join()

    def distinctWords(self, indexNo, sorted_keys,tf_list):
        with open("words/"+str(indexNo) + ".txt", "w") as f:
            i = 0
            for key in sorted_keys:
                f.write(key + " " + str(tf_list[key])+"\n") # add tf and idf  as well
                i += 1
            f.close()
    # def docIndex(self, data, indexNo, sorted_keys):
    #     with open("docs/" + str(indexNo) + ".txt", "w") as f:
    #         i = 0
    #         l = len(data.keys())
    #         k = list(data.keys())
    #         for key in sorted_keys:
    #             if i < l and k[i] == key:
    #                 f.write(" ".join(data[key]) + "\n")
    #             else: f.write("\n")
    #             i += 1
    #         f.close()
    def titleIndex(self, data, indexNo, sorted_keys):
        with open("title/" + str(indexNo) + ".txt", "w") as f:
            i = 0
            l = len(data.keys())
            k = list(data.keys())
            for key in sorted_keys:
                if i < l and k[i] == key:
                    for doc in data[key]:
                        f.write(doc + "-" + str(data[key][doc]) + " ")
                    i += 1
                f.write("\n")
            f.close()
    def bodyIndex(self, data, indexNo, sorted_keys):
        with open("body/" + str(indexNo) + ".txt", "w") as f:
            i = 0
            l = len(data.keys())
            k = list(data.keys())
            for key in sorted_keys:
                if i < l and k[i] == key:
                    for doc in data[key]:
                        f.write(doc + "-" + str(data[key][doc]) + " ")
                    i += 1
                f.write("\n")
            f.close()
    def categoryIndex(self, data, indexNo, sorted_keys):
        with open("category/" + str(indexNo) + ".txt", "w") as f:
            i = 0
            l = len(data.keys())
            k = list(data.keys())
            for key in sorted_keys:
                if i < l and k[i] == key:
                    for doc in data[key]:
                        f.write(doc + "-" + str(data[key][doc]) + " ")
                    i += 1
                f.write("\n")
            f.close()
    def infoboxIndex(self, data, indexNo, sorted_keys):
        with open("infobox/" + str(indexNo) + ".txt", "w") as f:
            i = 0
            l = len(data.keys())
            k = list(data.keys())
            for key in sorted_keys:
                if i < l and k[i] == key:
                    for doc in data[key]:
                        f.write(doc + "-" + str(data[key][doc]) + " ")
                    i += 1
                f.write("\n")
            f.close()
    def referenceIndex(self, data, indexNo, sorted_keys):
        with open("reference/" + str(indexNo) + ".txt", "w") as f:
            i = 0
            l = len(data.keys())
            k = list(data.keys())
            for key in sorted_keys:
                if i < l and k[i] == key:
                    for doc in data[key]:
                        f.write(doc + "-" + str(data[key][doc]) + " ")
                    i += 1
                f.write("\n")
            f.close()
    def linkIndex(self, data, indexNo, sorted_keys):
        with open("link/" + str(indexNo) + ".txt", "w") as f:
            i = 0
            l = len(data.keys())
            k = list(data.keys())
            for key in sorted_keys:
                if i < l and k[i] == key:
                    for doc in data[key]:
                        f.write(doc + "-" + str(data[key][doc]) + " ")
                    i += 1
                f.write("\n")
            f.close()


class TextProcesser:
    def Stem(self, words):  # vectorize with np arrays
        return stemmer.stemWords(
            words
        )  # apparently stemword-> (gives unhashable type list error) and stemwords are different

    def tokenize(self, text):
        text = text.encode("ascii", "ignore").decode()
        text = re.sub(r"http[^\ ]*\ ", r" ", text)  # removing urls
        text = re.sub(
            r"&nbsp;|&lt;|&gt;|&amp;|&quot;|&apos;", r" ", text
        )  # removing html entities
        text = re.sub(r'[^a-zA-Z0-9]',r' ',text)  # removing special characters
        # Add support for foreign language words (don't remove accents (aigu, grave, etc.))
        return text.split()  # rematch returns a list not a string

    def CleanText(self, title, text, id):  # case folding and cleaning

        references = []
        category = []
        links = []
        text = text.lower()
        title = title.lower()
        # remove references section
        data = text.split("==references==")
        if len(data) == 1:
            data = text.split("== references ==")
        if len(data) == 1:
            references = []
            links = []
            category = []
        else:
            links = self.getExternalLinks(data[1])
            category = self.getCategory(data[1])
            references = self.getReferences(data[1])

        title = self.getTitle(title)
        infobox = self.getInfobox(data[0])
        body = self.getBody(data[0])

        return title, infobox, body, category, references, links, id

    def StopWordRemover(self, words):
        return [w for w in words if stop_map[w] != 1]

    def getExternalLinks(self, links):
        out = links.split('\n')
        link_text = []
        for i in out:
            if re.match(
                r'\*[\ ]*\[', i
            ):  # match for * + zero or more spaces + [
                link_text.append(i)

        out = self.tokenize(
            ' '.join(link_text)
        )  # join operator is wayyy faster than string concatenation
        global total_bef_stem
        total_bef_stem += len(out)
        out = self.StopWordRemover(out)
        out = self.Stem(out)
        return out

    def getTitle(self, title):
        out = self.tokenize(title)
        global total_bef_stem
        total_bef_stem += len(out)
        out = self.StopWordRemover(out)
        out = self.Stem(out)
        return out

    def getInfobox(self, infobox):
        out = infobox.split('\n')
        infobox_open = False
        infobox_text = []
        for i in out:
            if re.match(r'\{\{infobox', i):
                infobox_open = True
                infobox_text.append(re.sub(r'\{\{infobox(.*)', r'\1', i))

            elif infobox_open:
                if i == '}}':
                    infobox_open = False
                    continue
                infobox_text.append(i)

        # out = re.match(r'{{infobox(.*?)}}', infobox, re.DOTALL).group(1)
        out = self.tokenize(' '.join(infobox_text))
        global total_bef_stem
        total_bef_stem += len(out)
        out = self.StopWordRemover(out)
        out = self.Stem(out)
        return out

    def getBody(self, body):

        # out = body.split('\n')
        # infobox_open = False
        # for i in out:
        #     if re.match(r"{{infobox", i):
        #         infobox_open = True

        #     elif infobox_open:
        #         if re.match(r"}}", i):
        #             infobox_open = False

        ## now the body starts?? or is the entire thing body

        out =  re.sub(r'\{\{.*\}\}', r' ', body)
        out = self.tokenize(out)
        global total_bef_stem
        total_bef_stem += len(out)
        out = self.StopWordRemover(out)
        out = self.Stem(out)

        return out

    def getCategory(self, category):
        out = category.split('\n')
        category_text = []
        for i in out:
            if re.match(r"\[\[category", i):
                category_text.append(
                    re.sub(r'\[\[category:(.*)\]\]', r'\1', i)
                )  # remove [[category: and ]] and everything in between, specifying \1 as the repl puts everything in the middle in between so i get the middle text

        # out = re.match(r'{{infobox(.*?)}}', infobox, re.DOTALL).group(1)
        out = self.tokenize(' '.join(category_text))
        global total_bef_stem
        total_bef_stem += len(out)
        out = self.StopWordRemover(out)
        out = self.Stem(out)
        return out

    def getReferences(self, references):
        out = references.split('\n')
        references_text = []
        search_param = "(?<=<ref)(.*?)(?=</ref>)"
        alt_search_param = "(?<=<ref)(.*?)(?=/>)"
        alt_search = False

        for i in out:
            if re.search(r"<ref", i):
                references_text.append(re.sub(
                    r'.*title[\ ]*=[\ ]*([^\|]*).*', r'\1', i
                ))
        # while re.search(r"<ref", out):
        #     if not re.search(search_param, out, re.DOTALL):
        #         alt_search = True
        #         break

        #     else:
        #         x = re.search(search_param, out, re.DOTALL).group(0)
        #         references_text += x
        #         out = re.sub(r"<ref" + re.escape(x) + r"</ref>", "", out)
        # print(references_text)
        # if alt_search:
        #     out = out.split('\n')
        #     for i in out:
        #         if re.search(r'<ref', i):
        #             i = re.sub(r'<ref', '', i)
        #             i = re.sub(r'name=', '', i)

        #             references_text += i
        #     exit(0)

        # out = re.match(r'{{infobox(.*?)}}', infobox, re.DOTALL).group(1)
        out = self.tokenize(' '.join(references_text))
        global total_bef_stem
        total_bef_stem += len(out)
        out = self.StopWordRemover(out)
        out = self.Stem(out)
        return out


class WikiDumpHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.current = ""
        self.title = ""
        self.id = ""
        self.text = ""
        self.inID = False

    def startElement(self, tag, attributes):
        self.current = tag
        if tag == "page":
            self.inPage = True

    def characters(self, content):

        if self.current == "title":
            self.title += content
        elif self.current == "text":
            self.text += content
        elif self.current == "id" and self.inID == False:
            self.id += content
            self.inID = True

    def endElement(self, tag):
        global index
        global t
        global pages
        global docTitle
        global docNo
        if tag == "page":

            title, infobox, body, category, references, links, id = t.CleanText(
                self.title, self.text, self.id
            )
            pages += 1
            docTitle[pages] = self.title.rstrip('\n').rstrip()
            if pages % 16000 == 0:
                with open("docs/" + str(docNo) + ".txt", "w" ) as f:
                    for key in sorted(docTitle.keys()):
                        f.write(str(key) + " " + str(docTitle[key])+"\n")
                docTitle = defaultdict(str)
                docNo += 1
            # index.index_creation(title, infobox, body, category, references, links, id)
            self.current = ""
            self.title = ""
            self.text = ""
            self.id = ""
            self.inID = False


try:
    os.mkdir("docs")
except:
    pass
index = Indexer("./index/doc")
t = TextProcesser()
handler = WikiDumpHandler()
parser = xml.sax.make_parser()
parser.setFeature(xml.sax.handler.feature_namespaces, 0)
parser.setContentHandler(handler)
parser.parse(sys.argv[1])
with open("docs/" + str(docNo) + ".txt", "w" ) as f:
    for key in sorted(docTitle.keys()):
        f.write(str(key) + " " + str(docTitle[key])+"\n")
docTitle = defaultdict(str)

writer = NonThreadedWrite(super_words)
writer.run()
super_words = defaultdict(list)
index.long_mem_store(fileNo)

total_index_size = 0
for i in os.listdir("words/"):
    total_index_size += os.path.getsize("words/" + i)
for i in os.listdir("title/"):
    total_index_size += os.path.getsize("title/" + i)
for i in os.listdir("infobox/"):
    total_index_size += os.path.getsize("infobox/" + i)
for i in os.listdir("category/"):
    total_index_size += os.path.getsize("category/" + i)
for i in os.listdir("body/"):
    total_index_size += os.path.getsize("body/" + i)
for i in os.listdir("reference/"):
    total_index_size += os.path.getsize("reference/" + i)
for i in os.listdir("link/"):
    total_index_size += os.path.getsize("link/" + i)
total_index_size = total_index_size/ 1000000000

with open("stats.txt", "w") as f:
   f.write(f"{total_index_size}\n{fileNo}\n{pages}\n{total}")
