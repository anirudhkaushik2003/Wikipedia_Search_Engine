Division of files
index - intermediate file, used for creating words with posting list, non unique unsorted
words - list of all the words, sorted in batches of 16000
title, body, category, etc - contains the documents and term frequency for each word indexed by the line number of that word
docs - contains a mapping of docID to actual title

there is a string matching error which i will fix and re submit