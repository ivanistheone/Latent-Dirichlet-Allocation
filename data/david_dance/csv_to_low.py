#!/usr/bin/env python


print "Reading the csv file and outputting each answer on a separate line"
print "also generating the ids.txt file"

import csv


from porter2 import stem


from stopwords import STOPWORDS

# setup the regex for words that consist of letters ONLY
import re
word=re.compile(r"[a-z]+")


data = csv.reader(open("parneringwithinterviews.csv"))
#data = csv.reader(open("partnering.csv"))

header = data.next()

low_file = open("docs.txt", 'w')
ids_file = open("ids.txt", 'w')


counter = 1
for line in data:
    
    if line[2]=='No Response':
        print "no response, skipping..."
        continue

    user_id = line[0]
    question_id = line[1]
    answer = line[2]

    # convert answer to lowercase
    answer = answer.lower()

    word_list = []
    # remove all punctuation 
    for m in word.finditer(answer):
        # we are on word w
        w = m.group(0)
        # skip stopwords
        if w in STOPWORDS:
            continue
        # skip two letter words
        if len(w)<3:
            continue

        w_stem = stem(w)
        word_list.append(w_stem)

    answer2 = " ".join(word_list) 

    low_file.write( answer2 + "\n" )
    ids_file.write( user_id+","+question_id + "\n" )
    
    counter += 1

print "processed ", counter, " entries"



