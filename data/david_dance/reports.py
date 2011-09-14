

import os,sys
import numpy as np

from liblda.topicviz.show_top import top_words_for_topic


os.chdir('/Users/ivan/Homes/master/Documents/Projects/LatentDirichletAllocation/data/david_dance')


MODEL = 'm1'
phi   = np.load(MODEL+'/phi.npy')
theta = np.load(MODEL+'/theta.npy')
labels = [ w.strip() for w in open(MODEL + '/labels.txt').readlines() ]


# CORPUS
id2word = dict( enumerate( [ w.strip() for w in open("vocab.txt").readlines() ] ))
word2id = dict( [(v,k) for k,v in id2word.iteritems() ] )
ids_str = [ w.strip() for w in open('ids.txt').readlines() ]
ids = [ tuple( map( int, idstr.split(",")) ) for idstr in ids_str ]


# BUG:  len(ids) != theta.shape[0] 
# lots of the docs (after stemming/ stopword removal)
# are 0 or 1 word long -- and the low2corpus automatically 
# removes such documents:

docs_list = [ doc.strip() for doc in open('docs.txt').readlines() ]
used_ids = []
doc_lengths = []

for row_id in range(0, len(ids)):
    w_list = docs_list[row_id].split(" ")
    if len(w_list) < 2:
        continue
    else:
        used_ids.append( ids[row_id] )
        doc_lengths.append( len(w_list) )


import csv
data = csv.reader(open("parneringwithinterviews.csv"))
header = data.next()
doc_first_words = []
for line in data:
    if line[2]=='No Response':
        continue
    user_id = line[0]
    question_id = line[1]
    answer = line[2]
    
    if (int(user_id), int(question_id)) in used_ids:
        doc_first_words.append( answer[0:13]+"..." )






# let's reproduce the excel stylesheet -- show top 5 words
print """

# let's reproduce the excel stylesheet -- show top 5 words
"""
print "Topic label".ljust(35), "Top words in topic"
for topic_id in range(0,29):
    print str(topic_id).ljust(2),  \
          ("<"+labels[topic_id]+">").ljust(35), \
          top_words_for_topic( phi, topic_id, num=7, id2word=id2word)









# doc = one answer by one person

numDocs = len(used_ids)
assert numDocs == theta.shape[0]
assert numDocs == len(doc_first_words)
assert theta.shape[1] == phi.shape[0]


print """



==== PART 1  ====

Treating each answer as a document

"""
print "Who".ljust(3), "Que".ljust(3), "First words...".ljust(16), "Top topics (prob)"
print "===", "===", "="*16, "="*80

for doc_id in range(0, numDocs):

    th = theta[doc_id,:]
    top_topics = sorted( enumerate(th), key=lambda t: t[1], reverse=True)[0:5]

    tt_strs_1 = [ labels[t[0]]+"(%1.2f)"%t[1] for t in top_topics if t[1]>0.07]
    tt_strs = [ w[0:50] for  w in tt_strs_1 ]


    print str(ids[doc_id][0]).ljust(3), \
          str(ids[doc_id][1]).ljust(3), \
          doc_first_words[doc_id], \
          ", ".join(tt_strs)

    



print """



==== PART 2  ====

Treating each PERSON as a document
we represent a person by the sum of his/her answers

-- note: the interview persons are treated as different
         people -- since they have different ids.
         if you don't want that -- you have to use
         the same id for their interview questions ....


"""

print "Who".ljust(3), "Top topics for this person (prob)"
print "===", "="*80

from collections import defaultdict
answers_for_user = defaultdict(list)
words_for_user = defaultdict( int )


for ans_id in range(0, numDocs):
    user_id, question_id = used_ids[ans_id]
    answers_for_user[user_id].append( doc_lengths[ans_id]*theta[ans_id] )
    words_for_user[user_id] += doc_lengths[ ans_id ]
    
numUsers = len(answers_for_user)
numT = theta.shape[1]

utheta = np.zeros( (numUsers + 1, numT) )           # 1- indexed
for user_id, ans_list in answers_for_user.iteritems():
    s = np.zeros( numT )
    for ans in ans_list:
        s += ans
    utheta[user_id,:] = s / words_for_user[user_id]

for user_id in range(1, numUsers+1):
    th = utheta[user_id,:]
    top_topics = sorted( enumerate(th), key=lambda t: t[1], reverse=True)[0:5]
    tt_strs_1 = [ labels[t[0]]+"(%1.2f)"%t[1] for t in top_topics if t[1]>0.05]
    tt_strs = [ w[0:50] for  w in tt_strs_1 ]

    print str(user_id).ljust(3), \
          ", ".join(tt_strs)

    


print """



==== PART 3  ====

Treating each QUESTION as a document
we represent a question by the sum of all the 
answers given by the different people.


"""




print "Que".ljust(3), "Top topics for this question (prob)"
print "===", "="*80

from collections import defaultdict
answers_for_question = defaultdict(list)
words_for_question = defaultdict( int )


for ans_id in range(0, numDocs):
    user_id, question_id = used_ids[ans_id]
    answers_for_question[question_id].append( doc_lengths[ans_id]*theta[ans_id] )
    words_for_question[question_id] += doc_lengths[ ans_id ]
    
numQuestions = len(answers_for_question)
numT = theta.shape[1]


qtheta = np.zeros( (numQuestions + 1, numT) )           # 1- indexed
for question_id, ans_list in answers_for_question.iteritems():
    s = np.zeros( numT )
    for ans in ans_list:
        s += ans
    qtheta[question_id,:] = s / words_for_user[question_id]

for question_id in range(1, numQuestions+1):
    th = utheta[question_id,:]
    top_topics = sorted( enumerate(th), key=lambda t: t[1], reverse=True)[0:5]
    tt_strs_1 = [ labels[t[0]]+"(%1.2f)"%t[1] for t in top_topics if t[1]>0.05]
    tt_strs = [ w[0:50] for  w in tt_strs_1 ]

    print str(question_id).ljust(3), \
          ", ".join(tt_strs)

    





