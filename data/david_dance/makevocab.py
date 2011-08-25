#!/usr/bin/env python
import argparse
import sys
import collections
import os
import cPickle
#import random
#import re
#import codecs

from stopwords import STOPWORDS

import numpy as np




def makevocab(args):
    """
        Goes through the "lines of words" file generating word counts and decides which
        are the most important words to keep on the basis of:

    """

    wsfile      = args.input
    vocabfile   = args.output
    minoccur    = args.minoccur
    maxwords    = args.maxwords

    # do all the counts and cache them
    if not args.cache or (args.cache and not os.path.exists('vocab.dat') ):
        # prepare default dict where we will store output
        freq_for_word =  collections.defaultdict(long)
        docs_with_word  =  collections.defaultdict(long)


        wsfile = open(wsfile, 'r')

        wcount = long(0)
        dcount = 0
        for doc in wsfile.readlines():
            words = doc.split()
            wuniq  = []
            for word in words:
                word = word.strip()
                freq_for_word[word] = freq_for_word[word]+1
                wcount += 1
                if word not in wuniq:
                    wuniq.append(word)
            for w in wuniq:
                docs_with_word[w] = docs_with_word[w]+1
            dcount += 1

        wsfile.close()

        if args.cache:
            # cache for later  [freq_for_word,docs_with_word]
            cPickle.dump([freq_for_word,docs_with_word, wcount, dcount], open("vocab.dat","w"))

    else:   # if .dat exits then we must load it -- already the vocab
        [freq_for_word,docs_with_word, wcount,dcount] = cPickle.load(open("vocab.dat","r"))



    # prune words that do not occur in at lear
    target_count = float( args.mindocperc )/100*dcount
    good_words = []
    freq_for_word2={}
    docs_with_word2={}
    for word,dococcur in docs_with_word.iteritems():
        if dococcur >= target_count:
            good_words.append(word)
    for word in good_words:
        freq_for_word2[word]  = freq_for_word[word]
        docs_with_word2[word] = docs_with_word[word]


    print "after rem. low counts in doc : ", len(freq_for_word2)

    # prunce stop words
    good_words = []
    freq_for_word3={}
    docs_with_word3={}
    for word in freq_for_word2.keys():
        if word not in STOPWORDS:
            good_words.append(word)
    for word in good_words:
        freq_for_word3[word]  = freq_for_word2[word]
        docs_with_word3[word] = docs_with_word2[word]

    print "after removing stopwords: ", len(freq_for_word3)


    # sanity check
    assert freq_for_word3.keys() == docs_with_word3.keys()


    # prune words that are do not satisfy minoccur
    vocab2 = {} # moving here things that pass the test...
    for word,count in freq_for_word3.iteritems():
        if count < minoccur:
            continue
#        elif len(word)==2 and word!="pi":
#            continue
#        elif len(word)==3 and count < 10000:
#            continue
#        elif len(word)==4 and count < 6000:
#            continue
#        elif len(word)==5 and count < 2000:
#            continue
        else:
            vocab2[word]=count


    # sort by freq
    wordtuples=sorted(vocab2.items(),key=lambda(word,count): (-count,word) )


    # cur off only maxwords worth
    if maxwords:
        finalvocab=wordtuples[0:maxords]
    else:
        finalvocab=wordtuples


    print "Final vocab length: ", len(finalvocab)

    # write in order --- one word per line
    fout = open(vocabfile,"w")
    for word, count in finalvocab:
        fout.write(word+"\n")
    fout.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Produces a tmp_vocab.txt file from a word stream" )
    parser.add_argument('-i', '--input',  help="wordsteam file")
    parser.add_argument('-o', '--output', help="name of vocab file",default="tmp_vocab.txt")
    parser.add_argument('--minoccur',  help="minimum times word must occur", type=int, default=1)
    parser.add_argument('--mindocperc',  help="minimum perc. of docs w word must occur in", type=int, default=0)
    parser.add_argument('--maxwords',  help="max number of words in vocab", type=int)
    parser.add_argument('--docstring',  help="Print the script  __doc__", action='store_true')
    parser.add_argument('-v', dest='verbose', action='store_true')
    parser.add_argument('-c', dest='cache',   action='store_true', help="cache counts to pickle file")

    args = parser.parse_args()

    if args.docstring:
        print makevocab.__doc__
        sys.exit(0)

    print "VOCAB extractor"
    if not args.input:
        print "no input specified,.... exiting "
        sys.exit(1)


    makevocab(args)
    #args.input, args.output, minoccur=args.minoccur, maxwords=args.maxwords)


