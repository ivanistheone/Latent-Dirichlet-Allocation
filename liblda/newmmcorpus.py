#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Corpus in the Matrix Market format.
"""


import logging

from gensim import interfaces, matutils



class NewMmCorpus(matutils.MmReader, interfaces.CorpusABC):
    """
    Corpus in the Matrix Market format.
    """
    def __init__(self, input, word2id=None, transposed=True):
        super(NewMmCorpus,self).__init__(input, transposed=transposed)

        if word2id:
            self.setVocabFromDict( word2id )
        else:
            self.word2id = None

        self.totalNwords = None

    def __len__(self):
        if not self.numDocs:
            self.doCounts()
        # ok now numDocs is available
        return self.numDocs

    def doCounts(self):
        """
        populate the fields
         self.numDocs
         self.totalNwords
        this could be expensive for large corpora....
        """
        # return the document, then forget it and move on to the next one
        # note that this way, only one doc is stored in memory at a time, not the whole corpus
        nwords = 0L
        terms = {}
        ndocs =0
        for doc in self:
            ndocs += 1
            for word, cnt in doc:
                nwords += cnt
                terms[word]=1
        self.totalNwords = long(nwords)
        self.numDocs = ndocs

    def setVocabFromList(self, wlist):
        """
        given a list of words (strings), sets it to vocab
         id2word and word2id

        """
        self.id2word = dict( enumerate(wlist) )
        self.word2id = dict( [(word,id)  for id,word in self.id2word.items()] )
        self.numTerms = len(self.id2word)


    def setVocabFromDict(self, word2id):
        """
        given a dict of word-to-id mappings sets up:
         id2word and word2id

        """

        assert type(word2id)==type({})

        self.word2id = word2id
        self.id2word = dict( [(int(id),word)  for word,id in self.word2id.items()] )
        self.numTerms = len(self.word2id)


    def __iter__(self):
        """
        Interpret a matrix in Matrix Market format as a streaming corpus.

        This simply wraps the I/O reader of MM format, to comply with the corpus
        interface.
        """
        for docId, doc in super(NewMmCorpus, self).__iter__():
            yield doc # get rid of docId, return the sparse vector only

    @staticmethod
    def saveCorpus(fname, corpus, id2word = None, progressCnt = 1000):
        """
        Save a corpus in the Matrix Market format to disk.
        """
        logging.info("storing corpus in Matrix Market format to %s" % fname)
        matutils.MmWriter.writeCorpus(fname, corpus, progressCnt = progressCnt)
#endclass MmCorpus


