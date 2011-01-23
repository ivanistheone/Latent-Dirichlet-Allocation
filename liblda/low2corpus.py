
"""
Corpus that consists of two files:
  vfile:    vocabulary file one word per line
  infile:   documents, one document per line...
"""

import logging

import operator


import liblda

from gensim import interfaces, utils


def splitOnSpace(s):
    return s.strip().split(' ')


class Low2Corpus(interfaces.CorpusABC):
    """
    Handles a document collection that appears one doc per line.
    """
    def __init__(self, infile, id2word = None, word2id=None):
        """
        Initialize the corpus from a file

        If provided, `id2word` is a dictionary mapping between wordIds (integers)
        and words (strings). If not provided, the mapping is constructed from
        the documents.
        """
        logging.info("loading corpus from %s" % infile)

        self.infile = infile    # input file, see class doc for format
        self.id2word = id2word
        self.word2id = word2id
        self.numDocs = None                    # we will not know until we reach the end...
        self.totalNwords = None

    def __len__(self):
        if not self.numDocs:
            self.doCounts()
        # ok now numDocs is available
        return self.numDocs


    def __bool__(self):
        if self.word2id:
            return True
        else:
            return False

    def buildVocabs(self, vfile):
        """
        Given a vocabulary file,
          freqw1  word1
          freqw2  word2
          ...
        sets the two dictionaries `id2word` and `word2id`.
        """
        logging.info("Readin in vocabulary file")

        v = open(vfile,'r')

        word2id    = {}       # word --> id in 1...numTerms
        id2word    = {}       # int id --> string
        index=1
        for line in v.readlines():
            tokens = line.split()
            if len(tokens) != 2:
                continue
                print "found a non two token line..."
            else:
                freq, word = line.split()
                word2id[word]=index
                id2word[index]=word
                index = index + 1

        logging.info("Read total of %d words." % int(index-1) )

        self.numTerms = index-1
        self.word2id  = word2id
        self.id2word  = id2word

    def doCounts(self):
        """
        populate the fields
         self.numDocs
         self.totalNwords
        this could be expensive for large corpora....
        """

        # return the document, then forget it and move on to the next one
        # note that this way, only one doc is stored in memory at a time, not the whole corpus
        ndocs = 0
        nwords = 0L
        terms = {}
        for doc in self:
            ndocs += 1
            for word, cnt in doc:
                nwords += cnt
                terms[word]=1

        self.numDocs = ndocs
        self.totalNwords = nwords

        #assert self.numTerms == len(terms)


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
        Iterate over the corpus, returning one bag-of-words vector at a time.
        """
        if not self.word2id:
            raise Exception("No word2id dict, run buildVocabs first.")

        #iterate through the whole vocab
        for lineNo, line in enumerate(open(self.infile)):

            words = line.split()
            d = {}
            if len(words)<2:    # no words or 1 word are not informative
                continue
            for word in words:
                wid =  self.word2id.get(word, None)   # get word id
                if not wid:     # skip words that are not in Vocab
                    continue
                d[wid]=d.get(wid,0)+1
            counts = sorted(tuple(d.iteritems()),key=operator.itemgetter(0) )

            if lineNo % 1000000 == 0:
                logging.info("done with docid " + str(lineNo))

            yield counts


#endclass Low2Corpus

