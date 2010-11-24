# things i need to run each time I stary ipython
import numpy, scipy
import os,sys

# INSTALLED APPS ;)
####################################

# original gensim
sys.path.insert(1, '/Projects/LatentDirichletAllocation/gensim/trunk/src')
from gensim import corpora, models, similarities

# ldalib
sys.path.insert(1, '/Projects/LatentDirichletAllocation/')
import liblda


# settings file with RUNDIRS path, topicmodel location and PROJECT_HOME
# from liblda import settings ?
from liblda import local_settings


# to see logging...
import logging
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)



# some taks specific config
PROJECT_PATH=os.path.realpath(os.path.join(os.path.dirname(__file__),".."))

DATA_PATH=os.path.join(PROJECT_PATH,"data/semrelwords/")

INFILE="ukwac-uniqmultiwordterms.SAMPLE.txt"
VOCABFILE="ukwac-vocabulary.SAMPLE.txt"


print "Now let's get started...."

logging.info("Readin in vocabulary file")


vfile = open(os.path.join(DATA_PATH, VOCABFILE) )


vocab={}
index=1
for line in vfile.readlines():
    tokens = line.split()
    if len(tokens) != 2:
        continue
    else:
        freq, word = line.split()
        vocab[index]=word
        index = index + 1

print  "total number of words:" + str(index-1)



logging.info("Setting up corpus")

corpusfilename = os.path.join(DATA_PATH, INFILE)

corpus = corpora.LowCorpus(corpusfilename, id2word=vocab)



from liblda.newmanLDAmodel import NewmanLdaModel

lda = NewmanLdaModel(numT=3,corpus=corpus)


