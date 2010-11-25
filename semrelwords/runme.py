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
from liblda.local_settings import *


# to see logging...
import logging
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)




DATA_PATH=os.path.join(PROJECT_PATH,"data/semrelwords/")
INFILE="ukwac-uniqmultiwordterms.SAMPLE.txt"
VOCABFILE="ukwac-vocabulary.SAMPLE.txt"



logging.info("Creating corpus")
infilename = os.path.join(DATA_PATH, INFILE)
vfilename =  os.path.join(DATA_PATH, VOCABFILE)
from liblda.low2corpus import Low2Corpus
c = Low2Corpus(infilename)
c.buildVocabs(vfilename)





logging.info("Importing NewmanLdaModel for you")
from liblda.newmanLDAmodel import NewmanLdaModel




