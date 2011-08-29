
"""
sets up some basic functions and loads up useful data

assume run from Latent-Dirichlet-Allocation/ folder...
"""

import numpy as np
import scipy as sp
import os,sys


from liblda.low2corpus import Low2Corpus

from liblda.LDAmodel import LdaModel


# distances
from scipy.stats.distributions import entropy as spKLdiv
from liblda.math.distances import KLdiv, JSdiv


# Phi based mappings
from liblda.ILDA.hungarian_algorithm import getCostMatrix, find_closest
# Theta  based mappings
from liblda.ILDA.hungarian_algorithm import getCostMatrix2, find_closest2





# data exploration, plotting  and reporting
from liblda.topicviz.show_top import show_top
from liblda.topicviz.show_top import top_words_for_topic
import pylab as p





#####
#####   MAIN SETTINGS FOR DATA SET
######################################################################

DATASET_NAME = "ArXiv16k"

print " LOADING DATA  for: " + DATASET_NAME

DATA_PARENT_DIR="/CurrentPorjects/LatentDirichletAllocation/data/arXiv_as_LOW2/"
VOCAB_FILE = DATA_PARENT_DIR+"vocab.txt"

DOCS_FILE =  DATA_PARENT_DIR+"arXiv_train_docs.txt"
IDS_FILE  =  DATA_PARENT_DIR+"arXiv_train_ids.txt"

TEST_DOCS_FILE =  DATA_PARENT_DIR+"arXiv_test_docs.txt"
TEST_IDS_FILE  =  DATA_PARENT_DIR+"arXiv_test_ids.txt"

######################################################################





# loaders....

# vocab, model and doc2id
arXiv_corpus = Low2Corpus(DOCS_FILE)
arXiv_corpus.setVocabFromList( [w.strip() for w in open(VOCAB_FILE, 'r').readlines() ] )
arXiv_corpus.doCounts()
id_list =  [w.strip() for w in open(IDS_FILE, 'r').readlines() ]
doc2id = dict(  enumerate(id_list) )

# vocab, model and doc2id
arXiv_test_corpus = Low2Corpus(TEST_DOCS_FILE)
arXiv_test_corpus.setVocabFromList( [w.strip() for w in open(VOCAB_FILE, 'r').readlines() ] )
arXiv_test_corpus.doCounts()
test_id_list =  [w.strip() for w in open(TEST_IDS_FILE, 'r').readlines() ]
test_doc2id = dict(  enumerate(test_id_list) )



# the original to compare with
#phiT60_1   = np.load("../runs/repeatedT60-1/phi.npy")
#thetaT60_1 = np.load("../runs/repeatedT60-1/theta.npy")
#zT60_1     = np.load("../runs/repeatedT60-1/z.npy")






# Mon 29 Aug 2011 12:02:14 EDT
# testing log like 



# hydrate from dir
morig = LdaModel( numT=60, corpus=arXiv_corpus, alpha=0.01, beta=0.01)
morig.allocate_arrays()
morig.read_dw_alphabetical()
#rd = os.path.join( RUNDIRS_ROOT, "../runs/repeatedT60-1/" )
rd = "/Users/ivan/Homes/master/Documents/Projects/runs/repeatedT60-1/"
morig.load_from_rundir(rd)

# same for merged topic model
mstar = LdaModel( numT=60, corpus=arXiv_corpus, alpha=0.01, beta=0.01)
mstar.allocate_arrays()
mstar.read_dw_alphabetical()
rd = "/Users/ivan/Homes/master/Documents/Projects/runs/new_merging_gibbs0"
mstar.load_from_rundir(rd)






