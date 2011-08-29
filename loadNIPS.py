
"""
sets up some basic functions and loads up useful data

assume run from Latent-Dirichlet-Allocation/ folder...
"""

import numpy as np
import scipy as sp
import os,sys


from liblda.low2corpus import Low2Corpus
from liblda.newmmcorpus import NewMmCorpus

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





# libs (relative to PROJECT_PATH)
from gensim import corpora, models, similarities    # original gensim
import liblda                                       # ldalib
from liblda.extlibs import argparse                 # for civilized command line options
from liblda.math.dirichlet_sparse_stats import get_sparse_stats



from liblda.LDAmodel import LdaModel
from liblda.ILDA.topic_merging import genrate_quality_F, classify_using, topic_merging_fixed_size, topic_assignment_merging_in_C
from liblda.extlibs.ordereddict import OrderedDict
from liblda.ILDA.topic_merging import mergekit_to_table, print_table








#####
#####   MAIN SETTINGS FOR DATA SET
######################################################################

DATASET_NAME = "ArXiv16k"

print " LOADING DATA  for: " + DATASET_NAME



#####   MAIN SETTINGS FOR DATA SET
######################################################################
DATA_PARENT_DIR="/CurrentPorjects/LatentDirichletAllocation/data/NIPS1-17/"
DATASET_NAME = "NIPS 1-17"
print " LOADING DATA  for: " + DATASET_NAME
#DATA_PARENT_DIR = os.path.join(DATA_DIR2, "NIPS1-17/")
VOCAB_FILE = DATA_PARENT_DIR+"NIPS_vocab.txt"
DOCS_FILE =  DATA_PARENT_DIR+"NIPS_counts.mm"
IDS_FILE  =  DATA_PARENT_DIR+"NIPS_doc_names.txt"


######################################################################


NUM_TOPICS = 50

# load corpus
NIPS_corpus = NewMmCorpus(DOCS_FILE)
NIPS_corpus.setVocabFromList( [w.strip() for w in open(VOCAB_FILE, 'r').readlines() ] )
NIPS_corpus.doCounts()
id_list =  [w.strip() for w in open(IDS_FILE, 'r').readlines() ]
doc2id = dict(  enumerate(id_list) )




RUNDIRS_ROOT = "../runs/"



# load the main model
mstar = LdaModel(numT=NUM_TOPICS, corpus=NIPS_corpus, alpha=0.01, beta=0.01)
mstar.allocate_arrays()
mstar.read_dw_alphabetical()
rd = os.path.join( RUNDIRS_ROOT, "lab7-24/run0021/" )
mstar.load_from_rundir(rd)


# load the merged model
mrgd = LdaModel(numT=NUM_TOPICS, corpus=NIPS_corpus, alpha=0.005, beta=0.01)
mrgd.allocate_arrays()
mrgd.read_dw_alphabetical()
rd = os.path.join( RUNDIRS_ROOT, "merge40-a0_005-b0_01/" )
mrgd.load_from_rundir(rd)



# setup the dirs models to be merged
m_dir_list = []
for num in range(22,41):
    rd = os.path.join( RUNDIRS_ROOT, "lab7-24/run00"+str(num)+"/" )
    m_dir_list.append(rd)
    # i am not preloading models because I fear i will run out of MEM



mdir = m_dir_list[0]
mnew = LdaModel(numT=1, corpus=NIPS_corpus)
mnew.load_from_rundir(mdir)


qFnew  = genrate_quality_F(mnew.phi, mnew.theta)
qFstar = genrate_quality_F(mstar.phi, mstar.theta)

print " In [15]: %run liblda/ILDA/topic_merging.py "

numT=50
mk = topic_merging_fixed_size(range(0,numT), range(0,numT), mstar.phi, mnew.phi, qFstar, qFnew, eps_match=0.2)


Metric = mk['costM']


#(M, Mperm, perm1, perm2) = bulgarian_algorithm_w_perms(Metric, 0.2, qF_rows=qFstar)







#phi_orig = phiT60_1
#theta_orig = thetaT60_1
#z_orig = zT60_1



# The experimen where phiT60_1 had phiT60_2 -- to phiT60_8
# merged into it with 0 60 and 200 steps of Gibbs reamplings
# in between merging steps

#phi_m0gibbs   = np.load("../runs/new_merging_gibbs0/phi.npy")
#theta_m0gibbs = np.load("../runs/new_merging_gibbs0/theta.npy")
z_m0gibbs     = np.load("../runs/new_merging_gibbs0/z.npy")




