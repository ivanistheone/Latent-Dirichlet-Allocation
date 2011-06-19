
"""
sets up some basic functions and loads up useful data

assume run from Latent-Dirichlet-Allocation/ folder...
"""

import numpy as np
import scipy as sp
import os,sys


from liblda.low2corpus import Low2Corpus


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

######################################################################





# loaders....

# vocab, model and doc2id
train_corpus = Low2Corpus(DOCS_FILE)
train_corpus.setVocabFromList( [w.strip() for w in open(VOCAB_FILE, 'r').readlines() ] )
train_corpus.doCounts()
id_list =  [w.strip() for w in open(IDS_FILE, 'r').readlines() ]
doc2id = dict(  enumerate(id_list) )


phiT60_1   = np.load("../runs/repeatedT60-1/phi.npy")
thetaT60_1 = np.load("../runs/repeatedT60-1/theta.npy")
zT60_1     = np.load("../runs/repeatedT60-1/z.npy")

phiT60_2   = np.load("../runs/repeatedT60-2/phi.npy")
thetaT60_2 = np.load("../runs/repeatedT60-2/theta.npy")
zT60_2     = np.load("../runs/repeatedT60-2/z.npy")

phiT60_3   = np.load("../runs/repeatedT60-3/phi.npy")
thetaT60_3 = np.load("../runs/repeatedT60-3/theta.npy")
zT60_3     = np.load("../runs/repeatedT60-3/z.npy")

phiT60_4   = np.load("../runs/repeatedT60-4/phi.npy")
thetaT60_4 = np.load("../runs/repeatedT60-4/theta.npy")
zT60_4     = np.load("../runs/repeatedT60-4/z.npy")

# 5 6 7 8


phiT60_9   = np.load("../runs/repeatedT60-9/phi.npy")
thetaT60_9 = np.load("../runs/repeatedT60-9/theta.npy")
zT60_9     = np.load("../runs/repeatedT60-9/z.npy")


phi_orig = phiT60_1
theta_orig = thetaT60_1
z_orig = zT60_1



# The experimen where phiT60_1 had phiT60_2 -- to phiT60_8
# merged into it with 0 60 and 200 steps of Gibbs reamplings
# in between merging steps

phi_m0gibbs   = np.load("../runs/new_merging_gibbs0/phi.npy")
theta_m0gibbs = np.load("../runs/new_merging_gibbs0/theta.npy")
z_m0gibbs     = np.load("../runs/new_merging_gibbs0/z.npy")

phi_m60gibbs   = np.load("../runs/new_merging_gibbs60/phi.npy")
theta_m60gibbs = np.load("../runs/new_merging_gibbs60/theta.npy")
z_m60gibbs     = np.load("../runs/new_merging_gibbs60/z.npy")

phi_m200gibbs   = np.load("../runs/new_merging_gibbs200/phi.npy")
theta_m200gibbs = np.load("../runs/new_merging_gibbs200/theta.npy")
z_m200gibbs     = np.load("../runs/new_merging_gibbs200/z.npy")

# same as 0gibbs, but in the end we do a 200 iterations
phi_m0gibbs_f200   = np.load("../runs/new_merging_gibbs0_f200/phi.npy")
theta_m0gibbs_f200 = np.load("../runs/new_merging_gibbs0_f200/theta.npy")
z_m0gibbs_f200     = np.load("../runs/new_merging_gibbs0_f200/z.npy")
# we want to test whether it is the Gibbs steps that is undoing
# the topic coherence that was done by the merging steps




# one w/ 200 for fun
phiT200   = np.load("../runs/subtopicsT200unseeded/phi.npy")
thetaT200 = np.load("../runs/subtopicsT200unseeded/theta.npy")
zT200     = np.load("../runs/subtopicsT200unseeded/z.npy")




print """
you might want to run these commands:

%run liblda/ILDA/doc_likelyhood_plots.py
"""

#seeded_m2 = find_closest2(thetacut, seeded_theta)


