
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
from liblda.subtopics.hungarian_algorithm import getCostMatrix, find_closest
# Theta  based mappings
from liblda.subtopics.hungarian_algorithm import getCostMatrix2, find_closest2





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
tcorpus3 = Low2Corpus(DOCS_FILE)
tcorpus3.setVocabFromList( [w.strip() for w in open(VOCAB_FILE, 'r').readlines() ] )
tcorpus3.doCounts()
id_list =  [w.strip() for w in open(IDS_FILE, 'r').readlines() ]
doc2id = dict(  enumerate(id_list) )


# T50 data
#phi1   = np.load("../runs/repeatedT50-1/phi.npy")
#theta1 = np.load("../runs/repeatedT50-1/theta.npy")
#phi2   = np.load("../runs/repeatedT50-2/phi.npy")
#theta2 = np.load("../runs/repeatedT50-2/theta.npy")
#phi3   = np.load("../runs/repeatedT50-3/phi.npy")
#theta3 = np.load("../runs/repeatedT50-3/theta.npy")



#phi1   = np.load("../runs/repeatedT400-1/phi.npy")
#theta1 = np.load("../runs/repeatedT400-1/theta.npy")
#phi2   = np.load("../runs/repeatedT400-2/phi.npy")
#theta2 = np.load("../runs/repeatedT400-2/theta.npy")
#phi3   = np.load("../runs/repeatedT400-3/phi.npy")
#theta3 = np.load("../runs/repeatedT400-3/theta.npy")

#phi4   = np.load("../runs/repeatedT400-4/phi.npy")
#theta4 = np.load("../runs/repeatedT400-4/theta.npy")
#phi5   = np.load("../runs/repeatedT400-5/phi.npy")
#theta5 = np.load("../runs/repeatedT400-5/theta.npy")
#phi6   = np.load("../runs/repeatedT400-6/phi.npy")
#theta6 = np.load("../runs/repeatedT400-6/theta.npy")

phiT30_1   = np.load("../runs/repeatedT30-1/phi.npy")
thetaT30_1 = np.load("../runs/repeatedT30-1/theta.npy")
thetaT30_1 = np.load("../runs/repeatedT30-1/theta.npy")
phiT30_2   = np.load("../runs/repeatedT30-2/phi.npy")
thetaT30_2 = np.load("../runs/repeatedT30-2/theta.npy")
phiT30_3   = np.load("../runs/repeatedT30-3/phi.npy")
thetaT30_3 = np.load("../runs/repeatedT30-3/theta.npy")
phiT30_4   = np.load("../runs/repeatedT30-4/phi.npy")
thetaT30_4 = np.load("../runs/repeatedT30-4/theta.npy")


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






print """
you might want to run these commands:

%run liblda/ILDA/doc_likelyhood_plots.py
"""

#seeded_m2 = find_closest2(thetacut, seeded_theta)


