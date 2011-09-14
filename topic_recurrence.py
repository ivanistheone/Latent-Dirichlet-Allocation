#
#
# TOPIC RECURRENCE
#
# We aim to investigate how many topics recur in LDA
# models started from different random seed.
#


import pylab as p
import numpy as np
import scipy as sp

import os, sys

# distances
from scipy.stats.distributions import entropy as spKLdiv
from liblda.math.distances import KLdiv, JSdiv
from liblda.topicviz.show_top import top_words_for_topic

# import the \phi metrics and topic matchging functionnality
from liblda.ILDA.hungarian_algorithm import getCostMatrix, find_closest, hungarian_algorithm


#from liblda.ILDA.topic_cleaning import NUM_BINS, BIN_SEP, TOPIC_METRICS
#from liblda.ILDA.topic_cleaning import compute_theta_metrics



# for table layout
from liblda.extlibs.prettytable import PrettyTable
from liblda.extlibs.ordereddict import OrderedDict




import logging
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
logger = logging.getLogger('RECUR')
logger.setLevel(logging.INFO)




ILDA_PATH=os.path.realpath(os.path.join(os.path.dirname(__file__),"."))
PROJECT_PATH="/Projects/LatentDirichletAllocation/"
# data dirs
RUNDIRS_ROOT = os.path.join(PROJECT_PATH, "../runs/")



from liblda.math.distances import KLdiv, JSdiv
from liblda.topicviz.show_top import show_top
from liblda.topicviz.show_top import top_words_for_topic

from liblda.math.dirichlet_sparse_stats import get_sparse_stats




# setup the dirs contianin fitted models
dir_list_T40 = []
for num_int in range(0,10):
    rd = os.path.join( RUNDIRS_ROOT, "newNIPST40"+str(num_int)+"/" )
    dir_list_T40.append(rd)

phi_list = []
for rd in dir_list_T40:
    phi = np.load( rd + "phi.npy")
    phi_list.append(phi)






def match_distances( phi_list ):
    """ Goes through the phi matrices and computes
        the KL divergence.
        The output `d` in a list of lists of np.array's 

          d[1][2][34,44]  = KL( t_34(m1) || t_44(m2) )

    """

    M = len(phi_list)
    numT, numTerms = phi_list[0].shape 

    # setup the output data structure
    d = []
    for i in range(0,M):
        row = []
        for j in range(0,M):
            numTrow = phi_list[i].shape[0]
            numTcol = phi_list[j].shape[0]
            row.append( np.zeros( (numTrow,numTcol) ) )
        d.append( row )


    # fill it in
    for i in range(0,M):
        for j in range(0,M):
            m = d[i][j]
            nR, nC = m.shape
            for t1 in range(0,nR):
                for t2 in range(0, nC):
                    m[t1,t2]= JSdiv( phi_list[i][t1,:], phi_list[j][t2,:] )

    return d


def match_stats( phi_list, eps_match ):
    """ Goes through the phi matrices and counts
        how many topic are matched in JSdiv in the other models """

    M = len(phi_list)
    numT, numTerms = phi_list[0].shape 

    # global match counts
    unique_matches = 0
    double_matches = 0
    multi_matches  = 0

    # detialed match counts for each topic, in each m
    match_counts = []
    for phi in phi_list:
        nT,nW = phi.shape
        count_store = np.zeros( (nT,3) )   #[ uniq, doubl, multi ]
        match_counts.append( count_store )

    for mid in range(0, M):
        
        print "processing model ", mid
        print ""

        other_phis = phi_list[0:mid] + phi_list[mid+1:]
        
        for tid in range(0,numT):
            
            ph_row = phi_list[mid][tid]

            for o_phi in other_phis:
                match_count = 0
                o_numT, o_numTerms = o_phi.shape
                for o_tid in range(0, o_numT):
                    o_ph_row = o_phi[o_tid]
                    if JSdiv(o_ph_row, ph_row) <= eps_match:
                        match_count += 1

                if match_count == 1:
                    unique_matches += 1
                    match_counts[mid][tid,0]+=1
                elif match_count == 2:
                    double_matches += 1
                    match_counts[mid][tid,1]+=1
                elif match_count >= 3:
                    multi_matches += 1
                    match_counts[mid][tid,2]+=1
                    
    print "unique matches", unique_matches
    print "double matches", double_matches
    print "multi matches", multi_matches
    return match_counts

def print_matches_n_words( m_cnts, id2word, phi_list, mid):
    numT, pfff  = m_cnts[mid].shape
    for ii in range(0,numT):
        print m_cnts[mid][ii], ",".join( top_words_for_topic( phi_list[mid], ii, id2word=id2word, num=10) )


def find_matching_topic( phi_list, eps_match, mid, tid ):
    """ Find matching topics for `tid` in `mid`th model in `phi_list`.
    """
    M = len(phi_list)
    numT, numTerms = phi_list[0].shape

    other_phis = phi_list[0:mid] + phi_list[mid+1:]
    ph_row = phi_list[mid][tid]

    matches = []   # list of tuples (mid,tid)
    for o_mid in range(0,M):
        if o_mid == mid:
            continue
        o_phi =  phi_list[o_mid]
        o_numT, o_numTerms = o_phi.shape
        for o_tid in range(0, o_numT):
            o_ph_row = o_phi[o_tid]
            if JSdiv(o_ph_row, ph_row) <= eps_match:
                matches.append( (o_mid, o_tid) )

    return matches



def print_matching_topics( matches, id2word, phi_list, mid, tid):
    print "The topic", tid, "in model", mid, "is:"
    print ",".join( top_words_for_topic( phi_list[mid], tid, id2word=id2word, num=10) )

    print "It has following matches in other models:"
    for o_mid, o_tid in matches:
        print ("("+str(o_mid)+","+str(o_tid)+")").ljust(8),  \
            ",".join( top_words_for_topic( phi_list[o_mid], o_tid, id2word=id2word, num=10) )







