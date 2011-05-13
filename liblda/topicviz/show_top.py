#!/usr/bin/env python

# general
import os,sys
import logging
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
logger = logging.getLogger('show-top')
logger.setLevel(logging.INFO)

import numpy as np
import scipy as sp

import operator

try:
    import json as simplejson
except ImportError:
    import simplejson




#
#    ASSUMPTION:    PROJACT_DIR/liblda/topicviz/show_top.py         !!!!!!!!!!!!!!!!!!!!!   <-----------
#
# assumes run.py is found at the root of the LDA directory
PROJECT_PATH=os.path.realpath(os.path.join(os.path.dirname(__file__),"../../"))
sys.path.insert(1, PROJECT_PATH)


# libs (relative to PROJECT_PATH)
from gensim import corpora, models, similarities    # original gensim
import liblda                                       # ldalib
from liblda.extlibs import argparse                 # for civilized command line options


from liblda.math.dirichlet_sparse_stats import get_sparse_stats



# data dirs
DATA_DIR = os.path.join(PROJECT_PATH, "../data/")
DATA_DIR2 = os.path.join(PROJECT_PATH, "data/")

# Location which we use to store run data
# outputs will be stored here
RUNDIRS_ROOT = os.path.join(PROJECT_PATH, "../runs/")

# FILENAMES for storing run output
RUN_FILENAMESS = {  "dp":"Ndt.npy",
                    "wp":"Nwt.npy",
                    "z":"z.npy",
                    "phi":"phi.npy",
                    "theta":"theta.npy",
                    "alpha":"alpha.npy",    # the whole alpha array = prior on p(t|d) distribution
                                            # if missing look in output.json["alpha"][0]
                                            # [1] contains the variance of alpha vector
                    "beta":"beta.npy"       # prior on p(w|t) distr.
                                            # same as output.json["beta"][0]  if constant
                 }








def show_top(phi, num=20, id2word=None):
    """
    Given a p(w|t) distribution, returns the top `num` words
    in each topic.
    """
    numT,numTerms = phi.shape
    topics = []
    for t in range(numT):
        pw_gt = phi[t,:]
        topwords = sorted(enumerate(pw_gt), key=operator.itemgetter(1), reverse=True)
        words = [id2word[id] for id,prb in topwords[0:num] ]
        topics.append(words)
    return topics


def print_top(phi, num=20, id2word=None, print_probs=False):
    """
    Same as the above, but prints the prob of each term in bracket
    """
    numT,numTerms = phi.shape

    topic_strings =[]
    numT,numTerms = phi.shape
    topics = [] # list of strings
    for t in range(numT):
        topic = ''
        topic += str(t) + ": "
        pw_gt = phi[t,:]
        topwords = sorted(enumerate(pw_gt), key=operator.itemgetter(1), reverse=True)
        for id in topwords[0:num]:
            topic += id2word[id] + '(%.4d), '%phi[t,id]
            #words = [id2word[id] for id,prb in topwords[0:num] ]
        topics.append(words)

    # print them all now
    for t in topics:
        print t






# one row version of the above
def top_words_for_topic(phi, t,  num=20, id2word=None):
    """
    Given a p(w|t) distribution, returns the top `num` words
    in topic `t`.
    """
    numT,numTerms = phi.shape
    pw_gt = phi[t,:]
    topwords = sorted(enumerate(pw_gt), key=operator.itemgetter(1), reverse=True)
    words = [id2word[id] for id,prb in topwords[0:num] ]
    return words




if __name__=="__main__":
    """
    Take all kinds of inputs on command line
    """

    parser = argparse.ArgumentParser(description='Prints top words in topics')


    # these are required
    parser.add_argument('--vocab', dest="vocab_file",
                        help="The vocab file document corpus [[ (w_id, count) ]] in .mm format")

    parser.add_argument('--num', type=int, default=10,
                        help='How many to print.')

    args = parser.parse_args()
    #print args



    vocab_file = None

    # if no vocab specified on cmd line, search for it in output.json
    if not args.vocab_file:
        f=open( os.path.join("output.json"), "r" )
        output = simplejson.load(f)
        f.close()
        logger.info("Read output.json")
        vocab_file=output["vocab"]

        if not os.path.exists(vocab_file):
            vocab_file = os.path.join(PROJECT_PATH, vocab_file)

            if not os.path.exists(vocab_file):
                print "ERROR: Can't resolve vocab file from output.json"
                print "Specify manually using --vocab option"
                sys.exit(-1)

    # or load from command line
    else:
        vocab_file = args.vocab_file


    logger.info("Using vocab file " + vocab_file )

    # LOAD VOCAB
    id2word = None
    if vocab_file[-5:]==".json":
        vjson = simplejson.load(open(vocab_file,'r'))
        vocab = vjson
    elif vocab_file[-4:]==".txt":       # one word per line
        vfile = open(vocab_file, 'r')
        wlist = [w.strip() for w in vfile.readlines() ]
        id2word = dict( enumerate(wlist) )
        #word2id = dict( [(word,id)  for id,word in id2word.items()] )
        #vocab = word2id
    else:
        print "Vocab format not recognized"
        sys.exit(-1)




    # LOAD Nwt.npy and convert to phi = p(w|t)
    wpf     = RUN_FILENAMESS["wp"]
    betaf  = RUN_FILENAMESS["beta"]

    wp      = np.load(wpf)
    beta    = np.load(betaf)

    # repetition of LDAmodel.conv_wp_to_phi(self):
    numTerms,numT = wp.shape
    input = np.array( wp, dtype=np.float )
    #                    N_wt   + beta[w]
    #  p(w|t)     =  - ----------------------
    #                 sum_w N_wt   + sumbeta
    sumbeta = np.sum(beta)
    ztot = np.sum(input,0)      # total number of words in corpus for topic t
    denom = ztot + sumbeta

    betarows = np.resize( beta, (numT,len(beta)) )  # numpy makes multiple copies of array on resize
    betacols = betarows.transpose()

    withbeta = input + betacols

    prob_w_given_t = np.dot(withbeta, np.diag(1.0/denom) )
    phi = prob_w_given_t.transpose()




    # get the topic list
    top_words_in_topics = show_top(phi, num=args.num, id2word=id2word)

    for topic in top_words_in_topics:
        words = ", ".join(topic)
        print words


    print "phi sparsensess"

    # compute sparseness and write it out
    sp = get_sparse_stats( phi )
    np.save("phi_sparseness.npy", sp)

    # report on sparseness statisitcs (assume single mode)
    nz = sp.nonzero()[0]                        # get the nonzero entries
    sp_avg = sum([sp[i]*i for i in nz])         # where are non-zero they concentrated ?
    sp_var = sum( [sp[i]*np.abs(i-sp_avg)**2 for i in nz] )
    sp_stdev = np.sqrt( sp_var )                # how concentrated they are around sp_avg

    logger.info("Phi sparseness. center=%d, width=%d" % (int(sp_avg),int(sp_stdev))  )
    #print list(sp)

    # exit with OK status
    sys.exit(0)



