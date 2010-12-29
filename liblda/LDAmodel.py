

import os,sys

import numpy as np
import scipy as sp


__doc__ = """
This module defines the class LdaModel which uses
a Gibbs sampling approach (in C using scipy.weave)
to infer the topics.
"""

from liblda import interfaces


# can be removed?
from liblda.util import rungen
from liblda.util import newman_topicmodel
import subprocess
# shouldn't be necessary
from local_settings import PROJECT_PATH, topicmodel_DIR, RUNDIRS_ROOT
import shutil
import datetime




import logging
logger = logging.getLogger('LdaModel')
logger.setLevel(logging.INFO)


# Errors accociated with this
class IncompleteInputError(Exception):
    """
    The user didn't supply a corpus or numT
    """
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)




class LdaModel(interfaces.LdaModelABC):
    """
    A simple Gibbs sampling LDA model.
    See FastLdaModel and SparseLdaModel for other variants. (TODO)

    Parameters:
        numT        : number of topics in model
        alpha       : Dirichlet prior on theta (topics in docs)
                      len: numT   or  constant scalar
        beta        : Dirichlet prior on phi (words in topics)

    Input data:
        corpus      : reference to a corpus object (anything that looks like [[ (termid,count) ]]
        numDocs     : len(corpus)

    Optional input:
        vocab       : ref to Vacabulary object  #calV
                      (which can map word id to word string)
        numTerms    : len(vocab) #V

    API:
        corpus=[[(1,1),(2,4)],[(1,1)]]
        lda = LdaModel( corpus=corpus, numT=17, alpha=50.0/17, beta=0.01 )
        lda.train()

    Then some time goes by...
    and the following things appear.

    Inferred model:
        phi         : topic-word distribution
                      size: numT x numTerms
                      alias: prob_w_given_t
        theta       : document-topic distibution
                      size: numDocs x numT
                      alias: prob_t_given_d
        z           : the topic assignment
                      size: ?

    """

    def __init__(self, numT=None, alpha=None, beta=None, corpus=None,  vocab=None):
        """
        This creates the LDA model.

        Must supply corpus and numT before you can train() it.
        """
        self.numT = int(numT)               # number of topics

        self.corpus= corpus
        self.numDocs = len(corpus)               # how do i make this lazy?
                                            # assume Corpus object is smart
                                            # and has cached its length ...

        # optional
        if not vocab:
            # then we need to go thgouh corpus and see how many different terms are used
            # assume corpus used sequential term-ids so just find the max value
            maxTermID = 0
            for doc in corpus:
                for termid, cnt in doc:
                    if termid > maxTermID:
                        maxTermID = termid
            # since 0-indexed, we must add 1 to get numTerms
            self.numTerms = maxTermID + 1
            self.vocab =    None
                # or setup a dummy vocab like
                # self.vocab = dict( (id,str(id)) for id in range(0,self.numTerms) )
        else:
            self.vocab    = vocab
            self.numTerms = len(vocab)
            # assert maxTermID == len(vocab)


        # set default values for \alpha, \beta
        if not alpha:
            alpha=  0.1             # default alpha
                                    # 0.05 * N / (numDocs * numT)   Newman choice
                                    #                                  N=total# words in corpus
                                    # 50.0 / numT                   Griffiths recommended
        if not beta:
            beta = 0.01

        # convert scalar \alpha,\beta to vectors (constant values)
        if not hasattr(alpha, "__iter__"):      # if \alpha is not list like
            self.alpha = alpha*np.ones(self.numT)
        else:
            self.alpha = alpha                  # Dirichlet prior on theta (topics in docs)
        if not hasattr(beta, "__iter__"):
            self.beta = beta*np.ones(self.numTerms)
        else:
            self.beta = beta                    # Dirichlet prior on phi (words in topics)

        # results set to None
        self.phi = None
        self.theta = None
        self.z = None


    def is_trained(self):
        """
        Tells you whether model has been trained or not.
        """
        raise NotImplementedError('cannot instantiate abstract base class')

    def train(self, iter=50 ,seed=None):
        """
        Runs a Gibbs sampler similar to Dave Newman's C code.

        1/ Create work arrays
        2/ Initialize to random topic assignment
        3/ Gibbs sample
        4/ Set self.phi, self.theta and self.z from the Nwt and Ndt arrays
                     Nwt.txt,  Ndt.txt,  z.txt
                     p(w|t)    p(t|d)    Z_{d_i,w_j}

        """
        if not self.numT:
            raise  IncompleInputError('Must specify number of topics: self.numT')
        if self.corpus ==None:
            raise  IncompleInputError('Must provide a corpus to train on: self.corpus')

        # training params
        self.iter = iter    # iter set to 50 by default..
        if not seed:
            seed = 777
        self.seed = 2*seed+1

        # subfunctions
        self.allocate_arrays()
        self.random_initialize()
        self.gibbs_sample(iter=self.iter, seed=self.seed )
        self.load_probs()
        self.deallocate_arrays()

    def countN(self):
        """ Count the total number of words in corpus
            this quantity is called N in Dave Newman's code
            and totalNwords in my code ... for now.
        """

        # maybe the corpus has cached the totalNwords ?
        if hasattr(self.corpus, "totalNwords") and self.corpus.totalNwords != None:
            return self.corpus.totalNwords

        else:   # corpus is not smart so must go through it
            total=0L
            for doc in self.corpus:
                for word,cnt in doc:
                    total += cnt
            # tag the totalNwords onto corpus object
            self.corpus.totalNwords = total
            return total


    def allocate_arrays(self):
        """
        Allocates memory for the arrays of counts which
        will be necessary for the Gibbs sampler
        """
        pass

    def random_initialize(self):
        pass

    def gibbs_sample(self, iter=None, seed=None ):
        pass

    def load_probs(self):
        pass

    def deallocate_arrays(self):
        pass





    def mkrundir(self,rundir):
        # sort out the rundir
        if not rundir:
            # ok so we will generate one
            if not rundir_root:
                rundir_root=RUNDIRS_ROOT
            rundir = rungen.mk_next_rundir(rundir_root)

        # ok so we have a rundir now
        assert os.path.exists(rundir), "rundir doesn't exist!"
        self.rundir = rundir    # tag it onto the LdaModel while we are at it

    def copy_topicmodel(self, rundir):
        # move topicmodel binary into place
        binary = os.path.join(topicmodel_DIR,"topicmodel")
        assert os.path.exists(binary), "topicmodel binary doesn't exist."
        shutil.copy(binary,rundir)

    def write_corpus_to_docword(self, rundir):
        # write corpus to docwords...
        dwfile = os.path.join(rundir,"docword.txt")
        newman_topicmodel.NewmanWriter.writeCorpus(dwfile, self.corpus)
        assert os.path.exists(dwfile), "docword.txt doesn't exist."

    def run(self,rundir,iter,seed):
        # cd to rundir
        os.chdir(rundir)


        # get the party started
        logger.info("Starting Newman's Gibbs sampler")
        #logger.info("Starting run in  "+ rundir )

        runcommand = os.path.join(rundir,"topicmodel") +" "+str(self.numT)+" "+str(iter)+" "+str(seed)
        logger.info("run command: "+runcommand )

        ###### CALLING STARTS  #####################################################
        start_time = datetime.datetime.now()
        (out,err)  = subprocess.Popen(runcommand, shell=True, \
                                                   stdout=subprocess.PIPE, \
                                                   stderr=subprocess.PIPE \
                                                   ).communicate()

        end_time = datetime.datetime.now()
        duration = (end_time-start_time).seconds
        logger.info("run duration: "+str(duration)+" seconds")
        ###### CALLING STOPS  ######################################################


    def load_probs(self,rundir):
        # read Nwt.txt
        fname1 = os.path.join(rundir,"Nwt.txt")
        Nwt=newman_topicmodel.loadsparsemat(fname1)   # teruns an array of counts
        self.Nwt = Nwt
        # make into normalized probability distirbution
        self.phi = newman_topicmodel.conv_Nwt_to_phi(Nwt)

        # read Ndt
        fname2 = os.path.join(rundir,"Ndt.txt")
        Ndt=newman_topicmodel.loadsparsemat(fname2)   # teruns an array of counts
        self.Ndt = Ndt
        # make into normalized probability distirbution
        self.theta = newman_topicmodel.conv_Ndt_to_theta(Ndt)

        logger.info("Done loading LDA model (phi,theta) from files" )




