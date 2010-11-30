

import os,sys
import shutil
import datetime


from liblda import interfaces

from liblda.util import rungen
from liblda.util import newman_topicmodel


import subprocess

from local_settings import PROJECT_PATH, topicmodel_DIR, RUNDIRS_ROOT


import logging
logger = logging.getLogger('NewmanLdaModel')
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

class ScriptError(Exception):
    """
    An error occurred somewhere in train(), i.e creating rundir
    writing input to disk, mv-ing the executable in place in rundir
    or running ./topicmodel
    """
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)


class NewmanLdaModel(interfaces.LdaModelABC):
    """
    A class that calls Dave Newman's Gibbs sampler to train the
    model.

    Parameters:
        numT        : number of topics in model
        alpha       : Dirichlet prior on theta (topics in docs)
                      len: numT   or  constant scalar
        beta        : Dirichlet prior on phi (words in topics)

    Input data:
        corpus      : reference to a corpus object (anything that looks like [[ (termid,count) ]]
        numDocs     : len(corpus)

    Optional input:
        vocab       : ref to Vacabulary object (which can map word id to word string)
        numTerms    : len(vocab)

    API:
        corpus=[[(1,1),(2,4)],[(1,1)]]
        lda = NewmanLdaModel( corpus=corpus, numT=17, alpha=50.0/17, beta=0.01 )
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
        self.numT = numT                    # number of topics
        self.alpha = alpha                  # Dirichlet prior on theta (topics in docs)
        self.beta  = beta                   # Dirichlet prior on phi (words in topics)

        self.corpus= corpus
        #numDocs = len(corpus)      how do i make this lazy?

        self.vocab = vocab

        self.phi = None
        self.theta = None
        self.z = None



    def is_trained(self):
        """
        Tells you whether model has been trained or not.
        """
        raise NotImplementedError('cannot instantiate abstract base class')

    def train(self, rundir=None, rundir_root=None, iter=50,seed=None):
        """
        Runs the Dave Newman's C code:
         1/ setup a rundir, if no rundir, then create one in /tmp
         2/ write the contents of corpus to  rundir/docword.txt
         3/ extract a copy of the executable for that OS
         4/ run topicmodel
                run topic model(T=10 topics, iter=200 iterations),
                ./topicmodel 10 200 777
         5/ read in  Nwt.txt,  Ndt.txt,  z.txt into
                     p(w|t)    p(t|d)    Z_{d_i,w_j}

        """
        if not self.numT:
            raise  IncompleInputError('Must specify number of topics: self.numT')
        if self.corpus ==None:
            raise  IncompleInputError('Must provide a corpus to train on: self.corpus')


        # params
        self.iter = iter    # iter set to 50 by default..

        if not seed:
            seed = 777
        self.seed = seed

        self.mkrundir(rundir)
        self.copy_topicmodel(rundir)
        self.write_corpus_to_docword(rundir)
        self.run(rundir,iter,seed)
        self.load_probs(rundir)





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




