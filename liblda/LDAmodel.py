

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


DEBUG = True



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
        self.read_dw_alphabetical()
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
        will be necessary for the Gibbs sampler.

        I am working with the assumption that the C code will be
        able to access the numpy arrays at "native speed",
        so I just have instantiate some numpy.zeros arrays.

        note:   On my machine (Mac OS 64 bit, python2.6)
                compiled using gcc, the C code thiks that sizeof(int) == 4
                So I am forcing the use of int32 as the data type,
                because otherwise numpy will use int64.
        """

        totalNwords = self.countN()     # this is the total n of tokens
                                        # in the corpus. i.e.  wc -w corpus.txt

        # The Gibbs sampling loop uses the "corpus" index i \in [0,1, ..., totalNwords -1 ]
        # sometimes we will use i = (m,n) where m is the document id and n is the n-th word
        # in the document m.  m \in [numDocs],   n \in [wc_of_dm]
        self.d = np.zeros(totalNwords, dtype=np.int32) # the document id of token i (m)
        self.w = np.zeros(totalNwords, dtype=np.int32) # the term id of token i  w[i] \in [numTerms]

        self.z    = np.zeros(totalNwords, dtype=np.int32) # the topic of token i,   z[i] \in [numT]
        self.ztot = np.zeros(self.numT,   dtype=np.int32)  # ztot[k] = total # of tokens in corpus
                                                            # that have been assigned to topic k


        # so far we have consumed  4*N  + T units of RAM
        # now come the two heavy hitters

        self.wp   = np.zeros( (self.numTerms, self.numT), dtype=np.int32 )
        # wp[w][t]  is the count of how many times the word w \in [numTerms]
        #           has been assigned to topic t \in [numT]

        self.dp   = np.zeros( (self.numDocs, self.numT), dtype=np.int32 )
        # dp[m][t]  is the count of words assigned to topic t \in [numT]
        #           in document m

        # RAM consumption += (numTerms + numDocs)*numT


    def read_dw_alphabetical(self):         # i.e. read_dw
        """
        Initizlizes the corpus arrays with the correct joint index
        i=(m,n)
        where m is the document id  \in [numDocs]
        and   n is a term id  \in [numTerms]

        note:   this funciton has the effect of assigning large
                contiguous patches of the same topic.
                TODO:   make a funciton read_dw that reads directly from the
                        original word stream (pre-corpus since corpus is already BOW).
                        read_dw_prebow ?
        """

        offset = 0          # running pointer through self.z, self.d
        curdoc=0            # current document id
        for doc in self.corpus:

            for w,cntw in doc:
                # word w occurs cntw times in doc
                # so have to loop put that many copies of it
            # SLICE assignment faster?
                for i in range(0,cntw):
                    self.w[offset+i]  = w
                    self.d[offset+i]  = curdoc
                offset += int(cntw)

            curdoc += 1




    def random_initialize(self):
        """
        The ease of writing this funciton is the main reason
        why I wanted to code all this up in python.
        Lets see if numpy is "really" that efficient for scientific exploration.

        on your marks, get set: r!date
        Thu 30 Dec 2010 20:09:30 EST
        """


        # ok now really starting -- had to creat read_dw_alphabetical first
        #Thu 30 Dec 2010 20:24:05 EST

        ntot = len(self.z)  # = N  = totalNwords

        self.z = np.random.randint(0, high=self.numT, size=ntot)

        for i in range(0,ntot):

            # pick a random topic

            # set it to current token, and
            t = self.z[i]
            self.ztot[t] +=1

            # update wp and dp via the self.d and self.w lookup tables
            self.wp[self.w[i]][t] += 1
            self.dp[self.d[i]][t] += 1

            if DEBUG and i % (ntot/20) ==0:
                print "progress %d/20 ..." % i




        assert sum(self.ztot) == ntot

        # finished....
        #Thu 30 Dec 2010 20:34:39 EST

        # OK and perhaps this is the BEAUTY of python
        # i just tried to run the code... and it runs... and I looked
        # around with the interpreter and seems to be all OK
        # done at Thu 30 Dec 2010 20:38:07 EST
        # 14 mins is decent ...



    def gibbs_sample(self, iter=None, seed=None ):

        extra_code = """
           // line 209 in LDAmodel.py
           double *dvec(int n) //
           {
             double *x = (double*)calloc(n,sizeof(double));
             assert(x);
             return x;
           }

            /*  given a T-vector of probs [0.1, 0.2, 0.33, 0.37 ]
                will return an int \in [T]

            int sample_probs( T, probs, total) {
                int i;
                i=2;
                // probably not a good idea to make a fn call so
                // cancelling this one and will do it in main loop
            }

                */



        """

        # longs
        N        = int( self.corpus.totalNwords )  # will be long long in C ?

        # ints
        numT     = int( self.numT )
        numDocs  = int( self.numDocs )
        numTerms = int( self.numTerms )

        # 1D arrays
        z = self.z
        ztot = self.ztot
        d = self.d
        w = self.w
        alpha = self.alpha
        beta = self.beta

        # 2D arrays
        dp = self.dp
        wp = self.wp

        # just to be sure let's intify the Gibbs algo. params
        if seed:
            seed = int(seed)
        else:
            seed = int(self.seed)
        if iter:
            iter = int(iter)
        else:
            iter = int(self.iter)


        # bonus
        debug = np.zeros( 100 , dtype=np.int32)
        debug2 = np.zeros( 100 , dtype=np.float64)




        code = """  // gibbs_sample C weave code  ///////////////////// START /////


            int i;      // counter within z,d,w
            int itr;    // Gibbs iteration counter

            int t, k, oldt, newt;             // topic indices
            int w_id, term;              // index over words --
            int doc_id;                 // index over documents

            int T;

            double *probs, prz, sumprz, currprob, U ;


            double sumalpha, sumbeta;



            // calculate total alpha and beta values
            sumalpha = 0.0;
            for(t=0; t<numT; t++) {
                sumalpha += alpha[t];
            }
            sumbeta  = 0.0;
            for(term=0; term<numTerms; term++) {
                sumbeta += alpha[term];
            }



            T = numT;
            /* note that both dp and wp
               are "wrongly" defined as (*int) by weave when
               in fact they are (**int), thus we have
               to "manually" do the row access logic.

               Each row of dp and wp is of size T=numT the number of topics.

                   dp[d][t] -->  dp[d*T + t]
                   wp[w][t] -->  wp[w*T + t]

                it could be cleaner but hey...
                */

            probs = dvec(N);

            for(itr=0; itr<iter; itr++) {

                for(i=0; i<N; i++) {

                    w_id        = (int) w[i];
                    doc_id      = (int) d[i];


                    // decrement all counts
                    oldt = (int) z[i];
                    dp[doc_id*T + oldt]--;
                    wp[  w_id*T + oldt]--;
                    ztot[oldt]--;

                    // fill up   probs = P(z| ...)
                    sumprz = 0.0;
                    for( t=0; t<numT; t++){
                        prz = (wp[w_id*T+t] + beta[w_id])/(ztot[t]+sumbeta)*(dp[doc_id*T+t] + alpha[t]);
                        probs[t] = prz;
                        sumprz  += prz;

                        //printf("Cur prob is eval at: %f", prz);
                    }

                    //sample from probs
                    U = sumprz * drand48();
                    currprob = probs[0];
                    newt = 0;
                    while( U > currprob){
                        newt ++;
                        currprob += probs[newt];
                    }
                    // TODO: bin_search -- up in support code


                    // increment back up  all counts
                    z[i] = newt;
                    dp[doc_id*T + newt]++;
                    wp[  w_id*T + newt]++;
                    ztot[newt]++;

                    /*

                    for( t=0; t<numT; t++){
                        debug2[t]=probs[t];
                    }

                    */

                }
           }


            free(probs);

            //////////////////////////////////////////////////////////////  END ///

        """

        out = sp.weave.inline( code,
               ['N','numT', 'numDocs','numTerms',   # constants
                'z', 'd', 'w',      # topic, document_id and term_id for i \in [totalNwords]
                'ztot',             # total # tokens in corpus with topit t \in [numT]
                'dp', 'wp',         #
                'alpha', 'beta',    # Dirichlet priors of self.theta and self.phi respectively
                'iter', 'seed',     # gibbs specific params
                'debug', 'debug2' ],          #
               support_code=extra_code,
               compiler='gcc')

        return out



    def wpdt_to_probs(self):
        """
                self.wp   ==> self.Nwt  ==> self.phi
                self.dp   ==> self.Ndt  ==> self.theta
        """
        # read Nwt.txt
        #self.Nwt = Nwt
        # make into normalized probability distirbution
        self.phi = newman_topicmodel.conv_Nwt_to_phi(wp)

        # read Ndt
        #self.Ndt = Ndt
        # make into normalized probability distirbution
        self.theta = newman_topicmodel.conv_Ndt_to_theta(dp)



    def deallocate_arrays(self):
        """
        free up some memory

        note: there is a sense in keeping sparse versions
              of the model variables that we have learned:
                self.wp   ==> self.Nwt  ==> self.phi
                self.dp   ==> self.Ndt  ==> self.theta
        """

        del self.d
        del self.w
        del self.z
        del self.ztot

        del self.wp
        del self.dp





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




