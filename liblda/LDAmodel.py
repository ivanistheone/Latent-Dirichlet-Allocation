

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
#from liblda.util import rungen
#from liblda.util import newman_topicmodel
#import subprocess
#import shutil


# shouldn't be necessary
from local_settings import PROJECT_PATH, topicmodel_DIR, RUNDIRS_ROOT

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
                      size: totalNwords x 1

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
        if not vocab and not hasattr(corpus, 'word2id'):
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
        elif not vocab and corpus.word2id:
            self.numTerms = len(corpus.word2id)

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
            1.1/ Allocate memory
            1.2/ Load the indicator lists self.w and self.d from corpus
        2/ Initialize to random topic assignment --> self.z
        3/ Gibbs sample
        4/ Set self.phi, self.theta and self.z from the Nwt and Ndt arrays
                     Nwt.txt,   Ndt.txt,    z.txt
                     wp         dp
                     ~p(w|t)    ~p(t|d)     Z_{d_i,w_j}

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
        self.wpdt_to_probs()
        #self.deallocate_arrays()

    def countN(self):
        """ Count the total number of words in corpus
            this quantity is called N in Dave Newman's code
            and totalNwords in my code ... for now.
        """

        
        if hasattr(self, "totalNwords") and self.totalNwords != None:
            return self.totalNwords
        # maybe the corpus has cached the totalNwords ?
        if hasattr(self.corpus, "totalNwords") and self.corpus.totalNwords != None:
            return self.corpus.totalNwords

        else:
            # corpus is not smart so must go through it
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

        logger.info("Loading corpus into self.w and self.d")

        offset = 0          # running pointer through self.z, self.d
        curdoc=0            # current document id
        for doc in self.corpus:
            for w,cntw in doc:
                # word w occurs cntw times in doc
                # so have to loop put that many copies of it
            # SLICE assignment faster?
                for i in range(0,cntw):
                    self.w[offset+i]  = w   # IS THIS THE FIX??? OMG!
                    self.d[offset+i]  = curdoc
                offset += int(cntw)

            curdoc += 1

        logger.info("Done loading corpus")




    def random_initialize(self):
        """
        Goes through `self.z` and assigns random choice of topic for
        each of the tokens in each of the documents.

        The ranzom choices of `z` and then 
        update the countsd in `ztot`, `wp` and `dp`.
        """
        logger.info("Random assignment of topic indicator variable self.z started")
        ntot = len(self.z)  # = N  = totalNwords

        # pick list of random topics
        randomz  = np.random.randint(0, high=self.numT, size=ntot)
        self.set_z_and_compute_counts_in_C( randomz )

        logger.info("Random assignment of self.z done")




    def set_z_and_compute_counts(self, z_new):
        """
        An appropirately sized topic assignment vector `z_new`,
        we will be used to replace the current `self.z`.
        
        The change in `z` is also accounted for by recomputing 
            `ztot`, `wp` and `dp`.

        TODO: rewrite in C -- it is kind of slow now.
        """

        logger.info("Replacing topic indicator variable self.z with z_new")

        ntot = len(self.z)      # = N  = totalNwords
        ntotbis = len(z_new)
        assert ntot == ntotbis

        # pick list of random topics
        self.z = z_new

        # reflect the `z` choice in the other arrays
        self.ztot.fill(0)
        self.wp.fill(0)
        self.dp.fill(0)
        for i in range(0,ntot):

            t = self.z[i]        # set it to current token, and

            # update total count topic occurence
            self.ztot[t] +=1

            # update wp and dp via the self.d and self.w lookup tables
            self.wp[self.w[i]][t] += 1
            self.dp[self.d[i]][t] += 1

            if DEBUG and i % (ntot/20) ==0 and not i==0:
                print "progress %d/20 ..." % int((i*20)/ntot +1)

        assert sum(self.ztot) == ntot

        logger.info("self.z has been set to z_new. ztot, wp, dp updated.")


    def set_z_and_compute_counts_in_C(self, z_new):
        """
        An appropirately sized topic assignment vector `z_new`,
        we will be used to replace the current `self.z`.
        
        The change in `z` is also accounted for by recomputing 
            `ztot`, `wp` and `dp`.

        """
        logger.info("Replacing topic indicator variable self.z with z_new")
        ntot = len(self.z)      # = N  = totalNwords
        ntotbis = len(z_new)
        assert ntot == ntotbis
        # pick list of random topics
        self.z = z_new
        # reflect the `z` choice in the other arrays
        self.ztot.fill(0)
        self.wp.fill(0)
        self.dp.fill(0)


        z  = self.z
        w  = self.w
        d  = self.d
        wp = self.wp
        dp = self.dp
        ztot = self.ztot

        code = """ // updating the counts wp and dp 
        int i, t;
        for(i=0; i<ntot; i++){
            t = z[i];   //        # set it to current token, and
            ztot[t] +=1;
            WP2(w[i],t) += 1;
            DP2(d[i],t) += 1;
        }
        """
        out = sp.weave.inline( code,
           ['ntot',                 # params
            'z', 'w', 'd',          # inputs
            'wp', 'dp','ztot'],     # outputs
           headers = ["<math.h>"],      # for isnan() ... but doesn't seem to work.
           compiler='gcc')

        assert sum(self.ztot) == ntot

        logger.info("self.z has been set to z_new. ztot, wp, dp updated.")




    def gibbs_sample(self, iter=None, seed=None ):
        """
        Scipy.weave gibbs sampler called by train()
        """

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


        logger.info("Preparing numpy variables to be passed to the C code")

        # longs
        N        = int( self.corpus.totalNwords )  # will be long long in C ?

            # FIXME!!!
            # This will not work if total n words in corpus is greater than 4 bytes
            # on a 32bit machine (like in SOCS)
            # C code is fine -- it uses long long which always turns out to be 8 bytes

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


        buf = sys.stdout        # to try to get output of printf scrolling

        logger.info("Starting scipy.weave Gibbs sampler")


        code = """  // gibbs_sample C weave code  ///////////////////// START /////


            int i;      // counter within z,d,w
            int itr;    // Gibbs iteration counter

            int t, k, oldt, newt;             // topic indices
            int w_id, term;              // index over words --
            int doc_id;                  // index over documents

            int T;

            // double *probs  ---> converted to cumprobs
            double *cumprobs;                       // new
            double prz, sumprz, currprob, U ;
            int bsmin, bsmax, bsmid;                // for binary search




            double sumalpha, sumbeta;


            // seed the random num generator
            srand48( seed );

            // calculate total alpha and beta values
            sumalpha = 0.0;
            for(t=0; t<numT; t++) {
                sumalpha += alpha[t];
            }
            sumbeta  = 0.0;
            for(term=0; term<numTerms; term++) {
                sumbeta += beta[term];
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

            cumprobs = dvec(T);



            // LETs see what-a-gwan- ooon --- the problem was the sumbeta calc above !!!
            printf("alpha[0]=%f alpha[1]=%f ... alpha[numT-1]=%f \\n", alpha[0], alpha[1], alpha[numT-1] );
            printf("beta[0]=%f beta[1]=%f ... beta[numTerms-1]=%f \\n", beta[0], beta[1], beta[numTerms-1] );

            printf("# of iterations = iter = %d\\n", iter);


            for(itr=0; itr<iter; itr++) {

                //printf("itr = %d\\n", itr);
                fprintf(buf, "itr = %d\\n", itr);

                for(i=0; i<N; i++) {

                    w_id        = (int) w[i];
                    doc_id      = (int) d[i];


                    // decrement all counts
                    oldt = (int) z[i];
                    dp[doc_id*T + oldt]--;
                    wp[  w_id*T + oldt]--;
                    ztot[oldt]--;

                    // fill up   cumprobs = CMF(  P(z| ...)  )
                    sumprz = 0.0;
                    for( t=0; t<numT; t++){
                            /* in case of debug
                            printf("doc_id: %d \\n", doc_id);
                            printf("w_id: %d \\n", w_id);
                            printf("t: %d \\n", t);
                            printf("wp[w_id][t] = %d \\n", wp[w_id*T+t]);
                            printf("dp[doc_id][t] = %d \\n", wp[doc_id*T+t]);
                            printf("ztot[t] = %d \\n", ztot[t] );
                            printf("sumbeta: %f \\n", sumbeta);
                            */
                        prz = (double)(wp[w_id*T+t] + beta[w_id])/(ztot[t]+sumbeta)*(dp[doc_id*T+t] + alpha[t]);
                        sumprz  += prz;
                        cumprobs[t] = sumprz;

                        //if (isnan(prz) ) {
                        //    printf("Current prob: %f \\n", prz);
                        //    printf("-----------------------------------------------------\\n");
                        //}

                    }

                                                            /* Old sampling code

                                                            //sample from probs
                                                            U = sumprz * drand48();
                                                            //    printf("max sample val: %f \\n", sumprz);
                                                            //    printf("sample: %f \\n", U);
                                                            currprob = probs[0];
                                                            newt = 0;
                                                            while( U > currprob){
                                                                newt ++;
                                                                currprob += probs[newt];
                                                            }

                                                            */


                    //sample from probs
                    U = sumprz * drand48();


                    // Binary search in cumprobs
                    // find index newt s.t. cumpobs[newt] => U
                    // and cumprobs[newt-1] < U  OR  newt==0

                       newt = 0;
                       bsmin = 0;
                       bsmax = numT-1;

                       for(;;)
                       {
                           bsmid =  (bsmin + bsmax) /2;
                           currprob = cumprobs[bsmid];
                           if (currprob  < U)
                               bsmin = bsmid  + 1;
                           else if (  bsmid==0   &&    currprob >= U  )
                           {
                             newt = 0;
                             break;
                           }
                           else if (  cumprobs[bsmid-1]<U   &&  currprob >= U  )
                           {
                             newt = bsmid;
                             break;
                           }
                           else if (currprob >  U)
                               bsmax = bsmid - 1;
                           else
                           {
                               printf("Shouldn't be here bro: %d \\n", bsmid);
                               break;
                           }
                       }




                    //printf("newt: %d \\n", newt);
                    // increment back up  all counts
                    z[i] = newt;
                    dp[doc_id*T + newt]++;
                    wp[  w_id*T + newt]++;
                    ztot[newt]++;

                    /*

                    for( t=0; t<numT; t++){
                        debug2[t]=cumprobs[t];
                    }

                    */

                }
           }


            free(cumprobs);


            //////////////////////////////////////////////////////////////  END ///

        """

        out = sp.weave.inline( code,
               ['N','numT', 'numDocs','numTerms',   # constants
                'z', 'd', 'w',      # topic, document_id and term_id for i \in [totalNwords]
                'ztot',             # total # tokens in corpus with topit t \in [numT]
                'dp', 'wp',         #
                'alpha', 'beta',    # Dirichlet priors of self.theta and self.phi respectively
                'iter', 'seed',     # gibbs specific params
                'buf',
                'debug', 'debug2' ],          #
               support_code=extra_code,
               headers = ["<math.h>"],      # for isnan() ... but doesn't seem to work.
               compiler='gcc')

        logger.info("Finished scipy.weave Gibbs sampler")

        return out





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



    def conv_dp_to_theta(self):
        """
        converts the topic counts per document in self.dp
        to a probability distibution
          p(t|d)
        rows are documents
        columns are topic proportions
        """
        numDocs,numT = self.dp.shape
        input = np.array( self.dp, dtype=np.float )


        #                    N_td   + alpha[t]
        #  p(t|d)     =  - ----------------------
        #                 sum_t N_td   + sumalpha

        sumalpha=np.sum(self.alpha)
        totWinDocs = np.sum(input,1)
        denom= totWinDocs + sumalpha

        normalizer = 1.0/denom
        for t in np.arange(0,numT):
            input[:,t]=(input[:,t]+self.alpha[t])*normalizer
        self.theta = input
        # i think this is the same as
        # (input + np.ones( (numT,numT))*alpha) * normalizer ... ;)



    def conv_wp_to_phi(self):
        """
        convers number of words in topic counts
        to a probability distibution
          p(w|t)
        rows are topics
        columns are word likelyhoods
        """

        numTerms,numT = self.wp.shape
        input = np.array( self.wp, dtype=np.float )

        #                    N_wt   + beta[w]
        #  p(w|t)     =  - ----------------------
        #                 sum_w N_wt   + sumbeta
        sumbeta = np.sum(self.beta)
        ztot = np.sum(input,0)      # total number of words in corpus for topic t
        denom = ztot + sumbeta

        betarows = np.resize( self.beta, (numT,len(self.beta)) )  # numpy makes multiple copies of array on resize
        betacols = betarows.transpose()

        withbeta = input + betacols

        prob_w_given_t = np.dot(withbeta, np.diag(1.0/denom) )
        self.phi = prob_w_given_t.transpose()


    def wpdt_to_probs(self):
        """
                self.wp   ==> self.Nwt  ==> self.phi
                self.dp   ==> self.Ndt  ==> self.theta
        """
        self.conv_dp_to_theta()
        self.conv_wp_to_phi()

        logger.info("Finished converting  wp  --> phi  and  dp --> theta ")


    def loglike(self, recompute=True):
        """
        Compute the log likelyhood of the corpus
        under current `phi` and `theta` distributions

        assumes that accessing the corpus is expensive
        so goes though the lists `self.w` and `self.d` instead

        if corpus is in RAM also, then more efficient to use term_counts
        """
        if hasattr(self, 'loglike_val') and not recompute:
            return self.loglike_val
        else:
            sum=0.0
            for i in range(0,self.corpus.totalNwords):
                sum += np.log( np.inner( self.phi[:,self.w[i]], self.theta[self.d[i],:] ) )
            self.loglike_val = sum
            return self.loglike_val


    def perplexity(self, recompute=True):
        """ Compute the perplexity of corpus = exp( - loglike / totalNwords ) """
        if hasattr(self, 'perplexity_val') and not recompute:
            return self.perplexity_val
        else:
            self.perplexity_val =  np.exp( -1.0*self.loglike()/self.corpus.totalNwords )
            return self.perplexity_val
                




    def seeded_initialize(self, seed_z, expand_factors):
        """
        Given a `z` vector from a previous LDA run, with topics assignments
        varying between 0 and seed_numT-1, we set the values of `self.z`
        to be a random "subtopics" of the original.

        The ratio numT/seed_numT is called the expand_factor.

        The ranzom choices of `z` are also accounted for in `ztot`, `wp` and `dp`.
        """

        logger.info("Seeded assignment of topic indicator variable self.z started")

        ntot = len(self.z)  # = N  = totalNwords
        seedntot = len(seed_z)  # = N  = totalNwords
        assert ntot == seedntot, "Seed z.npy must be of same length as self.z"

        # calculate seed_numT
        maxt = 0
        for t in seed_z:
            if t> maxt:
                maxt=t
        seed_numT = maxt + 1    # assumes all topics appear at least once

        # setup uniform expand_factors if None is specified
        if expand_factors is  None:
            expand_factors = np.zeros(seed_numT, dtype=np.int)
            unif_expand = int( self.numT/seed_numT )
            for i in range(0,seed_numT-1):
                expand_factors[i]= unif_expand
            expand_factors[seed_numT-1] = self.numT - np.sum(expand_factors)


        # do some checking
        assert np.sum(expand_factors) == self.numT, "expand_factors must sum up to total of topics in new model"
        assert len(expand_factors) == seed_numT, "must specify exactly one expand_factor per seed topic"


        # go through topic assignment list and assign to one of expand_factors subtopics
        shift_right_one = np.concatenate( (np.zeros(1), expand_factors) )
        prev_expand = np.cumsum( shift_right_one )
        for i in range(0,ntot):
            seed_t = seed_z[i]
            self.z[i] = prev_expand[seed_t] + np.random.randint(0, expand_factors[seed_t])

        # update the counts dp, wp and ztot
        self.set_z_and_compute_counts_in_C( self.z )

        assert sum(self.ztot) == ntot

        logger.info("Seeded assignment of self.z done")





    def load_from_rundir(self, rundir):
        """ re-hydrates a LdaModel object from the infor
            stored in rundir:
                minimum requirements:
                    z.npy
                    alpha.npy
                    beta.npy
                if the following "counts" are present we load them too
                    Nwt.npy
                    Ndt.npy
                finally the probs can be computed from the counts
                    phi.npy
                    theta.npy
        """
        if not os.path.exists( rundir ):
            raise IncompleteInputError('Rundir does not exist, so cannot load')

        logger.info("Loading LDA model from disk")

        # load the essential ones
        for var in ["z"]: #,"alpha", "beta"]:
            fname = os.path.join( rundir, var+".npy")
            if not os.path.exists( fname ):
                raise IncompleteInputError('Cannot find '+var+".npy in rundir "+ rundir )
            npy_var = np.load(fname)
            setattr(self, var, npy_var)
        
        # warning CORPUS not loaded
        #         VOCAB  not loaded 
        # hence   w and d not loaded ...
        #
        # IDEA:  We can recurse up the rundir hierarchy
        #        to look for a corpus / vocab file
        #      or   
        #        read it from output.json
        #



        # compute the counts
        # self.allocate_arrays()
        # self.set_z_and_compute_counts_in_C( self.z )
        # NOT FEASIBLE since depends on w and d which 
        # are not available -- since we don't know corpus yet :(

        # load counts 
        self.dp = np.load(  os.path.join( rundir, "Ndt.npy" ) )
        self.wp = np.load(  os.path.join( rundir, "Nwt.npy")  )
        assert  all( np.sum(self.dp,0) ==  np.sum(self.wp,0) )
        self.ztot = np.sum(self.dp,0)


        # set  params
        self.numDocs, self.numT = self.dp.shape
        self.numTerms = self.wp.shape[0]
        self.totalNwords = len(self.z)


        # setup alpha and beta
        if not hasattr(self.alpha, "__iter__"):      # if \alpha is not list like
            self.alpha = self.alpha*np.ones(self.numT)
        if not hasattr(self.beta, "__iter__"):
            self.beta = beta*np.ones(self.numTerms)

        # compute the probs
        self.wpdt_to_probs()


        logger.info("Loaded LDA model. Need to set CORPUS and VOCAB manually.")








    ##### INFERENCE ################################################################################









    def inference(self, qcorpus, iter=100, seed=3):
        """ Returns the inferred theta matrix for the query corpus """

        #        if not hasattr(self, 'iter'):
        self.iter=iter
        #        if not hasattr(self, 'seed'):
        self.seed=seed

        self.qcorpus = qcorpus
        self.numQDocs = len(qcorpus)               # how do i make this lazy?
                                            # assume Corpus object is smart
                                            # and has cached its length ...
        self.qtheta = None
        self.qz = None


        # subfunctions
        self.allocate_qarrays()
        self.read_qdw_alphabetical()
        self.qrandom_initialize()
        self.qgibbs_sample(iter=self.iter, seed=self.seed )
        #self.wpdt_to_probs()
        self.conv_qdp_to_qtheta()

        logger.info("Finished inference.")
        return self.qtheta
        
        
        
    def countQN(self):
        """ Count the total number of words in query corpus        """
        if hasattr(self, "totalQNwords") and self.totalQNwords != None:
            return self.totalQNwords
        # maybe the corpus has cached the totalNwords ?
        if hasattr(self.qcorpus, "totalNwords") and self.qcorpus.totalNwords != None:
            return self.qcorpus.totalNwords
        else:
            # corpus is not smart so must go through it
            total=0L
            for doc in self.qcorpus:
                for word,cnt in doc:
                    total += cnt
            self.qcorpus.totalNwords = total
            return total


    def allocate_qarrays(self):
        """
        Allocates memory for the query arrays of counts which
        will be necessary for the Gibbs sampler.

        """
        totalQNwords = self.countQN()     # this is the total n of tokens
                                          # in the query corpus.
        # The Gibbs sampling loop uses the "corpus" index i \in [0,1, ..., totalNwords -1 ]
        # sometimes we will use i = (m,n) where m is the document id and n is the n-th word
        # in the document m.  m \in [numDocs],   n \in [wc_of_dm]
        self.qd = np.zeros(totalQNwords, dtype=np.int32) # the document id of token i (m)
        self.qw = np.zeros(totalQNwords, dtype=np.int32) # the term id of token i  w[i] \in [numTerms]
        self.qz    = np.zeros(totalQNwords, dtype=np.int32) # the topic of token i,   z[i] \in [numT]
        self.qdp   = np.zeros( (self.numQDocs, self.numT), dtype=np.int32 )


    def read_qdw_alphabetical(self):         # i.e. read_dw
        """
        Initizlizes the query corpus arrays with the correct joint index
        i=(m,n)
        where m is the document id  \in [numDocs]
        and   n is a term id  \in [numTerms]
        """
        logger.info("Loading query corpus into self.qw and self.qd")

        offset = 0          # running pointer through self.z, self.d
        curdoc=0            # current document id
        for doc in self.qcorpus:
            for w,cntw in doc:
                for i in range(0,cntw):
                    self.qw[offset+i]  = w   # IS THIS THE FIX??? OMG!
                    self.qd[offset+i]  = curdoc
                offset += int(cntw)
            curdoc += 1
        logger.info("Done loading query corpus")



    def qrandom_initialize(self):
        """
        Goes through `self.qz` and assigns random choice of topic 
        for each of the tokens according to self.phi \equiv self.wp

        """

        ntotq = len(self.qz)  # = QN  = totalQNwords
        
        # pick list of random topics
        randomqz  = np.random.randint(0, high=self.numT, size=ntotq)
        self.set_qz_and_compute_counts_in_C( randomqz )
        logger.info("Random assignment of self.qz done")

    def set_qz_and_compute_counts_in_C(self, z_new):
        """
        An appropirately sized topic assignment vector `z_new`,
        we will be used to replace the current `self.qz`.
        
        The change in `z` is also accounted for by recomputing 
        `qdp`.
        """
        logger.info("Replacing topic indicator variable self.qz with z_new")
        qntot = len(self.qz)      # = N  = totalNwords
        ntotbis = len(z_new)
        assert qntot == ntotbis
        # pick list of random topics
        self.qz = z_new
        self.qdp.fill(0)
        qz  = self.qz
        qw  = self.qw
        qd  = self.qd
        qdp = self.qdp
    
        code = """ // updating qdp 
        int i, t;
        for(i=0; i<qntot; i++){
            t = qz[i];   //        # set it to current token, and
            QDP2(qd[i],t) += 1;
        }
        """
        out = sp.weave.inline( code,
           [ 'qntot',
             'qz', 'qw', 'qd',          # inputs
             'qdp'],                  # outputs
           headers = ["<math.h>"],      # for isnan() ... but doesn't seem to work.
           compiler='gcc')
        logger.info("self.qz has been set to z_new. qdp updated.")
            

    def qgibbs_sample(self, iter=None, seed=None ):
        """
        Scipy.weave gibbs sampler called by inference()
        """
        
        extra_code = """
           // line 1089 in LDAmodel.py
           double *dvec(int n) //
           {
             double *x = (double*)calloc(n,sizeof(double));
             assert(x);
             return x;
           }
        """

        logger.info("Preparing numpy variables to be passed to the C code")

        # longs
        QN        = int( self.qcorpus.totalNwords )  # will be long long in C ?
        # ints
        numT     = int( self.numT )
        numQDocs  = int( self.numQDocs )
        numTerms = int( self.numTerms )

        # 1D arrays
        qz = self.qz
        ztot = self.ztot
        qd = self.qd
        qw = self.qw
        alpha = self.alpha
        beta = self.beta

        # 2D arrays
        qdp = self.qdp
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


        buf = sys.stdout        # to try to get output of printf scrolling

        logger.info("Starting scipy.weave Gibbs sampler")

        code = """  // query gibbs_sample C weave code  ///////////////////// START /////

            int i;      // counter within z,d,w
            int itr;    // Gibbs iteration counter

            int t, k, oldt, newt;             // topic indices
            int w_id, term;              // index over words --
            int doc_id;                  // index over documents

            int T;

            // double *probs  ---> converted to cumprobs
            double *cumprobs;                       // new
            double prz, sumprz, currprob, U ;
            int bsmin, bsmax, bsmid;                // for binary search

            double sumalpha, sumbeta;

            // seed the random num generator
            srand48( seed );

            // calculate total alpha and beta values
            sumalpha = 0.0;
            for(t=0; t<numT; t++) {
                sumalpha += alpha[t];
            }
            sumbeta  = 0.0;
            for(term=0; term<numTerms; term++) {
                sumbeta += beta[term];
            }

            T = numT;
            cumprobs = dvec(T);

            printf("# of iterations = iter = %d\\n", iter);

            for(itr=0; itr<iter; itr++) {

                printf("itr = %d\\n", itr);

                for(i=0; i<QN; i++) {

                    w_id        = (int) qw[i];
                    doc_id      = (int) qd[i];

                    // decrement all counts
                    oldt = (int) qz[i];
                    QDP2(doc_id,oldt)=QDP2(doc_id,oldt)-1;
                    //wp[  w_id*T + oldt]--;
                    //ztot[oldt]--;

                    // fill up   cumprobs = CMF(  P(z| ...)  )
                    sumprz = 0.0;
                    for( t=0; t<numT; t++){
                        prz = (double)(wp[w_id*T+t] + beta[w_id])/(ztot[t]+sumbeta)*(qdp[doc_id*T+t] + alpha[t]);
                        sumprz  += prz;
                        cumprobs[t] = sumprz;
                    }

                    //sample from probs
                    U = sumprz * drand48();


                    // Binary search in cumprobs
                    newt = 0;
                    bsmin = 0;
                    bsmax = numT-1;                    
                    for(;;)
                    {
                       bsmid =  (bsmin + bsmax) /2;
                       currprob = cumprobs[bsmid];
                       if (currprob  < U)
                           bsmin = bsmid  + 1;
                       else if (  bsmid==0   &&    currprob >= U  )
                       {
                         newt = 0;
                         break;
                       }
                       else if (  cumprobs[bsmid-1]<U   &&  currprob >= U  )
                       {
                         newt = bsmid;
                         break;
                       }
                       else if (currprob >  U)
                           bsmax = bsmid - 1;
                       else
                       {
                           printf("Shouldn't be here bro: %d \\n", bsmid);
                           break;
                       }
                    }

                    qz[i] = newt;
                    //qdp[doc_id*T + newt]++;
                    QDP2(doc_id,newt)=QDP2(doc_id,newt)+1;
                }
           }


            free(cumprobs);


            //////////////////////////////////////////////////////////////  END ///

        """

        out = sp.weave.inline( code,
               ['QN','numT', 'numQDocs','numTerms',   # constants
                'qz', 'qd', 'qw',      # topic, document_id and term_id for i \in [totalNwords]
                'ztot',             # total # tokens in corpus with topit t \in [numT]
                'qdp', 'wp',         #
                'alpha', 'beta',    # Dirichlet priors of self.theta and self.phi respectively
                'iter', 'seed',     # gibbs specific params
                'buf',
                'debug', 'debug2' ],          #
               support_code=extra_code,
               headers = ["<math.h>"],      # for isnan() ... but doesn't seem to work.
               compiler='gcc')

        logger.info("Finished query Gibbs sampling")

        return out



    def conv_qdp_to_qtheta(self):
        """
        converts the query topic counts per document in self.qdp
        to a probability distibution
          p(t|d)
        rows are documents columns are topic proportions
        """
        numQDocs,numT = self.qdp.shape
        input = np.array( self.qdp, dtype=np.float )

        #                    N_td   + alpha[t]
        #  p(t|d)     =  - ----------------------
        #                 sum_t N_td   + sumalpha

        sumalpha=np.sum(self.alpha)
        totWinDocs = np.sum(input,1)
        denom= totWinDocs + sumalpha
        normalizer = 1.0/denom
        for t in np.arange(0,numT):
            input[:,t]=(input[:,t]+self.alpha[t])*normalizer
        self.qtheta = input



    def qloglike(self, recompute=True):
        """
        Compute the log likelyhood of the corpus
        under current `phi` and `theta` distributions

        assumes that accessing the corpus is expensive
        so goes though the lists `self.w` and `self.d` instead

        if corpus is in RAM also, then more efficient to use term_counts
        """
        if hasattr(self, 'qloglike_val') and not recompute:
            return self.qloglike_val
        else:
            sum=0.0
            for i in range(0,self.qcorpus.totalNwords):
                sum += np.log( np.inner( self.phi[:,self.qw[i]], self.qtheta[self.qd[i],:] ) )
            self.qloglike_val = sum
            return self.qloglike_val


    def qperplexity(self, recompute=True):
        """ Compute the perplexity of corpus = exp( - loglike / totalNwords ) """
        if hasattr(self, 'qperplexity_val') and not recompute:
            return self.qperplexity_val
        else:
            self.qperplexity_val =  np.exp( -1.0*self.qloglike()/self.qcorpus.totalNwords )
            return self.qperplexity_val
                
                
                
