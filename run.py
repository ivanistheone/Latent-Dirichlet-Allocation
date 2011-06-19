#!/usr/bin/env python2.6


# general
import os,sys
import logging
import datetime
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
logger = logging.getLogger('lda-run')
logger.setLevel(logging.INFO)


try:
    import json as simplejson
except ImportError:
    import simplejson

#
#    ASSUMPTION:    PROJACT_DIR/run.py              !!!!!!!!!!!!!!!!!!!!!!!!!!!!!   <-----------
#
# assumes run.py is found at the root of the LDA directory
PROJECT_PATH=os.path.realpath(os.path.join(os.path.dirname(__file__),"."))
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

# example:
# runs/
#   lab7-4/
#       run0003/
#           inputs.json             the specific parameters set for this run
#               numT  = int   number of topics to find
#               iter
#               seed
#               alpha = list( float )    or  float
#               beta  = list( float )    or  float
#               override of default options:    corpus-format=".mm"
#                                               vocab-format=".json"
#
#           docs.mm         --s-->  ../data/some_corpus_collection.mm
#           vocab.json      --s-->  ../data/the_vocab_for_corpus.json  dict like { "thetoken":int("w_id"), ... }
#
#           results.json            appears after run is done
#               numTerms
#               numDocs
#               totalNwords
#               numT
#               duration, perplexity,
#               likelyhood?, top30wintopic,
#               post processing functions read and write to this file
#
#           //gibbs sampler state
#           Nwt.npy
#           Ndt.npy
#           z.npy
#
#           //outputs
#           topics_in_docs.json         which topics are present in each document
#           words_in_topics.json        the inv. vocab lookup of w_id's in each row of \Phi=p(w|t)
#
#           phi.npy         a 2D numpy array pickled   row i, column j has   Pr{word=j when topic=i}
#           theta.npy       a 2D numpy array pickled   row i, column j has   Pr{topic=j when document=i}
#
#
#
#           // autotagging
#           tags.json                   labels for mixtures of topics (these document-like
#           tags_in_docs_timeline.html
#               a long scrollable repr of topics_in_docs.j (sorted by doc_id -- which we assume correlates to "time")
#
####################################



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


# this is the main work horse !
from liblda.LDAmodel import LdaModel
####################################
####################################





import types







# Errors accociated with this
class ListInputError(Exception):
    """
    The user supplied a bad list file
    """
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)









def smart_list_reader( fname ):         #,  items_type=None):
    """ Tries to read a filename and return an iterable.
        Supported formats are:
            comma separated list of words ALL ON ONE LINE
            one item per line
            numpyarray np.save( ...
            the keys of a pickled dict cPickle.dump ( {"item1":[some metadata for item1], "item2":[meta2], ... } )

        Returned list for .txt formats will be a list of strings, so if you know
        that items should be ints of floats you have to convert by yourself.
    """

    items = None

    if fname.endswith(".json"):         # Q? Can we guarantee order is preserved in json serialization?
        file = open(fname, 'r')
        vjson = simplejson.load(file)
        items = vjson
    elif fname.endswith(".txt"):
        file = open(fname, 'r')
        lines = file.readlines()

        if len(lines)==2 and lines[1].strip()=='':   # handle newline at EOL just in case
            items = [ it.strip() for it in  lines[0].split(",") ]
        elif len(lines)==1:
            items = [ it.strip() for it in  lines[0].split(",") ]
        else:
            items = [l.strip() for l in lines if len(l.strip())>0 ]
    elif fname.endswith(".npy"):
        loaded = np.load(fname)
        if loaded.shape == ():      # handles a dict saved by numpy
            maybe_items = loaded.item()
            if type(maybe_items) == types.DictType:
                items = maybe_items.keys()
            elif type(maybe_items) == types.ListType:
                items = maybe_items
        else:
            items = loaded
    else:
        raise ListInputError("List file type not recognized")


    return items









def run(args):
    """
    The command-line run script for LDA experiments.


    """

    # scientific
    import numpy as np
    import scipy as sp


    # display what run got in args
    for tup in args.__dict__.iteritems():
        print tup


    # LOAD VOCAB
    wlist = smart_list_reader( args.vocab_file )
    if not wlist:
        print "Vocab format not recognized"
        sys.exit(-1)
    # convert from list [term1, term2, ...] to dicts
    # [term1:0, term2:1, ... ] and the inverse mapping
    id2word = dict( enumerate(wlist) )
    word2id = dict( [(word,id)  for id,word in id2word.items()] )
    vocab = word2id


    # SETUP CORPUS (LAZY)
    # doCounts -- not so lazy...
    if args.docs_file[-3:]==".mm":
        from liblda.newmmcorpus import NewMmCorpus
        corpus = NewMmCorpus(args.docs_file)
        corpus.setVocabFromDict( vocab )
        corpus.doCounts()
    elif args.docs_file[-4:]==".txt":
        from liblda.low2corpus import Low2Corpus
        corpus = Low2Corpus(args.docs_file)
        corpus.setVocabFromDict( vocab )
        corpus.doCounts()
    else:
        print "Corpus format not recognized"
        sys.exit(-1)



    # Create rundir
    from socket import gethostname
    from liblda.util import rungen

    full_hostname = gethostname()
    host_id = full_hostname.rstrip(".cs.mcgill.ca")

    if not args.rundirs_root:
        rundirs_root = RUNDIRS_ROOT
    else:
        rundirs_root = args.rundirs_root
    if not os.path.exists(rundirs_root):
        print "Error, rundirs_root %s doesn't exist" % rundirs_root
        sys.exit(-1)

    # create the host-specific rundir if necessary
    host_rundirs_root = os.path.join(rundirs_root, host_id)
    if not os.path.exists(host_rundirs_root):
        os.mkdir( host_rundirs_root )

    # create a new (sequential) rundir for this host
    rundir = rungen.mk_next_rundir(host_rundirs_root)
    logger.info("rundir: " + rundir  )

    # prepare a dict which will become input.json
    input = {}
    input["rundir"]=rundir
    input["numT"]=args.numT
    input["iter"]=args.iter
    input["corpus"]=args.docs_file
    input["vocab"]=args.vocab_file
    input["alpha"]=args.alpha
    input["beta"]= args.beta
    input["seed"]=args.seed
    input["host_id"]=host_id
    # and write it to disk
    f=open( os.path.join(rundir, "input.json"), "w" )
    simplejson.dump( input, f, indent=0 )
    f.close()




    start_time = datetime.datetime.now()



    # setup the lda model
    lda = LdaModel( numT=args.numT,    alpha=args.alpha, beta=args.beta,  corpus=corpus, vocab=vocab )


    # if not in seeded mode run as usual
    if not args.seed_z_from:
        if not args.save_perplexity_every:
            lda.train(iter=args.iter, seed=args.seed )
        else:
            lda.allocate_arrays()
            lda.read_dw_alphabetical()
            lda.random_initialize() 
            cum=0
            perp_hist = []
            while( cum < args.iter):

                lda.gibbs_sample(iter=args.save_perplexity_every, seed=args.seed+cum )
                lda.wpdt_to_probs()
                perp_hist.append( lda.perplexity() )   # = np.exp( -1 * loglike() / totalNwords )

                cum += args.save_perplexity_every


            
    # NEW: S
    else:
        logger.info("Using seeded z training ... ")

        # training params
        if not args.iter:
            lda.iter = 50
        else:
            lda.iter = args.iter

        if not args.seed:
            seed = 777
            lda.seed = 2*seed+1
        else:
            lda.seed = 2*args.seed + 1


        # loadup the seed_z_from file into seed_z np array
        seed_z = np.load( args.seed_z_from)
        if args.expand_factors:
            expand_factors_str = smart_list_reader( args.expand_factors )
            expand_factors = np.array( [int(i) for i in expand_factors_str ] )
        else:
            expand_factors = None    # let lda.seeded_initialize() handle it

        # custom train sequence
        lda.allocate_arrays()
        lda.read_dw_alphabetical()
        #self.random_initialize()   # NO -- we want a seeded initialization!
        lda.seeded_initialize(seed_z, expand_factors )
        lda.gibbs_sample(iter=lda.iter, seed=lda.seed )
        lda.wpdt_to_probs()
        #self.deallocate_arrays()


    # record how long it took
    end_time = datetime.datetime.now()
    duration = (end_time-start_time).seconds







    # save word counts and topic assignment counts (these are sparse)
    if args.save_counts:    # TRUE by default
        state = ["dp", "wp", "alpha", "beta" ]
        for var_name in state:
            f_name = os.path.join(rundir, RUN_FILENAMESS[var_name] )
            np.save( f_name, lda.__getattribute__(var_name) )
        logger.info("Done writing out Nwt+beta, Ndt+alpha")

    # Gibbs sampler state, which consists of
    # the full  topic assignments "z.npy"
    if args.save_z:
        var_name="z"
        f_name = os.path.join(rundir, RUN_FILENAMESS[var_name] )
        np.save( f_name, lda.__getattribute__(var_name) )
        logger.info("Done writing out z.npy")

    # save probs
    if args.save_probs:
        probs = ["phi", "theta"]
        for var_name in probs:
            f_name = os.path.join(rundir, RUN_FILENAMESS[var_name] )
            np.save( f_name, lda.__getattribute__(var_name) )
        logger.info("Done writing out probabilities phi.npy and theta.npy")



    # prepare a dict which will become output.json
    output = {}
    # run details
    output["rundir"]=rundir
    output["host_id"]=host_id
    output["iter"]=args.iter
    output["seed"]=args.seed
    output["start_time"]=start_time.isoformat()  # ISO format string
                                    # to read ISO time stamps use dateutil
                                    #>>> from dateutil import parser
                                    #>>> parser.parse("2011-01-25T23:36:43.373248")
                                    # datetime.datetime(2011, 1, 25, 23, 36, 43, 373247)
    output["duration"]=int(duration)
    # corpus info
    output["corpus"]=args.docs_file
    output["vocab"]=args.vocab_file
    output["numDocs"] = lda.numDocs
    output["numTerms"] = lda.numTerms
    output["totalNterms"] = lda.corpus.totalNwords
    # model parameters
    output["numT"]=lda.numT
    # the hyperparameters are too long to store in full here,
    # use separate .npy files if alpha/beta non uniform
    output["alpha"]= lda.alpha[0] #[np.average(lda.alpha), float(np.cov(lda.alpha)) ]  # [avg, var]
    output["beta"]=  lda.beta[0]  #[np.average(lda.beta), float(np.cov(lda.beta)) ]  # [avg, var]
    #
    # calculate likelyhood
    output["loglike"]=lda.loglike()
    output["perplexity"]=lda.perplexity()   # = np.exp( -1 * loglike() / totalNwords )
    if args.save_perplexity_every:
        output["perplexity_history"]=perp_hist
    logger.info("Log likelyhood: %f" % output["loglike"] )
    logger.info("Perplexity: %f" % output["perplexity"] )
    #
    # special seeding info
    if args.seed_z_from:
        output["seed_z_from"]= args.seed_z_from
    if args.expand_factors:
        output["expand_factors"]= args.expand_factors



    # compute sparseness and write it out
    sp = get_sparse_stats( lda.phi )
    np.save(  os.path.join(rundir, "phi_sparseness.npy"), sp)
    # report on sparseness statisitcs (assume single mode)
    nz = sp.nonzero()[0]                        # get the nonzero entries
    sp_avg = sum([sp[i]*i for i in nz])         # where are non-zero they concentrated ?
    sp_var = sum( [sp[i]*np.abs(i-sp_avg)**2 for i in nz] )
    sp_stdev = np.sqrt( sp_var )                # how concentrated they are around sp_avg
    output["phi_sparseness_avg"]=sp_avg
    output["phi_sparseness_stdev"]=sp_stdev
    logger.info("Phi sparseness. center=%d, width=%d" % (int(sp_avg),int(sp_stdev))  )

    # same for theta
    sp = get_sparse_stats( lda.theta )
    np.save( os.path.join(rundir, "theta_sparseness.npy"), sp)
    # report on sparseness statisitcs (assume single mode)
    nz = sp.nonzero()[0]                        # get the nonzero entries
    sp_avg = sum([sp[i]*i for i in nz])         # where are non-zero they concentrated ?
    sp_var = sum( [sp[i]*np.abs(i-sp_avg)**2 for i in nz] )
    sp_stdev = np.sqrt( sp_var )                # how concentrated they are around sp_avg
    output["theta_sparseness_avg"]=sp_avg
    output["theta_sparseness_stdev"]=sp_stdev
    logger.info("Theta sparseness. center=%d, width=%d" % (int(sp_avg),int(sp_stdev))  )

    # write all output data to disk
    f=open( os.path.join(rundir, "output.json"), "w" )
    simplejson.dump( output, f, indent=0 )
    f.close()
    logger.info("Done saving output.json")



    if args.print_topics:
        from liblda.topicviz.show_top import show_top
        top_words_in_topics = show_top(lda.phi, num=args.print_topics, id2word=lda.corpus.id2word)

        for topic in top_words_in_topics:
            words = ", ".join(topic)
            print words


    logger.info("Done! --> thank you come again")




if __name__=="__main__":
    """
    Take all kinds of inputs on command line
    """



    parser = argparse.ArgumentParser(description='Latent Dirichlet Allocation runner.')


    # these are required
    parser.add_argument('--docs', dest="docs_file",     required=True,
                        help="The document corpus [[ (w_id, count) ]] in .mm format")
    parser.add_argument('--vocab', dest="vocab_file",   required=True,
                        help="The vocab file document corpus [[ (w_id, count) ]] in .mm format")
    parser.add_argument('--numT', type=int,             required=True,
                        help="Number of topics.")


    # for seeding one LDA run with the topic assignments
    # of another LDA run on the same corpus
    parser.add_argument('--seed_z_from', dest='seed_z_from',
                        help='specify a saved topic assignment vector (z.npy) to use as seed')
    parser.add_argument('--expand_factors', dest='expand_factors',
                        help='a file contaning a list that specifies into how many subtopics each seed topic should be split')

    # these are optional
    parser.add_argument('--seed', type=int,
                        help="Seed value for rand. num generator.")
    parser.add_argument('--iter', type=int,
                        help="Number of iterations of Gibbs sampling.")
    parser.add_argument('--alpha', type=float,
                        help="Specify uniform Dirichlet prior on theta (topics in docs)")
    parser.add_argument('--beta', type=float,
                        help="Specify uniform prior on phi (words in topics)")

    # NEW
    parser.add_argument('--save_perplexity_every', type=int,
                        help="Calculate the model perplexity and print it to disk at this interval.")

    parser.add_argument('--rundirs_root',
                        help="Parent folder where runs are to be stored")

    parser.add_argument('--save_z', action='store_true', default=False, dest="save_z",
                        help='save z.npy (the topic assignments for each word in corpus) (large file!) ')
    parser.add_argument('--save_probs', action='store_true', default=False, dest="save_probs",
                        help='save phi.npy and theta.npy. These can be produced from Nwt.npy+beta.npy ' + \
                             'and Ndt.py+alpha.npy respectively (probs are large files since not sparse) ')
    parser.add_argument('--dont_save_counts', action='store_false', default=True, dest="save_counts",
                        help='save Nwt.npy and Ndt.py  (True by default since they are relatively sparse) ')
    parser.add_argument('--print_topics', type=int, default=None,
                        help='Print top words in each topic that was learned.')
#    parser.add_argument('--corpus-format',
#                        help="Specify different corpus format, ex: lines-of-words, newman_docword, ... default: .mm matrix market")
#    parser.add_argument('--vocab-format',
#                        help='Specify different vocab format. default: json list of tuples [ "term":int(term_id) ]')

    args = parser.parse_args()
    #print args


    run(args)




