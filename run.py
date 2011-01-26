#!/usr/bin/env python2.6


# general
import os,sys
import logging
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
    if args.vocab_file[-5:]==".json":
        vjson = simplejson.load(open(args.vocab_file,'r'))
        _ocab = vjson
    elif args.vocab_file[-4:]==".txt":       # one word per line
        vfile = open(args.vocab_file, 'r')
        wlist = [w.strip() for w in vfile.readlines() ]
        id2word = dict( enumerate(wlist) )
        word2id = dict( [(word,id)  for id,word in id2word.items()] )
        vocab = word2id
    else:
        print "Vocab format not recognized"
        sys.exit(-1)


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

    # prepare a dict which will become input.json
    input = {}
    input["rundir"]=rundir
    input["numT"]=args.numT
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


    # setup the lda model
    lda = LdaModel( numT=args.numT,    alpha=args.alpha, beta=args.beta,  corpus=corpus, vocab=vocab )
    lda.train(iter=args.iter, seed=args.seed )









    # save word counts and topic assignment counts (these are sparse)
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
    output["rundir"]=rundir
    output["host_id"]=host_id
    output["seed"]=lda.seed
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
    output["alpha"]=[np.average(lda.alpha), float(np.cov(lda.alpha)) ]  # [avg, var]
    output["beta"]=[np.average(lda.beta), float(np.cov(lda.beta)) ]  # [avg, var]
    #
    # calculate likelyhood
    output["loglike"]=lda.loglike()
    output["perplexity"]=lda.perplexity()   # = np.exp( -1 * loglike() / totalNwords )
    logger.info("Log likelyhood: %f" % output["loglike"] )
    logger.info("Perplexity: %f" % output["perplexity"] )
    #
    # and write it to disk
    f=open( os.path.join(rundir, "output.json"), "w" )
    simplejson.dump( output, f, indent=0 )
    f.close()
    logger.info("Done saving output.json")


    if args.print_topics:
        from liblda.topicviz.show_top import show_top
        top_words_in_topics = show_top(lda.phi, num=args.print_topics, corpus=lda.corpus)
        print "id of word quantum ", lda.corpus.word2id["quantum"]
        print "word associated w id 0 (first word) ", lda.corpus.id2word[0]
        print "id of word quantum ", lda.corpus.word2id["hilbertspace"]
        print "word associated with last id 10010 ", lda.corpus.id2word[lda.numTerms-1]

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



    # these are optional
    parser.add_argument('--seed', type=int,
                        help="Seed value for rand. num generator.")
    parser.add_argument('--iter', type=int,
                        help="Number of iterations of Gibbs sampling.")
    parser.add_argument('--alpha', type=float,
                        help="Specify uniform Dirichlet prior on theta (topics in docs)")
    parser.add_argument('--beta', type=float,
                        help="Specify uniform prior on phi (words in topics)")

    parser.add_argument('--rundirs_root',
                        help="Parent folder where runs are to be stored")

    parser.add_argument('--save_z', action='store_true', default=False, dest="save_z",
                        help='save z.npy (the topic assignments for each word in corpus) (large file!) ')
    parser.add_argument('--save_probs', action='store_true', default=False, dest="save_probs",
                        help='save phi.npy and theta.npy. These can be produced from Nwt.npy+beta.npy ' + \
                             'and Ndt.py+alpha.npy respectively (probs are large files since not sparse) ')
    parser.add_argument('--print_topics', type=int, default=None,
                        help='Print top words in each topic that was learned.')
#    parser.add_argument('--corpus-format',
#                        help="Specify different corpus format, ex: lines-of-words, newman_docword, ... default: .mm matrix market")
#    parser.add_argument('--vocab-format',
#                        help='Specify different vocab format. default: json list of tuples [ "term":int(term_id) ]')

    args = parser.parse_args()
    #print args


    run(args)




