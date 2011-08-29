


import numpy as np
import os, sys

from liblda.supervised.common import random_split


def load_data( rundir, dataset , normalize_cols=False):
    """ Given a rundir with an LDA output theta.npy in it,
        and a dataset dir which contains a categories.txt 
        this function sets up the
            ( (X_train, y_train, train_ids),  (X_test, y_test, test_ids)  )

        """

    th = np.load( os.path.join( rundir, "theta.npy") )

    ndata, nfeatures = th.shape

    if normalize_cols:
        for t in range(0,nfeatures):
            th[:,t]=1.0/np.std( th[:,t] ) * th[:, t]



    #####   MAIN SETTINGS FOR DATA SET
    ######################################################################
    if not os.path.exists( dataset ):
        print "Error: dataset dir does not exist..."
        return -1

    DATA_PARENT_DIR=dataset
    print " LOADING Corpus Data from: " + DATA_PARENT_DIR
    VOCAB_FILE = DATA_PARENT_DIR+"NIPS_vocab.txt"
    DOCS_FILE =  DATA_PARENT_DIR+"NIPS_counts.mm"
    IDS_FILE  =  DATA_PARENT_DIR+"NIPS_doc_names.txt"
    # none of these are used here... but nice block of code to keep :)


    catsf = open( os.path.join(DATA_PARENT_DIR, 'NIPS_categories.txt'), 'r' )
    categories = np.array( [ int( cid.strip() ) for cid in catsf.readlines() ] )
    catsf.close()



    

     
    # X is theta   [ [ p(t1|d1) p(t2|d1) ...      p(tT|d1)  ] 
    #                [ p(t1|d2) p(t2|d2) ...      p(tT|d2)  ] 
    #                ...
    #                [ p(t1|dD)          ...      p(tT|dD)  ] ]

    # y are the labels [1 ... 8 ]
    # 

    # custom function that ensures that exactly 10 samples of each label
    # are being put in the test set
    ( (trainX,trainY,train_ids), (testX, testY,test_ids) ) =  random_split(th, categories, size=10)
    return ( (trainX,trainY,train_ids), (testX, testY,test_ids) )



def load_titles(dataset, truncate_to=70):
    """ Given a dataset dir which contains a NIPS_doc_titles.txt 
        this function will
        return the document titles as a list.

        """
    if not os.path.exists( dataset ):
        print "Error: dataset dir does not exist..."
        return -1

    DATA_PARENT_DIR=dataset
    print " LOADING Corpus Data from: " + DATA_PARENT_DIR
    #VOCAB_FILE = DATA_PARENT_DIR+"NIPS_vocab.txt"
    #DOCS_FILE =  DATA_PARENT_DIR+"NIPS_counts.mm"
    #IDS_FILE  =  DATA_PARENT_DIR+"NIPS_doc_names.txt"
    TITLES_FILE = os.path.join(DATA_PARENT_DIR, "NIPS_doc_titles.txt" )

    # none of these are used here... but nice block of code to keep :)

    tits_file = open( TITLES_FILE )
    titles = [ title.strip() for title in tits_file.readlines() ]
    tits_file.close()

    return np.array(titles)



def load_labels(dataset, truncate_to=70):
    """ Given a dataset dir which contains a NIPS_doc_titles.txt 
        this function will
        return the document titles as a list.

        """
    if not os.path.exists( dataset ):
        print "Error: dataset dir does not exist..."
        return -1

    dir=dataset
 
    labelsf = open( os.path.join(dir, 'NIPS_labels.txt') )
    labels  = [ int(w.strip()) for w in labelsf.readlines() ]
    labelsf.close()

    return np.array(labels)

