
import numpy as np
import sys, os


from liblda.supervised.load_data import load_data, load_titles, load_labels
from liblda.supervised import logistic_regression
from liblda.supervised import support_vector_machines


from liblda.supervised.common import evaluate




NIPSDIR = "/Projects/LatentDirichletAllocation/data/NIPS1-17/"
catsf = open( os.path.join(NIPSDIR, 'NIPS_categories.txt'), 'r' )
categories = np.array( [ int( cid.strip() ) for cid in catsf.readlines() ] )
catsf.close()


import logging
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
logger = logging.getLogger('LABELLED')
logger.setLevel(logging.INFO)




class SVMdemo(object):

    def __init__(self, rundir):
        self.rundir = rundir
        self.SVMparams = None

    def grid_search(self, allX, allY):
        SVMparams = support_vector_machines.do_grid_search(allX, allY)
        self.SVMparams = SVMparams
        print "performed grid search to find: "
        print self.SVMparams

    def run_SVM(self, rundir=None):
        """ Demo script that evaluates the supervised prediction
            performance of the p(t|d) (stored in theta.npy) of
            an the LDA model in `rundir`.
        """

        if not rundir:
            rundir = self.rundir

        ( (trainX,trainY,train_ids), (testX, testY,test_ids) ) = load_data(rundir, dataset=NIPSDIR)
        allX = np.vstack((trainX,testX))
        allY = np.concatenate((trainY,testY))

        if not self.SVMparams:
            self.grid_search(allX,allY)

        sv = support_vector_machines.train_svpipe(trainX, trainY, self.SVMparams)

        allLabels = load_labels(NIPSDIR)
        allTitles = load_titles(NIPSDIR)
        evaluate(sv, testX, testY, testTitles=allTitles[test_ids], testLabels=allLabels[test_ids])








class LRdemo(object):

    def __init__(self, rundir, penalty="l2"):
        self.penalty = penalty
        self.rundir = rundir
        self.LRparams = None

    def grid_search(self, allX, allY):
        
        if self.penalty == "l1":
            gs_params = { 
                    'logreg__C': (1, 8, 10, 20,  50), 
                    'logreg__penalty': ('l1',) , 
                    } 
        elif self.penalty == "l2":
            gs_params = { 
                    'logreg__C': (0.1, 1, 7, 10, 20, 30, 40, 100), 
                    'logreg__penalty': ('l2',) , 
                    } 
        else:
            print "ERROR"
    
        LRparams = logistic_regression.do_grid_search(allX, allY, gs_params )
        self.LRparams = LRparams
        print "performed grid search to find: "
        print self.LRparams

    def run_LR(self, rundir=None):
        """ Demo script that evaluates the supervised prediction
            performance of the p(t|d) (stored in theta.npy) of
            an the LDA model in `rundir`.
        """

        if not rundir:
            rundir = self.rundir

        ( (trainX,trainY,train_ids), (testX, testY,test_ids) ) = load_data(rundir, dataset=NIPSDIR)
        allX = np.vstack((trainX,testX))
        allY = np.concatenate((trainY,testY))

        if not self.LRparams:
            self.grid_search(allX,allY)

        lr = logistic_regression.train_lrpipe(trainX, trainY, self.LRparams)

        allLabels = load_labels(NIPSDIR)
        allTitles = load_titles(NIPSDIR)
        evaluate(lr, testX, testY, testTitles=allTitles[test_ids], testLabels=allLabels[test_ids])


