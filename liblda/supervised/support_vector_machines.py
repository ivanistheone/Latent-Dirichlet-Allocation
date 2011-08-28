



from collections import defaultdict
import os,sys
import numpy as np





from scikits.learn.svm import  SVC


from scikits.learn.grid_search import GridSearchCV
from scikits.learn import metrics
from scikits.learn.pipeline import Pipeline

from liblda.supervised.load_data import load_data




import logging
logger = logging.getLogger('SVM')
logger.setLevel(logging.INFO)


rundir = '/Users/ivan/Homes/master/Documents/Projects/runs/reduced1/'
NIPSDIR = "/CurrentPorjects/LatentDirichletAllocation/data/NIPS1-17/"

( (trainX,trainY,train_ids), (testX, testY,test_ids) ) = load_data(rundir, dataset=NIPSDIR, normalize_cols=False)
allX = np.vstack((trainX,testX))
allY = np.concatenate((trainY,testY))

print "called:"
print "( (trainX,trainY,train_ids), (testX, testY,test_ids) ) = load_data(rundir, dataset=NIPSDIR, normalize_cols=False)"
 


def do_grid_search(X,Y, gs_params=None):
    """ Given data (X,Y) will perform a grid search on g_params
        for a LogisticRegression called logreg
        """
    svpipe = Pipeline([
        ('rbfsvm',  SVC()  )
        ])
    if not gs_params: 
        gs_params = {
                'rbfsvm__C': (1.5, 2, 5, 10, 20),
                'rbfsvm__gamma': (0.01, 0.1, 0.3, 0.6, 1, 1.5, 2, 5 ) ,
                }
    gs = GridSearchCV( svpipe, gs_params , n_jobs=-1)
    #print gs
    gs = gs.fit(X,Y)

    best_parameters, score = max(gs.grid_scores_, key=lambda x: x[1])
    logger.info("best_parameters: " +str( best_parameters ) )
    logger.info("expected score: "+str( score ) )

    return best_parameters


def train_svpipe(trainX, trainY,  params ):
    """ trains LogisiticRegression model with params
        logreg_C specified by params 
        """
    svpipe = Pipeline([
        ('rbfsvm',  SVC()  )
        ])
    svpipe = svpipe.fit(trainX,trainY, **params)
    return svpipe




