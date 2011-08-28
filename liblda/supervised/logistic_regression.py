



from collections import defaultdict
import os,sys
import numpy as np





from scikits.learn.linear_model import  LogisticRegression



from scikits.learn.grid_search import GridSearchCV
from scikits.learn import metrics
from scikits.learn.pipeline import Pipeline

from liblda.supervised.load_data import load_data




import logging
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
logger = logging.getLogger('LogReg')
logger.setLevel(logging.INFO)



rundir = '/Users/ivan/Homes/master/Documents/Projects/runs/reduced1/'
NIPSDIR = "/CurrentPorjects/LatentDirichletAllocation/data/NIPS1-17/"

( (trainX,trainY,train_ids), (testX, testY,test_ids) ) = load_data(rundir, dataset=NIPSDIR, normalize_cols=False)
allX = np.vstack((trainX,testX))
allY = np.concatenate((trainY,testY))

print "called:"
print "( (trainX,trainY,train_ids), (testX, testY,test_ids) ) = load_data(rundir, dataset=NIPSDIR, normalize_cols=False)"
 
LRparametersL1 = {
        'logreg__C': (0.1, 1, 2, 5, 10, 50),
        'logreg__penalty': ('l1',) ,
        }

LRparametersL2 = {
        'logreg__C': (0.001, 0.01, 0.1, 1),
        'logreg__penalty': ('l2',) ,
        }


def do_grid_search(X,Y, gs_params):
    """ Given data (X,Y) will perform a grid search on g_params
        for a LogisticRegression called logreg
        """
    lrpipe = Pipeline([
        ('logreg',  LogisticRegression()  )
        ])
    gs = GridSearchCV( lrpipe, gs_params , n_jobs=-1)
    #print gs
    gs = gs.fit(X,Y)

    best_parameters, score = max(gs.grid_scores_, key=lambda x: x[1])
    logger.info("best_parameters: " +str( best_parameters ) )
    logger.info("expected score: "+str( score ) )

    return best_parameters


def train_lrpipe(trainX, trainY,  params ):
    """ trains LogisiticRegression model with params
        logreg_C specified by params 
        """
    lrpipe = Pipeline([
        ('logreg',  LogisticRegression(penalty="l1", C=1)  )
        ])
    lrpipe = lrpipe.fit(trainX,trainY, **params)
    return lrpipe







def reg_param_search( th1000, categories, dir=None):

    if not dir:
        dir = "/Projects/LatentDirichletAllocation/data/NIPS1-17/"
    lnamesf = open( os.path.join(dir, 'NIPS_label_names.txt') )
    label_names = dict(enumerate( [w.strip() for w in lnamesf.readlines() ] ))
    cnamesf = open( os.path.join(dir, 'NIPS_category_names.txt') )
    category_names = dict(enumerate( [w.strip().split(" ",1)[1] for w in cnamesf.readlines() ] ))
    category_names[99]=category_names[9]
    del(category_names[9])

    labelsf = open( os.path.join(dir, 'NIPS_labels.txt') ) 
    labels  = [ int(w.strip()) for w in labelsf.readlines() ]


    ((Trx,Try,train_ids), (tex,tey,test_ids) ) = random_split( th1000, categories )

    Cs =  [1, 2, 3, 5, 10 ] #, 23, 10, 20, 30, 40]
    #Cs = [30]
    #np.arange(0.1,120,5)
    Cscore = np.zeros( len(Cs) )

    for Cid in range(0,len(Cs) ):
        
        cp = CategoryPredictor(Trx, Try, reg_param=Cs[Cid])

        score=0
        item_id=0
        for d in range(0,len(tey)):
            doc_id = test_ids[item_id]
            #print cp.classify(tex[d,:][np.newaxis]), tey[d]
            if cp.classify(tex[d,:][np.newaxis])==tey[d]: 
                score+=1
            else:
                print "CLASSIF ERR: ", \
                        "predicted ", cp.classify(tex[d,:][np.newaxis]),  \
                        ",      true cat=", tey[d], ", label=", label_names[labels[doc_id]]
            item_id += 1
            

        Cscore[Cid] = score

    return (Cs, Cscore)


class CategoryPredictor(object):

    # 22 was decided by reg_param_search
    def __init__(self, train_thetas, train_cats, reg_type="l1", reg_param=22.0): 
        """ each row is a feature vector
            the corresponding label 1...8 is in the cats array.

            we do all possible pairwise combinations
        """
        self.reg_type = reg_type
        self.reg_param = reg_param

        CATEGORIES = range(1,9) # 1... 8
        self.CATEGORIES = CATEGORIES

        betasM = np.ndarray([9,9], dtype=object )
        for ii in range(1,9):           # col of matrix
            for jj in range(ii+1, 9):   # row
                lr = self.pairwise_regress( train_thetas, train_cats, ii, jj, self.reg_type, self.reg_param)
                betasM[ii,jj] = lr

        self.betasM = betasM


    def classify(self, theta ):
        """ does the majority election thing """

        votes = np.zeros(9, dtype=np.int32)
        for ii in range(1,9):           # col of matrix
            for jj in range(ii+1, 9):   # row
                pjj = self.pairwise_predict( self.betasM[ii,jj], theta)
                if pjj>=0.5: # outcome 1
                    votes[jj]+=1
                else:
                    votes[ii]+=1

        return np.argmax( votes )


    @staticmethod
    def pairwise_regress( thetas,cats, ii, jj , reg_type, reg_param):


        en_cats = enumerate( cats )
        set0 = [idx for (idx,cat) in en_cats if cat==ii]
        en_cats = enumerate( cats )
        set1 = [idx for (idx,cat) in en_cats if cat==jj]

        X0 = thetas[set0,:]
        Y0 = np.zeros(X0.shape[0])
        X1 = thetas[set1,:]
        Y1 = np.ones(X1.shape[0])

        X = np.vstack( [X0,X1] )
        Y = np.concatenate( [Y0, Y1] )

        lr = LogisticRegression(penalty=reg_type, tol=0.00000000001, C=reg_param)
        lr.fit( X, Y)

        return lr

        #bet, J, logl   = logistic_regression(X.T,Y)



    @staticmethod
    def pairwise_predict( lr, theta):
        """ given a row of theta
            and a bet ii=0 jj=1 uses the linear_regress to predict
            the prob of 1
        """
        aa = lr.predict_proba(theta)

        return aa[0][1]










