
from collections import defaultdict
import os,sys
import numpy as np



from liblda.supervised.common import random_split

from scikits.learn.linear_model import  LogisticRegression



nips_dir = "/Projects/LatentDirichletAllocation/data/NIPS1-17/"
catsf = open( os.path.join(nips_dir, 'NIPS_categories.txt'), 'r' )
pre_categories = np.array( [ int( cid.strip() ) for cid in catsf.readlines() ] )
cat_labels = np.bincount(pre_categories)
sparse_labels = []
for id in range(0,len(cat_labels)):
    if cat_labels[id] > 0:
        sparse_labels.append(id)
catsf.close()



def calculate_success( theta, categories):
    ((TrainX,TrainY,TrainIDs), (TestX,TestY,TestIDs) ) = random_split( theta, categories )
    tc = ThresholdClassfier( TrainX, TrainY)

    numTestDocs = TestX.shape[0]

    scores = np.zeros(10)
    for t_id in range(0,numTestDocs):
        (g,top) =  tc.classify(TestX[t_id])
        truth = TestY[t_id]
        if g == truth:
            scores[truth] += 1
        
    print sum(scores), "/ 80"
    print scores


        


class ThresholdClassfier(object):
    """
    Consider the hyperplaneas
        \sum_t  m_dt \theta_d(t) = 1 
    defined for each document, i.e. each row of the theta matrix.

    Define the average theta vector of all the theta vectors in 
    the category c:
        theta_c = \sum_{d in c} \theta_d           \in R^T

    then the average hyperplane is
        m_c \cdot theta_c = 1

    We will define a set of yes-no classifiers for each
    category defined as a test for a query document \theta_q

        score_c =   m_c \cdot \theta_q 


    extensaions:
     -  compute m'_c which cuts off only the most significant 
        `order` components of m_c
     -  differentially update m'_c to only the mose specific parts
        tf/idf ?
        direct subtraciton?

     
    """
    
    def __init__(self,theta, categories,use_categories=[1,2,3,4,5,6,7,8]):

        numDocs, numT = theta.shape

        self.theta = theta
        self.categories = categories
        self.use_categories = use_categories


        cat_ids_for = {}
        for cat_id in use_categories:
            en_cats = enumerate( categories )
            ids = [idx for (idx,cat) in en_cats if cat==cat_id]
            cat_ids_for[cat_id] = ids
        self.cat_ids_for = cat_ids_for 

        cat_ms = np.zeros([max(use_categories)+1, numT] )
        # compute theta averages
        for cat_id in use_categories:
            en_cats = enumerate( categories )
            set1 = [idx for (idx,cat) in en_cats if cat==cat_id]
            X1 = theta[set1,:]
            m_c = 1.0/X1.shape[0]**np.sum(X1, 0)
            cat_ms[cat_id] = m_c
        self.cat_ms = cat_ms

        

    def classify(self, theta ):
        """ computes all the dot products
                m_c \cdot theta 
            and reports the largest index
            as well as the list of closest.
        """

        cat_ms = self.cat_ms
        use_categories = self.use_categories 

        scores = np.zeros(max(use_categories)+1)
        for cat_id in use_categories:
            score = np.dot( cat_ms[cat_id], theta )
            scores[cat_id] = score


        guess = np.argmax(scores)
        en_scores = enumerate(scores)
        highest_scores = sorted(en_scores, key=lambda t: t[1], reverse=True )

        return (guess, highest_scores)








