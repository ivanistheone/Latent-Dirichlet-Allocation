import numpy as np
from liblda.math.distances import JSdiv



def getCostMatrix2(th, th_t):
    """ computes the KL divergence (actually JS) between
        each pair columns of \Theta and \Theta_tilde

        return matrix is of size  numT x numT_t
        """

    numDocs, numT  = th.shape
    numDocsBis, numT_t = th_t.shape

    assert numDocs==numDocsBis, "Two topic distributions must be over the same documents"

    cost =np.zeros( [numT,numT_t] )
    for t1 in np.arange(0,numT):
        for t2 in np.arange(0,numT_t):
            cost[ t1, t2 ] = JSdiv( th[:,t1], th_t[:,t2] )

    return cost



def find_closest2(th, th_t):
    """
    Let the cols of theta be indexed by t \in 0 ... numT-1
    and the cols of theta_t by tt \in 0 ... numT_t-1

    For each topic tt, we find the topic t that is closest,
    i.e. most similar in terms of document appearence

    Returns a dict with keys 0 ... numT-1 and
    values = lists of topics tt.
    """

    M = getCostMatrix2(th, th_t)
    numT, numT_t  = M.shape

    # setup the output data structure
    mapping = {}
    for t in np.arange(0,numT):
        mapping[t]=[]

    for tt in np.arange(0,numT_t):
        col = M[:,tt]
        t = np.argmin(col)
        val = col[t]
        mapping[t].append( (tt,"%.3f" % val) )


    return mapping





def getCostMatrix(phi, phi_t):
    """ computes the KL divergence (actually JS) between each of the probs in
        phi and phi_t

        return matrix is of size  numT x numT_t
        """

    numT, numTerms  = phi.shape
    numT_t, numTermsBis = phi_t.shape

    assert numTerms==numTermsBis, "Two topic distributions must be over the same vocabulary"

    cost =np.zeros( [numT,numT_t] )
    for t1 in np.arange(0,numT):
        for t2 in np.arange(0,numT_t):
            cost[ t1, t2 ] = JSdiv( phi[t1,:], phi_t[t2,:] )

    return cost




def find_closest(phi, phi_t):
    """
    Let the rows of phi be indexed by t \in 0 ... numT-1
    and the rows of tt \in 0 ... numT_t-1

    For each topic tt, we find the topic t that is closest.

    Returns a dict with keys 0 ... numT-1 and
    values = lists of topics tt.
    """

    M = getCostMatrix(phi, phi_t)
    numT, numT_t  = M.shape

    # setup the output data structure
    mapping = {}
    for t in np.arange(0,numT):
        mapping[t]=[]

    for tt in np.arange(0,numT_t):
        col = M[:,tt]
        t = np.argmin(col)
        val = col[t]
        mapping[t].append( (tt,"%.3f" % val) )


    return mapping


def hungarianAlgorithm(topics_A, topics_B):
    """ returns a mapping
        the optimal matching between the two distributions of the same size
        ( (0,3), (1,4), ... , )
        which means topicA0 is best matched to topicB3, etc...

        see: http://en.wikipedia.org/wiki/Hungarian_algorithm
    """
    pass



