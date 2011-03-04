import numpy as np
from liblda.math.distances import JSdiv


from liblda.extlibs import munkres



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


def hungarian_algorithm(cost_matrix, mapping_only=False):
    """
        inputs:     A matrix of distances ( either d_\Theta or d_\Phi )
                    cost_matrix[i][j] is some measure of how different
                    topic i is from topic j.

        outputs:    A triple of the form
                    (rows_unused, mapping, cols_unused)
                    where mapping is of the form
                      [(0, m(0)), (1, m(1)), ...   ]
                    where `m` is the minimum cost matching of rows to columns.

                    If the cost_matrix is square then the _unused  will be empty,
                    else they will contain the elements that were not used by the mapping.

        I decided to use the code from:
            https://github.com/bmc/munkres

        For algo description see:
            http://en.wikipedia.org/wiki/Hungarian_algorithm

    """

    # if cost_matrix specified as some non-numpy array of array format
    # like a list of lists for example we should convert:
    if not type(cost_matrix) == type(np.array([1])):
        cost_matrix =  np.array( cost_matrix )

    nrows, ncols = cost_matrix.shape
    allrows = set(range(0,nrows))
    allcols = set(range(0,ncols))

    halgo = munkres.Munkres()

    indices = halgo.compute( cost_matrix.tolist()  )    # algo assumes list of lists

    usedrows, usedcols = zip(*indices)      # i am sorry but I have to show off
                                            # my newly learned python kung fu

    rows_unused = sorted(list(  allrows - set(usedrows)  ))
    cols_unused = sorted(list(  allcols - set(usedcols)  ))

    if mapping_only:
        return indices
    else:
        return (rows_unused, indices, cols_unused)








