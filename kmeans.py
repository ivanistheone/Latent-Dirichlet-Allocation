#!/usr/bin/env python
# kmeans.py using scipy.spatial.distance.cdist, any metric
# kmeanssample 2 pass, first sample sqrt(N)
# http://stackoverflow.com/questions/5529625/is-it-possible-to-specify-your-own-distance-function-using-scikits-learn-k-means

from __future__ import division
import random
import numpy as np
from scipy.spatial.distance import cdist  # $scipy/spatial/distance.py
    # http://docs.scipy.org/doc/scipy/reference/spatial.html
    # cf http://docs.scipy.org/doc/scipy/reference/cluster.vq.html

__date__ = "2011-06-30 jun denis"

#...............................................................................
def kmeans( X, centres, delta=.001, maxiter=10, metric="euclidean", p=2, verbose=1 ):
    """ centres, Xtocentre, distances = kmeans( X, initial centres ... )
    in:
        X N x dim
        centres k x dim: initial centres, e.g. random.sample( X, k )
        delta: relative error, iterate until the average distance to centres
            is within delta of the previous average distance
        maxiter
        metric: any of the 20-odd in scipy.spatial.distance
            "chebyshev" = max, "cityblock" = L1, "minkowski" with p=
            or a function( Xvec, centrevec ), e.g. Lqmetric below
        p: for minkowski metric -- local mod cdist for 0 < p < 1 too
        verbose: 0 silent, 2 prints running distances
    out:
        centres, k x dim
        Xtocentre: each X -> its nearest centre, ints N -> k
        distances, N
    see also: kmeanssample below, class Kmeans below.
    """
        # seed like learn ?
    X = np.asarray(X)
    centres = np.copy(centres)
    N, dim = X.shape
    k, cdim = centres.shape
    if dim != cdim:
        raise ValueError( "kmeans: X %s and centres %s must have the same number of columns" % (
            X.shape, centres.shape ))
    if verbose:
        print "kmeans: X %s  centres %s  delta=%.2g  maxiter=%d  metric=%s" % (
            X.shape, centres.shape, delta, maxiter, metric)
    allx = np.arange(N)
    prevdist = 0
    for jiter in range(maxiter+1):
        D = cdist( X, centres, metric=metric, p=p )  # |X| x |centres|
        xtoc = D.argmin(axis=1)  # X -> nearest centre
        distances = D[allx,xtoc]
            # todo: cython cdist w argmin, min only
        avdist = distances.mean()  # median ?
        if verbose >= 2:
            print "kmeans: av |X - nearest centre| = %.4g" % avdist
        if (1 - delta) * prevdist <= avdist <= prevdist \
        or jiter == maxiter:
            break
        prevdist = avdist
        for jc in range(k):  # (1 pass in C)
            c = (xtoc == jc)
            if c.any():
                centres[jc] = X[c].mean( axis=0 )
    if verbose:
        print "kmeans: cluster sizes", np.bincount(xtoc)
    if verbose >= 2:  # 21jun --
        r50 = np.ones(k) * -1
        r90 = np.ones(k) * -1
        for j in range(k):
            dist = distances[ xtoc == j ]
            closest = np.sort(dist)
            #print type(dist)
            #print dist 
            #print closest
            #print type(closest)
            #sys.exit(-1)
            ii =0
            while ii < 0.5*len(dist):
                r50[j] = closest[ii]
                ii += 1
            while ii < 0.9*len(dist):
                r90[j] = closest[ii] 
                ii += 1
            # r50[j], r90[j] = np.percentile( dist, (50, 90) )
        print "kmeans: cluster 50 % radius", r50  # np.set_printoptions
        print "kmeans: cluster 90 % radius", r90
            # scale L1 / dim, L2 / sqrt(dim) ?
    return centres, xtoc, distances

#...............................................................................
def kmeanssample( X, k, nsample=0, **kwargs ):
    """ 2-pass kmeans, fast for large N:
        1) kmeans a random sample of nsample ~ sqrt(N) from X
        2) full kmeans, starting from those centres
    """
        # merge w kmeans ? mttiw
        # v large N: sample N^1/2, N^1/2 of that
    N, dim = X.shape
    if nsample == 0:
        nsample = max( 2*np.sqrt(N), 10*k )
    Xsample = random.sample( X, int(nsample) )
    pass1centres = random.sample( X, int(k) )
    samplecentres = kmeans( Xsample, pass1centres, **kwargs )[0]
    return kmeans( X, samplecentres, **kwargs )


def distancestocentres( X, centres, metric="euclidean", p=2 ):
    """ all distances X -> nearest centre, any metric
            euclidean2 (~ withinss) is more sensitive to outliers,
            cityblock (manhattan, L1) less sensitive
    """
    D = cdist( X, centres, metric=metric, p=p )  # |X| x |centres|
    return D.min(axis=1)

def Lqmetric( x, y=None, q=.5 ):
    # yes a metric, may increase weight of near matches; see ...
    return (np.abs(x - y) ** q) .mean() if y is not None \
        else (np.abs(x) ** q) .mean()

#...............................................................................
class Kmeans:
    """ km = Kmeans( X, k= or centres=, ... )
        in: either initial centres= for kmeans
            or k= [nsample=] for kmeanssample
        out: km.centres, km.Xtocentre, km.distances
        iterator:
            for jcentre, J in km:
                clustercentre = centres[jcentre]
                J indexes e.g. X[J], classes[J]
    """
    def __init__( self, X, k=0, centres=None, nsample=0, **kwargs ):
        self.X = X
        if centres is None:
            self.centres, self.Xtocentre, self.distances = kmeanssample(
                X, k=k, nsample=nsample, **kwargs )
        else:
            self.centres, self.Xtocentre, self.distances = kmeans(
                X, centres, **kwargs )

    def __iter__(self):
        for jc in range(len(self.centres)):
            yield jc, (self.Xtocentre == jc)

#...............................................................................
if __name__ == "__main__":
    import random
    import sys
    from time import time

    N = 10000
    dim = 10
    ncluster = 10
    kmsample = 100  # 0: random centres, > 0: kmeanssample
    kmdelta = .001
    kmiter = 10
    metric = "cityblock"  # "chebyshev" = max, "cityblock" L1,  Lqmetric
    seed = 1

    exec( "\n".join( sys.argv[1:] ))  # run this.py N= ...
    np.set_printoptions( 1, threshold=200, edgeitems=5, suppress=True )
    np.random.seed(seed)
    random.seed(seed)

    print "N %d  dim %d  ncluster %d  kmsample %d  metric %s" % (
        N, dim, ncluster, kmsample, metric)
    X = np.random.exponential( size=(N,dim) )
        # cf scikits-learn datasets/
    t0 = time()
    if kmsample > 0:
        centres, xtoc, dist = kmeanssample( X, ncluster, nsample=kmsample,
            delta=kmdelta, maxiter=kmiter, metric=metric, verbose=2 )
    else:
        randomcentres = random.sample( X, ncluster )
        centres, xtoc, dist = kmeans( X, randomcentres,
            delta=kmdelta, maxiter=kmiter, metric=metric, verbose=2 )
    print "%.0f msec" % ((time() - t0) * 1000)

    # also ~/py/np/kmeans/test-kmeans.py




