
#

import numpy
import itertools


def is_k_sparse(vec, k):
    """
    checks if 95% of the weight of a normalized vector vec
    is in less than k of its entries.
    """
    WEIGHT=0.95     # 95%

    n=len(vec)

    if k<0 or k>n:
        raise ValueError(" k has to be in 1 .. len(vec) ")

    # a filter with k entries set to 1
    exfilt = numpy.concatenate( (numpy.ones(k), numpy.zeros(n-k) ) )

    out =  False

    # go though all possible permutations of filter
    for filt in itertools.permutations(exfilt):
        if numpy.dot(vec,filt) >= WEIGHT:
            out=True
            break

    return out


def get_sparse_stats(dsamples):
    """
    Takes the output of numpy.random.dirichlet
    and spits out the proportions of the samples
    that are k-sparse for k=1 ... n

    returns an array of counts of sparseness
     [ # 1-sparse, # 2-sparse, # 3-sprase, ...,  # n-sparse ]

    note that we subtract from k-sparse all the k-1 sparse
    since we don't want to double count.
    """
    N,n = dsamples.shape        # N samples from, n-dim Dirichlet

    out = numpy.zeros(n)
    for k in range(1,n+1):

        count=0
        for sample in dsamples:
            if is_k_sparse(sample, k):
                count +=1
        if k==1:
            out[0]=float(count)
        else:
            out[k-1]=float(count)-numpy.sum( out[0:k-1] )


    outf = numpy.divide(out,N)
    return outf


# usage

# ONE ==================
# see sparseness as a function of \alpha for n=6
#
# get_sparse_stats(numpy.random.dirichlet(0.01*numpy.ones(6), 5000) )
# array([ 0.8612,  0.1322,  0.0066,  0.    ,  0.    ,  0.    ])
# i.e., 86% of entries have one dominant entry when alpha = 0.01 symmetric

# get_sparse_stats(numpy.random.dirichlet(0.1*numpy.ones(6), 5000) )
# array([ 0.233 ,  0.444 ,  0.2582,  0.0596,  0.0052,  0.    ])
# when \alpha = 0.1 we have 23% 1-sparse and 67% 2-sparse and nearly all 3-sparse

# when \alpha = 0.5
# get_sparse_stats(numpy.random.dirichlet(0.5*numpy.ones(6), 5000) )
# array([ 0.001 ,  0.0306,  0.205 ,  0.4424,  0.292 ,  0.029 ])
# NOT very sparse at all... mostly 3 4 5 entries



# TWO ===================
# see sparseness as a function of dimension for alpha = 0.1
#
# n=7  ==>  [ 0.1804,  0.4046,  0.3074,  0.0948,  0.0122,  0.0006,  0.    ]
# n=6  ==>  [ 0.233 ,  0.444 ,  0.2582,  0.0596,  0.0052,  0.    ])
# n=5  ==>  [ 0.3358,  0.45  ,  0.1862,  0.027 ,  0.001 ]
# n=4  ==>  [ 0.4234,  0.4558,  0.111 ,  0.0098]
# n=3  ==>  [ 0.5872,  0.365 ,  0.0478]
# reading the above it seems like \alpha=0.1 selects 3-sparse vectors
# must play with it more ...




# THREE  ===========
# for \alpha = 50/n  explore what happens for different number of n
# not currently possible with the itertools.permutations n=7 takes long already !
# assuming T > 50 should be in the sparse regime ... but how sparse?

















