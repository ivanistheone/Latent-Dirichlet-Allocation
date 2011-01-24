

import numpy as np
import itertools

__doc__ = """
Dirichlet explorer .. v 0.1

"""


DIR_DIM_START  = 1     # interested in many topics
DIR_DIM_STOP   = 202    # but not too many ;)


def scaled_constf(n):
     return 1.0/n* np.ones(n)

def scaled_constf(n):
     return 10.0/n* np.ones(n)

def exp_function_maker(rate, scale):
    """
    scale*exp( - rate * t ) / sum( exp(-rate*t) )
    return as function suitable for the alpha input of get_sparse_for_alpha
    """

    def expf(n):
        """
        returns a numpy array of length n with weights
        decaying exponentially fast at `rate`,
        normalized so that the total sum is `scale`.
        """
        x = np.arange(0,n)
        return scale*np.exp(-rate*x)/sum( np.exp(-rate*x) )

    return expf


def n_scaled_exp_function_maker(rate, multiplier):
    """
    multiplier*n* exp( - rate * t ) / sum( exp(-rate*t) )
    return as function suitable for the alpha input of get_sparse_for_alpha
    """

    def expf(n):
        """
        returns a numpy array of length n with weights
        decaying exponentially fast at `rate`,
        normalized so that the total sum is `multiplier` times `n`
        this is to mimick the constant \alpha scenario,
        where the sum of all alphas grows with the number of dimensions.
        """
        x = np.arange(0,n)
        return multiplier*n*np.exp(-rate*x)/sum( np.exp(-rate*x) )

    return expf




def get_sparse_stats(dsamples):
    """
    Takes the output of np.random.dirichlet
    and spits out the proportions of the samples
    that are k-sparse for k=1 ... n

    define: k-sparse
    checks if 95% of the weight of a normalized vector vec
    is in less than k of its entries.

    returns an array of counts of sparseness
     [ # 1-sparse, # 2-sparse, # 3-sprase, ...,  # n-sparse ]

    note that we subtract from k-sparse all the k-1 sparse
    since we don't want to double count.
    """

    WEIGHT=0.95     # 95%

    N,n = dsamples.shape        # N samples from, n-dim Dirichlet

    largelast = np.sort(dsamples,1)        #

    counts = np.zeros(n)                # number of i-sparse entties
    for sample in largelast:

        k=1                 #
        val=sample[n-k]     # start at last entry
        # if it is already larger than 0.95 wont go into loop
        while val < WEIGHT:     # loop until get enough bins to break out
            k+=1
            val = val + sample[n-k]
        #
        counts[k-1] +=1     # 0-based index

    outf = np.divide(counts,N)
    return outf


# usage

# ONE ==================
# see sparseness as a function of \alpha for n=6
#
# get_sparse_stats(np.random.dirichlet(0.01*np.ones(6), 5000) )
# array([ 0.8612,  0.1322,  0.0066,  0.    ,  0.    ,  0.    ])
# i.e., 86% of entries have one dominant entry when alpha = 0.01 symmetric

# get_sparse_stats(np.random.dirichlet(0.1*np.ones(6), 5000) )
# array([ 0.233 ,  0.444 ,  0.2582,  0.0596,  0.0052,  0.    ])
# when \alpha = 0.1 we have 23% 1-sparse and 67% 2-sparse and nearly all 3-sparse

# when \alpha = 0.5
# get_sparse_stats(np.random.dirichlet(0.5*np.ones(6), 5000) )
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




# let's refactor -- alpha is now number OR function-like
#
#
#
#

def get_sparse_for_alpha(alpha=None, alphaval=None, nrange=None, sample_size=None):
    """
     let's see a plot of sparseness
     for different n
     as a funciton of \alpha
    """

    if not sample_size:
        N = 5000
    else:
        N = sample_size



    # set default values for \alpha if nothing is specified
    if not alphaval and not alpha:
        alphaval = 0.1             # default alpha
                                # 0.05 * N / (numDocs * numT)   Newman choice
                                #                               N=total# words in corpus
                                # 50.0 / numT                   Griffiths recommended

    #turn into constant funciton --i.e. "callable" python object
    if not alpha: #hasattr(alpha, "__call__"):
        alphaval = float(alphaval)
        def const_fun(n):
            return alphaval * np.ones(n)
        # set alpha to the
        alpha = const_fun




    # the randge of dimensions we will probe
    # defaults to 1 - 202
    # which begs a question, whther a prob dist with a single outcome is sparse
    # on the one hand it has all its weight "on very few terms"
    # on the other hand it has all its weight on all terms :)
    # -- we just use 1 becaue the plot axis will be easier to set
    if not nrange:
        nrange = np.arange(DIR_DIM_START, DIR_DIM_STOP,1, dtype=float)
    totaln = len(nrange)


    interesting = 70    # show only sparseness up to k=30

    results  =  []

    #np.zeros( (totaln, interesting), dtype=float)         # each row represents the sparse
                                                    # vector for a different n

    for n in nrange:

        # alpha is some python callable
        # alpha(n) ---> numpy array with suitable alpha weights for n dimensional Dirichlet
        dirsamples = np.random.dirichlet( alpha(n), N)
        spvec = get_sparse_stats(dirsamples)

        # in order to plot all n's together will pad with zeros
        spvec_head = np.zeros(interesting)
        for i in range(interesting):
            if i <n:
                spvec_head[i] = spvec[i]

        results.append(spvec_head)  #[counter,:]= spvec_head

    return np.array(results)




def plot_results( res , case=""):
    """
    given a matrix of sparseness features
    one in each row, plots a color plot

    TODO: 3D figure with it
    #fig=p.figure()
    #ax = p3.Axes3D(fig)
    #ax.plot_wireframe(X,Y, res)
    #ax.scatter3D(np.ravel(X), np.ravel(Y), np.ravel(res) )

    #p.set_xlabel('Dirichlet dimension')
    #x.set_ylabel('sparseness')
    #ax.set_zlabel('prop of vectors of dim n with this sparsity')
    #ax.

    # fig.add_axes(ax)

    """

    import pylab as p
    import mpl_toolkits.mplot3d.axes3d as p3

    # new fig
    p.figure()

    # setup the domain
    x = np.arange(0, res.shape[0], 1)
    y = np.arange(0, res.shape[1],1)
    X, Y = p.meshgrid(x, y)


    p.pcolor(X,Y, res.T)

    p.xlabel('Dirichlet dimension')
    p.ylabel('# entries that contain 95% of prob')
    p.title('Sparseness of Dirichlet distribution $\\vec{\\alpha}(i)$ = %s' % case )

    p.show()





#
# mrecommend = get_sparse_for_alpha(0.1, recomendalpha=True)
# plot_reults(mrecommend)
# prior around small window... is this desired?



# how important is the prior?
# shouldn't it expect certain # of topics AND less
# why does it have to be such a narrow window?

# \alpha uniform seems to be mostly one corridor ...
# can you derive the slope of the line analytically?????
#






















