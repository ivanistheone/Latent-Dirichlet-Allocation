
"""
Basic way to measure similarity between probability distributions.
    - Kullback-Liebler
    - Jensen-Shannon

KL is not defined whenever support(p_Q)  \  support(p_P) is non-empty,
but the JSdiv is good all the time.

"""


import numpy as np






from scipy.stats.distributions import entropy as KLdiv
#scipy entropy can be called with two arguments gives relative entropy
# = Kullback Liebler divergence
#def KLdiv(prob_p, prob_q):
#    """ Computes the KL distance betwen two distributions represented as
#        two arrays of same length
#
#    """
#
#    #return np.sum( p*np.log(p/q) )
#    #vec = np.where(
#    # symmetrized version
#    #return 0.5*sum(p*log(p/q)+q*log(q/p))
#



def JSdiv(prob_p, prob_q):
    """
    Jensen-Shannon divergence = symmetrized KL divergence
    """
    p = prob_p
    q = prob_q
    m = (p+q)/2

    return 0.5*KLdiv(p,m) + 0.5*KLdiv(q,m)



# For reference
#In [37]: %psource spst.distributions.entropy
#def entropy(pk,qk=None):
#    """S = entropy(pk,qk=None)
#
#    calculate the entropy of a distribution given the p_k values
#    S = -sum(pk * log(pk), axis=0)
#
#    If qk is not None, then compute a relative entropy
#    S = sum(pk * log(pk / qk), axis=0)
#
#    Routine will normalize pk and qk if they don't sum to 1
#    """
#    pk = arr(pk)
#    pk = 1.0* pk / sum(pk,axis=0)
#    if qk is None:
#        vec = where(pk == 0, 0.0, pk*log(pk))
#    else:
#        qk = arr(qk)
#        if len(qk) != len(pk):
#            raise ValueError, "qk and pk must have same length."
#        qk = 1.0*qk / sum(qk,axis=0)
#        # If qk is zero anywhere, then unless pk is zero at those places
#        #   too, the relative entropy is infinite.
#        if any(take(pk,nonzero(qk==0.0),axis=0)!=0.0, 0):
#            return inf
#        vec = where (pk == 0, 0.0, -pk*log(pk / qk))
#    return -sum(vec,axis=0)
#
#
#
