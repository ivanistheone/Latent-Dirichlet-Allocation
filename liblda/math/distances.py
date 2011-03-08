
"""
Basic way to measure similarity between probability distributions.
    - Kullback-Liebler
    - Jensen-Shannon

KL is not defined whenever support(p_Q)  \  support(p_P) is non-empty,
but the JSdiv is good all the time.

"""


import numpy as np




#from scipy.stats.distributions import entropy as KLdiv
# %psource spst.distributions.entropy
def KLdiv(pk,qk):
    """KL = KLdiv(pk,qk)

    Then compute a relative entropy
    KLdiv  = sum(pk * log(pk / qk), axis=0)

    Routine will NOT normalize pk and qk if they don't sum to 1
    """
    pk = np.array(pk, dtype=np.float)
    qk = np.array(qk, dtype=np.float)
    #pk = 1.0* pk / sum(pk,axis=0)
    #qk = 1.0*qk / sum(qk,axis=0)
    if len(qk) != len(pk):
        raise ValueError, "qk and pk must have same length."
    # If qk is zero anywhere, then unless pk is zero at those places
    #   too, the relative entropy is infinite.
    if np.any(np.take(pk,np.nonzero(qk==0.0),axis=0)!=0.0, 0):
        return inf
    vec = np.where (pk == 0, 0.0, pk*np.log(pk / qk))
    return np.sum(vec,axis=0)




def JSdiv(prob_p, prob_q):
    """
    Jensen-Shannon divergence = symmetrized KL divergence
    """
    p = prob_p
    q = prob_q
    m = (p+q)/2

    return 0.5*KLdiv(p,m) + 0.5*KLdiv(q,m)


